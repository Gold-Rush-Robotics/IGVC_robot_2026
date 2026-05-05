"""Lifecycle-style task runner for IGVC autonomous operation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import Twist

from igvc_task_runner.behavior import TaskBehaviorController
from igvc_task_runner.task_config import (
    TaskProfile,
    load_task_profiles,
    select_task_profile,
)

from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, NavSatFix

from std_msgs.msg import String

from std_srvs.srv import Trigger


class RunnerState(str, Enum):
    """High-level runner states."""

    CONFIGURING = 'configuring'
    READY = 'ready'
    ARMED = 'armed'
    RUNNING = 'running'
    RECOVERING = 'recovering'
    COMPLETED = 'completed'
    FAILED = 'failed'
    ABORTED = 'aborted'
    SAFE_STOPPED = 'safe_stopped'


@dataclass
class HealthCheck:
    """Runtime health check for topic freshness and basic content."""

    name: str
    topic: str
    msg_type: type
    timeout_sec: float
    required: bool = True
    last_msg: Any = None
    last_time_ns: int | None = None

    def observe(self, msg: Any, now_ns: int) -> None:
        """Record the newest message observed for this check."""
        self.last_msg = msg
        self.last_time_ns = now_ns

    def is_fresh(self, now_ns: int) -> bool:
        """Return whether the newest message is recent enough."""
        if self.last_time_ns is None:
            return False
        age_sec = (now_ns - self.last_time_ns) / 1e9
        return age_sec <= self.timeout_sec

    def has_content(self) -> bool:
        """Return whether the newest message has useful content."""
        if self.last_msg is None:
            return False
        if isinstance(self.last_msg, OccupancyGrid):
            return bool(self.last_msg.data) and any(
                cost >= 0 for cost in self.last_msg.data)
        if isinstance(self.last_msg, NavPath):
            return bool(self.last_msg.poses)
        if isinstance(self.last_msg, Image):
            return bool(self.last_msg.data)
        if isinstance(self.last_msg, Odometry):
            return math.isfinite(self.last_msg.pose.pose.position.x)
        if isinstance(self.last_msg, NavSatFix):
            return self.last_msg.status.status >= 0
        if isinstance(self.last_msg, String):
            return bool(self.last_msg.data)
        return True

    def ready(self, now_ns: int) -> bool:
        """Return whether this health check is satisfied."""
        return self.is_fresh(now_ns) and self.has_content()


MESSAGE_TYPES = {
    'Image': Image,
    'NavSatFix': NavSatFix,
    'OccupancyGrid': OccupancyGrid,
    'Odometry': Odometry,
    'Path': NavPath,
    'String': String,
}


DEFAULT_HEALTH_SPECS = {
    'odom': {'topic': '/odom', 'type': 'Odometry', 'timeout_sec': 1.0},
    'front_camera': {
        'topic': '/front_zed_camera_x/rgb/image_raw',
        'type': 'Image',
        'timeout_sec': 2.0,
    },
    'lane_costmap': {
        'topic': '/lane_costmap',
        'type': 'OccupancyGrid',
        'timeout_sec': 1.5,
    },
    'lane_path': {'topic': '/lane_path', 'type': 'Path', 'timeout_sec': 1.5},
    'localization_status': {
        'topic': '/localization_status',
        'type': 'String',
        'timeout_sec': 1.5,
    },
    'gps_fix': {'topic': '/gps/fix', 'type': 'NavSatFix', 'timeout_sec': 2.5},
}


class IGVCTaskRunner(Node):
    """State machine that arms, runs, monitors, and stops IGVC tasks."""

    def __init__(self) -> None:
        """Create the runner, load profiles, and advertise control APIs."""
        super().__init__('igvc_task_runner')
        self._declare_parameters()

        self._robot_mode = self.get_parameter('robot_mode').value
        self._task_mode = self.get_parameter('task_mode').value
        self._selected_task = self.get_parameter('selected_task').value
        self._safe_stop_topics = list(
            self.get_parameter('safe_stop_topics').value)
        self._maneuver_command_topic = self.get_parameter(
            'maneuver_command_topic').value
        self._status_period_sec = float(
            self.get_parameter('status_period_sec').value)
        self._run_log_path = self.get_parameter('run_log_path').value

        task_dir = self.get_parameter('task_config_dir').value
        if not task_dir:
            task_dir = (
                Path(get_package_share_directory('igvc_task_runner'))
                / 'config'
                / 'tasks'
            )
        self._profiles = load_task_profiles(task_dir)
        self._profile: TaskProfile | None = None
        self._behavior: TaskBehaviorController | None = None

        self._state = RunnerState.CONFIGURING
        self._state_reason = 'initializing'
        self._started_ns: int | None = None
        self._last_odom_xy: tuple[float, float] | None = None
        self._distance_m = 0.0
        self._events: list[dict[str, Any]] = []

        self._status_pub = self.create_publisher(
            String, '/igvc/task_status', 10)
        self._maneuver_command_pub = self.create_publisher(
            String, self._maneuver_command_topic, 10)
        self._cmd_publishers = [
            self.create_publisher(Twist, topic, 10)
            for topic in self._safe_stop_topics
        ]

        self._health_checks: dict[str, HealthCheck] = {}
        self._configure_task()
        self._configure_health_checks()

        self.create_service(Trigger, '~/arm', self._on_arm)
        self.create_service(Trigger, '~/start', self._on_start)
        self.create_service(Trigger, '~/pause', self._on_pause)
        self.create_service(Trigger, '~/resume', self._on_resume)
        self.create_service(Trigger, '~/abort', self._on_abort)
        self.create_service(Trigger, '~/safe_stop', self._on_safe_stop)
        self.create_service(
            Trigger, '~/complete_maneuver', self._on_complete_maneuver)

        self.create_timer(self._status_period_sec, self._tick)
        self._transition(RunnerState.READY, 'configured')

    def _declare_parameters(self) -> None:
        self.declare_parameter('robot_mode', 'sim')
        self.declare_parameter('task_mode', 'selected')
        self.declare_parameter('selected_task', 'full_course_2026')
        self.declare_parameter('task_config_dir', '')
        self.declare_parameter('safe_stop_topics', ['/cmd_vel_nav'])
        self.declare_parameter(
            'maneuver_command_topic', '/igvc/maneuver_command')
        self.declare_parameter('status_period_sec', 0.2)
        self.declare_parameter('run_log_path', '')

    def _configure_task(self) -> None:
        self._profile = select_task_profile(
            self._profiles,
            task_mode=self._task_mode,
            selected_task=self._selected_task,
            robot_mode=self._robot_mode,
        )
        self._behavior = TaskBehaviorController(self._profile, self._profiles)
        self._event('task_selected', task_id=self._profile.task_id)

    def _configure_health_checks(self) -> None:
        assert self._profile is not None
        for check_name in self._profile.required_checks:
            spec = DEFAULT_HEALTH_SPECS.get(check_name)
            if spec is None:
                self.get_logger().warn(f'Unknown health check: {check_name}')
                continue
            msg_type = MESSAGE_TYPES[str(spec['type'])]
            check = HealthCheck(
                name=check_name,
                topic=str(spec['topic']),
                msg_type=msg_type,
                timeout_sec=float(spec['timeout_sec']),
            )
            self._health_checks[check_name] = check
            self.create_subscription(
                msg_type,
                check.topic,
                lambda msg, name=check_name: self._observe_health(name, msg),
                10,
            )

    def _observe_health(self, name: str, msg: Any) -> None:
        now_ns = self.get_clock().now().nanoseconds
        self._health_checks[name].observe(msg, now_ns)
        if isinstance(msg, Odometry):
            self._update_distance(msg)

    def _update_distance(self, msg: Odometry) -> None:
        xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if (self._last_odom_xy is not None
                and self._state == RunnerState.RUNNING):
            self._distance_m += math.hypot(
                xy[0] - self._last_odom_xy[0],
                xy[1] - self._last_odom_xy[1],
            )
        self._last_odom_xy = xy

    def _missing_health(self) -> list[str]:
        now_ns = self.get_clock().now().nanoseconds
        return [
            name for name, check in self._health_checks.items()
            if check.required and not check.ready(now_ns)
        ]

    def _on_arm(self, _request, response):
        if self._state not in {RunnerState.READY, RunnerState.SAFE_STOPPED}:
            response.success = False
            response.message = f'Cannot arm from {self._state.value}.'
            return response
        missing = self._missing_health()
        if missing:
            response.success = False
            response.message = 'Missing health checks: ' + ', '.join(missing)
            return response
        self._transition(RunnerState.ARMED, 'operator armed')
        response.success = True
        response.message = 'Runner armed.'
        return response

    def _on_start(self, _request, response):
        if self._state != RunnerState.ARMED:
            response.success = False
            response.message = f'Cannot start from {self._state.value}.'
            return response
        self._started_ns = self.get_clock().now().nanoseconds
        self._distance_m = 0.0
        assert self._behavior is not None
        self._behavior.start(self._started_ns)
        self._transition(RunnerState.RUNNING, 'operator start')
        response.success = True
        response.message = 'Runner started.'
        return response

    def _on_pause(self, _request, response):
        if self._state != RunnerState.RUNNING:
            response.success = False
            response.message = f'Cannot pause from {self._state.value}.'
            return response
        self._publish_safe_stop()
        self._transition(RunnerState.ARMED, 'paused')
        response.success = True
        response.message = 'Runner paused and stopped.'
        return response

    def _on_resume(self, _request, response):
        if self._state != RunnerState.ARMED:
            response.success = False
            response.message = f'Cannot resume from {self._state.value}.'
            return response
        missing = self._missing_health()
        if missing:
            response.success = False
            response.message = 'Missing health checks: ' + ', '.join(missing)
            return response
        self._transition(RunnerState.RUNNING, 'resumed')
        response.success = True
        response.message = 'Runner resumed.'
        return response

    def _on_abort(self, _request, response):
        self._publish_safe_stop()
        if self._behavior is not None:
            self._behavior.stop()
        self._transition(RunnerState.ABORTED, 'operator abort')
        self._write_run_log()
        response.success = True
        response.message = 'Runner aborted and stopped.'
        return response

    def _on_safe_stop(self, _request, response):
        self._publish_safe_stop()
        if self._behavior is not None:
            self._behavior.stop()
        self._transition(RunnerState.SAFE_STOPPED, 'operator safe stop')
        self._write_run_log()
        response.success = True
        response.message = 'Safe stop published.'
        return response

    def _on_complete_maneuver(self, _request, response):
        if self._state not in {RunnerState.RUNNING, RunnerState.ARMED}:
            response.success = False
            response.message = (
                f'Cannot complete maneuver from {self._state.value}.')
            return response
        assert self._behavior is not None
        advanced = self._behavior.complete_current_step(
            self.get_clock().now().nanoseconds)
        if not advanced:
            response.success = False
            response.message = 'No active maneuver to complete.'
            return response
        if self._behavior.is_complete():
            self._publish_safe_stop()
            self._transition(RunnerState.COMPLETED, 'all maneuvers completed')
            self._write_run_log()
            response.message = 'Final maneuver completed.'
        else:
            active = self._behavior.active_step
            assert active is not None
            response.message = f'Advanced to {active.maneuver}.'
        self._event('maneuver_completed')
        response.success = True
        return response

    def _tick(self) -> None:
        if self._state == RunnerState.RUNNING:
            missing = self._missing_health()
            if missing:
                self._publish_safe_stop()
                self._transition(
                    RunnerState.RECOVERING,
                    'health lost: ' + ', '.join(missing),
                )
            elif self._completion_reached():
                self._publish_safe_stop()
                self._transition(
                    RunnerState.COMPLETED,
                    'completion policy reached',
                )
                self._write_run_log()
            else:
                self._tick_behavior()
        elif self._state == RunnerState.RECOVERING:
            missing = self._missing_health()
            if not missing:
                self._transition(RunnerState.ARMED, 'health restored')
        self._publish_status()

    def _tick_behavior(self) -> None:
        assert self._behavior is not None
        status = self._behavior.tick(
            self.get_clock().now().nanoseconds,
            self._distance_m,
        )
        if status is None:
            return
        msg = String()
        msg.data = json.dumps(status.command.to_dict(), sort_keys=True)
        self._maneuver_command_pub.publish(msg)
        if status.command.stop_required:
            self._publish_safe_stop()

    def _completion_reached(self) -> bool:
        assert self._profile is not None
        policy = self._profile.completion
        if self._behavior is not None and self._behavior.is_complete():
            return True
        if policy.policy_type == 'manual':
            return False
        if policy.policy_type == 'distance':
            return self._distance_m >= policy.distance_m
        if policy.policy_type == 'timeout' and self._started_ns is not None:
            elapsed = (
                self.get_clock().now().nanoseconds - self._started_ns) / 1e9
            return elapsed >= policy.timeout_sec
        return False

    def _transition(self, state: RunnerState, reason: str) -> None:
        if self._state == state and self._state_reason == reason:
            return
        old_state = self._state
        self._state = state
        self._state_reason = reason
        self._event(
            'state_transition',
            old=old_state.value,
            new=state.value,
            reason=reason,
        )
        self.get_logger().info(
            f'Task runner {old_state.value} -> {state.value}: {reason}')

    def _event(self, event_type: str, **fields: Any) -> None:
        self._events.append({
            'event': event_type,
            'time_ns': self.get_clock().now().nanoseconds,
            **fields,
        })

    def _publish_safe_stop(self) -> None:
        stop = Twist()
        for publisher in self._cmd_publishers:
            publisher.publish(stop)

    def _publish_status(self) -> None:
        assert self._profile is not None
        status = {
            'state': self._state.value,
            'reason': self._state_reason,
            'robot_mode': self._robot_mode,
            'task_mode': self._task_mode,
            'task_id': self._profile.task_id,
            'distance_m': round(self._distance_m, 3),
            'missing_health': self._missing_health(),
            'behavior': (
                self._behavior.status_dict()
                if self._behavior is not None else None
            ),
        }
        msg = String()
        msg.data = json.dumps(status, sort_keys=True)
        self._status_pub.publish(msg)

    def _write_run_log(self) -> None:
        if not self._run_log_path:
            return
        path = Path(self._run_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'final_state': self._state.value,
            'reason': self._state_reason,
            'distance_m': self._distance_m,
            'behavior': (
                self._behavior.status_dict()
                if self._behavior is not None else None
            ),
            'events': self._events,
        }
        path.write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding='utf-8',
        )


def main(args: list[str] | None = None) -> None:
    """Run the task runner node."""
    rclpy.init(args=args)
    node = IGVCTaskRunner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
