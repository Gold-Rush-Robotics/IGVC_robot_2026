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

from yolo_msgs.msg import DetectionArray


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

# Maps IGVC-specific YOLO class names to relevant task IDs.
# Class names must match what the custom IGVC YOLO model outputs.
DETECTION_CLASS_TO_TASK_AFFINITIES: dict[str, list[str]] = {
    'stop_sign': ['ii1_stop_sign_detection'],
    'pedestrian': [
        'i1_pedestrian_detection',
        'v1_unobstructed_static_pedestrian_detection',
        'v2_obstructed_dynamic_pedestrian_detection',
        'v3_static_pedestrian_lane_changing',
    ],
    'person': [
        'i1_pedestrian_detection',
        'v1_unobstructed_static_pedestrian_detection',
        'v2_obstructed_dynamic_pedestrian_detection',
        'v3_static_pedestrian_lane_changing',
    ],
    'pothole': ['vii1_pothole_detection'],
    'tire': ['i2_tire_detection'],
    'barrel': ['v4_obstacle_detection_lane_changing'],
    'cone': ['v4_obstacle_detection_lane_changing'],
    'obstacle': ['v4_obstacle_detection_lane_changing'],
    'lane_line': ['iii1_lane_keeping', 'full_course_2026'],
    'white_lane': ['iii1_lane_keeping', 'full_course_2026'],
    'curved_lane': ['vi1_curved_road_lane_keeping', 'vi2_curved_road_lane_changing'],
    'parking_space': ['iv1_parking_pull_out', 'iv2_parking_pull_in', 'iv3_parking_parallel'],
    'intersection': ['iii2_left_turn', 'iii3_right_turn', 'q3_left_turn', 'q4_right_turn'],
}

# Maps YOLO class names to the override maneuver to inject at runtime.
DETECTION_CLASS_TO_OVERRIDE_MANEUVER: dict[str, str] = {
    'stop_sign': 'stop_at_sign',
    'pedestrian': 'stop_or_yield',
    'person': 'stop_or_yield',
    'pothole': 'obstacle_avoidance',
    'tire': 'lane_change',
    'barrel': 'lane_change',
    'cone': 'lane_change',
    'obstacle': 'obstacle_avoidance',
}

# Override command properties keyed by override maneuver name.
_OVERRIDE_CONTROLLER_MODES: dict[str, str] = {
    'stop_at_sign': 'stop_at_sign',
    'stop_or_yield': 'yield_to_target',
    'obstacle_avoidance': 'avoid_obstacle',
    'lane_change': 'lane_change',
}
_OVERRIDE_SPEED_LIMITS: dict[str, float] = {
    'stop_at_sign': 0.0,
    'stop_or_yield': 0.0,
    'obstacle_avoidance': 0.55,
    'lane_change': 0.7,
}
_OVERRIDE_STOP_REQUIRED: dict[str, bool] = {
    'stop_at_sign': True,
    'stop_or_yield': True,
    'obstacle_avoidance': False,
    'lane_change': False,
}

# Maneuvers that should never be preempted by a YOLO override.
_NO_OVERRIDE_MANEUVERS = frozenset({
    'stop_at_sign',
    'stop_or_yield',
    'parking_pull_in',
    'parking_pull_out',
    'parallel_parking',
    'detect_lanes',
})


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
        self.create_service(Trigger, '~/reconfigure', self._on_reconfigure)

        # YOLO prediction and runtime override state
        self._yolo_detection_buffer: list[tuple[int, str, float]] = []
        self._yolo_override_command: ManeuverCommand | None = None
        self._yolo_override_expires_ns: int | None = None
        self._yolo_override_last_set_ns: int | None = None
        self._last_prediction: dict[str, Any] = {}

        self._prediction_pub = self.create_publisher(
            String, '/igvc/yolo_prediction', 10)

        if self.get_parameter('yolo_enabled').value:
            yolo_topic = str(
                self.get_parameter('yolo_detection_topic').value)
            self.create_subscription(
                DetectionArray, yolo_topic, self._on_detections, 10)
            self.get_logger().info(
                f'YOLO prediction enabled on {yolo_topic}')

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
        # YOLO prediction parameters
        self.declare_parameter('yolo_enabled', True)
        self.declare_parameter('yolo_detection_topic', '/detections')
        self.declare_parameter('yolo_confidence_threshold', 0.65)
        self.declare_parameter('yolo_prediction_window_sec', 3.0)
        self.declare_parameter('yolo_startup_prediction', True)
        self.declare_parameter('yolo_runtime_overrides', True)
        self.declare_parameter('yolo_override_duration_sec', 4.0)
        self.declare_parameter('yolo_override_cooldown_sec', 8.0)

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
        now_ns = self.get_clock().now().nanoseconds
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
                if self.get_parameter('yolo_runtime_overrides').value:
                    self._check_yolo_runtime_overrides(now_ns)
                self._tick_behavior()
        elif self._state == RunnerState.RECOVERING:
            missing = self._missing_health()
            if not missing:
                self._transition(RunnerState.ARMED, 'health restored')
        elif self._state in (
            RunnerState.READY,
            RunnerState.ARMED,
        ):
            if self.get_parameter('yolo_startup_prediction').value:
                self._publish_yolo_prediction(now_ns)
        self._publish_status()

    def _tick_behavior(self) -> None:
        assert self._behavior is not None
        now_ns = self.get_clock().now().nanoseconds
        status = self._behavior.tick(now_ns, self._distance_m)
        if status is None:
            return
        # Apply YOLO override if one is active and has not expired.
        if (self._yolo_override_command is not None
                and self._yolo_override_expires_ns is not None
                and now_ns <= self._yolo_override_expires_ns):
            command = self._yolo_override_command
        else:
            if (self._yolo_override_command is not None
                    and self._yolo_override_expires_ns is not None
                    and now_ns > self._yolo_override_expires_ns):
                self._yolo_override_command = None
                self._yolo_override_expires_ns = None
            command = status.command
        msg = String()
        msg.data = json.dumps(command.to_dict(), sort_keys=True)
        self._maneuver_command_pub.publish(msg)
        if command.stop_required:
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
        yolo_active = (self._yolo_override_command is not None
                       and self._yolo_override_expires_ns is not None
                       and self.get_clock().now().nanoseconds
                       <= self._yolo_override_expires_ns)
        status = {
            'state': self._state.value,
            'reason': self._state_reason,
            'robot_mode': self._robot_mode,
            'task_mode': self._task_mode,
            'task_id': self._profile.task_id,
            'distance_m': round(self._distance_m, 3),
            'missing_health': self._missing_health(),
            'yolo_override_active': yolo_active,
            'yolo_override_maneuver': (
                self._yolo_override_command.maneuver
                if yolo_active and self._yolo_override_command else None
            ),
            'yolo_prediction': self._last_prediction,
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


    # ------------------------------------------------------------------
    # YOLO detection subscription and prediction helpers
    # ------------------------------------------------------------------

    def _on_detections(self, msg: DetectionArray) -> None:
        """Append incoming YOLO detections to the rolling buffer."""
        now_ns = self.get_clock().now().nanoseconds
        for det in msg.detections:
            self._yolo_detection_buffer.append(
                (now_ns, det.class_name.lower(), float(det.score)))
        self._prune_detection_buffer(now_ns)

    def _prune_detection_buffer(self, now_ns: int) -> None:
        """Remove buffer entries older than the prediction window."""
        window_ns = int(
            float(self.get_parameter('yolo_prediction_window_sec').value)
            * 1e9)
        cutoff = now_ns - window_ns
        self._yolo_detection_buffer = [
            entry for entry in self._yolo_detection_buffer
            if entry[0] >= cutoff
        ]

    def _compute_startup_prediction(self, now_ns: int) -> dict[str, Any]:
        """Score all task profiles by detected class evidence.

        Returns a ranked prediction dict suitable for JSON serialisation.
        """
        self._prune_detection_buffer(now_ns)
        threshold = float(
            self.get_parameter('yolo_confidence_threshold').value)

        class_counts: dict[str, int] = {}
        for _t, class_name, score in self._yolo_detection_buffer:
            if score >= threshold:
                class_counts[class_name] = (
                    class_counts.get(class_name, 0) + 1)

        if not class_counts:
            return {}

        task_scores: dict[str, float] = {}
        for class_name, count in class_counts.items():
            for task_id in DETECTION_CLASS_TO_TASK_AFFINITIES.get(
                    class_name, []):
                task_scores[task_id] = (
                    task_scores.get(task_id, 0.0) + count)

        if not task_scores:
            return {}

        total = float(sum(class_counts.values()))
        ranked = sorted(
            task_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_with_conf = [
            {'task_id': tid, 'confidence': round(score / total, 3)}
            for tid, score in ranked
        ]
        top = ranked_with_conf[0]
        return {
            'mode': 'startup_suggestion',
            'suggested_task_id': top['task_id'],
            'confidence': top['confidence'],
            'detection_evidence': class_counts,
            'ranked_tasks': ranked_with_conf[:5],
        }

    def _publish_yolo_prediction(self, now_ns: int) -> None:
        """Compute, cache, and publish the current task prediction."""
        prediction = self._compute_startup_prediction(now_ns)
        if not prediction:
            return
        self._last_prediction = prediction

        # Auto-select task when running in auto mode with high confidence.
        if (self._task_mode == 'auto'
                and self._state == RunnerState.READY
                and prediction.get('confidence', 0.0) >= float(
                    self.get_parameter('yolo_confidence_threshold').value)):
            suggested = prediction['suggested_task_id']
            if (self._profile is not None
                    and suggested != self._profile.task_id
                    and suggested in self._profiles):
                self._selected_task = suggested
                self._configure_task()
                self._configure_health_checks()
                self.get_logger().info(
                    f'YOLO auto-selected task: {suggested} '
                    f'(confidence={prediction["confidence"]:.2f})')

        msg = String()
        msg.data = json.dumps(prediction, sort_keys=True)
        self._prediction_pub.publish(msg)

    def _check_yolo_runtime_overrides(self, now_ns: int) -> None:
        """Check live YOLO detections and set a maneuver override if warranted.

        Overrides are suppressed when within the cooldown window, when the
        current maneuver is already a priority stop/yield, or when no
        relevant detection is present above the confidence threshold.
        """
        # Respect cooldown between successive overrides.
        cooldown_ns = int(
            float(self.get_parameter('yolo_override_cooldown_sec').value)
            * 1e9)
        if (self._yolo_override_last_set_ns is not None
                and (now_ns - self._yolo_override_last_set_ns) < cooldown_ns):
            return

        if self._behavior is None:
            return
        step = self._behavior.active_step
        if step is None or step.maneuver in _NO_OVERRIDE_MANEUVERS:
            return

        threshold = float(
            self.get_parameter('yolo_confidence_threshold').value)
        window_ns = int(
            float(self.get_parameter('yolo_prediction_window_sec').value)
            * 1e9)

        # Collect best score per class seen in the recent window.
        recent_classes: dict[str, float] = {}
        for t, class_name, score in self._yolo_detection_buffer:
            if (now_ns - t) <= window_ns and score >= threshold:
                if score > recent_classes.get(class_name, 0.0):
                    recent_classes[class_name] = score

        # Evaluate override candidates in priority order.
        override_maneuver: str | None = None
        for class_name in (
            'stop_sign',
            'pedestrian',
            'person',
            'pothole',
            'tire',
            'barrel',
            'cone',
            'obstacle',
        ):
            if class_name not in recent_classes:
                continue
            candidate = DETECTION_CLASS_TO_OVERRIDE_MANEUVER.get(class_name)
            if candidate and candidate != step.maneuver:
                override_maneuver = candidate
                break

        if override_maneuver is None:
            return

        self._yolo_override_command = ManeuverCommand(
            maneuver=override_maneuver,
            controller_mode=_OVERRIDE_CONTROLLER_MODES[override_maneuver],
            targets=step.perception_targets,
            speed_limit_mps=_OVERRIDE_SPEED_LIMITS[override_maneuver],
            stop_required=_OVERRIDE_STOP_REQUIRED[override_maneuver],
            notes=('yolo_override',),
        )
        duration_ns = int(
            float(self.get_parameter('yolo_override_duration_sec').value)
            * 1e9)
        self._yolo_override_expires_ns = now_ns + duration_ns
        self._yolo_override_last_set_ns = now_ns
        self._event('yolo_override', maneuver=override_maneuver)
        self.get_logger().info(
            f'YOLO runtime override: {step.maneuver} → {override_maneuver}')

    # ------------------------------------------------------------------
    # Reconfigure service — hot-swap the active task profile
    # ------------------------------------------------------------------

    def _on_reconfigure(self, _request, response):
        """Re-load the task profile from the current selected_task parameter.

        Only permitted from READY or SAFE_STOPPED states so the runner is
        never reconfigured mid-run.
        """
        if self._state not in {
            RunnerState.READY,
            RunnerState.SAFE_STOPPED,
        }:
            response.success = False
            response.message = (
                f'Cannot reconfigure from {self._state.value}. '
                'Must be READY or SAFE_STOPPED.')
            return response
        self._task_mode = self.get_parameter('task_mode').value
        self._selected_task = self.get_parameter('selected_task').value
        try:
            self._configure_task()
            self._configure_health_checks()
        except (KeyError, ValueError) as exc:
            response.success = False
            response.message = f'Reconfigure failed: {exc}'
            return response
        assert self._profile is not None
        response.success = True
        response.message = f'Reconfigured to task: {self._profile.task_id}'
        self.get_logger().info(response.message)
        return response


def main(args: list[str] | None = None) -> None:
    """Run the task runner node."""
    rclpy.init(args=args)
    node = IGVCTaskRunner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
