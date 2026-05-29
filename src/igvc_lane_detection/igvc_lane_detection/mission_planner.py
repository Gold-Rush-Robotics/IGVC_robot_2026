"""
mission_planner.py
==================

Drives the 4-waypoint IGVC AutoNav obstacle-avoidance mission:

    waypoint 1 — start         (mission entry)
    waypoint 2 — virtual obs.  (skip; stamp into /obstacle_map instead)
    waypoint 3 — virtual obs.  (skip; stamp into /obstacle_map instead)
    waypoint 4 — end           (real Nav2 goal)

The mission begins when the robot crosses an operator-controlled
trigger (a ``std_srvs/Trigger`` call on ``~/start_mission``, or the
``auto_start`` parameter set true).  It ends when Nav2 reports the
final waypoint reached, after which the planner publishes
``lane_follow`` on ``/mission/state`` and the lane-following
``navigator_node`` takes over again.

GPS ↔ odom conversion
---------------------
We treat the *first valid GPS fix* received as the origin of the
ENU frame, which (by design — see ``static_map_to_odom`` in
``lane_segmentation.launch.py``) coincides with the odom origin.
Waypoint lat/lon → (x, y) is then ``gps_to_map`` (WGS-84 flat-earth)
into the ``odom`` frame.

Topics
------
* ``/gps/fix``                       ``sensor_msgs/NavSatFix``        (in)
* ``/front_zed_camera_x/zed_node/odom`` ``nav_msgs/Odometry``        (in)
* ``/mission/state``                 ``std_msgs/String``  (latched, out)
* ``/mission/virtual_obstacles``     ``geometry_msgs/PoseArray`` (latched, out)
* ``/mission/clear_virtual_obstacles`` ``std_msgs/Empty``           (out)

States published on ``/mission/state``:

    ``idle``         — waiting for GPS origin / start trigger
    ``active``       — driving to waypoint 4, lane follower paused
    ``done``         — final goal reached
    ``lane_follow``  — handed back to the lane navigator

Services
--------
* ``~/start_mission``  ``std_srvs/Trigger``  — begin the mission
* ``~/abort_mission``  ``std_srvs/Trigger``  — abort, return to lane follow
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose

# Reuse the navigator's WGS-84 flat-earth converter so all GPS↔odom
# logic stays in one place.
from igvc_lane_detection.navigator import gps_to_map


# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Waypoint:
    lat: float
    lon: float
    kind: str            # 'start' | 'obstacle' | 'end'

    # Filled in after the GPS origin is anchored.
    x: Optional[float] = None
    y: Optional[float] = None


STATE_IDLE        = 'idle'
STATE_ACTIVE      = 'active'
STATE_DONE        = 'done'
STATE_LANE_FOLLOW = 'lane_follow'
# Ramp-handling states.  The lane navigator yields (treats these like the
# GPS-mission 'active' state) while mission_planner aligns the robot with the
# ramp fall line and drives it up via injected NavigateToPose goals.
STATE_RAMP_ALIGN  = 'ramp_align'
STATE_RAMP_CLIMB  = 'ramp_climb'

# States in which a GPS waypoint mission owns navigation; ramp handling is
# suppressed so it never fights the mission.
_GPS_MISSION_STATES = (STATE_ACTIVE, STATE_DONE)


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (rad) about +z from a quaternion."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _pitch_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Pitch (rad) about +y from a quaternion."""
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    return math.asin(sinp)


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion for a pure yaw rotation."""
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


class MissionPlannerNode(Node):

    def __init__(self) -> None:
        super().__init__('mission_planner_node')

        p = self.declare_parameter
        self._frame              = p('frame_id',           'odom').value
        self._gps_topic          = p('gps_topic',          '/gps/fix').value
        self._odom_topic         = p('odom_topic',
            '/front_zed_camera_x/zed_node/odom').value
        self._state_topic        = p('state_topic',        '/mission/state').value
        self._virtual_topic      = p('virtual_obstacles_topic',
                                     '/mission/virtual_obstacles').value
        self._clear_topic        = p('clear_virtual_obstacles_topic',
                                     '/mission/clear_virtual_obstacles').value
        self._auto_start         = bool(p('auto_start',     False).value)
        # If true, fall back to declared lat/lon origin instead of waiting
        # for first GPS fix.  Useful for sim playback.
        self._origin_lat_param   = float(p('origin_lat',    float('nan')).value)
        self._origin_lon_param   = float(p('origin_lon',    float('nan')).value)
        self._arrival_tolerance  = float(p('arrival_tolerance_m', 1.0).value)

        # ── Ramp handling ──────────────────────────────────────────
        self._ramp_enabled       = bool(p('ramp_enabled', True).value)
        self._ramp_status_topic  = p('ramp_status_topic', '/ramp/state').value
        self._imu_topic          = p('imu_topic', '/zed/zed_node/imu/data').value
        # Lane-seg forward slope (deg) that arms ramp handling.
        self._ramp_detect_slope_deg = float(p('ramp_detect_slope_deg', 8.0).value)
        # Min lane-seg confidence (Vector3Stamped.z) to trust the slope reading.
        self._ramp_min_confidence   = float(p('ramp_min_confidence', 0.3).value)
        # IMU pitch (deg) confirming the robot is physically on the ramp, and
        # the pitch below which the ramp is considered cleared.
        self._ramp_confirm_pitch_deg = float(p('ramp_confirm_pitch_deg', 5.0).value)
        self._ramp_exit_pitch_deg    = float(p('ramp_exit_pitch_deg', 3.0).value)
        # Heading error (deg) below which alignment is complete → climb.
        self._ramp_align_yaw_tol_deg = float(p('ramp_align_yaw_tol_deg', 8.0).value)
        # How far ahead (m) up the fall line to place the injected goal.
        self._ramp_climb_distance_m  = float(p('ramp_climb_distance_m', 6.0).value)
        # Re-inject the climb goal at most this often (s).
        self._ramp_goal_period_sec   = float(p('ramp_goal_period_sec', 1.0).value)
        # Slope freshness: ignore lane-seg readings older than this (s).
        self._ramp_status_timeout_sec = float(p('ramp_status_timeout_sec', 1.0).value)
        # Pitch must stay below the exit threshold this long before declaring
        # the ramp cleared (avoids early release at the crest).
        self._ramp_exit_hold_sec      = float(p('ramp_exit_hold_sec', 1.5).value)

        # Waypoints declared as flat arrays so they round-trip through
        # the ROS 2 YAML param loader cleanly.
        wp_lats = list(p('waypoint_lats', [0.0, 0.0, 0.0, 0.0]).value)
        wp_lons = list(p('waypoint_lons', [0.0, 0.0, 0.0, 0.0]).value)
        wp_kinds = list(p('waypoint_kinds',
                          ['start', 'obstacle', 'obstacle', 'end']).value)
        if not (len(wp_lats) == len(wp_lons) == len(wp_kinds)):
            raise ValueError(
                'mission_planner: waypoint_lats / waypoint_lons / '
                'waypoint_kinds must have equal length')
        if len(wp_lats) < 2:
            raise ValueError('mission_planner: need at least 2 waypoints')
        self._waypoints: List[_Waypoint] = [
            _Waypoint(lat=float(la), lon=float(lo), kind=str(k))
            for la, lo, k in zip(wp_lats, wp_lons, wp_kinds)]

        # ── State ──────────────────────────────────────────────────
        self._state = STATE_IDLE
        self._origin_lat: Optional[float] = None
        self._origin_lon: Optional[float] = None
        if (math.isfinite(self._origin_lat_param)
                and math.isfinite(self._origin_lon_param)):
            self._origin_lat = self._origin_lat_param
            self._origin_lon = self._origin_lon_param
            self._anchor_waypoints()
        self._start_requested = bool(self._auto_start)
        self._goal_handle = None
        self._goal_pending = False
        self._reached_end = False
        self._robot_xy: Optional[tuple[float, float]] = None
        self._robot_yaw: Optional[float] = None

        # Ramp runtime state.
        self._ramp_slope_deg   = 0.0
        self._ramp_fall_yaw    = 0.0        # base-frame steepest-ascent heading
        self._ramp_conf        = 0.0
        self._ramp_status_mono: Optional[float] = None
        self._imu_pitch_deg    = 0.0
        self._ramp_goal_handle = None
        self._ramp_goal_mono: Optional[float] = None
        self._ramp_exit_below_since: Optional[float] = None
        self._state_before_ramp = STATE_LANE_FOLLOW

        # ── QoS ────────────────────────────────────────────────────
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── I-O ────────────────────────────────────────────────────
        self._state_pub = self.create_publisher(
            String, self._state_topic, latched)
        self._virtual_pub = self.create_publisher(
            PoseArray, self._virtual_topic, latched)
        self._clear_pub = self.create_publisher(
            Empty, self._clear_topic, 10)

        self.create_subscription(
            NavSatFix, self._gps_topic,
            self._on_gps, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, self._odom_topic, self._on_odom, odom_qos)

        if self._ramp_enabled:
            self.create_subscription(
                Vector3Stamped, self._ramp_status_topic,
                self._on_ramp_status, 10)
            self.create_subscription(
                Imu, self._imu_topic, self._on_imu, qos_profile_sensor_data)

        self._start_srv = self.create_service(
            Trigger, '~/start_mission', self._on_start_srv)
        self._abort_srv = self.create_service(
            Trigger, '~/abort_mission', self._on_abort_srv)

        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 2 Hz tick drives the GPS mission state machine.
        self._timer = self.create_timer(0.5, self._tick)
        # 5 Hz tick drives ramp alignment/climb for snappier heading control.
        if self._ramp_enabled:
            self._ramp_timer = self.create_timer(0.2, self._ramp_tick)

        self._publish_state()
        self.get_logger().info(
            f'mission_planner_node: {len(self._waypoints)} waypoints '
            f'(kinds={[w.kind for w in self._waypoints]}), '
            f'auto_start={self._auto_start}, frame={self._frame}')

    # ── Origin / waypoint setup ───────────────────────────────────────────

    def _anchor_waypoints(self) -> None:
        assert self._origin_lat is not None and self._origin_lon is not None
        for wp in self._waypoints:
            wp.x, wp.y = gps_to_map(
                wp.lat, wp.lon, self._origin_lat, self._origin_lon)
        self.get_logger().info(
            f'mission_planner: origin anchored at '
            f'({self._origin_lat:.6f}, {self._origin_lon:.6f}); '
            + ', '.join(f'wp{i}({w.kind})=({w.x:.1f},{w.y:.1f})'
                        for i, w in enumerate(self._waypoints)))
        self._publish_virtual_obstacles()

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_gps(self, msg: NavSatFix) -> None:
        if msg.status.status < 0:
            return
        if self._origin_lat is None:
            self._origin_lat = msg.latitude
            self._origin_lon = msg.longitude
            self._anchor_waypoints()

    def _on_odom(self, msg: Odometry) -> None:
        self._robot_xy = (msg.pose.pose.position.x,
                          msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self._robot_yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)

    def _on_ramp_status(self, msg: Vector3Stamped) -> None:
        # Vector3Stamped: x=slope_deg, y=fall_line_yaw_rad (base), z=confidence.
        self._ramp_slope_deg = float(msg.vector.x)
        self._ramp_fall_yaw  = float(msg.vector.y)
        self._ramp_conf      = float(msg.vector.z)
        self._ramp_status_mono = time.monotonic()

    def _on_imu(self, msg: Imu) -> None:
        q = msg.orientation
        self._imu_pitch_deg = math.degrees(
            _pitch_from_quat(q.x, q.y, q.z, q.w))

    def _on_start_srv(self, _req, resp):
        if self._state in (STATE_ACTIVE, STATE_DONE):
            resp.success = False
            resp.message = f'mission already in state {self._state}'
            return resp
        self._start_requested = True
        resp.success = True
        resp.message = 'mission start requested'
        return resp

    def _on_abort_srv(self, _req, resp):
        self._abort()
        resp.success = True
        resp.message = 'mission aborted; returned to lane_follow'
        return resp

    # ── Mission control ──────────────────────────────────────────────────

    def _tick(self) -> None:
        # Heart-beat the latched state every few ticks for late subscribers.
        if self._state == STATE_IDLE and self._start_requested:
            self._begin()
            return
        if self._state == STATE_ACTIVE and self._reached_end:
            self._finish()
            return

    def _waypoints_anchored(self) -> bool:
        return all(w.x is not None and w.y is not None for w in self._waypoints)

    def _end_waypoint(self) -> Optional[_Waypoint]:
        for w in reversed(self._waypoints):
            if w.kind == 'end':
                return w
        return self._waypoints[-1] if self._waypoints else None

    def _begin(self) -> None:
        if not self._waypoints_anchored():
            self.get_logger().warn(
                'mission_planner: start requested but waypoints not yet '
                'anchored (waiting for GPS origin)',
                throttle_duration_sec=2.0)
            return
        if not self._nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn(
                'mission_planner: navigate_to_pose action server not ready',
                throttle_duration_sec=2.0)
            return

        end = self._end_waypoint()
        if end is None or end.x is None or end.y is None:
            self.get_logger().error('mission_planner: no end waypoint configured')
            return

        self._publish_virtual_obstacles()  # re-stamp so it's fresh

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self._frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(end.x)
        goal.pose.pose.position.y = float(end.y)
        goal.pose.pose.orientation.w = 1.0

        self._goal_pending = True
        self.get_logger().info(
            f'mission_planner: navigating to end waypoint '
            f'({end.x:.2f}, {end.y:.2f}) in {self._frame}')
        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

        self._state = STATE_ACTIVE
        self._publish_state()

    def _on_goal_response(self, future) -> None:
        self._goal_pending = False
        try:
            handle = future.result()
        except Exception as ex:  # pragma: no cover
            self.get_logger().error(f'mission_planner: send_goal failed: {ex}')
            self._abort()
            return
        if not handle.accepted:
            self.get_logger().error('mission_planner: NavigateToPose rejected')
            self._abort()
            return
        self._goal_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future) -> None:
        try:
            result = future.result()
            status = result.status
        except Exception as ex:  # pragma: no cover
            self.get_logger().error(f'mission_planner: goal result error: {ex}')
            self._abort()
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('mission_planner: end waypoint reached')
            self._reached_end = True
        else:
            self.get_logger().warn(
                f'mission_planner: NavigateToPose ended with status={status}; '
                'aborting mission')
            self._abort()

    def _finish(self) -> None:
        self._state = STATE_DONE
        self._publish_state()
        # Clear virtual obstacles and hand control back to the lane follower.
        self._clear_pub.publish(Empty())
        self._virtual_pub.publish(PoseArray(header=self._stamped_header()))
        self._state = STATE_LANE_FOLLOW
        self._publish_state()
        self.get_logger().info(
            'mission_planner: handing control back to lane_follow')

    def _abort(self) -> None:
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass
        self._goal_handle = None
        self._goal_pending = False
        self._reached_end = False
        self._clear_pub.publish(Empty())
        self._virtual_pub.publish(PoseArray(header=self._stamped_header()))
        self._state = STATE_LANE_FOLLOW
        self._publish_state()

    # ── Ramp alignment / climb ────────────────────────────────────────────

    def _ramp_tick(self) -> None:
        if not self._ramp_enabled:
            return
        # A GPS waypoint mission always wins; never interfere with it.
        if self._state in _GPS_MISSION_STATES:
            return

        now = time.monotonic()
        fresh = (self._ramp_status_mono is not None
                 and (now - self._ramp_status_mono) <= self._ramp_status_timeout_sec)
        slope = self._ramp_slope_deg if fresh else 0.0
        conf  = self._ramp_conf if fresh else 0.0
        pitch = abs(self._imu_pitch_deg)

        if self._state in (STATE_IDLE, STATE_LANE_FOLLOW):
            armed = (slope >= self._ramp_detect_slope_deg
                     and conf >= self._ramp_min_confidence)
            confirmed = pitch >= self._ramp_confirm_pitch_deg
            if armed and confirmed:
                self._enter_ramp(now)
            return

        if self._state == STATE_RAMP_ALIGN:
            self._ramp_align_step(now, fresh)
            return

        if self._state == STATE_RAMP_CLIMB:
            self._ramp_climb_step(now, fresh, pitch)
            return

    def _enter_ramp(self, now: float) -> None:
        if self._robot_xy is None or self._robot_yaw is None:
            self.get_logger().warn(
                'mission_planner: ramp detected but no odom pose yet',
                throttle_duration_sec=2.0)
            return
        self._state_before_ramp = (
            self._state if self._state == STATE_LANE_FOLLOW else STATE_LANE_FOLLOW)
        self._ramp_exit_below_since = None
        self._state = STATE_RAMP_ALIGN
        self._publish_state()
        self.get_logger().info(
            f'mission_planner: ramp detected (slope={self._ramp_slope_deg:.1f}°, '
            f'pitch={self._imu_pitch_deg:.1f}°) → aligning to fall line')
        self._send_ramp_goal(now, fresh=True)

    def _ramp_align_step(self, now: float, fresh: bool) -> None:
        # Heading error to the fall line equals the base-frame fall-line yaw.
        yaw_err = abs(self._ramp_fall_yaw) if fresh else 0.0
        if yaw_err <= math.radians(self._ramp_align_yaw_tol_deg):
            self._state = STATE_RAMP_CLIMB
            self._publish_state()
            self.get_logger().info(
                'mission_planner: aligned with ramp → climbing')
            self._send_ramp_goal(now, fresh=True)
            return
        self._maybe_refresh_ramp_goal(now)

    def _ramp_climb_step(self, now: float, fresh: bool, pitch: float) -> None:
        # Keep steering up the (re-fitted) fall line.
        self._maybe_refresh_ramp_goal(now)
        # Exit once the chassis has flattened out for a sustained period.
        if pitch < self._ramp_exit_pitch_deg:
            if self._ramp_exit_below_since is None:
                self._ramp_exit_below_since = now
            elif (now - self._ramp_exit_below_since) >= self._ramp_exit_hold_sec:
                self._exit_ramp()
        else:
            self._ramp_exit_below_since = None

    def _maybe_refresh_ramp_goal(self, now: float) -> None:
        if (self._ramp_goal_mono is None
                or (now - self._ramp_goal_mono) >= self._ramp_goal_period_sec):
            self._send_ramp_goal(now, fresh=False)

    def _send_ramp_goal(self, now: float, fresh: bool) -> None:
        if self._robot_xy is None or self._robot_yaw is None:
            return
        if not self._nav_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn(
                'mission_planner: navigate_to_pose not ready for ramp goal',
                throttle_duration_sec=2.0)
            return
        status_fresh = (self._ramp_status_mono is not None
                        and (now - self._ramp_status_mono)
                        <= self._ramp_status_timeout_sec)
        fall = self._ramp_fall_yaw if status_fresh else 0.0
        world_yaw = self._robot_yaw + fall
        gx = self._robot_xy[0] + self._ramp_climb_distance_m * math.cos(world_yaw)
        gy = self._robot_xy[1] + self._ramp_climb_distance_m * math.sin(world_yaw)
        qx, qy, qz, qw = _yaw_to_quat(world_yaw)

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self._frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(gx)
        goal.pose.pose.position.y = float(gy)
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        # Cancel any previous ramp goal before issuing the refreshed one.
        if self._ramp_goal_handle is not None:
            try:
                self._ramp_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self._ramp_goal_handle = None

        self._ramp_goal_mono = now
        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(self._on_ramp_goal_response)

    def _on_ramp_goal_response(self, future) -> None:
        # Ignore stale responses once we've left ramp handling.
        if self._state not in (STATE_RAMP_ALIGN, STATE_RAMP_CLIMB):
            return
        try:
            handle = future.result()
        except Exception as ex:  # pragma: no cover
            self.get_logger().error(f'mission_planner: ramp send_goal failed: {ex}')
            return
        if not handle.accepted:
            self.get_logger().warn('mission_planner: ramp NavigateToPose rejected')
            return
        self._ramp_goal_handle = handle

    def _exit_ramp(self) -> None:
        if self._ramp_goal_handle is not None:
            try:
                self._ramp_goal_handle.cancel_goal_async()
            except Exception:
                pass
        self._ramp_goal_handle = None
        self._ramp_goal_mono = None
        self._ramp_exit_below_since = None
        self._state = self._state_before_ramp or STATE_LANE_FOLLOW
        self._publish_state()
        self.get_logger().info(
            'mission_planner: ramp cleared → handing control back to '
            f'{self._state}')

    # ── Publishers ────────────────────────────────────────────────────────

    def _publish_state(self) -> None:
        self._state_pub.publish(String(data=self._state))

    def _publish_virtual_obstacles(self) -> None:
        msg = PoseArray()
        msg.header = self._stamped_header()
        for wp in self._waypoints:
            if wp.kind != 'obstacle' or wp.x is None or wp.y is None:
                continue
            pose = Pose()
            pose.position.x = float(wp.x)
            pose.position.y = float(wp.y)
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        self._virtual_pub.publish(msg)

    def _stamped_header(self):
        from std_msgs.msg import Header
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = self._frame
        return h


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = MissionPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
