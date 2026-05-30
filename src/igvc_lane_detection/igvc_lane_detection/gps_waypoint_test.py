#!/usr/bin/env python3
"""
Standalone GPS-waypoint test node.

Drives the robot to a single GPS coordinate via Nav2's ``NavigateToPose``
action, ignoring lane lines and obstacle avoidance logic in the higher-level
mission/lane stack — it simply hands Nav2 a goal pose and lets the global
planner take the robot there.  Intended for bring-up / field testing of the
GPS → map localisation chain only.

The target is given as ``target_lat`` / ``target_lon`` parameters.  The goal is
expressed in the ``map`` frame using the same equirectangular ENU conversion
(:func:`navigator.gps_to_map`) the production navigator uses, so the map origin
**must** match the rest of the system:

* Set ``origin_lat`` / ``origin_lon`` to the same values the navigator uses, or
* leave them at 0.0 to anchor the origin to this node's first GPS fix (only
  correct if the navigator also anchors to that same first fix).

Example::

    ros2 run igvc_lane_detection gps_waypoint_test_node --ros-args \
        -p target_lat:=42.678920 -p target_lon:=-83.195610 \
        -p origin_lat:=42.678000 -p origin_lon:=-83.195000
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateThroughPoses
from sensor_msgs.msg import Imu, NavSatFix
from tf2_ros import StaticTransformBroadcaster

from igvc_lane_detection.navigator import gps_to_map


def _wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class GpsWaypointTestNode(Node):
    """
    Two-phase GPS waypoint tester.

    Phase 1 — HEADING INIT
        GPS gives position but no heading, and the ``odom`` frame's yaw is
        whatever the VIO initialised to (not true North).  The robot drives
        straight forward a short distance, then compares the GPS displacement
        direction (true ENU heading) against the odom displacement direction.
        The difference is the ``map -> odom`` yaw offset; combined with a
        translation that pins the robot's map pose to its GPS-ENU pose, this
        node publishes the corrected ``map -> odom`` static transform.

    Phase 2 — NAVIGATE
        Sends the target as a series of evenly spaced intermediate waypoints
        via ``NavigateThroughPoses`` so RPP always has a nearby carrot.
    """

    # State machine
    _S_WAIT       = 'wait'          # waiting for first GPS fix + odom
    _S_CALIB_START = 'calib_start'  # averaging start fixes
    _S_DRIVE      = 'drive'         # driving forward
    _S_CALIB_END  = 'calib_end'     # averaging end fixes
    _S_NAVIGATE   = 'navigate'      # sending waypoints
    _S_DONE       = 'done'

    def __init__(self) -> None:
        super().__init__('gps_waypoint_test')

        self.declare_parameter('target_lat', 42.400510946)   # 5004 Practice Mid
        self.declare_parameter('target_lon', -83.130640432)  # 5004 Practice Mid
        self.declare_parameter('origin_lat', 0.0)
        self.declare_parameter('origin_lon', 0.0)
        self.declare_parameter('gps_topic', '/fix')
        self.declare_parameter('odom_topic', '/front_zed_camera_x/zed_node/odom')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('nav_action', 'navigate_through_poses')
        self.declare_parameter('goal_tolerance_m', 1.0)
        # Re-send the goal at most this often while we wait for arrival.
        self.declare_parameter('resend_period_sec', 5.0)
        # Intermediate waypoints are inserted every this many metres along the
        # straight-line path to the target so RPP always has a nearby carrot.
        self.declare_parameter('waypoint_spacing_m', 3.0)

        # ── Heading-initialisation parameters ──────────────────────────────
        # Skip the drive-forward calibration and assume map==odom (only valid
        # if the robot already starts facing true East / odom is ENU-aligned).
        self.declare_parameter('heading_init', True)
        # Use the ZED magnetometer (IMU orientation) for heading instead of
        # a drive-forward calibration manoeuvre.
        self.declare_parameter('use_mag_heading', True)
        self.declare_parameter('mag_topic', '/front_zed_camera_x/zed_node/imu/data')
        # How far to drive forward to establish heading.  Must be well above
        # the GPS noise floor: ~1 m is the minimum for RTK, use 2-3 m for
        # standard GPS so the displacement dominates position noise.
        self.declare_parameter('calib_distance_m', 1.5)
        # Open-loop forward speed during calibration (m/s).
        self.declare_parameter('calib_speed_mps', 0.3)
        # Seconds of GPS samples to average at the start/end of the drive to
        # beat down single-fix noise.
        self.declare_parameter('calib_settle_sec', 1.0)
        # Topic to publish the calibration drive command on.  Default routes
        # through the velocity_smoother + collision_monitor safety chain.
        self.declare_parameter('drive_cmd_topic', 'cmd_vel_nav')
        # Abort calibration if the GPS-measured displacement is below this —
        # means the robot didn't actually move (blocked) or GPS is too noisy.
        self.declare_parameter('min_gps_displacement_m', 0.5)
        # If, while navigating, the distance to the target grows this far past
        # the closest we've ever been, the heading estimate is wrong and the
        # robot is driving away — cancel, re-calibrate heading, and retry.
        self.declare_parameter('recovery_dist_increase_m', 2.0)
        # Cap re-calibration attempts so a hopeless GPS/heading situation
        # doesn't loop forever.
        self.declare_parameter('max_recoveries', 5)
        # Robot-relative mode: skip GPS entirely and navigate to a fixed pose
        # expressed in the map/odom frame (identity transform).  target_x and
        # target_y are metres forward/lateral from the robot start position.
        self.declare_parameter('use_gps', True)
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)

        self._target_lat = float(self.get_parameter('target_lat').value)
        self._target_lon = float(self.get_parameter('target_lon').value)
        self._origin_lat = float(self.get_parameter('origin_lat').value)
        self._origin_lon = float(self.get_parameter('origin_lon').value)
        self._origin_set = (self._origin_lat != 0.0 or self._origin_lon != 0.0)
        self._gps_topic = self.get_parameter('gps_topic').value
        self._odom_topic = self.get_parameter('odom_topic').value
        self._map_frame = self.get_parameter('map_frame').value
        self._odom_frame = self.get_parameter('odom_frame').value
        self._nav_action = self.get_parameter('nav_action').value
        self._goal_tol = float(self.get_parameter('goal_tolerance_m').value)
        self._resend_period = float(self.get_parameter('resend_period_sec').value)
        self._waypoint_spacing = float(self.get_parameter('waypoint_spacing_m').value)
        self._heading_init = bool(self.get_parameter('heading_init').value)
        self._use_mag_heading = bool(self.get_parameter('use_mag_heading').value)
        self._mag_topic = self.get_parameter('mag_topic').value
        self._calib_distance = float(self.get_parameter('calib_distance_m').value)
        self._calib_speed = float(self.get_parameter('calib_speed_mps').value)
        self._calib_settle = float(self.get_parameter('calib_settle_sec').value)
        self._drive_cmd_topic = self.get_parameter('drive_cmd_topic').value
        self._min_gps_disp = float(self.get_parameter('min_gps_displacement_m').value)
        self._recovery_increase = float(
            self.get_parameter('recovery_dist_increase_m').value)
        self._max_recoveries = int(self.get_parameter('max_recoveries').value)
        self._use_gps = bool(self.get_parameter('use_gps').value)
        self._target_x = float(self.get_parameter('target_x').value)
        self._target_y = float(self.get_parameter('target_y').value)
        if not self._use_gps:
            # Map frame == odom frame (identity).  No GPS origin required.
            self._origin_set = True
            self._heading_init = False

        if self._use_gps and self._target_lat == 0.0 and self._target_lon == 0.0:
            self.get_logger().error(
                'target_lat/target_lon not set — nothing to navigate to. '
                'Pass them with --ros-args -p target_lat:=.. -p target_lon:=..')

        self._latest_fix: Optional[tuple[float, float]] = None
        self._latest_status: int = -1
        self._latest_odom: Optional[tuple[float, float]] = None  # (x, y) in odom
        self._latest_imu_yaw: Optional[float] = None            # yaw from ZED magnetometer
        self._odom_start: Optional[tuple[float, float]] = None  # odom position at nav start
        self._goal_handle = None
        self._goal_pending = False
        self._reached = False
        self._last_send_sec: Optional[float] = None
        # Closest we have ever been to the target, and recovery attempt count.
        self._min_dist_seen: Optional[float] = None
        self._recoveries = 0

        # Calibration bookkeeping
        self._state = self._S_WAIT
        self._settle_fixes: list[tuple[float, float]] = []
        self._settle_deadline: Optional[float] = None
        self._calib_gps_start: Optional[tuple[float, float]] = None
        self._calib_odom_start: Optional[tuple[float, float]] = None
        self._drive_cmd = Twist()  # current open-loop command (zero = stop)

        # map -> odom transform, owned by this node.  Identity until calibrated.
        self._tf_static = StaticTransformBroadcaster(self)
        self._map_to_odom_yaw = 0.0
        self._map_to_odom_t = (0.0, 0.0)
        self._publish_map_to_odom()
        if not self._heading_init:
            self._state = self._S_NAVIGATE

        self._nav = ActionClient(self, NavigateThroughPoses, self._nav_action)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)
        if self._use_gps:
            self.create_subscription(
                NavSatFix, self._gps_topic, self._on_gps, sensor_qos)
        self.create_subscription(
            Odometry, self._odom_topic, self._on_odom, sensor_qos)
        if self._heading_init and self._use_mag_heading:
            self.create_subscription(
                Imu, self._mag_topic, self._on_imu, sensor_qos)

        self._drive_pub = self.create_publisher(Twist, self._drive_cmd_topic, 10)

        # 1 Hz state-machine tick + 10 Hz drive-command republish so the
        # diff_drive controller's cmd_vel timeout never trips mid-calibration.
        self._timer = self.create_timer(1.0, self._tick)
        self._drive_timer = self.create_timer(0.1, self._publish_drive_cmd)
        if self._use_gps:
            self.get_logger().info(
                f'gps_waypoint_test [GPS]: target=({self._target_lat:.6f}, '
                f'{self._target_lon:.6f}); heading_init={self._heading_init}; '
                f'gps={self._gps_topic} odom={self._odom_topic}')
        else:
            self.get_logger().info(
                f'gps_waypoint_test [odom-relative]: target=({self._target_x:.2f},'
                f' {self._target_y:.2f}) m in {self._map_frame}; '
                f'odom={self._odom_topic}')

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _on_gps(self, msg: NavSatFix) -> None:
        if msg.status.status < 0:
            return
        self._latest_status = msg.status.status
        self._latest_fix = (msg.latitude, msg.longitude)
        if not self._origin_set:
            self._origin_lat = msg.latitude
            self._origin_lon = msg.longitude
            self._origin_set = True
            self.get_logger().warn(
                f'GPS origin anchored to first fix '
                f'({self._origin_lat:.6f}, {self._origin_lon:.6f}).')

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _on_imu(self, msg: Imu) -> None:
        """Extract yaw from the ZED IMU orientation (magnetometer-fused)."""
        q = msg.orientation
        # Standard ZYX Euler yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._latest_imu_yaw = math.atan2(siny_cosp, cosy_cosp)

    # ── map -> odom transform ───────────────────────────────────────────────

    def _publish_map_to_odom(self) -> None:
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self._map_frame
        tf.child_frame_id = self._odom_frame
        tf.transform.translation.x = float(self._map_to_odom_t[0])
        tf.transform.translation.y = float(self._map_to_odom_t[1])
        tf.transform.translation.z = 0.0
        tf.transform.rotation.z = math.sin(self._map_to_odom_yaw * 0.5)
        tf.transform.rotation.w = math.cos(self._map_to_odom_yaw * 0.5)
        self._tf_static.sendTransform(tf)

    # ── Open-loop drive ─────────────────────────────────────────────────────

    def _publish_drive_cmd(self) -> None:
        # Only actively command motion while driving; otherwise stay silent so
        # we don't fight Nav2's controller once it takes over.
        if self._state == self._S_DRIVE:
            self._drive_pub.publish(self._drive_cmd)
        elif self._state in (self._S_CALIB_START, self._S_CALIB_END):
            self._drive_pub.publish(Twist())  # hold still while settling

    # ── State machine ────────────────────────────────────────────────────────

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _tick(self) -> None:
        if self._use_gps and self._target_lat == 0.0 and self._target_lon == 0.0:
            return

        if self._state == self._S_WAIT:
            self._tick_wait()
        elif self._state == self._S_CALIB_START:
            self._tick_calib_settle(start=True)
        elif self._state == self._S_DRIVE:
            self._tick_drive()
        elif self._state == self._S_CALIB_END:
            self._tick_calib_settle(start=False)
        elif self._state == self._S_NAVIGATE:
            self._tick_navigate()
        # _S_DONE: nothing to do

    def _tick_wait(self) -> None:
        if not self._origin_set or self._latest_fix is None or self._latest_odom is None:
            self.get_logger().info(
                'Waiting for GPS fix + odom before calibrating…',
                throttle_duration_sec=3.0)
            return
        if not self._heading_init:
            self._enter_navigate()
            return
        if self._use_mag_heading:
            self._calibrate_from_mag()
            return
        if self._latest_status < 1:
            self.get_logger().warn(
                'GPS has no augmentation (status < 1); heading estimate from a '
                f'{self._calib_distance:.1f} m drive may be poor. Consider RTK '
                'or a longer calib_distance_m.', throttle_duration_sec=5.0)
        self.get_logger().info(
            f'Starting heading calibration: averaging GPS for '
            f'{self._calib_settle:.1f} s, then driving {self._calib_distance:.1f} m.')
        self._begin_settle()
        self._state = self._S_CALIB_START

    def _calibrate_from_mag(self) -> None:
        """Compute map->odom transform from ZED magnetometer heading + GPS fix."""
        if self._latest_imu_yaw is None:
            self.get_logger().info(
                f'Waiting for magnetometer heading on {self._mag_topic}…',
                throttle_duration_sec=3.0)
            return
        yaw = self._latest_imu_yaw
        gps_e, gps_n = gps_to_map(
            self._latest_fix[0], self._latest_fix[1],
            self._origin_lat, self._origin_lon)
        ox, oy = self._latest_odom
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        # t = gps_pos - R(yaw) * odom_pos  (pins robot map pose to GPS-ENU pose)
        tx = gps_e - (cos_y * ox - sin_y * oy)
        ty = gps_n - (sin_y * ox + cos_y * oy)
        self._map_to_odom_yaw = yaw
        self._map_to_odom_t = (tx, ty)
        self._publish_map_to_odom()
        self.get_logger().info(
            f'Heading from magnetometer: {math.degrees(yaw):.1f}° '
            f'(odom offset t=({tx:.2f}, {ty:.2f}) m). Navigating to target.')
        self._enter_navigate()

    def _begin_settle(self) -> None:
        self._settle_fixes = []
        self._settle_deadline = self._now() + self._calib_settle

    def _collect_settle(self) -> Optional[tuple[float, float]]:
        """Accumulate fixes until the settle window closes; return the mean."""
        if self._latest_fix is not None:
            self._settle_fixes.append(self._latest_fix)
        if self._settle_deadline is None or self._now() < self._settle_deadline:
            return None
        if not self._settle_fixes:
            return None
        n = len(self._settle_fixes)
        mean_lat = sum(f[0] for f in self._settle_fixes) / n
        mean_lon = sum(f[1] for f in self._settle_fixes) / n
        return (mean_lat, mean_lon)

    def _tick_calib_settle(self, start: bool) -> None:
        mean = self._collect_settle()
        if mean is None:
            return
        gps_e, gps_n = gps_to_map(mean[0], mean[1],
                                  self._origin_lat, self._origin_lon)
        if start:
            self._calib_gps_start = (gps_e, gps_n)
            self._calib_odom_start = self._latest_odom
            self.get_logger().info(
                f'Calibration start logged at map ({gps_e:.2f}, {gps_n:.2f}); '
                f'driving forward at {self._calib_speed:.2f} m/s.')
            self._drive_cmd = Twist()
            self._drive_cmd.linear.x = self._calib_speed
            self._state = self._S_DRIVE
        else:
            self._finish_calibration((gps_e, gps_n), self._latest_odom)

    def _tick_drive(self) -> None:
        if self._calib_odom_start is None or self._latest_odom is None:
            return
        dx = self._latest_odom[0] - self._calib_odom_start[0]
        dy = self._latest_odom[1] - self._calib_odom_start[1]
        travelled = math.hypot(dx, dy)
        if travelled < self._calib_distance:
            self.get_logger().info(
                f'Calibrating: {travelled:.2f} / {self._calib_distance:.2f} m',
                throttle_duration_sec=1.0)
            return
        # Reached calibration distance — stop and settle for the end sample.
        self._drive_cmd = Twist()
        self.get_logger().info(
            f'Drove {travelled:.2f} m; stopping to log end fix.')
        self._begin_settle()
        self._state = self._S_CALIB_END

    def _finish_calibration(self, gps_end: tuple[float, float],
                            odom_end: Optional[tuple[float, float]]) -> None:
        if (self._calib_gps_start is None or self._calib_odom_start is None
                or odom_end is None):
            self.get_logger().error(
                'Calibration failed: missing start/end samples. Retrying.')
            self._state = self._S_WAIT
            return

        g_de = gps_end[0] - self._calib_gps_start[0]
        g_dn = gps_end[1] - self._calib_gps_start[1]
        o_dx = odom_end[0] - self._calib_odom_start[0]
        o_dy = odom_end[1] - self._calib_odom_start[1]
        gps_disp = math.hypot(g_de, g_dn)
        odom_disp = math.hypot(o_dx, o_dy)

        if gps_disp < self._min_gps_disp:
            self.get_logger().error(
                f'Calibration aborted: GPS moved only {gps_disp:.2f} m '
                f'(< {self._min_gps_disp:.2f} m). Robot blocked or GPS too '
                'noisy. Retrying from scratch.')
            self._state = self._S_WAIT
            return
        if odom_disp < 0.5 * self._calib_distance:
            self.get_logger().warn(
                f'Odom moved only {odom_disp:.2f} m vs commanded '
                f'{self._calib_distance:.2f} m — collision monitor may have '
                'stopped the robot. Heading estimate may be unreliable.')

        true_heading = math.atan2(g_dn, g_de)
        odom_heading = math.atan2(o_dy, o_dx)
        yaw = _wrap(true_heading - odom_heading)

        # map_p = R(yaw) * odom_p + t, pinned so robot's map pose == GPS-ENU pose.
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        tx = gps_end[0] - (cos_y * odom_end[0] - sin_y * odom_end[1])
        ty = gps_end[1] - (sin_y * odom_end[0] + cos_y * odom_end[1])

        self._map_to_odom_yaw = yaw
        self._map_to_odom_t = (tx, ty)
        self._publish_map_to_odom()
        self.get_logger().info(
            f'Heading calibrated: true={math.degrees(true_heading):.1f}°, '
            f'odom={math.degrees(odom_heading):.1f}°, '
            f'map->odom yaw={math.degrees(yaw):.1f}°, '
            f't=({tx:.2f}, {ty:.2f}). Navigating to target.')
        self._enter_navigate()

    def _enter_navigate(self) -> None:
        """Reset goal/progress state and switch to the navigation phase."""
        self._drive_cmd = Twist()
        self._goal_handle = None
        self._goal_pending = False
        self._last_send_sec = None
        self._min_dist_seen = None
        # Capture starting odom position so it can be added to odom-relative goals.
        if self._odom_start is None and self._latest_odom is not None:
            self._odom_start = self._latest_odom
            self.get_logger().info(
                f'Odom start captured: ({self._odom_start[0]:.3f}, '
                f'{self._odom_start[1]:.3f}) m')
        self._state = self._S_NAVIGATE

    # ── Navigation ───────────────────────────────────────────────────────────

    def _tick_navigate(self) -> None:
        # Re-publish map->odom so the static transform survives late TF
        # subscribers that joined after the one-shot send above.
        self._publish_map_to_odom()

        if self._reached:
            return
        if self._use_gps:
            if not self._origin_set or self._latest_fix is None:
                return
            cur_e, cur_n = gps_to_map(self._latest_fix[0], self._latest_fix[1],
                                      self._origin_lat, self._origin_lon)
            tgt_e, tgt_n = gps_to_map(self._target_lat, self._target_lon,
                                      self._origin_lat, self._origin_lon)
        else:
            if self._latest_odom is None:
                return
            cur_e, cur_n = self._latest_odom
            # Add starting odom position to the goal so the target is correct
            # even when odom doesn't start at (0, 0).
            start_x = self._odom_start[0] if self._odom_start is not None else 0.0
            start_y = self._odom_start[1] if self._odom_start is not None else 0.0
            tgt_e, tgt_n = self._target_x + start_x, self._target_y + start_y
        dist = math.hypot(tgt_e - cur_e, tgt_n - cur_n)
        if dist <= self._goal_tol:
            self.get_logger().info(
                f'Target reached (dist={dist:.2f} m ≤ {self._goal_tol:.2f} m).')
            self._reached = True
            self._state = self._S_DONE
            return

        # Progress watchdog: if we drift well past the closest approach we've
        # ever made, the heading estimate is wrong and the robot is driving
        # away from the goal.  Re-calibrate heading and try again.
        if self._min_dist_seen is None or dist < self._min_dist_seen:
            self._min_dist_seen = dist
        elif (dist - self._min_dist_seen) > self._recovery_increase:
            self._trigger_recovery(dist)
            return

        if self._goal_pending:
            return

        now_sec = self._now()
        if (self._goal_handle is not None and self._last_send_sec is not None
                and (now_sec - self._last_send_sec) < self._resend_period):
            return

        self._send_goal(tgt_e, tgt_n, cur_e, cur_n, dist)

    def _trigger_recovery(self, dist: float) -> None:
        """Cancel the active goal and re-run heading calibration."""
        if self._recoveries >= self._max_recoveries:
            self.get_logger().error(
                f'Distance still increasing ({dist:.1f} m) after '
                f'{self._recoveries} recovery attempts; giving up. Check GPS '
                'fix quality and that odom is moving.')
            self._drive_cmd = Twist()
            self._state = self._S_DONE
            return
        self._recoveries += 1
        self.get_logger().warn(
            f'Moving AWAY from target: dist={dist:.1f} m vs best '
            f'{self._min_dist_seen:.1f} m. Heading estimate likely wrong — '
            f're-calibrating (attempt {self._recoveries}/{self._max_recoveries}).')
        # Stop Nav2 driving the wrong way.
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:  # pragma: no cover
                pass
        self._goal_handle = None
        self._goal_pending = False
        self._last_send_sec = None
        if self._use_gps:
            # Fresh forward drive to re-estimate heading.
            self._calib_gps_start = None
            self._calib_odom_start = None
            self._begin_settle()
            self._state = self._S_CALIB_START
        else:
            # In odom-relative mode there is no heading to re-estimate;
            # re-enter navigation so the goal is re-sent.
            self._enter_navigate()

    # ── Goal handling ──────────────────────────────────────────────────────

    def _send_goal(self, tgt_e: float, tgt_n: float,
                   cur_e: float, cur_n: float, dist: float) -> None:
        if not self._nav.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(
                f'{self._nav_action} action server not available yet.',
                throttle_duration_sec=3.0)
            return

        # Build intermediate waypoints spaced every waypoint_spacing_m metres
        # along the straight line to the target.  Each waypoint faces the final
        # target so the robot doesn't spin between sub-goals.
        yaw = math.atan2(tgt_n - cur_n, tgt_e - cur_e)
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)

        spacing = max(0.5, self._waypoint_spacing)
        n_steps = max(1, round(dist / spacing))
        now = self.get_clock().now().to_msg()
        poses: list[PoseStamped] = []
        for i in range(1, n_steps + 1):
            frac = i / n_steps
            p = PoseStamped()
            p.header.frame_id = self._map_frame
            p.header.stamp = now
            p.pose.position.x = cur_e + frac * (tgt_e - cur_e)
            p.pose.position.y = cur_n + frac * (tgt_n - cur_n)
            p.pose.orientation.z = qz
            p.pose.orientation.w = qw
            poses.append(p)

        goal = NavigateThroughPoses.Goal()
        goal.poses = poses

        self._goal_pending = True
        self._last_send_sec = self._now()
        mode = 'GPS' if self._use_gps else 'odom-rel'
        self.get_logger().info(
            f'Sending [{mode}] goal: map ({tgt_e:.2f}, {tgt_n:.2f}), '
            f'dist={dist:.2f} m via {len(poses)} waypoint(s) '
            f'(spacing={spacing:.1f} m)')
        future = self._nav.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future) -> None:
        self._goal_pending = False
        try:
            handle = future.result()
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f'send_goal failed: {exc}')
            return
        if not handle.accepted:
            self.get_logger().warn('NavigateThroughPoses goal rejected.')
            return
        self._goal_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future) -> None:
        self._goal_handle = None
        try:
            future.result()
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f'goal result error: {exc}')
            return
        # Arrival is confirmed by the distance check in _tick_navigate; the
        # result just frees us to (re)send if Nav2 finished short of tolerance.
        self.get_logger().info(
            'NavigateThroughPoses goal finished; re-checking distance.')



def main(args=None) -> None:
    rclpy.init(args=args)
    node = GpsWaypointTestNode()
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
