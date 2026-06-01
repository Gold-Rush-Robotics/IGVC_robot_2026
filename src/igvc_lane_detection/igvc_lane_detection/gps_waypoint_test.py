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
from collections import deque
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateThroughPoses
from sensor_msgs.msg import Imu, MagneticField, NavSatFix
from tf2_ros import StaticTransformBroadcaster

from igvc_lane_detection.navigator import gps_to_map


def _wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_quat(q: tuple[float, float, float, float]) -> float:
    """Yaw (rotation about +z) from a (x, y, z, w) quaternion."""
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_vec_by_quat(q: tuple[float, float, float, float],
                        v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate vector ``v`` by quaternion ``q`` (x, y, z, w):  v' = q v q*."""
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least-squares line fit.  Returns (slope, intercept)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0, mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / den
    return slope, mean_y - slope * mean_x


class GpsWaypointTestNode(Node):
    """
    Two-phase GPS waypoint tester.

    Phase 1 — MAGNETOMETER HEADING INIT (stationary)
        GPS gives position but no heading, and the ``odom`` frame's yaw is
        whatever the VIO initialised to (not true North).  Rather than driving
        a calibration leg, the node averages the ZED2i magnetometer for a few
        seconds while stationary and rotates that body-frame field into the
        ``odom`` frame using the fast, low-noise fused IMU orientation (tilt
        compensation falls out of the full 3-D rotation).  The bearing of
        magnetic north, corrected by the magnetic declination, gives the true
        ENU heading, which pins the ``map -> odom`` static transform.  The IMU
        then maintains odom yaw for the rest of the run.

    Phase 2 — NAVIGATE
        Sends the target as a series of evenly spaced intermediate waypoints
        via ``NavigateThroughPoses`` so RPP always has a nearby carrot.
    """

    # State machine
    _S_WAIT     = 'wait'      # waiting for first GPS fix + odom + mag + imu
    _S_CALIB    = 'calib'     # averaging magnetometer + GPS while stationary
    _S_NAVIGATE = 'navigate'  # sending waypoints
    _S_DONE     = 'done'

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
        # Intermediate waypoints are inserted every this many meters along the
        # straight-line path to the target so RPP always has a nearby carrot.
        self.declare_parameter('waypoint_spacing_m', 3.0)

        # ── Heading-initialisation parameters ──────────────────────────────
        # When true, derive the absolute (true-North) heading from the ZED2i
        # magnetometer once, while stationary, and pin the map->odom transform.
        # When false, assume map==odom (only valid if odom is already
        # ENU-aligned / the robot starts facing true East).
        self.declare_parameter('heading_init', True)
        # ZED2i magnetometer + fused-IMU topics.  The magnetometer supplies the
        # absolute heading reference; the faster, lower-noise fused IMU
        # orientation tilt-compensates it and then maintains odom yaw.
        self.declare_parameter(
            'mag_topic', '/front_zed_camera_x/zed_node/imu/mag')
        self.declare_parameter(
            'imu_topic', '/front_zed_camera_x/zed_node/imu/data')
        # Seconds to average the magnetometer + GPS while stationary.
        self.declare_parameter('mag_settle_sec', 2.0)
        # Minimum magnetometer samples required before a heading is accepted.
        self.declare_parameter('mag_min_samples', 20)
        # Magnetic declination (deg, +east) for the venue.  Rochester, MI is
        # roughly -7.5 (7.5° west).  Converts magnetic north to true north.
        self.declare_parameter('mag_declination_deg', 0.0)
        # One-time mounting/frame calibration to absorb the constant offset
        # between the magnetometer sensor frame and base_link.  Tune in the
        # field so the reported map->odom yaw matches a known heading.
        self.declare_parameter('mag_heading_offset_deg', 0.0)
        # Hard-iron offsets (same units as the MagneticField message, Tesla)
        # subtracted from the raw horizontal magnetometer components.
        self.declare_parameter('mag_offset_x', 0.0)
        self.declare_parameter('mag_offset_y', 0.0)
        # If, while navigating, the distance to the target grows this far past
        # the closest we've ever been, the heading estimate is wrong and the
        # robot is driving away — cancel, re-estimate heading, and retry.
        self.declare_parameter('recovery_dist_increase_m', 2.0)
        # Cap re-calibration attempts so a hopeless GPS/heading situation
        # doesn't loop forever.
        self.declare_parameter('max_recoveries', 5)
        # ── Closed-loop GPS regression ──────────────────────────────────────
        # Rolling time window (seconds) of GPS fixes fed into the linear
        # regression that smooths the robot's current position estimate.
        self.declare_parameter('gps_regression_window_sec', 5.0)
        # Minimum number of fixes required before regression is used;
        # falls back to the raw latest fix while the buffer is filling.
        self.declare_parameter('gps_min_samples', 3)
        # Resend Nav2 goal whenever the robot's regressed GPS position has
        # shifted at least this far (meters) from where the last goal was sent.
        # Keeps the goal current as the robot moves through GPS space.
        self.declare_parameter('goal_update_distance_m', 0.5)
        # When false, do not send a fresh NavigateThroughPoses goal while an
        # existing goal is active.  Sending another active goal preempts Nav2
        # and can keep the planner from executing the current route.
        self.declare_parameter('allow_active_goal_refresh', False)
        # Robot-relative mode: skip GPS entirely and navigate to a fixed pose
        # expressed in the map/odom frame (identity transform).  target_x and
        # target_y are meters forward/lateral from the robot start position.
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
        self._mag_topic = self.get_parameter('mag_topic').value
        self._imu_topic = self.get_parameter('imu_topic').value
        self._mag_settle = float(self.get_parameter('mag_settle_sec').value)
        self._mag_min_samples = int(self.get_parameter('mag_min_samples').value)
        self._mag_declination = float(
            self.get_parameter('mag_declination_deg').value)
        self._mag_heading_offset = float(
            self.get_parameter('mag_heading_offset_deg').value)
        self._mag_off_x = float(self.get_parameter('mag_offset_x').value)
        self._mag_off_y = float(self.get_parameter('mag_offset_y').value)
        self._recovery_increase = float(
            self.get_parameter('recovery_dist_increase_m').value)
        self._max_recoveries = int(self.get_parameter('max_recoveries').value)
        self._gps_regression_window = float(
            self.get_parameter('gps_regression_window_sec').value)
        self._gps_min_samples = int(self.get_parameter('gps_min_samples').value)
        self._goal_update_dist = float(
            self.get_parameter('goal_update_distance_m').value)
        self._allow_active_goal_refresh = bool(
            self.get_parameter('allow_active_goal_refresh').value)
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
        self._latest_mag: Optional[tuple[float, float, float]] = None
        self._latest_imu_quat: Optional[tuple[float, float, float, float]] = None
        # Rolling buffer of (timestamp_sec, lat, lon) for GPS regression.
        self._gps_buffer: deque[tuple[float, float, float]] = deque()
        self._goal_handle = None
        self._goal_pending = False
        self._reached = False
        self._last_send_sec: Optional[float] = None
        # ENU position (map frame) from which the last Nav2 goal was sent;
        # used to decide when to refresh the goal as the robot moves.
        self._last_send_pos: Optional[tuple[float, float]] = None
        # Closest we have ever been to the target, and recovery attempt count.
        self._min_dist_seen: Optional[float] = None
        self._recoveries = 0

        # Heading-calibration bookkeeping (stationary magnetometer averaging).
        self._state = self._S_WAIT
        self._collecting = False
        self._settle_fixes: list[tuple[float, float]] = []
        self._mag_samples: list[tuple[float, float, float]] = []
        self._settle_deadline: Optional[float] = None

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
            if self._heading_init:
                self.create_subscription(
                    MagneticField, self._mag_topic, self._on_mag, sensor_qos)
                self.create_subscription(
                    Imu, self._imu_topic, self._on_imu, sensor_qos)
        self.create_subscription(
            Odometry, self._odom_topic, self._on_odom, sensor_qos)

        # 1 Hz state-machine tick.  Heading init is stationary, so there is no
        # open-loop drive command to republish.
        self._timer = self.create_timer(1.0, self._tick)
        if self._use_gps:
            self.get_logger().info(
                f'gps_waypoint_test [GPS]: target=({self._target_lat:.6f}, '
                f'{self._target_lon:.6f}); heading_init={self._heading_init}; '
                f'gps={self._gps_topic} odom={self._odom_topic} '
                f'mag={self._mag_topic} imu={self._imu_topic}')
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
        # Append to rolling regression buffer and trim stale entries.
        t_now = self._now()
        self._gps_buffer.append((t_now, msg.latitude, msg.longitude))
        cutoff = t_now - self._gps_regression_window
        while self._gps_buffer and self._gps_buffer[0][0] < cutoff:
            self._gps_buffer.popleft()
        if self._collecting:
            self._settle_fixes.append(self._latest_fix)
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
        q = msg.orientation
        self._latest_imu_quat = (q.x, q.y, q.z, q.w)

    def _on_mag(self, msg: MagneticField) -> None:
        m = msg.magnetic_field
        self._latest_mag = (m.x, m.y, m.z)
        if self._collecting:
            self._mag_samples.append(self._latest_mag)

    # ── GPS regression ──────────────────────────────────────────────────────

    def _smooth_gps_position(self) -> Optional[tuple[float, float]]:
        """
        Return a regression-smoothed (lat, lon) estimate for the current time.

        Fits independent linear models  lat(t) = a·t + b  and
        lon(t) = c·t + d  to the rolling GPS buffer, then evaluates them at
        ``now``.  This cancels random fix-to-fix noise and, when the robot is
        moving, extrapolates the velocity trend slightly forward to compensate
        for GPS latency.

        Falls back to the raw latest fix when the buffer contains fewer than
        ``gps_min_samples`` entries.
        """
        buf = list(self._gps_buffer)
        if len(buf) < self._gps_min_samples:
            return self._latest_fix  # not enough data yet
        t0   = buf[0][0]
        t_now = self._now() - t0
        ts   = [e[0] - t0 for e in buf]
        lats = [e[1] for e in buf]
        lons = [e[2] for e in buf]
        a_lat, b_lat = _ols(ts, lats)
        a_lon, b_lon = _ols(ts, lons)
        return (a_lat * t_now + b_lat, a_lon * t_now + b_lon)

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

    # ── State machine ────────────────────────────────────────────────────────

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _tick(self) -> None:
        if self._use_gps and self._target_lat == 0.0 and self._target_lon == 0.0:
            return

        if self._state == self._S_WAIT:
            self._tick_wait()
        elif self._state == self._S_CALIB:
            self._tick_calib()
        elif self._state == self._S_NAVIGATE:
            self._tick_navigate()
        # _S_DONE: nothing to do

    def _tick_wait(self) -> None:
        if (not self._origin_set or self._latest_fix is None
                or self._latest_odom is None):
            self.get_logger().info(
                'Waiting for GPS fix + odom before heading init…',
                throttle_duration_sec=3.0)
            return
        if not self._heading_init:
            self._enter_navigate()
            return
        if self._latest_mag is None or self._latest_imu_quat is None:
            self.get_logger().info(
                'Waiting for ZED magnetometer + IMU before heading init…',
                throttle_duration_sec=3.0)
            return
        if self._latest_status < 1:
            self.get_logger().warn(
                'GPS has no augmentation (status < 1); the position pin may be '
                'coarse, but the magnetometer heading is unaffected.',
                throttle_duration_sec=5.0)
        self.get_logger().info(
            f'Starting magnetometer heading init: averaging mag + GPS for '
            f'{self._mag_settle:.1f} s while stationary.')
        self._begin_settle()
        self._state = self._S_CALIB

    def _begin_settle(self) -> None:
        """Open the stationary sampling window (mag + GPS gathered in callbacks)."""
        self._settle_fixes = []
        self._mag_samples = []
        self._collecting = True
        self._settle_deadline = self._now() + self._mag_settle

    def _tick_calib(self) -> None:
        if self._settle_deadline is None or self._now() < self._settle_deadline:
            self.get_logger().info(
                f'Settling: {len(self._mag_samples)} mag / '
                f'{len(self._settle_fixes)} gps samples…',
                throttle_duration_sec=1.0)
            return
        self._collecting = False
        self._finish_heading_init()

    def _finish_heading_init(self) -> None:
        """Average mag + GPS, derive true heading, and pin map->odom."""
        if (len(self._mag_samples) < self._mag_min_samples
                or not self._settle_fixes
                or self._latest_imu_quat is None
                or self._latest_odom is None):
            self.get_logger().error(
                f'Heading init failed: mag={len(self._mag_samples)} '
                f'(need {self._mag_min_samples}), gps={len(self._settle_fixes)}, '
                f'imu={"ok" if self._latest_imu_quat else "missing"}. Retrying.')
            self._state = self._S_WAIT
            return

        # Averaged, hard-iron-corrected magnetometer vector (body frame).
        n = len(self._mag_samples)
        mx = sum(s[0] for s in self._mag_samples) / n - self._mag_off_x
        my = sum(s[1] for s in self._mag_samples) / n - self._mag_off_y
        mz = sum(s[2] for s in self._mag_samples) / n

        # Rotate the body-frame field into the odom frame using the IMU
        # orientation; tilt compensation falls out of the full 3-D rotation.
        mox, moy, _moz = _rotate_vec_by_quat(self._latest_imu_quat, (mx, my, mz))
        if math.hypot(mox, moy) < 1e-12:
            self.get_logger().error(
                'Magnetometer horizontal component ~0 (sensor saturated or '
                'pointing straight down?). Retrying heading init.')
            self._state = self._S_WAIT
            return
        # Bearing of magnetic north as expressed in the odom frame.
        alpha = math.atan2(moy, mox)

        # ENU bearing of magnetic north = 90 deg (true north) - declination(+east).
        decl = math.radians(self._mag_declination)
        offset = math.radians(self._mag_heading_offset)
        yaw = _wrap((math.pi / 2.0 - decl) - alpha + offset)

        # Averaged GPS → ENU to pin the translation.
        n_gps = len(self._settle_fixes)
        mean_lat = sum(f[0] for f in self._settle_fixes) / n_gps
        mean_lon = sum(f[1] for f in self._settle_fixes) / n_gps
        gps_e, gps_n = gps_to_map(mean_lat, mean_lon,
                                  self._origin_lat, self._origin_lon)

        # map_p = R(yaw) * odom_p + t, pinned so robot's map pose == GPS-ENU pose.
        ox, oy = self._latest_odom
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        tx = gps_e - (cos_y * ox - sin_y * oy)
        ty = gps_n - (sin_y * ox + cos_y * oy)

        self._map_to_odom_yaw = yaw
        self._map_to_odom_t = (tx, ty)
        self._publish_map_to_odom()
        imu_yaw = _yaw_from_quat(self._latest_imu_quat)
        self.get_logger().info(
            f'Heading set from magnetometer: mag-north odom bearing='
            f'{math.degrees(alpha):.1f} deg, imu yaw={math.degrees(imu_yaw):.1f} deg, '
            f'map->odom yaw={math.degrees(yaw):.1f} deg, t=({tx:.2f}, {ty:.2f}). '
            'Navigating to target.')
        self._enter_navigate()

    def _enter_navigate(self) -> None:
        """Reset goal/progress state and switch to the navigation phase."""
        self._goal_handle = None
        self._goal_pending = False
        self._last_send_sec = None
        self._last_send_pos = None
        self._min_dist_seen = None
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
            # Use regression-smoothed GPS position as the current estimate.
            smooth = self._smooth_gps_position()
            if smooth is None:
                return
            cur_e, cur_n = gps_to_map(smooth[0], smooth[1],
                                      self._origin_lat, self._origin_lon)
            tgt_e, tgt_n = gps_to_map(self._target_lat, self._target_lon,
                                      self._origin_lat, self._origin_lon)
        else:
            if self._latest_odom is None:
                return
            cur_e, cur_n = self._latest_odom
            tgt_e, tgt_n = self._target_x, self._target_y
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

        # ── Closed-loop goal refresh ────────────────────────────────────────
        # Only refresh an active goal when explicitly requested.  Otherwise,
        # let Nav2 finish or abort the current NavigateThroughPoses request;
        # replacing it here shows up as a preempted goal and can starve the
        # planner/controller before the robot gets between course elements.
        if self._goal_handle is not None and not self._allow_active_goal_refresh:
            return

        # Resend when no goal is active, or when active refresh is enabled and
        # the robot has moved goal_update_distance_m from where the last goal
        # was issued, or the resend_period timer expires.
        now_sec = self._now()
        pos_moved = (
            self._last_send_pos is None
            or math.hypot(cur_e - self._last_send_pos[0],
                          cur_n - self._last_send_pos[1]) >= self._goal_update_dist
        )
        timer_elapsed = (
            self._goal_handle is None
            or self._last_send_sec is None
            or (now_sec - self._last_send_sec) >= self._resend_period
        )
        if not pos_moved and not timer_elapsed:
            return

        self._send_goal(tgt_e, tgt_n, cur_e, cur_n, dist)

    def _trigger_recovery(self, dist: float) -> None:
        """Cancel the active goal and re-estimate heading from the magnetometer."""
        if self._recoveries >= self._max_recoveries:
            self.get_logger().error(
                f'Distance still increasing ({dist:.1f} m) after '
                f'{self._recoveries} recovery attempts; giving up. Check GPS '
                'fix quality and magnetometer calibration.')
            self._state = self._S_DONE
            return
        self._recoveries += 1
        self.get_logger().warn(
            f'Moving AWAY from target: dist={dist:.1f} m vs best '
            f'{self._min_dist_seen:.1f} m. Heading estimate likely wrong — '
            f're-running magnetometer heading init '
            f'(attempt {self._recoveries}/{self._max_recoveries}).')
        # Stop Nav2 driving the wrong way.
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:  # pragma: no cover
                pass
        self._goal_handle = None
        self._goal_pending = False
        self._last_send_sec = None
        if self._use_gps and self._heading_init:
            # Re-sample the magnetometer (stationary) to re-estimate heading.
            self._begin_settle()
            self._state = self._S_CALIB
        else:
            # No heading to re-estimate; re-enter navigation so the goal is
            # re-sent.
            self._enter_navigate()

    # ── Goal handling ──────────────────────────────────────────────────────

    def _send_goal(self, tgt_e: float, tgt_n: float,
                   cur_e: float, cur_n: float, dist: float) -> None:
        if not self._nav.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(
                f'{self._nav_action} action server not available yet.',
                throttle_duration_sec=3.0)
            return

        # Build intermediate waypoints spaced every waypoint_spacing_m meters
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
        self._last_send_pos = (cur_e, cur_n)  # record GPS position at send time
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
