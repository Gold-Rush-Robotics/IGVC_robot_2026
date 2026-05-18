"""
igvc_navigator.py

Drives the robot along the IGVC AutoNav course without requiring a
pre-supplied waypoint file.

GPS mode  (gps_enabled: true)
    Converts incoming NavSatFix messages to map-frame targets using a
    flat-earth projection anchored to the first fix received (or an
    explicit origin supplied via parameters).  Targets are sent to Nav2
    via NavigateToPose.  When the robot arrives within goal_tolerance the
    next fix in the queue is consumed.  If /gps/fix stops arriving the
    node falls back to lane-only forward progress automatically.

Sim / GPS-denied mode  (gps_enabled: false)
    No GPS targets are used.  The detected lane centreline is published as
    /lane_path and, by default, sent to Nav2's FollowPath controller action.
    The legacy rolling NavigateToPose carrot can be re-enabled with
    follow_path_enabled:=false while bringing up controller-server configs.

In both modes the lane costmap already carries lethal costs on the
boundaries, so Nav2's RegulatedPurePursuit controller + inflation layer
handle obstacle / boundary avoidance without any additional logic here.

Parameters
    gps_enabled           bool    true
    gps_topic             str     /gps/fix
    origin_lat            float   0.0     explicit origin; 0 = use first fix
    origin_lon            float   0.0
    map_frame             str     map
    odom_frame            str     odom
    base_frame            str     base_link
    goal_tolerance_m      float   1.2     Nav2 arrival radius
    waypoint_horizon_m    float   1.8     sim: how far ahead to place carrot
    replan_dist_m         float   0.6     sim: re-send goal when carrot moves this far
    lane_hold_sec         float   1.0     sim: reuse the last valid lane carrot briefly
    path_lookahead_m      float   4.0     centreline extraction depth
    grid_resolution       float   0.05
    grid_width_m          float   10.0
    nav_action            str     navigate_to_pose
    follow_path_enabled   bool    true    sim: use Nav2 FollowPath directly
    follow_path_action    str     follow_path
    controller_id         str     ''      Nav2 default controller
    goal_checker_id       str     ''      Nav2 default goal checker
    progress_checker_id   str     ''      Nav2 default progress checker
    min_follow_path_poses int     5       reject short/noisy local paths
    min_follow_path_length_m float 1.5
    path_sample_spacing_m float   0.10    controller path spacing
    path_smooth_window    int     5       moving-average window, 1 = off
    path_change_tolerance_m float 0.25    FollowPath resend hysteresis
    path_change_tolerance_rad float 0.25
    max_path_lateral_jump_m float 0.5

Subscriptions
    /gps/fix              NavSatFix       (GPS mode only)
    /lane_costmap         OccupancyGrid
    /front_zed_camera_x/zed_node/odom                 Odometry
    /localization_status  std_msgs/String  from igvc_localization

Publications
    /lane_path            nav_msgs/Path    dense centreline for visualisation
"""

from __future__ import annotations

import math
import json
from collections import deque
from typing import Optional

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy,
                        QoSProfile, ReliabilityPolicy)
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowPath, NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401


# ── WGS-84 flat-earth helpers ─────────────────────────────────────────────────

_WGS84_A  = 6_378_137.0
_WGS84_E2 = 0.006_694_379_990_14


def _ecef(lat_deg: float, lon_deg: float, alt: float = 0.0) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N   = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
    c   = math.cos(lat)
    return ((N + alt) * c * math.cos(lon),
            (N + alt) * c * math.sin(lon),
            (N * (1.0 - _WGS84_E2) + alt) * math.sin(lat))


def gps_to_map(lat: float, lon: float,
               origin_lat: float, origin_lon: float) -> tuple[float, float]:
    """
    Convert a GPS coordinate to local East/North metres relative to an origin.
    Accurate to ~1 cm within a few km.
    """
    ox, oy, oz = _ecef(origin_lat, origin_lon)
    px, py, pz = _ecef(lat, lon)
    dx, dy, dz = px - ox, py - oy, pz - oz
    lat0, lon0 = math.radians(origin_lat), math.radians(origin_lon)
    east  = -math.sin(lon0) * dx + math.cos(lon0) * dy
    north = (-math.sin(lat0) * math.cos(lon0) * dx
             - math.sin(lat0) * math.sin(lon0) * dy
             + math.cos(lat0) * dz)
    return east, north


# ── Small data class ──────────────────────────────────────────────────────────

class _Waypoint:
    __slots__ = ('x', 'y', 'yaw')

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.yaw: Optional[float] = None

    def dist_to(self, other: '_Waypoint') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_pose_stamped(self, frame: str, stamp) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.header.stamp    = stamp
        ps.pose.position.x = self.x
        ps.pose.position.y = self.y
        if self.yaw is None:
            ps.pose.orientation.w = 1.0
        else:
            ps.pose.orientation.z = math.sin(self.yaw / 2.0)
            ps.pose.orientation.w = math.cos(self.yaw / 2.0)
        return ps


# ── Node ──────────────────────────────────────────────────────────────────────

class IGVCNavigatorNode(Node):

    def __init__(self) -> None:
        super().__init__('igvc_navigator')

        # ── Parameters ────────────────────────────────────────────────────
        self._declare_params()
        self._gps_enabled     = self._p('gps_enabled',        True)
        self._gps_topic       = self._p('gps_topic',          '/gps/fix')
        self._odom_topic      = self._p('odom_topic',         '/front_zed_camera_x/zed_node/odom')
        self._origin_lat      = self._p('origin_lat',          0.0)
        self._origin_lon      = self._p('origin_lon',          0.0)
        self._origin_set      = (self._origin_lat != 0.0 or self._origin_lon != 0.0)
        self._map_frame       = self._p('map_frame',           'map')
        self._base_frame      = self._p('base_frame',          'base_link')
        self._goal_tol        = self._p('goal_tolerance_m',    1.2)
        self._horizon         = self._p('waypoint_horizon_m',  1.8)
        self._replan_dist     = self._p('replan_dist_m',       0.6)
        self._lane_hold_sec   = self._p('lane_hold_sec',       1.0)
        self._replan_min_dt   = self._p('replan_min_dt_sec',   1.0)
        self._lookahead       = self._p('path_lookahead_m',    4.0)
        self._grid_res        = self._p('grid_resolution',     0.05)
        self._grid_w          = self._p('grid_width_m',       10.0)
        self._max_costmap_age = self._p('max_costmap_age_sec', 0.1)
        self._nav_action_name = self._p('nav_action',         'navigate_to_pose')
        self._follow_path_enabled = self._p('follow_path_enabled', True)
        self._follow_path_action_name = self._p('follow_path_action', 'follow_path')
        self._controller_id = self._p('controller_id', '')
        self._goal_checker_id = self._p('goal_checker_id', '')
        self._progress_checker_id = self._p('progress_checker_id', '')
        self._min_follow_path_poses = self._p('min_follow_path_poses', 5)
        self._min_follow_path_length_m = self._p('min_follow_path_length_m', 1.5)
        self._path_sample_spacing_m = self._p('path_sample_spacing_m', 0.10)
        self._path_smooth_window = self._p('path_smooth_window', 5)
        self._path_change_tolerance_m = self._p('path_change_tolerance_m', 0.25)
        self._path_change_tolerance_rad = self._p('path_change_tolerance_rad', 0.25)
        self._max_path_lateral_jump_m = self._p('max_path_lateral_jump_m', 0.35)
        self._centreline_gap_tolerance_m = self._p('centreline_gap_tolerance_m', 0.25)
        self._fallback_path_length_m = self._p('fallback_path_length_m', 2.0)
        self._allow_straight_fallback = self._p('allow_straight_fallback', False)
        self._max_odom_age = self._p('max_odom_age_sec', self._max_costmap_age)
        self._max_odom_costmap_skew = self._p(
            'max_odom_costmap_skew_sec', self._max_costmap_age)

        # Centerline-waypoint strategy: rolling NavigateToPose goals along a
        # pre-known centerline (from track_points.json).  Lets SmacPlanner2D
        # plan around obstacles via /lane_map instead of relying on a fragile
        # local centreline extraction at sharp curves.
        self._nav_strategy = (self._p('nav_strategy', '') or '').strip().lower()
        self._centerline_source_json = self._p('centerline_source_json', '')
        self._centerline_lookahead_m = self._p('centerline_lookahead_m', 6.0)
        self._centerline_advance_m   = self._p('centerline_advance_m',   0.6)
        self._centerline_loop        = self._p('centerline_loop',        True)
        self._centerline_direction_param = (
            self._p('centerline_direction', 'auto') or 'auto').strip().lower()
        self._centerline_search_window_m = self._p(
            'centerline_search_window_m', 8.0)
        self._centerline_reacquire_dist_m = self._p(
            'centerline_reacquire_dist_m', 3.0)
        self._centerline_max_goal_dist_m = self._p(
            'centerline_max_goal_dist_m', 12.0)

        # ── Internal state ────────────────────────────────────────────────
        # GPS mode: queue of _Waypoint in map frame, consumed as robot arrives
        self._gps_queue: deque[_Waypoint] = deque()

        # Shared: the waypoint currently being executed by Nav2
        self._active_wp: Optional[_Waypoint] = None
        self._goal_handle   = None
        self._goal_pending  = False
        self._next_goal_seq = 0
        self._current_goal_seq = 0
        self._last_lane_wp: Optional[_Waypoint] = None
        self._last_lane_wp_time = None
        self._loc_status    = 'sim' if not self._gps_enabled else 'initializing'
        self._last_goal_send_time = None
        self._last_sent_path: Optional[Path] = None
        self._last_lane_path_reason = 'not evaluated yet'
        # Backoff: don't re-send a new goal immediately after an ABORT.
        self._abort_backoff_until = None
        self._consecutive_aborts = 0

        # Latest sensor data
        self._grid: Optional[OccupancyGrid] = None
        self._robot_xy: Optional[tuple[float, float]] = None  # odom frame
        self._robot_yaw: Optional[float] = None
        self._odom_stamp = None

        # Pre-known centerline (map frame), arc-length cache, and progress
        # index.  Populated lazily on first use in centerline_waypoints mode.
        self._centerline_xy: Optional[list[tuple[float, float]]] = None
        self._centerline_s: Optional[list[float]] = None
        self._centerline_total_s: float = 0.0
        self._centerline_progress_idx: int = 0
        self._centerline_direction_sign: int = 0
        self._centerline_load_attempted: bool = False
        self._centerline_unavailable_warned: bool = False
        self._centerline_reacquire_warned: bool = False

        # Resolve the active strategy now (param may have been explicitly
        # set, otherwise infer from gps_enabled).
        if not self._nav_strategy:
            self._nav_strategy = 'gps' if self._gps_enabled else 'local_lane'
        elif self._gps_enabled and self._nav_strategy != 'gps':
            self.get_logger().warn(
                f"nav_strategy='{self._nav_strategy}' overridden to 'gps' because gps_enabled=true")
            self._nav_strategy = 'gps'

        # ── TF ────────────────────────────────────────────────────────────
        self._tf_buf      = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        # ── Nav2 action client ────────────────────────────────────────────
        self._nav = ActionClient(self, NavigateToPose, self._nav_action_name)
        self._path_nav = ActionClient(
            self, FollowPath, self._follow_path_action_name)

        # ── QoS ───────────────────────────────────────────────────────────
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(OccupancyGrid, '/lane_costmap',
                                 self._on_grid, map_qos)
        self.create_subscription(Odometry, self._odom_topic,
                     self._on_odom, odom_qos)
        self.create_subscription(String, '/localization_status',
                                 self._on_loc_status, 10)

        if self._gps_enabled:
            self.create_subscription(NavSatFix, self._gps_topic,
                                     self._on_gps, sensor_qos)
            self.get_logger().info(
                f'Navigator: GPS mode — listening on {self._gps_topic}')
        else:
            self.get_logger().info(
                'Navigator: sim mode — autonomous lane waypoint generation.')

        # ── Publishers ────────────────────────────────────────────────────
        self._path_pub = self.create_publisher(Path, '/lane_path', 10)

        # ── Main loop ─────────────────────────────────────────────────────
        self.create_timer(0.1, self._update)   # 10 Hz

    # ── Parameter helpers ─────────────────────────────────────────────────

    def _declare_params(self) -> None:
        for name, default in [
            ('gps_enabled',       True),
            ('gps_topic',         '/gps/fix'),
            ('odom_topic',        '/front_zed_camera_x/zed_node/odom'),
            ('origin_lat',         0.0),
            ('origin_lon',         0.0),
            ('map_frame',         'map'),
            ('odom_frame',        'odom'),
            ('base_frame',        'base_link'),
            ('goal_tolerance_m',   1.2),
            ('waypoint_horizon_m', 1.8),
            ('replan_dist_m',      0.6),
            ('lane_hold_sec',      1.0),
            ('replan_min_dt_sec',  1.0),
            ('path_lookahead_m',   4.0),
            ('grid_resolution',    0.05),
            ('grid_width_m',      10.0),
            ('nav_action',        'navigate_to_pose'),
            ('follow_path_enabled', True),
            ('follow_path_action', 'follow_path'),
            ('controller_id',      ''),
            ('goal_checker_id',    ''),
            ('progress_checker_id', ''),
            ('min_follow_path_poses', 5),
            ('min_follow_path_length_m', 1.5),
            ('path_sample_spacing_m', 0.10),
            ('path_smooth_window', 5),
            ('path_change_tolerance_m', 0.10),   # tightened from 0.25 — reduces stale-path tracking on turns
            ('path_change_tolerance_rad', 0.10),  # tightened from 0.25
            ('max_path_lateral_jump_m', 0.35),
            ('centreline_gap_tolerance_m', 0.25),
            ('fallback_path_length_m', 2.0),
            ('allow_straight_fallback', False),
            ('max_costmap_age_sec', 2.0),
            ('max_odom_age_sec', 0.75),
            ('max_odom_costmap_skew_sec', 1.5),
            # Centerline-waypoint navigation strategy
            ('nav_strategy', ''),               # '' (auto) | 'local_lane' | 'centerline_waypoints' | 'gps'
            ('centerline_source_json', ''),     # path to track_points.json
            ('centerline_lookahead_m', 6.0),
            ('centerline_advance_m', 0.6),
            ('centerline_loop', True),
            ('centerline_direction', 'auto'),  # 'auto' | 'forward' | 'reverse'
            ('centerline_search_window_m', 8.0),
            ('centerline_reacquire_dist_m', 3.0),
            ('centerline_max_goal_dist_m', 12.0),
        ]:
            self.declare_parameter(name, default)

    def _p(self, name: str, _default):
        return self.get_parameter(name).value

    def _stamp_age_sec(self, stamp) -> float:
        stamp_t = Time.from_msg(stamp)
        if stamp_t.nanoseconds == 0:
            return float('inf')
        return abs((self.get_clock().now() - stamp_t).nanoseconds / 1e9)

    @staticmethod
    def _stamp_delta_sec(lhs, rhs) -> float:
        lhs_t = Time.from_msg(lhs)
        rhs_t = Time.from_msg(rhs)
        if lhs_t.nanoseconds == 0 or rhs_t.nanoseconds == 0:
            return float('inf')
        return abs((lhs_t - rhs_t).nanoseconds / 1e9)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_grid(self, msg: OccupancyGrid) -> None:
        if self._stamp_age_sec(msg.header.stamp) > self._max_costmap_age:
            self.get_logger().warn(
                f'Dropping stale lane costmap older than {self._max_costmap_age:.3f}s.',
                throttle_duration_sec=2.0)
            return
        self._grid = msg

    def _on_odom(self, msg: Odometry) -> None:
        if self._stamp_age_sec(msg.header.stamp) > self._max_odom_age:
            self.get_logger().warn(
                f'Dropping unstamped/stale odom older than {self._max_odom_age:.3f}s.',
                throttle_duration_sec=2.0)
            return
        self._robot_xy = (msg.pose.pose.position.x,
                          msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self._robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self._odom_stamp = msg.header.stamp

    def _on_loc_status(self, msg: String) -> None:
        self._loc_status = msg.data

    def _on_gps(self, msg: NavSatFix) -> None:
        if msg.status.status < 0:
            return

        # Anchor origin to first good fix if not pre-configured
        if not self._origin_set:
            self._origin_lat = msg.latitude
            self._origin_lon = msg.longitude
            self._origin_set = True
            self.get_logger().info(
                f'GPS origin set: ({self._origin_lat:.6f}, {self._origin_lon:.6f})')

        x, y = gps_to_map(msg.latitude, msg.longitude,
                           self._origin_lat, self._origin_lon)
        wp = _Waypoint(x, y)

        # Deduplicate: only enqueue if meaningfully different from the last
        if not self._gps_queue or self._gps_queue[-1].dist_to(wp) > 0.5:
            self._gps_queue.append(wp)
            self.get_logger().info(
                f'GPS waypoint enqueued: map ({x:.2f}, {y:.2f})  '
                f'queue depth={len(self._gps_queue)}')

    # ── Main update ───────────────────────────────────────────────────────

    def _update(self) -> None:
        # Publish controller-facing lane paths independently from the
        # rolling NavigateToPose goal state.  This is the first migration
        # seam toward feeding Nav2's controller with a path directly.
        lane_path = self._lane_path_from_costmap()
        self._path_pub.publish(lane_path)

        # Don't navigate until TF chain is alive
        if self._loc_status == 'initializing':
            self.get_logger().warn(
                'Waiting for localization...', throttle_duration_sec=3.0)
            return

        # Don't spam Nav2 while a goal is being accepted
        if self._goal_pending:
            return

        # Honour ABORT backoff so we don't thrash NavigateToPose at ~3 Hz.
        now = self.get_clock().now()
        if self._abort_backoff_until is not None and now < self._abort_backoff_until:
            return

        if self._uses_follow_path():
            invalid_reason = self._path_invalid_reason(lane_path)
            if invalid_reason is not None:
                self.get_logger().warn(
                    f'No valid lane path visible ({invalid_reason}) — holding current course.',
                    throttle_duration_sec=2.0)
                return

            goal_active = self._goal_handle is not None
            if goal_active and not self._path_changed_enough(lane_path):
                return
            if self._last_goal_send_time is not None and (
                (now - self._last_goal_send_time).nanoseconds / 1e9
                < self._replan_min_dt
            ):
                return
            self._send_path_goal(lane_path, force=not goal_active)
            return

        # Check arrival at active waypoint
        if self._active_wp is not None and self._robot_xy is not None:
            robot_map = self._robot_in_map()
            if robot_map is not None:
                dist = math.hypot(robot_map[0] - self._active_wp.x,
                                  robot_map[1] - self._active_wp.y)
                if dist < self._goal_tol:
                    self.get_logger().info(
                        f'Waypoint reached (dist={dist:.2f} m).')
                    self._active_wp = None
                    self._cancel_goal()

        # Pick next waypoint
        if self._active_wp is None:
            # Rate-limit the initial-send path too, otherwise a rejected
            # goal causes 10 Hz spam (update timer fires every 100 ms).
            if self._last_goal_send_time is not None and (
                (now - self._last_goal_send_time).nanoseconds / 1e9
                < self._replan_min_dt
            ):
                return
            wp = self._next_waypoint()
            if wp is None:
                return
            self._send_goal(wp)
            return

        # Sim mode: re-issue goal when the lane carrot has moved enough
        if not self._gps_enabled:
            new_wp = self._rolling_carrot()
            if new_wp is not None and self._active_wp.dist_to(new_wp) > self._replan_dist:
                if self._last_goal_send_time is None or (
                    (now - self._last_goal_send_time).nanoseconds / 1e9 >= self._replan_min_dt
                ):
                    self._send_goal(new_wp)

    # ── Waypoint generation ───────────────────────────────────────────────

    def _next_waypoint(self) -> Optional[_Waypoint]:
        """
        GPS mode: pop the front of the GPS queue.
        Centerline mode: rolling lookahead along the pre-known centerline.
        Sim mode (local_lane): project a carrot along the lane centreline.
        Falls back to a straight-ahead carrot if the lane is not visible.
        """
        strategy = self._active_strategy()
        if strategy == 'gps':
            if not self._gps_queue:
                self.get_logger().info(
                    'GPS queue empty — waiting for fix.',
                    throttle_duration_sec=3.0)
                return None
            return self._gps_queue.popleft()

        if strategy == 'centerline_waypoints':
            wp = self._centerline_goal()
            if wp is not None:
                return wp
            # Fall through to local_lane chain if centerline unavailable
            if not self._centerline_unavailable_warned:
                self.get_logger().warn(
                    'centerline strategy unavailable — falling back to local lane carrot.')
                self._centerline_unavailable_warned = True

        return self._local_lane_carrot_chain()

    def _local_lane_carrot_chain(self) -> Optional[_Waypoint]:
        carrot = self._lane_carrot()
        if carrot is None:
            carrot = self._held_lane_carrot()
        if carrot is None:
            if getattr(self, '_allow_straight_fallback', False):
                carrot = self._straight_ahead_carrot()
        if carrot is None:
            self.get_logger().warn(
                'No lane visible — holding current course.',
                throttle_duration_sec=2.0)
        return carrot

    def _active_strategy(self) -> str:
        """Return the resolved navigation strategy name."""
        s = getattr(self, '_nav_strategy', '') or ''
        if s:
            return s
        return 'gps' if getattr(self, '_gps_enabled', False) else 'local_lane'

    def _lane_carrot(self) -> Optional[_Waypoint]:
        """
        Find the lane centreline point at horizon_m ahead in base_link,
        then transform to map frame.
        Returns None if the grid is missing or has no free cells.
        """
        pts = self._extract_centreline()
        if not pts:
            return None

        # Walk the centreline to find the point nearest to horizon_m
        target_fwd, target_lat = pts[-1]  # default: furthest visible point
        for fwd, lat in pts:
            if fwd >= self._horizon:
                target_fwd, target_lat = fwd, lat
                break

        carrot = self._base_link_to_map(target_fwd, target_lat)
        if carrot is not None:
            self._last_lane_wp = carrot
            self._last_lane_wp_time = self.get_clock().now()
        return carrot

    def _held_lane_carrot(self) -> Optional[_Waypoint]:
        if self._last_lane_wp is None or self._last_lane_wp_time is None:
            return None
        age = (self.get_clock().now() - self._last_lane_wp_time).nanoseconds / 1e9
        if age > self._lane_hold_sec:
            return None
        return self._last_lane_wp

    def _straight_ahead_carrot(self) -> Optional[_Waypoint]:
        """Project a point directly ahead in base_link → map."""
        return self._base_link_to_map(self._horizon, 0.0)

    def _uses_follow_path(self) -> bool:
        """Return whether sim mode should drive Nav2 with FollowPath."""
        if self._active_strategy() == 'centerline_waypoints':
            return False
        return (not self._gps_enabled) and self._follow_path_enabled

    def _rolling_carrot(self) -> Optional[_Waypoint]:
        """Strategy-aware re-issue carrot used by _update to track motion."""
        if self._active_strategy() == 'centerline_waypoints':
            wp = self._centerline_goal()
            if wp is not None:
                return wp
        return self._lane_carrot()

    # ── Centerline-waypoint strategy ──────────────────────────────────────

    def _load_centerline_from_json(self) -> bool:
        """
        Load the pre-known centerline from track_points.json, convert it to
        the map frame (by subtracting robot_start_pose.position_m so that
        the spawn becomes (0,0) — matching track_ground_truth_node), and
        precompute arc-length and per-point yaw caches.

        Returns True on success.  Logs ERROR and returns False on failure.
        """
        if self._centerline_load_attempted:
            return self._centerline_xy is not None
        self._centerline_load_attempted = True

        path = (self._centerline_source_json or '').strip()
        if not path:
            self.get_logger().error(
                "centerline strategy requested but 'centerline_source_json' is empty")
            return False
        try:
            with open(path, 'r') as fh:
                payload = json.load(fh)
        except Exception as exc:
            self.get_logger().error(
                f'Failed to read centerline JSON {path!r}: {exc}')
            return False

        raw = payload.get('centerline_m')
        if not raw or not isinstance(raw, list):
            self.get_logger().error(
                f"centerline JSON {path!r} is missing 'centerline_m' list")
            return False

        # Match track_ground_truth_node: shift so robot spawn maps to (0, 0).
        start_pose = payload.get('robot_start_pose') or {}
        start_pos  = start_pose.get('position_m') or {}
        ox = float(start_pos.get('x', 0.0))
        oy = float(start_pos.get('y', 0.0))

        xy: list[tuple[float, float]] = []
        for pt in raw:
            try:
                xy.append((float(pt['x']) - ox, float(pt['y']) - oy))
            except (KeyError, TypeError, ValueError):
                continue
        if len(xy) < 2:
            self.get_logger().error(
                f"centerline JSON {path!r} contains <2 usable points")
            return False

        # Drop a trailing duplicate point if the JSON closes the loop
        # by repeating the first sample.
        if math.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1]) < 1e-6:
            xy.pop()
            if len(xy) < 2:
                self.get_logger().error('centerline collapsed after dedup')
                return False

        # Arc length cache (cumulative)
        s = [0.0]
        for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
            s.append(s[-1] + math.hypot(x1 - x0, y1 - y0))
        # Closing leg for loop wrap distance
        if self._centerline_loop:
            total = s[-1] + math.hypot(xy[0][0] - xy[-1][0], xy[0][1] - xy[-1][1])
        else:
            total = s[-1]

        self._centerline_xy = xy
        self._centerline_s = s
        self._centerline_total_s = total
        self._centerline_progress_idx = 0
        self._centerline_direction_sign = 0
        self.get_logger().info(
            f'Loaded centerline: {len(xy)} points, total arc-length '
            f'{total:.2f} m (loop={self._centerline_loop}) from {path}')
        return True

    @staticmethod
    def _angle_diff(lhs: float, rhs: float) -> float:
        return math.atan2(math.sin(lhs - rhs), math.cos(lhs - rhs))

    def _centerline_heading_at(self, idx: int, direction: int) -> float:
        xy = self._centerline_xy
        assert xy is not None
        n = len(xy)
        direction = 1 if direction >= 0 else -1
        nxt = idx + direction
        if nxt >= n:
            nxt = 0 if self._centerline_loop else idx
        elif nxt < 0:
            nxt = n - 1 if self._centerline_loop else idx
        if nxt == idx:
            prev = idx - direction
            if prev >= n:
                prev = 0 if self._centerline_loop else idx
            elif prev < 0:
                prev = n - 1 if self._centerline_loop else idx
            return math.atan2(xy[idx][1] - xy[prev][1], xy[idx][0] - xy[prev][0])
        return math.atan2(xy[nxt][1] - xy[idx][1], xy[nxt][0] - xy[idx][0])

    def _resolve_centerline_direction(self, idx: int) -> int:
        """
        Choose centerline traversal direction, auto-matching robot yaw.

        For an explicit ``forward``/``reverse`` config value the sign is
        latched once.  In ``auto`` mode we re-evaluate every call against
        the robot's current yaw so the robot can traverse the loop in
        either direction (e.g. after a U-turn or when spawned facing
        backwards), but apply hysteresis so small yaw noise around the
        ±90° crossover cannot flip the direction cycle-to-cycle and pump
        Nav2 between forward and backward goals.
        """
        setting = getattr(self, '_centerline_direction_param', 'auto')
        if setting in ('reverse', 'backward', 'backwards', '-1'):
            new_sign = -1
        elif setting in ('forward', 'forwards', '1'):
            new_sign = 1
        elif self._robot_yaw is None:
            # Hold whatever we already had (default to forward on first call).
            return self._centerline_direction_sign or 1
        else:
            fwd = self._centerline_heading_at(idx, 1)
            rev = self._centerline_heading_at(idx, -1)
            fwd_err = abs(self._angle_diff(self._robot_yaw, fwd))
            rev_err = abs(self._angle_diff(self._robot_yaw, rev))
            cur = self._centerline_direction_sign
            if cur == 0:
                # First evaluation: take the closer alignment outright.
                new_sign = -1 if rev_err < fwd_err else 1
            else:
                # Hysteresis: only flip when the *other* direction is
                # better-aligned by a clear margin.  This stops yaw
                # jitter around the 90° boundary from oscillating the
                # traversal sign while the robot is roughly broadside
                # to the centerline.
                margin = math.radians(30.0)
                cur_err = fwd_err if cur > 0 else rev_err
                alt_err = rev_err if cur > 0 else fwd_err
                if alt_err + margin < cur_err:
                    new_sign = -cur
                else:
                    new_sign = cur

        if new_sign != self._centerline_direction_sign:
            label = 'forward' if new_sign > 0 else 'reverse'
            self.get_logger().info(
                f'Centerline traversal direction: {label} '
                f'(param={setting!r})')
            self._centerline_direction_sign = new_sign
        return self._centerline_direction_sign

    def _centerline_full_closest_index(self, x: float, y: float) -> tuple[int, float]:
        """Return nearest centerline index and distance by scanning all samples."""
        xy = self._centerline_xy
        assert xy is not None
        best_i = 0
        best_d2 = float('inf')
        for i, (px, py) in enumerate(xy):
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i, math.sqrt(best_d2)

    def _centerline_reacquire_index(self, x: float, y: float) -> tuple[int, float]:
        """
        Yaw-aware global reacquisition of centerline progress.

        On U-turn / hairpin loops, two strands of the centerline run close
        together near the apex.  Pure-Euclidean nearest then flips between
        the two strands as the robot pose jitters, which makes the goal
        bounce ~half-a-lookahead apart in opposite directions and locks
        Nav2 into thrashing.

        To disambiguate, we collect all centerline samples within a small
        slack of the global minimum distance and prefer the one whose
        forward tangent (in the committed traversal direction) is most
        aligned with the robot's current yaw.  Falls back to plain nearest
        when robot yaw or direction sign are not yet known.
        """
        xy = self._centerline_xy
        assert xy is not None

        dists_sq = [(px - x) ** 2 + (py - y) ** 2 for (px, py) in xy]
        best_idx_global = min(range(len(xy)), key=dists_sq.__getitem__)
        best_dist = math.sqrt(dists_sq[best_idx_global])

        yaw = self._robot_yaw
        direction = self._centerline_direction_sign
        if yaw is None or direction == 0:
            return best_idx_global, best_dist

        # Slack: only consider candidates whose distance is within
        # ~1.5 m of the absolute nearest (covers the typical lane
        # half-width on the IGVC track, where the two strands of a
        # horseshoe are within a couple of meters of each other).
        slack = 1.5
        threshold_sq = (best_dist + slack) ** 2

        best_idx = best_idx_global
        best_score = float('inf')
        for i, d2 in enumerate(dists_sq):
            if d2 > threshold_sq:
                continue
            tang = self._centerline_heading_at(i, direction)
            yaw_err = abs(self._angle_diff(yaw, tang))
            # Heavily prefer tangent alignment; mild distance penalty
            # keeps absolute nearest winning when both strands match.
            score = yaw_err + 0.5 * (math.sqrt(d2) - best_dist)
            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx, math.sqrt(dists_sq[best_idx])

    def _centerline_closest_index(self, x: float, y: float) -> int:
        """
        Return the centerline index closest to (x, y).

        Always performs a yaw-aware global nearest scan.  The previous
        rolling-window implementation could let ``_centerline_progress_idx``
        drift across the loop wrap (idx N-1 → 0) — once stuck near the
        loop seam, the ±window_m arc range no longer covered the robot's
        true nearest strand, so the rolling search returned a bogus
        ``best_i`` many meters away.  On some cycles this tripped the
        >3 m drift reacquire and snapped progress back to the true
        nearest; on other cycles it didn't (best within window happened
        to be slightly under the reacquire threshold) and the goal was
        computed from a stale ``cur_idx`` on the far side of the loop.
        That alternation pumped Nav2 between two goals ~12 m apart on
        opposite ends of the track and drove the robot into the lane
        line.  An O(n) scan over a 1000-sample track at 10 Hz is
        negligible and eliminates the wrap-drift class of bugs entirely.
        """
        xy = self._centerline_xy
        s = self._centerline_s
        assert xy is not None and s is not None  # checked by caller

        # First-call bookkeeping: commit traversal direction from the
        # robot's initial yaw the first time we ever see it.
        if not getattr(self, '_centerline_seen_robot', False):
            best_i, _ = self._centerline_full_closest_index(x, y)
            self._centerline_progress_idx = best_i
            self._centerline_seen_robot = True
            self._resolve_centerline_direction(best_i)
            return best_i

        # Yaw-aware global nearest on every call.  Tie-breaking by
        # tangent alignment keeps progress on the same strand of the
        # loop the robot is actually traveling, so closed-loop seams
        # and parallel U-turn strands cannot flip the progress index.
        idx, _dist = self._centerline_reacquire_index(x, y)
        self._centerline_progress_idx = idx
        return idx

    def _centerline_advance_index(
        self,
        start_idx: int,
        advance_m: float,
        direction: Optional[int] = None,
    ) -> Optional[int]:
        """
        Walk along the centerline by ``advance_m`` and return the
        resulting index.  Returns None past the end when not looping.
        """
        xy = self._centerline_xy
        if xy is None:
            return None
        n = len(xy)
        direction = direction if direction is not None else self._resolve_centerline_direction(start_idx)
        direction = 1 if direction >= 0 else -1
        target = max(0.0, float(advance_m))
        i = start_idx
        walked = 0.0
        while True:
            nxt = i + direction
            if nxt >= n:
                if self._centerline_loop:
                    nxt = 0
                else:
                    return n - 1
            elif nxt < 0:
                if self._centerline_loop:
                    nxt = n - 1
                else:
                    return 0
            walked += math.hypot(xy[nxt][0] - xy[i][0], xy[nxt][1] - xy[i][1])
            if walked >= target:
                return nxt
            i = nxt
            if i == start_idx:
                return start_idx  # safety stop after a full lap

    def _centerline_goal(self) -> Optional[_Waypoint]:
        """Return the NavigateToPose target along the pre-known centerline."""
        if not self._load_centerline_from_json():
            return None
        robot = self._robot_in_map()
        if robot is None:
            return None
        cur_idx = self._centerline_closest_index(robot[0], robot[1])
        direction = self._resolve_centerline_direction(cur_idx)
        goal_idx = self._centerline_goal_index_from(cur_idx, direction)
        if goal_idx is None:
            return None
        goal_idx = self._validate_centerline_goal_index(
            goal_idx, robot[0], robot[1], direction)
        if goal_idx is None:
            return None
        xy = self._centerline_xy
        gx, gy = xy[goal_idx]
        # Yaw: heading from goal to the next sample in the chosen direction.
        nxt = goal_idx + direction
        if nxt >= len(xy):
            nxt = 0 if self._centerline_loop else goal_idx
        elif nxt < 0:
            nxt = len(xy) - 1 if self._centerline_loop else goal_idx
        if nxt == goal_idx:
            prev = goal_idx - direction
            if prev < 0:
                prev = len(xy) - 1 if self._centerline_loop else 0
            elif prev >= len(xy):
                prev = 0 if self._centerline_loop else len(xy) - 1
            yaw = math.atan2(gy - xy[prev][1], gx - xy[prev][0])
        else:
            yaw = math.atan2(xy[nxt][1] - gy, xy[nxt][0] - gx)
        wp = _Waypoint(gx, gy)
        wp.yaw = yaw
        return wp

    def _centerline_goal_index_from(self, cur_idx: int, direction: int) -> Optional[int]:
        return self._centerline_advance_index(
            cur_idx, self._centerline_lookahead_m, direction)

    def _validate_centerline_goal_index(
        self,
        goal_idx: int,
        robot_x: float,
        robot_y: float,
        direction: int,
    ) -> Optional[int]:
        """Reject impossible centerline goals that are far beyond lookahead."""
        xy = self._centerline_xy
        assert xy is not None
        gx, gy = xy[goal_idx]
        goal_dist = math.hypot(gx - robot_x, gy - robot_y)
        max_dist = max(
            float(getattr(self, '_centerline_max_goal_dist_m', 12.0)),
            float(self._centerline_lookahead_m) + 2.0,
        )
        if goal_dist <= max_dist:
            return goal_idx

        nearest_i, nearest_dist = self._centerline_reacquire_index(robot_x, robot_y)
        self._centerline_progress_idx = nearest_i
        retry_idx = self._centerline_goal_index_from(nearest_i, direction)
        if retry_idx is None:
            return None
        rx, ry = xy[retry_idx]
        retry_dist = math.hypot(rx - robot_x, ry - robot_y)
        if retry_dist <= max_dist:
            self.get_logger().warn(
                'Dropped stale centerline goal '
                f'{goal_dist:.2f} m away; reacquired nearest centerline '
                f'point {nearest_dist:.2f} m from robot.',
                throttle_duration_sec=1.0)
            return retry_idx

        self.get_logger().warn(
            'Suppressing centerline goal because computed target is '
            f'{retry_dist:.2f} m away after reacquire (limit {max_dist:.2f} m).',
            throttle_duration_sec=1.0)
        return None

    # ── Centreline extraction ─────────────────────────────────────────────

    def _extract_centreline(self) -> list[tuple[float, float]]:
        """
        Derive the lane centreline from /lane_costmap.  Ground-truth mode
        marks the drivable corridor as FREE (0) and stamps obstacles as
        LETHAL (100), while the camera pipeline marks detected lane paint as
        LETHAL.  Prefer the free corridor when it is available so simulated
        obstacles do not get mistaken for lane lines.

        Algorithm:
          1. Collect all lethal cells (cost == 100) within the lookahead.
          2. Split into left side (lat > 0) and right side (lat < 0).
          3. Fit a polynomial lat = f(fwd) to each side.
          4. Sample (fwd, (left(fwd) + right(fwd)) / 2) along the horizon.
             If only one side is detected, offset by half corridor width.

        The previous row-by-row free-cell centroid approach failed when
        the corridor between detected lane lines was not filled in as
        free cells: lane_segmentation only fills free between detected
        lane lines in rows where a lane is actually present in that
        row, so on diagonal or sparse line segments most rows between
        the robot and the visible lines stayed at -1 (unknown), causing
        centreline extraction to find nothing and the robot to stall.
        """
        g = self._grid
        if g is None:
            return []

        W, H   = g.info.width, g.info.height
        res    = g.info.resolution
        orig_y = g.info.origin.position.y  # lateral offset of col 0 in base_link

        data = np.frombuffer(bytes(g.data), dtype=np.int8).reshape(H, W)

        # 1. Prefer FREE corridor boundaries when present.
        #    GT mode:     corridor=FREE(0), outside=UNKNOWN(-1), obstacles=LETHAL(100).
        #    Camera mode: lane lines=LETHAL(100), corridor mixed FREE/UNKNOWN.
        #
        #    Scan every row in the lookahead window; for each row use the
        #    outermost free cells as the left/right corridor boundary.
        #    Obstacle cells (100) punch holes in the free region but the
        #    outermost free cells on each side still represent the corridor
        #    edges correctly.  No per-row gap or lateral-jump filtering —
        #    those checks fired prematurely on sharp bends where the corridor
        #    shifts quickly in the base_link frame.  The polynomial fit and
        #    the corridor_max_lat guard below handle outliers.
        max_row = min(H, int(self._lookahead / res) + 1)
        free_rows_all, free_cols_all = np.where(data[:max_row] == 0)
        if free_rows_all.size >= 3:
            syn_rows: list[int] = []
            syn_cols: list[int] = []
            for r in np.unique(free_rows_all):
                c_in_row = free_cols_all[free_rows_all == r]
                if c_in_row.size >= 2:
                    syn_rows.extend([int(r), int(r)])
                    syn_cols.extend([int(c_in_row.max()), int(c_in_row.min())])
            if len(syn_rows) >= 3:
                lane_rows = np.array(syn_rows, dtype=np.intp)
                lane_cols = np.array(syn_cols, dtype=np.intp)
            else:
                lane_rows, lane_cols = np.where(data[:max_row] == 100)
        else:
            # No free corridor cells — camera pipeline: lane lines are LETHAL.
            lane_rows, lane_cols = np.where(data[:max_row] == 100)
        if lane_rows.size < 3:
            return []

        fwds = lane_rows.astype(float) * res
        lats = orig_y + (lane_cols.astype(float) + 0.5) * res

        # 2. Filter to plausible corridor extent (±2.0m). Anything beyond
        #    is almost certainly stray detection or an adjacent corridor.
        corridor_max_lat = 2.0
        keep = np.abs(lats) <= corridor_max_lat
        fwds = fwds[keep]
        lats = lats[keep]
        if fwds.size < 3:
            return []

        # 3. Split sides. ROS REP-103: +y is left of the robot, -y is right.
        #    Small dead-band around 0 so a stray cell on the centreline
        #    isn't force-classified.
        left_mask  = lats >  0.10
        right_mask = lats < -0.10

        def _fit(mask: np.ndarray):
            n = int(mask.sum())
            if n < 3:
                return None
            # Degree 1 if few points, degree 2 otherwise (handles turns).
            deg = 1 if n < 8 else 2
            try:
                return np.polyfit(fwds[mask], lats[mask], deg=deg)
            except (np.linalg.LinAlgError, ValueError):
                return None

        left_poly  = _fit(left_mask)
        right_poly = _fit(right_mask)

        if left_poly is None and right_poly is None:
            return []

        # 4. Sample centreline over the range with actual lane evidence so
        #    we don't extrapolate past the last visible lane segment.
        if left_poly is not None and right_poly is not None:
            fwd_max = float(min(fwds[left_mask].max(), fwds[right_mask].max()))
        elif left_poly is not None:
            fwd_max = float(fwds[left_mask].max())
        else:
            fwd_max = float(fwds[right_mask].max())
        fwd_max = min(fwd_max, self._lookahead)

        # Start a little ahead of the robot to avoid extrapolation near 0.
        fwd_min = 0.4
        if fwd_max <= fwd_min + 0.5:
            return []

        spacing = max(0.10, float(self._path_sample_spacing_m))
        sample_count = max(2, int(math.ceil((fwd_max - fwd_min) / spacing)) + 1)
        sample_fwds = np.linspace(fwd_min, fwd_max, sample_count)

        half_corridor = 1.0  # m — offset from a single visible line

        pts: list[tuple[float, float]] = []
        for fwd in sample_fwds:
            left_lat  = float(np.polyval(left_poly,  fwd)) if left_poly  is not None else None
            right_lat = float(np.polyval(right_poly, fwd)) if right_poly is not None else None

            if left_lat is not None and right_lat is not None:
                sep = left_lat - right_lat
                if sep < 0.5:
                    # Lines have crossed / collapsed — past corridor end.
                    break
                if sep > 4.0:
                    # One fit is suspect; fall back to the denser side.
                    if int(left_mask.sum()) >= int(right_mask.sum()):
                        lat = left_lat - half_corridor
                    else:
                        lat = right_lat + half_corridor
                else:
                    lat = 0.5 * (left_lat + right_lat)
            elif left_lat is not None:
                lat = left_lat - half_corridor
            else:
                lat = right_lat + half_corridor  # type: ignore[operator]

            # Clamp — guards against polynomial extrapolation blow-up.
            if abs(lat) > corridor_max_lat:
                break
            pts.append((float(fwd), float(lat)))

        return pts

    # ── Path message builder ──────────────────────────────────────────────

    def _lane_path_from_costmap(self) -> Path:
        """Build a controller-facing lane path from the latest costmap."""
        if self._grid is None:
            self._last_lane_path_reason = 'no /lane_costmap received'
            return self._build_path([], self.get_clock().now().to_msg())

        grid_age = self._stamp_age_sec(self._grid.header.stamp)
        if grid_age > self._max_costmap_age:
            self._last_lane_path_reason = (
                f'/lane_costmap stale: age {grid_age:.3f}s > {self._max_costmap_age:.3f}s')
            return self._build_path([], self.get_clock().now().to_msg())

        if self._odom_stamp is None:
            self._last_lane_path_reason = 'no /odom received'
            self.get_logger().warn(
                f'No fresh stamped odom within {self._max_odom_age:.3f}s; suppressing lane path.',
                throttle_duration_sec=2.0)
            return self._build_path([], self._grid.header.stamp)

        odom_age = self._stamp_age_sec(self._odom_stamp)
        if odom_age > self._max_odom_age:
            self._last_lane_path_reason = (
                f'/odom stale: age {odom_age:.3f}s > {self._max_odom_age:.3f}s')
            self.get_logger().warn(
                f'No fresh stamped odom within {self._max_odom_age:.3f}s; suppressing lane path.',
                throttle_duration_sec=2.0)
            return self._build_path([], self._grid.header.stamp)

        stamp_skew = self._stamp_delta_sec(self._grid.header.stamp, self._odom_stamp)
        if stamp_skew > self._max_odom_costmap_skew:
            self._last_lane_path_reason = (
                f'costmap/odom stamp skew {stamp_skew:.3f}s > {self._max_odom_costmap_skew:.3f}s')
            self.get_logger().warn(
                'Lane costmap and odom stamps differ by more than '
                f'{self._max_odom_costmap_skew:.3f}s; suppressing lane path.',
                throttle_duration_sec=2.0)
            return self._build_path([], self._grid.header.stamp)

        raw_pts = self._extract_centreline()
        # Use the odom stamp so Nav2 resolves base_link→odom TF at the moment
        # that matches the robot pose used for costmap extraction.  Using the
        # (possibly stale) costmap stamp, then overriding it to now() in
        # _send_path_goal, was anchoring the path to the wrong robot pose on
        # turns and causing the robot to track across lane lines.
        path_stamp = self._odom_stamp
        if not raw_pts:
            self._last_lane_path_reason = 'centreline extraction found no connected free band ahead'
            if self._allow_straight_fallback:
                return self._straight_ahead_path(path_stamp)
            return self._build_path([], path_stamp)
        if len(raw_pts) < self._min_follow_path_poses:
            self._last_lane_path_reason = f'centreline has {len(raw_pts)} raw point(s)'
            if self._allow_straight_fallback:
                return self._straight_ahead_path(path_stamp)
            return self._build_path([], path_stamp)
        else:
            self._last_lane_path_reason = f'centreline has {len(raw_pts)} raw point(s)'
        return self._build_path(
            self._condition_path_points(raw_pts),
            path_stamp)

    def _straight_ahead_path(self, stamp=None) -> Path:
        length = max(self._min_follow_path_length_m, self._fallback_path_length_m)
        spacing = max(0.10, self._path_sample_spacing_m)
        count = max(self._min_follow_path_poses, int(math.ceil(length / spacing)) + 1)
        pts = [(i * length / float(count - 1), 0.0) for i in range(count)]
        return self._build_path(pts, stamp)

    def _build_path(self, pts: list[tuple[float, float]], stamp=None) -> Path:
        path = Path()
        path.header.stamp    = stamp if stamp is not None else self.get_clock().now().to_msg()
        path.header.frame_id = self._base_frame

        prev_yaw = 0.0
        for i, (fwd, lat) in enumerate(pts):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = fwd
            ps.pose.position.y = lat

            if i < len(pts) - 1:
                df = pts[i + 1][0] - fwd
                dl = pts[i + 1][1] - lat
                prev_yaw = math.atan2(dl, df)

            ps.pose.orientation.z = math.sin(prev_yaw / 2.0)
            ps.pose.orientation.w = math.cos(prev_yaw / 2.0)
            path.poses.append(ps)

        return path

    # ── Path processing and validation ───────────────────────────────────

    def _condition_path_points(
        self,
        pts: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Smooth and resample raw centreline points for controller tracking."""
        if len(pts) < 2:
            return pts
        return self._resample_path_points(
            self._smooth_path_points(pts), self._path_sample_spacing_m)

    def _smooth_path_points(
        self,
        pts: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Apply a small moving average to lateral jitter."""
        window = int(self._path_smooth_window)
        if window <= 1 or len(pts) < 3:
            return pts
        if window % 2 == 0:
            window += 1
        window = min(window, len(pts) if len(pts) % 2 == 1 else len(pts) - 1)
        if window <= 1:
            return pts

        half = window // 2
        laterals = np.asarray([lat for _, lat in pts], dtype=float)
        padded = np.pad(laterals, (half, half), mode='edge')
        kernel = np.ones(window, dtype=float) / float(window)
        smooth_lat = np.convolve(padded, kernel, mode='valid')
        smooth_lat[0] = laterals[0]
        smooth_lat[-1] = laterals[-1]
        return [(fwd, float(lat)) for (fwd, _), lat in zip(pts, smooth_lat)]

    def _resample_path_points(
        self,
        pts: list[tuple[float, float]],
        spacing_m: float,
    ) -> list[tuple[float, float]]:
        """Resample points at approximately uniform arc-length spacing."""
        if len(pts) < 2 or spacing_m <= 0.0:
            return pts

        distances = [0.0]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            distances.append(distances[-1] + math.hypot(x1 - x0, y1 - y0))
        total = distances[-1]
        if total <= spacing_m:
            return pts

        samples = list(np.arange(0.0, total, spacing_m))
        if not samples or samples[-1] < total:
            samples.append(total)

        out: list[tuple[float, float]] = []
        seg = 0
        for sample in samples:
            while seg < len(distances) - 2 and distances[seg + 1] < sample:
                seg += 1
            d0, d1 = distances[seg], distances[seg + 1]
            ratio = 0.0 if d1 == d0 else (sample - d0) / (d1 - d0)
            x0, y0 = pts[seg]
            x1, y1 = pts[seg + 1]
            out.append((x0 + ratio * (x1 - x0), y0 + ratio * (y1 - y0)))
        return out

    def _path_length(self, path: Path) -> float:
        """Return accumulated path length in metres."""
        if len(path.poses) < 2:
            return 0.0
        return sum(
            math.hypot(
                b.pose.position.x - a.pose.position.x,
                b.pose.position.y - a.pose.position.y,
            )
            for a, b in zip(path.poses[:-1], path.poses[1:])
        )

    def _path_invalid_reason(self, path: Path) -> Optional[str]:
        """Return None when path is valid, otherwise explain the rejection."""
        if path.header.frame_id != self._base_frame:
            return f'frame {path.header.frame_id!r} != base frame {self._base_frame!r}'

        pose_count = len(path.poses)
        if pose_count < self._min_follow_path_poses:
            return (
                f'{self._last_lane_path_reason}; poses {pose_count} < '
                f'{self._min_follow_path_poses}')

        path_length = self._path_length(path)
        if path_length < self._min_follow_path_length_m:
            return (
                f'{self._last_lane_path_reason}; length {path_length:.2f}m < '
                f'{self._min_follow_path_length_m:.2f}m')

        prev_x = path.poses[0].pose.position.x
        prev_y = path.poses[0].pose.position.y
        for index, pose in enumerate(path.poses):
            x = pose.pose.position.x
            y = pose.pose.position.y
            if not math.isfinite(x) or not math.isfinite(y):
                return f'pose {index} has non-finite coordinate ({x}, {y})'
            if x + self._grid_res < prev_x:
                return f'pose {index} moves backward: x {x:.2f} after {prev_x:.2f}'
            if abs(y - prev_y) > self._max_path_lateral_jump_m:
                return (
                    f'pose {index} lateral jump {abs(y - prev_y):.2f}m > '
                    f'{self._max_path_lateral_jump_m:.2f}m')
            prev_x, prev_y = x, y
        return None

    def _path_is_valid(self, path: Path) -> bool:
        """Validate that a path is plausible enough to send to Nav2."""
        return self._path_invalid_reason(path) is None

    def _path_changed_enough(self, path: Path) -> bool:
        """Return true when a path differs enough to resend FollowPath."""
        old = self._last_sent_path
        if old is None:
            return True
        if not old.poses or not path.poses:
            return True
        if (abs(self._path_length(path) - self._path_length(old))
            > self._path_change_tolerance_m):
            return True

        count = min(len(old.poses), len(path.poses), 20)
        if count <= 1:
            return True
        old_idx = np.linspace(0, len(old.poses) - 1, count).astype(int)
        new_idx = np.linspace(0, len(path.poses) - 1, count).astype(int)
        for oi, ni in zip(old_idx, new_idx):
            old_pose = old.poses[int(oi)].pose
            new_pose = path.poses[int(ni)].pose
            shift = math.hypot(
                new_pose.position.x - old_pose.position.x,
                new_pose.position.y - old_pose.position.y,
            )
            if shift > self._path_change_tolerance_m:
                return True
            heading_delta = abs(
                self._wrap_angle(
                    self._pose_yaw(new_pose.orientation)
                    - self._pose_yaw(old_pose.orientation)))
            if heading_delta > self._path_change_tolerance_rad:
                return True
        return False

    @staticmethod
    def _pose_yaw(q) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    # ── Nav2 interaction ──────────────────────────────────────────────────

    def _send_path_goal(self, path: Path, force: bool = False) -> bool:
        """Dispatch a Nav2 FollowPath goal for the current lane path."""
        if not self._path_is_valid(path):
            self.get_logger().warn(
                'Refusing to send invalid FollowPath goal.',
                throttle_duration_sec=2.0)
            return False
        if not force and not self._path_changed_enough(path):
            return False
        if not self._path_nav.server_is_ready():
            self.get_logger().warn(
                'FollowPath server not ready.', throttle_duration_sec=2.0)
            return False

        if self._goal_handle is not None:
            self._cancel_goal()

        send_time = self.get_clock().now()
        # Do NOT override path.header.stamp here.  The stamp was set to
        # self._odom_stamp in _lane_path_from_costmap so that Nav2 resolves
        # base_link→odom at the correct robot pose.  Overriding it to now()
        # was shifting the entire path by however far the robot moved between
        # the odom capture time and the goal send time.

        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = self._controller_id
        goal.goal_checker_id = self._goal_checker_id
        # Humble FollowPath goal has no progress_checker_id field.
        if hasattr(goal, 'progress_checker_id'):
            goal.progress_checker_id = self._progress_checker_id

        self._goal_pending = True
        self._last_goal_send_time = send_time
        self._next_goal_seq += 1
        goal_seq = self._next_goal_seq
        future = self._path_nav.send_goal_async(goal)
        future.add_done_callback(
            lambda done, seq=goal_seq: self._on_path_goal_response(done, seq))
        self.get_logger().info(
            f'Sending FollowPath goal with {len(path.poses)} poses.')
        self._last_sent_path = path
        return True

    def _on_path_goal_response(self, future, goal_seq: int) -> None:
        self._goal_pending = False
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'FollowPath send error: {exc}')
            return

        if not handle.accepted:
            self._consecutive_aborts = min(self._consecutive_aborts + 1, 4)
            backoff_s = min(2.0 ** self._consecutive_aborts * 0.25, 2.0)
            self._abort_backoff_until = (
                self.get_clock().now() + Duration(seconds=backoff_s))
            self.get_logger().warn(
                f'Nav2 rejected FollowPath; backing off {backoff_s:.2f}s.',
                throttle_duration_sec=1.0)
            return

        self._current_goal_seq = goal_seq
        self._goal_handle = handle
        self._active_wp = None
        handle.get_result_async().add_done_callback(
            lambda done, seq=goal_seq: self._on_goal_result(done, seq))

    def _send_goal(self, wp: _Waypoint) -> bool:
        """Dispatch a NavigateToPose goal. Returns True iff the goal was
        actually sent to the action server."""
        if not self._nav.server_is_ready():
            self.get_logger().warn(
                'NavigateToPose server not ready.', throttle_duration_sec=2.0)
            return False

        send_time = self.get_clock().now()
        goal = NavigateToPose.Goal()
        goal.pose = wp.to_pose_stamped(self._map_frame, send_time.to_msg())
        self._goal_pending = True
        self._last_goal_send_time = send_time
        self._next_goal_seq += 1
        goal_seq = self._next_goal_seq
        future = self._nav.send_goal_async(goal)
        future.add_done_callback(
            lambda done, seq=goal_seq, waypoint=wp: self._on_goal_response(done, seq, waypoint))
        self.get_logger().info(
            f'Sending goal: map ({wp.x:.2f}, {wp.y:.2f})')
        return True

    def _on_goal_response(self, future, goal_seq: int, wp: _Waypoint) -> None:
        self._goal_pending = False
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'Goal send error: {exc}')
            return

        if not handle.accepted:
            # Back off on rejection the same way we do on abort, otherwise
            # the _update timer immediately retries and spams the action
            # server at 10 Hz.
            self._consecutive_aborts = min(self._consecutive_aborts + 1, 4)
            backoff_s = min(2.0 ** self._consecutive_aborts * 0.25, 2.0)
            self._abort_backoff_until = (
                self.get_clock().now() + Duration(seconds=backoff_s))
            self.get_logger().warn(
                f'Nav2 rejected goal; backing off {backoff_s:.2f}s before retry.',
                throttle_duration_sec=1.0)
            return

        self._current_goal_seq = goal_seq
        self._goal_handle = handle
        self._active_wp = wp
        handle.get_result_async().add_done_callback(
            lambda done, seq=goal_seq: self._on_goal_result(done, seq))

    def _on_goal_result(self, future, goal_seq: int) -> None:
        try:
            result = future.result()
            status = result.status
        except Exception as exc:
            self.get_logger().warn(f'Goal result error: {exc}')
            status = GoalStatus.STATUS_ABORTED

        if goal_seq != self._current_goal_seq:
            return

        self._goal_handle = None
        self._current_goal_seq = 0

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Nav2 goal succeeded.')
            self._consecutive_aborts = 0
            self._abort_backoff_until = None
            if self._uses_follow_path():
                self._last_sent_path = None
        elif status == GoalStatus.STATUS_CANCELED:
            pass  # expected when we replan
        else:
            # ABORTED or similar — apply exponential backoff before retrying.
            self._last_sent_path = None
            self._consecutive_aborts = min(self._consecutive_aborts + 1, 4)
            backoff_s = min(2.0 ** self._consecutive_aborts * 0.25, 2.0)
            self._abort_backoff_until = (
                self.get_clock().now() + Duration(seconds=backoff_s))
            self.get_logger().warn(
                f'received non-success goal result: status={status}; '
                f'backing off {backoff_s:.2f}s before retry',
                throttle_duration_sec=1.0)

        # Clear so _update picks the next waypoint on the next tick
        if self._active_wp is not None and status != GoalStatus.STATUS_CANCELED:
            self._active_wp = None

    def _cancel_goal(self) -> None:
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()

    # ── TF helpers ────────────────────────────────────────────────────────

    def _base_link_to_map(self, forward: float, lateral: float) -> Optional[_Waypoint]:
        """Transform a (forward, lateral) point in base_link to map frame."""
        ps = PoseStamped()
        ps.header.frame_id = self._base_frame
        ps.header.stamp    = Time().to_msg()  # latest available
        ps.pose.position.x = forward
        ps.pose.position.y = lateral
        ps.pose.orientation.w = 1.0
        try:
            out = self._tf_buf.transform(ps, self._map_frame,
                                         timeout=Duration(seconds=0.05))
            return _Waypoint(out.pose.position.x, out.pose.position.y)
        except Exception as exc:
            if (not self._gps_enabled
                    and self._robot_xy is not None
                    and self._robot_yaw is not None):
                robot_x, robot_y = self._robot_xy
                cy = math.cos(self._robot_yaw)
                sy = math.sin(self._robot_yaw)
                x_map = robot_x + forward * cy - lateral * sy
                y_map = robot_y + forward * sy + lateral * cy
                return _Waypoint(x_map, y_map)
            self.get_logger().warn(
                f'TF base_link→map failed: {exc}', throttle_duration_sec=1.0)
            return None

    def _robot_in_map(self) -> Optional[tuple[float, float]]:
        """Return robot (x, y) in map frame, or None on TF failure."""
        ps = PoseStamped()
        ps.header.frame_id = self._base_frame
        ps.header.stamp    = Time().to_msg()
        ps.pose.orientation.w = 1.0
        try:
            out = self._tf_buf.transform(ps, self._map_frame,
                                         timeout=Duration(seconds=0.05))
            return out.pose.position.x, out.pose.position.y
        except Exception:
            if not self._gps_enabled and self._robot_xy is not None:
                return self._robot_xy
            return None


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = IGVCNavigatorNode()

    try:
        try:
            from rclpy.experimental import EventsExecutor
            executor = EventsExecutor()
        except ImportError:
            from rclpy.executors import SingleThreadedExecutor
            node.get_logger().warn(
                'EventsExecutor is not available in this rclpy install; '
                'falling back to SingleThreadedExecutor.')
            executor = SingleThreadedExecutor()

        executor.add_node(node)
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()