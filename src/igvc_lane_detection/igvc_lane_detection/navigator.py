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
    No GPS targets are used.  Instead, a waypoint is synthesised every
    update cycle by projecting a point ahead along the detected lane
    centreline.  The projection distance is kept short (waypoint_horizon_m)
    so the robot is always chasing a fresh target that reflects the current
    lane geometry.  Nav2 is driven via a rolling NavigateToPose goal — as
    soon as the carrot moves far enough (replan_dist_m) the old goal is
    cancelled and a new one sent.

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

Subscriptions
    /gps/fix              NavSatFix       (GPS mode only)
    /lane_costmap         OccupancyGrid
    /odom                 Odometry
    /localization_status  std_msgs/String  from igvc_localization

Publications
    /lane_path            nav_msgs/Path    dense centreline for visualisation
"""

from __future__ import annotations

import math
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
from nav2_msgs.action import NavigateToPose
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
    __slots__ = ('x', 'y')

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def dist_to(self, other: '_Waypoint') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_pose_stamped(self, frame: str, stamp) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.header.stamp    = stamp
        ps.pose.position.x = self.x
        ps.pose.position.y = self.y
        ps.pose.orientation.w = 1.0
        return ps


# ── Node ──────────────────────────────────────────────────────────────────────

class IGVCNavigatorNode(Node):

    def __init__(self) -> None:
        super().__init__('igvc_navigator')

        # ── Parameters ────────────────────────────────────────────────────
        self._declare_params()
        self._gps_enabled     = self._p('gps_enabled',        True)
        self._gps_topic       = self._p('gps_topic',          '/gps/fix')
        self._origin_lat      = self._p('origin_lat',          0.0)
        self._origin_lon      = self._p('origin_lon',          0.0)
        self._origin_set      = (self._origin_lat != 0.0 or self._origin_lon != 0.0)
        self._map_frame       = self._p('map_frame',           'map')
        self._base_frame      = self._p('base_frame',          'base_link')
        self._goal_tol        = self._p('goal_tolerance_m',    1.2)
        self._horizon         = self._p('waypoint_horizon_m',  1.8)
        self._replan_dist     = self._p('replan_dist_m',       0.6)
        self._lane_hold_sec   = self._p('lane_hold_sec',       1.0)
        self._replan_min_dt   = self._p('replan_min_dt_sec',   0.7)
        self._lookahead       = self._p('path_lookahead_m',    4.0)
        self._grid_res        = self._p('grid_resolution',     0.05)
        self._grid_w          = self._p('grid_width_m',       10.0)
        self._nav_action_name = self._p('nav_action',         'navigate_to_pose')

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
        # Backoff: don't re-send a new goal immediately after an ABORT.
        self._abort_backoff_until = None
        self._consecutive_aborts = 0

        # Latest sensor data
        self._grid: Optional[OccupancyGrid] = None
        self._robot_xy: Optional[tuple[float, float]] = None  # odom frame
        self._robot_yaw: Optional[float] = None

        # ── TF ────────────────────────────────────────────────────────────
        self._tf_buf      = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        # ── Nav2 action client ────────────────────────────────────────────
        self._nav = ActionClient(self, NavigateToPose, self._nav_action_name)

        # ── QoS ───────────────────────────────────────────────────────────
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(OccupancyGrid, '/lane_costmap',
                                 self._on_grid, map_qos)
        self.create_subscription(Odometry, '/odom',
                                 self._on_odom, 10)
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
            ('origin_lat',         0.0),
            ('origin_lon',         0.0),
            ('map_frame',         'map'),
            ('odom_frame',        'odom'),
            ('base_frame',        'base_link'),
            ('goal_tolerance_m',   1.2),
            ('waypoint_horizon_m', 1.8),
            ('replan_dist_m',      0.6),
            ('lane_hold_sec',      1.0),
            ('replan_min_dt_sec',  0.7),
            ('path_lookahead_m',   4.0),
            ('grid_resolution',    0.05),
            ('grid_width_m',      10.0),
            ('nav_action',        'navigate_to_pose'),
        ]:
            self.declare_parameter(name, default)

    def _p(self, name: str, _default):
        return self.get_parameter(name).value

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_grid(self, msg: OccupancyGrid) -> None:
        self._grid = msg

    def _on_odom(self, msg: Odometry) -> None:
        self._robot_xy = (msg.pose.pose.position.x,
                          msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self._robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

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
            new_wp = self._lane_carrot()
            if new_wp is not None and self._active_wp.dist_to(new_wp) > self._replan_dist:
                if self._last_goal_send_time is None or (
                    (now - self._last_goal_send_time).nanoseconds / 1e9 >= self._replan_min_dt
                ):
                    self._send_goal(new_wp)

        # Always publish the lane path for visualisation / RPP
        pts = self._extract_centreline()
        self._path_pub.publish(self._build_path(pts))

    # ── Waypoint generation ───────────────────────────────────────────────

    def _next_waypoint(self) -> Optional[_Waypoint]:
        """
        GPS mode: pop the front of the GPS queue.
        Sim mode: project a carrot along the lane centreline.
        Falls back to a straight-ahead carrot if the lane is not visible.
        """
        if self._gps_enabled:
            if not self._gps_queue:
                self.get_logger().info(
                    'GPS queue empty — waiting for fix.',
                    throttle_duration_sec=3.0)
                return None
            return self._gps_queue.popleft()
        else:
            carrot = self._lane_carrot()
            if carrot is None:
                carrot = self._held_lane_carrot()
                if carrot is None:
                    self.get_logger().warn(
                        'No lane visible — holding current course.',
                        throttle_duration_sec=2.0)
            return carrot

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

    # ── Centreline extraction ─────────────────────────────────────────────

    def _extract_centreline(self) -> list[tuple[float, float]]:
        """
        Walk the lane costmap forward from the robot.  Each row's centreline
        is the centroid of free cells in the *connected* free band straddling
        the robot's lateral position (col 0 in base_link coords).

        The walk terminates as soon as one of the following happens:
          * a row has no free cells (blocked — e.g. the closed end of a
            U-turn);
          * the next row's centroid jumps laterally by more than
            ``max_lateral_jump_m`` compared with the previous row (the free
            region ahead has "teleported" across a closure).

        This prevents the carrot from leaping across a U-turn's closed end
        into the return lane, which is what was causing the robot to drive
        straight into the far line.
        """
        g = self._grid
        if g is None:
            return []

        W, H   = g.info.width, g.info.height
        res    = g.info.resolution
        orig_y = g.info.origin.position.y  # lateral offset of col 0 in base_link

        data = np.frombuffer(bytes(g.data), dtype=np.int8).reshape(H, W)
        # Strictly-free cells only (cost == 0).  Treating unknown (-1) as
        # drivable made the free band balloon out to the grid edge
        # whenever one lane was missing from the costmap — the centroid
        # then sat well outside the corridor and the carrot got sent
        # across the opposing lane.
        free_mask = (data == 0)

        # Expected lane half-width, in cells.  We clamp the free band to a
        # window of this size on either side of the previous centroid so a
        # one-sided lane detection can't yank the carrot sideways.
        lane_half_m        = 1.2
        lane_half_cols     = max(4, int(round(lane_half_m / res)))
        max_lateral_jump_m = 0.5

        pts: list[tuple[float, float]] = []
        prev_lat: Optional[float] = None
        started = False  # True once we've found the first row with free cells

        # Column that corresponds to lateral = 0 (robot centreline).
        centre_col0 = int(round(-orig_y / res))

        # Close-range rows (roughly within ``min_detection_depth_m``) are
        # always unknown because the depth camera can't see under its own
        # nose.  Don't terminate on those — skip forward until we find the
        # first row with any free cells, then start tracking the corridor.
        entered_band = False

        for row in range(H):
            fwd = row * res
            if fwd > self._lookahead:
                break
            row_free = free_mask[row]
            if not row_free.any():
                if not entered_band:
                    # Still in the blind close-range zone — keep searching.
                    continue
                # Row is entirely unknown/blocked after we've already seen
                # the drivable band — stop extending the centreline rather
                # than guessing past the sensed region.
                break

            # Window around the target column — this is the key fix.
            target_col = centre_col0 if prev_lat is None else int(round(
                (prev_lat - orig_y) / res))
            target_col = max(0, min(W - 1, target_col))
            lo = max(0, target_col - lane_half_cols)
            hi = min(W, target_col + lane_half_cols + 1)

            window_free = row_free[lo:hi]
            if not window_free.any():
                if not entered_band:
                    continue
                break  # corridor closed off around the robot's heading

            diff   = np.diff(window_free.astype(np.int8))
            starts = np.where(diff ==  1)[0] + 1
            ends   = np.where(diff == -1)[0] + 1
            if window_free[0]:
                starts = np.r_[0, starts]
            if window_free[-1]:
                ends = np.r_[ends, hi - lo]

            rel_target = target_col - lo
            picked = None
            for s, e in zip(starts, ends):
                if s <= rel_target < e:
                    picked = (s, e)
                    break
            if picked is None:
                if not entered_band:
                    continue
                break
            entered_band = True
            s, e = picked
            centre_col = lo + 0.5 * (s + e - 1)
            lateral = orig_y + centre_col * res

            if prev_lat is not None and abs(lateral - prev_lat) > max_lateral_jump_m:
                break  # centroid jumped — likely stepped across a line
            pts.append((fwd, lateral))
            prev_lat = lateral
            started = True

        return pts

    # ── Path message builder ──────────────────────────────────────────────

    def _build_path(self, pts: list[tuple[float, float]]) -> Path:
        path = Path()
        path.header.stamp    = self.get_clock().now().to_msg()
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

    # ── Nav2 interaction ──────────────────────────────────────────────────

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
            self._consecutive_aborts = min(self._consecutive_aborts + 1, 6)
            backoff_s = min(2.0 ** self._consecutive_aborts * 0.25, 4.0)
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
        elif status == GoalStatus.STATUS_CANCELED:
            pass  # expected when we replan
        else:
            # ABORTED or similar — apply exponential backoff before retrying.
            self._consecutive_aborts = min(self._consecutive_aborts + 1, 6)
            backoff_s = min(2.0 ** self._consecutive_aborts * 0.25, 4.0)
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
    rclpy.spin(IGVCNavigatorNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()