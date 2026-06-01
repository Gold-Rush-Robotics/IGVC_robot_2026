"""Unit tests for costmap-to-centerline navigator behavior."""

from __future__ import annotations

import math

from builtin_interfaces.msg import Time

from costmap_fixtures import erase_rows, make_corridor_costmap

from igvc_lane_detection.navigator import IGVCNavigatorNode, _Waypoint


class _Now:
    nanoseconds = 10_000_000_000

    def to_msg(self):
        return Time()

    def __lt__(self, other):
        return False

    def __sub__(self, other):
        return self


class _Clock:
    def now(self):
        return _Now()


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _Future:
    def add_done_callback(self, callback):
        self.callback = callback


class _ActionClient:
    def __init__(self, ready=True):
        self.ready = ready
        self.goals = []

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return _Future()


def make_navigator(grid, *, lookahead=4.0, horizon=1.8):
    """Create a navigator shell without ROS subscriptions or Nav2 clients."""
    node = object.__new__(IGVCNavigatorNode)
    node._grid = grid
    node._lookahead = lookahead
    node._horizon = horizon
    node._grid_res = grid.info.resolution
    node._base_frame = 'base_link'
    node._gps_enabled = False
    node._follow_path_enabled = True
    node._controller_id = ''
    node._goal_checker_id = ''
    node._progress_checker_id = ''
    node._min_follow_path_poses = 5
    node._min_follow_path_length_m = 1.5
    node._path_sample_spacing_m = 0.10
    node._path_smooth_window = 5
    node._path_change_tolerance_m = 0.25
    node._path_change_tolerance_rad = 0.25
    node._max_path_lateral_jump_m = 0.5
    node._max_path_heading_rad = 0.75
    node._goal_pending = False
    node._goal_handle = None
    node._next_goal_seq = 0
    node._current_goal_seq = 0
    node._active_wp = None
    node._robot_xy = None
    node._last_goal_send_time = None
    node._last_sent_path = None
    node._replan_min_dt = 0.7
    node._abort_backoff_until = None
    node._path_nav = _ActionClient()
    node._path_pub = _Publisher()
    node._base_link_to_map = lambda forward, lateral: _Waypoint(
        forward, lateral)
    node._last_lane_wp = None
    node._last_lane_wp_time = None
    node.get_clock = lambda: _Clock()
    node.get_logger = lambda: _Logger()
    return node


def test_extract_centerline_follows_straight_corridor():
    """The navigator should recover a centerd straight lane corridor."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=2.4)
    node = make_navigator(grid)

    points = node._extract_centerline()

    assert points
    assert points[0][0] >= 0.25
    assert points[-1][0] <= 4.0
    assert max(abs(lateral) for _, lateral in points) <= 0.03


def test_extract_centerline_follows_smooth_curve():
    """The navigator should follow a smooth local curve without drift."""
    def centerline(forward):
        return 0.45 * math.sin(forward / 4.0 * math.pi / 2.0)

    grid = make_corridor_costmap(centerline=centerline, lane_width_m=2.4)
    node = make_navigator(grid)

    points = node._extract_centerline()

    errors = [
        abs(lateral - centerline(forward))
        for forward, lateral in points
    ]

    assert len(points) > 50
    assert max(errors) <= 0.08


def test_extract_centerline_ignores_gt_obstacle_stamps():
    """GT obstacles are lethal cells but should not become lane boundaries."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=2.4)
    data = list(grid.data)
    width = grid.info.width
    res = grid.info.resolution
    center_col = int(round((0.0 - grid.info.origin.position.y) / res))
    obstacle_row = int(round(1.8 / res))
    obstacle_radius_cells = int(round(0.35 / res))
    for row in range(obstacle_row - obstacle_radius_cells, obstacle_row + obstacle_radius_cells + 1):
        for col in range(center_col - obstacle_radius_cells, center_col + obstacle_radius_cells + 1):
            if 0 <= row < grid.info.height and 0 <= col < grid.info.width:
                if (row - obstacle_row) ** 2 + (col - center_col) ** 2 <= obstacle_radius_cells ** 2:
                    data[row * width + col] = 100
    grid.data = data
    node = make_navigator(grid)

    points = node._extract_centerline()

    assert points
    assert points[-1][0] >= 3.5
    assert max(abs(lateral) for _, lateral in points) <= 0.04


def test_extract_centerline_stops_at_blocked_gap_after_entering_lane():
    """The navigator should stop extending paths through sensed gaps."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=2.4)
    grid = erase_rows(grid, 1.5, 1.8)
    node = make_navigator(grid)

    points = node._extract_centerline()

    assert points
    assert points[-1][0] < 1.5


def test_extract_centerline_rejects_large_lateral_jump():
    """The navigator should not jump sideways into a disconnected band."""
    def centerline(forward):
        return 0.0 if forward < 1.6 else 0.8

    grid = make_corridor_costmap(centerline=centerline, lane_width_m=1.0)
    node = make_navigator(grid)

    points = node._extract_centerline()

    assert points
    assert points[-1][0] < 1.65
    assert max(abs(lateral) for _, lateral in points) <= 0.05


def test_lane_carrot_uses_first_centerline_point_at_horizon():
    """The lane carrot should pick the first centerline point past horizon."""
    grid = make_corridor_costmap(
        centerline=lambda forward: 0.1 * forward,
        lane_width_m=2.4,
    )
    node = make_navigator(grid, horizon=1.8)

    carrot = node._lane_carrot()

    assert carrot is not None
    assert carrot.x >= 1.8
    assert abs(carrot.x - 1.8) <= grid.info.resolution
    assert abs(carrot.y - 0.18) <= 0.08


def test_lane_path_from_costmap_exposes_controller_path():
    """The navigator should expose centerline points as a Path message."""
    grid = make_corridor_costmap(
        centerline=lambda forward: 0.1 * forward,
        lane_width_m=2.4,
    )
    node = make_navigator(grid)

    path = node._lane_path_from_costmap()

    assert path.header.frame_id == 'base_link'
    assert len(path.poses) > 20
    assert path.poses[0].pose.position.x >= 0.25
    assert abs(path.poses[-1].pose.position.y - 0.4) <= 0.08
    assert all(pose.pose.orientation.w != 0.0 for pose in path.poses)


def test_conditioned_path_uses_stable_sample_spacing():
    """Controller paths should be uniformly spaced, not raw costmap rows."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=2.4)
    node = make_navigator(grid)

    path = node._lane_path_from_costmap()
    spacings = [
        b.pose.position.x - a.pose.position.x
        for a, b in zip(path.poses[:-1], path.poses[1:])
    ]

    assert min(spacings[:-1]) >= 0.08
    assert max(spacings) <= 0.12


def test_path_validation_rejects_short_paths():
    """Valid paths need enough geometry for the controller to track."""
    grid = make_corridor_costmap(sensed_end_m=0.5)
    node = make_navigator(grid)

    path = node._lane_path_from_costmap()

    assert not node._path_is_valid(path)


def test_path_validation_rejects_lateral_jump():
    """Discontinuous path geometry should not reach Nav2."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=2.4)
    node = make_navigator(grid)
    path = node._lane_path_from_costmap()
    path.poses[5].pose.position.y += 1.0

    assert not node._path_is_valid(path)


def test_conditioned_path_limits_sideways_slew_for_wide_lanes():
    """Wide straight lanes should not produce sideways/U-turn-like path starts."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=4.5)
    node = make_navigator(grid)
    raw = [(0.05, 0.0), (0.10, 2.0), (1.0, 2.0), (2.0, 2.0)]

    points = node._condition_path_points(raw)
    headings = [
        abs(math.atan2(b[1] - a[1], b[0] - a[0]))
        for a, b in zip(points[:-1], points[1:])
        if math.hypot(b[0] - a[0], b[1] - a[1]) > 1.0e-6
    ]

    assert headings
    assert max(headings) <= node._max_path_heading_rad + 0.05


def test_path_validation_rejects_sideways_heading():
    """A path that starts by sweeping sideways should be blocked."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=2.4)
    node = make_navigator(grid)
    path = node._build_path([(0.0, 0.0), (0.10, 1.0), (1.0, 1.0)])

    assert not node._path_is_valid(path)


def test_path_change_hysteresis_ignores_small_shifts():
    """Small perception jitter should not cause a fresh FollowPath goal."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=2.4)
    node = make_navigator(grid)
    first = node._lane_path_from_costmap()
    shifted_grid = make_corridor_costmap(centerline=0.03, lane_width_m=2.4)
    node._last_sent_path = first
    node._grid = shifted_grid

    assert not node._path_changed_enough(node._lane_path_from_costmap())


def test_path_change_hysteresis_detects_meaningful_shift():
    """Real path movement should trigger a fresh FollowPath goal."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=2.4)
    node = make_navigator(grid)
    first = node._lane_path_from_costmap()
    shifted_grid = make_corridor_costmap(centerline=0.4, lane_width_m=2.4)
    node._last_sent_path = first
    node._grid = shifted_grid

    assert node._path_changed_enough(node._lane_path_from_costmap())


def test_update_publishes_lane_path_while_localization_initializes():
    """Path output should not depend on Nav2 goal dispatch readiness."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=2.4)
    node = make_navigator(grid)
    publisher = _Publisher()
    node._path_pub = publisher
    node._loc_status = 'initializing'

    node._update()

    assert len(publisher.messages) == 1
    assert len(publisher.messages[0].poses) > 20


def test_update_sends_follow_path_in_sim_mode():
    """Sim mode should send lane paths to Nav2 FollowPath by default."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=2.4)
    node = make_navigator(grid)
    node._loc_status = 'sim'

    node._update()

    assert len(node._path_nav.goals) == 1
    assert len(node._path_nav.goals[0].path.poses) > 20
    assert node._path_nav.goals[0].controller_id == ''


def test_update_skips_follow_path_when_path_is_unchanged():
    """A stable path should not be resent on every update tick."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=2.4)
    node = make_navigator(grid)
    node._loc_status = 'sim'

    node._update()
    node._last_goal_send_time = None
    node._update()

    assert len(node._path_nav.goals) == 1


def test_update_uses_carrot_when_follow_path_disabled():
    """The legacy carrot path should remain available as a fallback."""
    grid = make_corridor_costmap(centerline=0.0, lane_width_m=2.4)
    node = make_navigator(grid)
    node._loc_status = 'sim'
    node._follow_path_enabled = False
    sent_waypoints = []
    node._send_goal = lambda waypoint: sent_waypoints.append(waypoint) or True

    node._update()

    assert sent_waypoints
    assert not node._path_nav.goals
