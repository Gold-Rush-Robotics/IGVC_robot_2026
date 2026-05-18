"""Unit tests for the centerline-waypoint navigation strategy."""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from igvc_lane_detection.navigator import IGVCNavigatorNode, _Waypoint


# ── Logger / clock stubs (no ROS init) ────────────────────────────────────────

class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _make_node(**overrides):
    """Build a navigator shell that bypasses ROS subscriptions."""
    node = object.__new__(IGVCNavigatorNode)
    node._gps_enabled = False
    node._nav_strategy = 'centerline_waypoints'
    node._centerline_source_json = ''
    node._centerline_lookahead_m = 4.0
    node._centerline_advance_m = 0.6
    node._centerline_loop = True
    node._centerline_search_window_m = 8.0
    node._centerline_xy = None
    node._centerline_s = None
    node._centerline_total_s = 0.0
    node._centerline_progress_idx = 0
    node._centerline_load_attempted = False
    node._centerline_unavailable_warned = False
    node._follow_path_enabled = True
    node._robot_xy = (0.0, 0.0)
    node._robot_yaw = 0.0
    node.get_logger = lambda: _Logger()
    for key, value in overrides.items():
        setattr(node, key, value)
    return node


def _write_track_json(tmp_path, centerline_xy, *, start=(0.0, 0.0), close_loop=False):
    pts = list(centerline_xy)
    if close_loop:
        pts = pts + [pts[0]]
    payload = {
        'schema': 'test',
        'centerline_m': [{'x': float(x) + start[0], 'y': float(y) + start[1]} for x, y in pts],
        'robot_start_pose': {'position_m': {'x': float(start[0]), 'y': float(start[1])}},
    }
    path = os.path.join(str(tmp_path), 'track.json')
    with open(path, 'w') as fh:
        json.dump(payload, fh)
    return path


# ── _load_centerline_from_json ────────────────────────────────────────────────

def test_load_centerline_from_json_basic(tmp_path):
    """Centerline points get shifted by robot_start_pose to map frame."""
    raw = [(10.0, 5.0), (12.0, 5.0), (14.0, 5.0)]
    path = _write_track_json(tmp_path, raw, start=(10.0, 5.0))
    node = _make_node(_centerline_source_json=path, _centerline_loop=False)

    assert node._load_centerline_from_json()
    assert node._centerline_xy == [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    # Arc length should reflect the 2-m spacing
    assert node._centerline_s == [0.0, 2.0, 4.0]
    assert node._centerline_total_s == pytest.approx(4.0)


def test_load_centerline_handles_missing_file():
    node = _make_node(_centerline_source_json='/no/such/file.json')
    assert not node._load_centerline_from_json()
    assert node._centerline_xy is None
    # Second call should be a no-op and still return False
    assert not node._load_centerline_from_json()


def test_load_centerline_handles_missing_centerline_key(tmp_path):
    path = os.path.join(str(tmp_path), 'bad.json')
    with open(path, 'w') as fh:
        json.dump({'schema': 'test'}, fh)
    node = _make_node(_centerline_source_json=path)
    assert not node._load_centerline_from_json()


def test_load_centerline_drops_duplicate_loop_close(tmp_path):
    raw = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    path = _write_track_json(tmp_path, raw, close_loop=True)
    node = _make_node(_centerline_source_json=path, _centerline_loop=True)

    assert node._load_centerline_from_json()
    # The repeated first point at the end should be dropped
    assert len(node._centerline_xy) == 3
    # Loop total length includes closing leg from last back to first
    assert node._centerline_total_s == pytest.approx(
        1.0 + 1.0 + math.hypot(1.0, 1.0))


# ── _centerline_closest_index ────────────────────────────────────────────────

def test_centerline_closest_index_first_call_does_full_scan(tmp_path):
    raw = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
    path = _write_track_json(tmp_path, raw, close_loop=False)
    node = _make_node(_centerline_source_json=path, _centerline_loop=False)
    assert node._load_centerline_from_json()

    # Robot is closest to index 3 — first call must find it via full scan
    assert node._centerline_closest_index(2.9, 0.1) == 3


def test_centerline_closest_index_advances_forward_only(tmp_path):
    raw = [(float(i), 0.0) for i in range(10)]
    path = _write_track_json(tmp_path, raw, close_loop=False)
    node = _make_node(
        _centerline_source_json=path,
        _centerline_loop=False,
        _centerline_search_window_m=3.0,
    )
    assert node._load_centerline_from_json()

    node._centerline_closest_index(0.0, 0.0)
    node._centerline_closest_index(2.0, 0.0)
    assert node._centerline_progress_idx == 2

    # A query 'behind' the cached index inside the forward window should
    # not jump backwards; nearest forward point wins.
    node._centerline_closest_index(1.5, 0.0)
    assert node._centerline_progress_idx == 2


def test_centerline_closest_index_loop_wraps(tmp_path):
    # 4-pt square loop with 1 m sides
    raw = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    path = _write_track_json(tmp_path, raw, close_loop=False)
    node = _make_node(
        _centerline_source_json=path,
        _centerline_loop=True,
        _centerline_search_window_m=10.0,
    )
    assert node._load_centerline_from_json()

    node._centerline_closest_index(0.0, 0.95)  # near idx 3
    assert node._centerline_progress_idx == 3
    # Robot moves to (0.05, 0.0): forward window must wrap to idx 0
    node._centerline_closest_index(0.05, 0.0)
    assert node._centerline_progress_idx == 0


# ── _centerline_advance_index ────────────────────────────────────────────────

def test_centerline_advance_index_within_segment(tmp_path):
    raw = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
    path = _write_track_json(tmp_path, raw, close_loop=False)
    node = _make_node(_centerline_source_json=path, _centerline_loop=False)
    assert node._load_centerline_from_json()

    # Lookahead 1.5 m from idx 0 should land at idx 2 (the first index
    # whose arc-length >= 1.5).
    assert node._centerline_advance_index(0, 1.5) == 2


def test_centerline_advance_index_no_loop_clamps_to_end(tmp_path):
    raw = [(float(i), 0.0) for i in range(5)]
    path = _write_track_json(tmp_path, raw, close_loop=False)
    node = _make_node(_centerline_source_json=path, _centerline_loop=False)
    assert node._load_centerline_from_json()

    # 100 m lookahead on a 4 m line should clamp to last index
    assert node._centerline_advance_index(0, 100.0) == 4


# ── _centerline_goal ──────────────────────────────────────────────────────────

def test_centerline_goal_returns_lookahead_point(tmp_path):
    raw = [(float(i), 0.0) for i in range(10)]
    path = _write_track_json(tmp_path, raw, close_loop=False)
    node = _make_node(
        _centerline_source_json=path,
        _centerline_loop=False,
        _centerline_lookahead_m=3.0,
    )
    node._robot_in_map = lambda: (0.0, 0.0)

    wp = node._centerline_goal()
    assert wp is not None
    # Should pick index 3 (arc-length 3.0 along the line at y=0)
    assert wp.x == pytest.approx(3.0)
    assert wp.y == pytest.approx(0.0)
    # Yaw points toward the next sample (+x)
    assert wp.yaw == pytest.approx(0.0, abs=1e-6)


def test_centerline_goal_missing_robot_pose_returns_none(tmp_path):
    raw = [(0.0, 0.0), (1.0, 0.0)]
    path = _write_track_json(tmp_path, raw, close_loop=False)
    node = _make_node(_centerline_source_json=path, _centerline_loop=False)
    node._robot_in_map = lambda: None
    assert node._centerline_goal() is None


def test_centerline_goal_missing_json_returns_none():
    node = _make_node(_centerline_source_json='')
    node._robot_in_map = lambda: (0.0, 0.0)
    assert node._centerline_goal() is None


# ── Strategy integration ──────────────────────────────────────────────────────

def test_next_waypoint_uses_centerline_strategy(tmp_path):
    raw = [(float(i), 0.0) for i in range(10)]
    path = _write_track_json(tmp_path, raw, close_loop=False)
    node = _make_node(
        _centerline_source_json=path,
        _centerline_loop=False,
        _centerline_lookahead_m=2.5,
    )
    node._robot_in_map = lambda: (0.0, 0.0)

    wp = node._next_waypoint()
    assert wp is not None
    assert wp.x == pytest.approx(3.0)  # first index with s >= 2.5
    assert wp.yaw is not None


def test_uses_follow_path_false_for_centerline_strategy(tmp_path):
    node = _make_node()
    assert node._uses_follow_path() is False


def test_waypoint_pose_stamped_includes_yaw():
    wp = _Waypoint(1.0, 2.0)
    wp.yaw = math.pi / 2
    ps = wp.to_pose_stamped('map', None)
    assert ps.pose.position.x == 1.0
    assert ps.pose.position.y == 2.0
    # quaternion z=sin(yaw/2), w=cos(yaw/2)
    assert ps.pose.orientation.z == pytest.approx(math.sin(math.pi / 4))
    assert ps.pose.orientation.w == pytest.approx(math.cos(math.pi / 4))
