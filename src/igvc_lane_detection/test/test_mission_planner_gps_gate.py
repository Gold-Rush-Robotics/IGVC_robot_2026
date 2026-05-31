"""Unit tests for mission-planner GPS startup gating."""

from __future__ import annotations

from igvc_lane_detection.mission_planner import MissionPlannerNode, _Waypoint


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _make_node(*, required_fixes: int = 1) -> MissionPlannerNode:
    node = object.__new__(MissionPlannerNode)
    node._waypoints = [
        _Waypoint(lat=42.293123, lon=-83.715456, kind='start'),
        _Waypoint(lat=42.293900, lon=-83.715999, kind='end'),
    ]
    node._gps_start_match_enabled = True
    node._gps_start_match_digits = 3
    node._gps_start_match_required_fixes = required_fixes
    node._gps_start_match_count = 0
    node.get_logger = lambda: _Logger()
    return node


def test_gps_start_area_rejects_wrong_lat_prefix():
    node = _make_node()

    assert not node._gps_start_area_matches(41.293123, -83.715456)
    assert node._gps_start_match_count == 0


def test_gps_start_area_rejects_wrong_lon_prefix():
    node = _make_node()

    assert not node._gps_start_area_matches(42.293123, -82.715456)
    assert node._gps_start_match_count == 0


def test_gps_start_area_accepts_matching_prefixes():
    node = _make_node()

    assert node._gps_start_area_matches(42.299999, -83.799999)
    assert node._gps_start_match_count == 1


def test_gps_start_area_can_require_consecutive_matches():
    node = _make_node(required_fixes=2)

    assert not node._gps_start_area_matches(42.299999, -83.799999)
    assert node._gps_start_area_matches(42.298888, -83.788888)
    assert node._gps_start_match_count == 2