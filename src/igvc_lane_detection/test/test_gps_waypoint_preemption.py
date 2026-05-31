"""Tests for gps_waypoint_test goal refresh behavior."""

from __future__ import annotations

from igvc_lane_detection.gps_waypoint_test import GpsWaypointTestNode


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _make_node(**overrides):
    node = object.__new__(GpsWaypointTestNode)
    node._publish_map_to_odom = lambda: None
    node.get_logger = lambda: _Logger()
    node._now = lambda: 10.0
    node._send_goal_calls = []
    node._send_goal = lambda *args: node._send_goal_calls.append(args)

    node._reached = False
    node._use_gps = False
    node._latest_odom = (1.0, 0.0)
    node._target_x = 10.0
    node._target_y = 0.0
    node._goal_tol = 0.1
    node._min_dist_seen = None
    node._recovery_increase = 100.0
    node._goal_pending = False
    node._goal_handle = None
    node._allow_active_goal_refresh = False
    node._last_send_pos = None
    node._last_send_sec = None
    node._goal_update_dist = 0.5
    node._resend_period = 5.0

    for key, value in overrides.items():
        setattr(node, key, value)
    return node


def test_tick_navigate_sends_initial_goal_without_active_goal():
    node = _make_node()

    node._tick_navigate()

    assert len(node._send_goal_calls) == 1


def test_tick_navigate_does_not_preempt_active_goal_by_default():
    node = _make_node(_goal_handle=object())

    node._tick_navigate()

    assert node._send_goal_calls == []


def test_tick_navigate_can_refresh_active_goal_when_enabled():
    node = _make_node(
        _goal_handle=object(),
        _allow_active_goal_refresh=True,
        _last_send_pos=(0.0, 0.0),
        _last_send_sec=9.0,
    )

    node._tick_navigate()

    assert len(node._send_goal_calls) == 1
