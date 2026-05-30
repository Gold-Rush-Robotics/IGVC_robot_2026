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

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateThroughPoses
from sensor_msgs.msg import NavSatFix

from igvc_lane_detection.navigator import gps_to_map


class GpsWaypointTestNode(Node):

    def __init__(self) -> None:
        super().__init__('gps_waypoint_test')

        self.declare_parameter('target_lat', 42.400510946)   # 5004 Practice Mid
        self.declare_parameter('target_lon', -83.130640432)  # 5004 Practice Mid
        self.declare_parameter('origin_lat', 0.0)
        self.declare_parameter('origin_lon', 0.0)
        self.declare_parameter('gps_topic', '/fix')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('nav_action', 'navigate_through_poses')
        self.declare_parameter('goal_tolerance_m', 1.0)
        # Re-send the goal at most this often while we wait for arrival.
        self.declare_parameter('resend_period_sec', 5.0)
        # Intermediate waypoints are inserted every this many metres along the
        # straight-line path to the target so RPP always has a nearby carrot.
        self.declare_parameter('waypoint_spacing_m', 3.0)

        self._target_lat = float(self.get_parameter('target_lat').value)
        self._target_lon = float(self.get_parameter('target_lon').value)
        self._origin_lat = float(self.get_parameter('origin_lat').value)
        self._origin_lon = float(self.get_parameter('origin_lon').value)
        self._origin_set = (self._origin_lat != 0.0 or self._origin_lon != 0.0)
        self._gps_topic = self.get_parameter('gps_topic').value
        self._map_frame = self.get_parameter('map_frame').value
        self._nav_action = self.get_parameter('nav_action').value
        self._goal_tol = float(self.get_parameter('goal_tolerance_m').value)
        self._resend_period = float(self.get_parameter('resend_period_sec').value)
        self._waypoint_spacing = float(self.get_parameter('waypoint_spacing_m').value)

        if self._target_lat == 0.0 and self._target_lon == 0.0:
            self.get_logger().error(
                'target_lat/target_lon not set — nothing to navigate to. '
                'Pass them with --ros-args -p target_lat:=.. -p target_lon:=..')

        self._latest_fix: Optional[tuple[float, float]] = None
        self._goal_handle = None
        self._goal_pending = False
        self._reached = False
        self._last_send_sec: Optional[float] = None

        self._nav = ActionClient(self, NavigateThroughPoses, self._nav_action)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)
        self.create_subscription(
            NavSatFix, self._gps_topic, self._on_gps, sensor_qos)

        self._timer = self.create_timer(1.0, self._tick)
        self.get_logger().info(
            f'gps_waypoint_test: target=({self._target_lat:.6f}, '
            f'{self._target_lon:.6f}); listening on {self._gps_topic}')

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _on_gps(self, msg: NavSatFix) -> None:
        if msg.status.status < 0:
            return
        self._latest_fix = (msg.latitude, msg.longitude)
        if not self._origin_set:
            self._origin_lat = msg.latitude
            self._origin_lon = msg.longitude
            self._origin_set = True
            self.get_logger().warn(
                f'GPS origin anchored to first fix '
                f'({self._origin_lat:.6f}, {self._origin_lon:.6f}); this must '
                'match the navigator origin for the goal to be correct.')

    def _tick(self) -> None:
        if self._reached or self._goal_pending:
            return
        if self._target_lat == 0.0 and self._target_lon == 0.0:
            return
        if not self._origin_set or self._latest_fix is None:
            self.get_logger().info(
                'Waiting for a GPS fix / origin before sending goal…',
                throttle_duration_sec=3.0)
            return

        # Distance-to-target purely for logging.
        cur_e, cur_n = gps_to_map(self._latest_fix[0], self._latest_fix[1],
                                  self._origin_lat, self._origin_lon)
        tgt_e, tgt_n = gps_to_map(self._target_lat, self._target_lon,
                                  self._origin_lat, self._origin_lon)
        dist = math.hypot(tgt_e - cur_e, tgt_n - cur_n)
        if dist <= self._goal_tol:
            self.get_logger().info(
                f'Target reached (dist={dist:.2f} m ≤ {self._goal_tol:.2f} m).')
            self._reached = True
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if (self._goal_handle is not None and self._last_send_sec is not None
                and (now_sec - self._last_send_sec) < self._resend_period):
            return

        self._send_goal(tgt_e, tgt_n, cur_e, cur_n, dist)

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
        self._last_send_sec = self.get_clock().now().nanoseconds / 1e9
        self.get_logger().info(
            f'Sending GPS goal: map ({tgt_e:.2f}, {tgt_n:.2f}), '
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
            self.get_logger().warn('NavigateToPose goal rejected.')
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
        # Arrival is confirmed by the distance check in _tick; the result just
        # frees us to (re)send if Nav2 finished short of the tolerance.
        self.get_logger().info('NavigateToPose goal finished; re-checking distance.')


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
