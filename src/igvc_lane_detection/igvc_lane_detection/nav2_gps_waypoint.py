#!/usr/bin/env python3
"""Follow GPS waypoint(s) using Nav2's GPS waypoint follower.

This mirrors the ``nav2_gps_waypoint_follower_demo`` nodes
(``logged_waypoint_follower`` / ``interactive_waypoint_follower``) from
https://github.com/ros-navigation/navigation2_tutorials but is driven by ROS
parameters so it can be launched the same way as the previous test node::

    ros2 run igvc_lane_detection nav2_gps_waypoint_node --ros-args \
        -p goal_lat:=42.400510946 -p goal_lon:=-83.130518968

It hands the goal to Nav2's ``/follow_gps_waypoints`` action server via
``nav2_simple_commander``.  Nav2 (planner/controller/bt_navigator) and
``robot_localization`` (``navsat_transform`` + dual EKF providing the
``map -> odom`` transform) MUST already be running, e.g. the demo's
``dual_ekf_navsat.launch.py`` + ``gps_waypoint_follower.launch.py`` equivalent
for this robot.

Provide either:
* ``goal_lat`` / ``goal_lon`` (+ optional ``goal_yaw_deg``) for a single
  waypoint, or
* ``waypoints_file`` pointing at a YAML file with the demo format::

      waypoints:
        - latitude: 42.400510946
          longitude: -83.130518968
          yaw: 0.0
"""

from __future__ import annotations

import math
from typing import List, Optional

import rclpy
import yaml
from geographic_msgs.msg import GeoPose
from geometry_msgs.msg import Quaternion
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.node import Node


def quaternion_from_yaw(yaw: float) -> Quaternion:
    """Build a quaternion from a planar yaw (rotation about +z)."""
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def lat_lon_yaw_to_geopose(latitude: float, longitude: float,
                           yaw: float = 0.0) -> GeoPose:
    """Create a ``geographic_msgs/GeoPose`` from lat/lon/yaw."""
    geopose = GeoPose()
    geopose.position.latitude = latitude
    geopose.position.longitude = longitude
    geopose.orientation = quaternion_from_yaw(yaw)
    return geopose


class Nav2GpsWaypoint(Node):
    """Send GPS waypoint(s) to Nav2's GPS waypoint follower."""

    def __init__(self) -> None:
        super().__init__('nav2_gps_waypoint')

        self.declare_parameter('goal_lat', 0.0)
        self.declare_parameter('goal_lon', 0.0)
        self.declare_parameter('goal_yaw_deg', 0.0)
        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('localizer', 'robot_localization')

        self._goal_lat = float(self.get_parameter('goal_lat').value)
        self._goal_lon = float(self.get_parameter('goal_lon').value)
        self._goal_yaw = math.radians(
            float(self.get_parameter('goal_yaw_deg').value))
        self._waypoints_file = str(self.get_parameter('waypoints_file').value)
        self._localizer = str(self.get_parameter('localizer').value)

        self._navigator = BasicNavigator('nav2_gps_waypoint_navigator')

    def _load_waypoints(self) -> Optional[List[GeoPose]]:
        if self._waypoints_file:
            with open(self._waypoints_file, 'r') as handle:
                data = yaml.safe_load(handle)
            waypoints = []
            for wp in data['waypoints']:
                waypoints.append(lat_lon_yaw_to_geopose(
                    float(wp['latitude']), float(wp['longitude']),
                    float(wp.get('yaw', 0.0))))
            self.get_logger().info(
                f'Loaded {len(waypoints)} waypoint(s) from '
                f'{self._waypoints_file}.')
            return waypoints

        if math.isclose(self._goal_lat, 0.0) and math.isclose(self._goal_lon,
                                                              0.0):
            self.get_logger().error(
                'No goal set: provide goal_lat/goal_lon or waypoints_file.')
            return None

        self.get_logger().info(
            f'Single GPS waypoint: ({self._goal_lat:.8f}, '
            f'{self._goal_lon:.8f}), yaw={math.degrees(self._goal_yaw):.1f} deg.')
        return [lat_lon_yaw_to_geopose(
            self._goal_lat, self._goal_lon, self._goal_yaw)]

    def run(self) -> bool:
        """Block until the waypoint(s) are followed; return success."""
        waypoints = self._load_waypoints()
        if not waypoints:
            return False

        self.get_logger().info(
            f"Waiting for Nav2 to become active (localizer='"
            f"{self._localizer}')...")
        self._navigator.waitUntilNav2Active(localizer=self._localizer)
        self.get_logger().info('Nav2 active; following GPS waypoint(s).')

        self._navigator.followGpsWaypoints(waypoints)
        while not self._navigator.isTaskComplete():
            feedback = self._navigator.getFeedback()
            if feedback is not None:
                self.get_logger().info(
                    f'Executing waypoint '
                    f'{feedback.current_waypoint + 1}/{len(waypoints)}',
                    throttle_duration_sec=2.0)
            rclpy.spin_once(self._navigator, timeout_sec=0.1)

        result = self._navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('GPS waypoint(s) completed successfully.')
            return True
        self.get_logger().error(f'GPS waypoint following ended: {result}.')
        return False


def main(args=None) -> None:
    """Run the Nav2 GPS waypoint follower client."""
    rclpy.init(args=args)
    node = Nav2GpsWaypoint()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
