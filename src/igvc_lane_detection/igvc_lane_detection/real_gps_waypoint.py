"""Turn toward one GPS waypoint, drive straight, then stop."""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from rclpy.time import Time
from sensor_msgs.msg import NavSatFix, NavSatStatus

from igvc_lane_detection.gps_geometry import (gps_to_enu, wrap_angle,
                                              yaw_from_quaternion_xyzw)


class RealGPSWaypoint(Node):
    """Simple GPS-to-TwistStamped waypoint driver."""

    _WAITING = 'waiting'
    _TURNING = 'turning'
    _DRIVING = 'driving'
    _STOPPED = 'stopped'

    def __init__(self) -> None:
        super().__init__('real_gps_waypoint')

        self.declare_parameter('goal_lat', 0.0)
        self.declare_parameter('goal_lon', 0.0)
        self.declare_parameter('gps_topic', '/gps/fix')
        self.declare_parameter('odom_topic', '/front_zed_camera_x/zed_node/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter('linear_speed_mps', 0.4)
        self.declare_parameter('approach_speed_mps', 0.12)
        self.declare_parameter('slowdown_distance_m', 2.0)
        self.declare_parameter('goal_tolerance_m', 0.75)
        self.declare_parameter('turn_tolerance_deg', 5.0)
        self.declare_parameter('reacquire_heading_tolerance_deg', 20.0)
        self.declare_parameter('angular_gain', 1.5)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('max_gps_age_sec', 2.0)
        self.declare_parameter('max_odom_age_sec', 1.0)
        self.declare_parameter('require_valid_fix', True)
        self.declare_parameter('odom_yaw_offset_deg', 0.0)

        self._goal_lat = float(self.get_parameter('goal_lat').value)
        self._goal_lon = float(self.get_parameter('goal_lon').value)
        self._gps_topic = str(self.get_parameter('gps_topic').value)
        self._odom_topic = str(self.get_parameter('odom_topic').value)
        self._publish_hz = float(self.get_parameter('publish_hz').value)
        self._linear_speed = float(self.get_parameter('linear_speed_mps').value)
        self._approach_speed = float(
            self.get_parameter('approach_speed_mps').value)
        self._slowdown_distance = float(
            self.get_parameter('slowdown_distance_m').value)
        self._goal_tolerance = float(self.get_parameter('goal_tolerance_m').value)
        self._turn_tolerance = math.radians(
            float(self.get_parameter('turn_tolerance_deg').value))
        self._reacquire_tolerance = math.radians(
            float(self.get_parameter('reacquire_heading_tolerance_deg').value))
        self._angular_gain = float(self.get_parameter('angular_gain').value)
        self._max_angular_speed = float(
            self.get_parameter('max_angular_speed').value)
        self._max_gps_age = float(self.get_parameter('max_gps_age_sec').value)
        self._max_odom_age = float(self.get_parameter('max_odom_age_sec').value)
        self._require_valid_fix = bool(
            self.get_parameter('require_valid_fix').value)
        self._odom_yaw_offset = math.radians(
            float(self.get_parameter('odom_yaw_offset_deg').value))

        self._latest_fix: Optional[NavSatFix] = None
        self._latest_fix_time: Optional[Time] = None
        self._latest_odom: Optional[Odometry] = None
        self._latest_odom_time: Optional[Time] = None
        self._state = self._WAITING

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(
            NavSatFix, self._gps_topic, self._on_gps, sensor_qos)
        self.create_subscription(
            Odometry, self._odom_topic, self._on_odom, sensor_qos)
        self._cmd_pub = self.create_publisher(TwistStamped, "/diff_drive_controller/cmd_vel", 10)

        timer_period = 1.0 / max(1.0, self._publish_hz)
        self.create_timer(timer_period, self._tick)

        self.get_logger().info(
            f'real_gps_waypoint: goal=({self._goal_lat:.8f}, '
            f'{self._goal_lon:.8f}), gps={self._gps_topic}, '
            f'odom={self._odom_topic}, cmd_vel={"/diff_drive_controller/cmd_vel"}')

    def _on_gps(self, msg: NavSatFix) -> None:
        if self._require_valid_fix and msg.status.status < NavSatStatus.STATUS_FIX:
            return
        self._latest_fix = msg
        self._latest_fix_time = self.get_clock().now()

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg
        self._latest_odom_time = self.get_clock().now()

    def _age_sec(self, timestamp: Time) -> float:
        age_sec = (self.get_clock().now() - timestamp).nanoseconds / 1e9
        return max(0.0, age_sec)

    def _goal_configured(self) -> bool:
        return not (math.isclose(self._goal_lat, 0.0)
                    and math.isclose(self._goal_lon, 0.0))

    def _inputs_ready(self) -> bool:
        if self._latest_fix is None or self._latest_fix_time is None:
            self.get_logger().info(
                'Waiting for GPS fix before waypoint drive.',
                throttle_duration_sec=2.0)
            return False
        if self._latest_odom is None or self._latest_odom_time is None:
            self.get_logger().info(
                'Waiting for odom before waypoint drive.',
                throttle_duration_sec=2.0)
            return False
        if self._age_sec(self._latest_fix_time) > self._max_gps_age:
            self.get_logger().warn(
                'GPS fix is stale; stopping.', throttle_duration_sec=2.0)
            return False
        if self._age_sec(self._latest_odom_time) > self._max_odom_age:
            self.get_logger().warn(
                'Odom is stale; stopping.', throttle_duration_sec=2.0)
            return False
        return True

    def _goal_vector(self) -> tuple[float, float]:
        fix = self._latest_fix
        assert fix is not None
        east_m, north_m = gps_to_enu(
            self._goal_lat, self._goal_lon, fix.latitude, fix.longitude)
        distance_m = math.hypot(east_m, north_m)
        bearing_rad = math.atan2(north_m, east_m)
        return distance_m, bearing_rad

    def _current_yaw(self) -> float:
        odom = self._latest_odom
        assert odom is not None
        orientation = odom.pose.pose.orientation
        yaw_rad = yaw_from_quaternion_xyzw(
            orientation.x, orientation.y, orientation.z, orientation.w)
        return wrap_angle(yaw_rad + self._odom_yaw_offset)

    def _publish_stop(self) -> None:
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x = 0.0
        cmd.twist.angular.z = 0.0
        self._cmd_pub.publish(cmd)

    def _publish_turn(self, heading_error: float) -> None:
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        angular_speed = self._angular_gain * heading_error
        cmd.twist.angular.z = max(
            -self._max_angular_speed,
            min(self._max_angular_speed, angular_speed),
        )
        self._cmd_pub.publish(cmd)

    def _drive_speed_for_distance(self, distance_m: float) -> float:
        if distance_m >= self._slowdown_distance:
            return self._linear_speed
        usable_range = max(0.01, self._slowdown_distance - self._goal_tolerance)
        scale = max(0.0, (distance_m - self._goal_tolerance) / usable_range)
        speed_mps = self._linear_speed * scale
        return min(self._linear_speed, max(self._approach_speed, speed_mps))

    def _publish_drive(self, distance_m: float) -> None:
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x = self._drive_speed_for_distance(distance_m)
        cmd.twist.angular.z = 0.0
        self._cmd_pub.publish(cmd)

    def _tick(self) -> None:
        if not self._goal_configured():
            self.get_logger().warn(
                'goal_lat/goal_lon are not set; publishing stop.',
                throttle_duration_sec=3.0)
            self._publish_stop()
            return
        if self._state == self._STOPPED:
            self._publish_stop()
            return
        if not self._inputs_ready():
            self._state = self._WAITING
            self._publish_stop()
            return

        distance_m, bearing_rad = self._goal_vector()
        if distance_m <= self._goal_tolerance:
            self.get_logger().info(
                f'Goal reached: distance={distance_m:.2f} m. Stopping.')
            self._state = self._STOPPED
            self._publish_stop()
            return

        heading_error = wrap_angle(bearing_rad - self._current_yaw())
        if self._state == self._WAITING:
            self._state = self._TURNING
        elif (self._state == self._DRIVING
              and abs(heading_error) > self._reacquire_tolerance):
            self.get_logger().info(
                f'Heading drifted {math.degrees(heading_error):.1f} deg; '
                'turning in place again.')
            self._state = self._TURNING

        if self._state == self._TURNING:
            if abs(heading_error) > self._turn_tolerance:
                self._publish_turn(heading_error)
                return
            self.get_logger().info(
                f'Heading aligned within '
                f'{math.degrees(self._turn_tolerance):.1f} deg; '
                'driving straight.')
            self._state = self._DRIVING

        self._publish_drive(distance_m)


def main(args=None) -> None:
    """Run the simple GPS waypoint driver."""
    rclpy.init(args=args)
    node = RealGPSWaypoint()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
 