"""Spoof GPS and absolute heading from relative ZED odometry."""

from __future__ import annotations

import math
import random

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus


ODOM_TOPIC = '/front_zed_camera_x/zed_node/odom'
GPS_TOPIC = '/fix'
HEADING_TOPIC = '/front_zed_camera_2i/imu/heading'
EARTH_RADIUS_M = 6_378_137.0


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_odom(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SimGpsSpoofer(Node):
    def __init__(self) -> None:
        super().__init__('sim_gps_spoofer')

        self.declare_parameter('origin_lat', 42.400510946)
        self.declare_parameter('origin_lon', -83.130640432)
        self.origin_lat = float(self.get_parameter('origin_lat').value)
        self.origin_lon = float(self.get_parameter('origin_lon').value)

        self.declare_parameter('initial_heading_deg', 0.0)
        self.declare_parameter('randomize_heading', False)
        if bool(self.get_parameter('randomize_heading').value):
            initial_heading = random.uniform(0.0, 2.0 * math.pi)
        else:
            initial_heading = math.radians(
                float(self.get_parameter('initial_heading_deg').value))

        self.start_xy: tuple[float, float] | None = None
        self.yaw_offset: float | None = None
        self.initial_heading = wrap(initial_heading)

        self.gps_pub = self.create_publisher(NavSatFix, GPS_TOPIC, 10)
        self.heading_pub = self.create_publisher(Imu, HEADING_TOPIC, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self.on_odom, 10)

        self.get_logger().info(
            f'origin=({self.origin_lat:.8f}, {self.origin_lon:.8f}), '
            f'initial_heading={math.degrees(self.initial_heading):.1f} deg')

    def on_odom(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        odom_yaw = yaw_from_odom(msg)

        if self.start_xy is None:
            self.start_xy = (x, y)
            self.yaw_offset = wrap(self.initial_heading - odom_yaw)

        assert self.start_xy is not None and self.yaw_offset is not None
        dx = x - self.start_xy[0]
        dy = y - self.start_xy[1]
        east = math.cos(self.yaw_offset) * dx - math.sin(self.yaw_offset) * dy
        north = math.sin(self.yaw_offset) * dx + math.cos(self.yaw_offset) * dy
        heading = wrap(odom_yaw + self.yaw_offset)

        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()

        fix = NavSatFix()
        fix.header.stamp = stamp
        fix.header.frame_id = 'gps_gps_antenna_link'
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude = self.origin_lat + math.degrees(north / EARTH_RADIUS_M)
        fix.longitude = self.origin_lon + math.degrees(
            east / (EARTH_RADIUS_M * math.cos(math.radians(self.origin_lat))))
        fix.position_covariance[0] = 0.25
        fix.position_covariance[4] = 0.25
        fix.position_covariance[8] = 1.0
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.gps_pub.publish(fix)

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = msg.child_frame_id or 'base_link'
        imu.orientation.z = math.sin(heading / 2.0)
        imu.orientation.w = math.cos(heading / 2.0)
        imu.orientation_covariance[8] = 0.01
        self.heading_pub.publish(imu)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimGpsSpoofer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()