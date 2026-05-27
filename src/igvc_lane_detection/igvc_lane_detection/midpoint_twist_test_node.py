from __future__ import annotations

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path


class MidpointTwistTestNode(Node):
    def __init__(self) -> None:
        super().__init__('midpoint_twist_test_node')

        self.declare_parameter('path_topic', '/lane_ground_truth_midpoint_path')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('output_topic', '/midpoint_cmd_vel')
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter('lookahead_points', 15)
        self.declare_parameter('linear_speed_mps', 0.6)
        self.declare_parameter('angular_gain', 1.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('max_odom_age_sec', 0.1)
        self.declare_parameter('deadman_timeout_sec', 0.3)

        self._path_topic = str(self.get_parameter('path_topic').value)
        self._odom_topic = str(self.get_parameter('odom_topic').value)
        self._output_topic = str(self.get_parameter('output_topic').value)
        self._hz = float(self.get_parameter('publish_hz').value)
        self._lookahead = int(self.get_parameter('lookahead_points').value)
        self._v = float(self.get_parameter('linear_speed_mps').value)
        self._k = float(self.get_parameter('angular_gain').value)
        self._w_max = float(self.get_parameter('max_angular_speed').value)
        self._max_odom_age = float(self.get_parameter('max_odom_age_sec').value)
        self._deadman_timeout = float(self.get_parameter('deadman_timeout_sec').value)

        self._path: Optional[Path] = None
        self._odom: Optional[Odometry] = None

        self._pub = self.create_publisher(TwistStamped, self._output_topic, 10)
        self.create_subscription(Path, self._path_topic, self._on_path, 10)
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, 10)
        self.create_timer(1.0 / max(0.1, self._hz), self._tick)

        self.get_logger().info(
            f'Midpoint twist test node: path={self._path_topic}, odom={self._odom_topic}, out={self._output_topic}')

    def _on_path(self, msg: Path) -> None:
        self._path = msg

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg

    @staticmethod
    def _yaw_from_q(x: float, y: float, z: float, w: float) -> float:
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    def _stamp_age(self, stamp) -> float:
        t = Time.from_msg(stamp)
        if t.nanoseconds == 0:
            return float('inf')
        return abs((self.get_clock().now() - t).nanoseconds / 1e9)

    def _publish_stop(self) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        self._pub.publish(msg)

    def _tick(self) -> None:
        if self._path is None or self._odom is None or len(self._path.poses) < 2:
            self._publish_stop()
            return

        if self._stamp_age(self._odom.header.stamp) > self._max_odom_age:
            self.get_logger().warn('Odom stale; publishing stop.', throttle_duration_sec=2.0)
            self._publish_stop()
            return

        path_age = self._stamp_age(self._path.header.stamp)
        if path_age > self._deadman_timeout:
            self.get_logger().warn('Path stale; publishing stop.', throttle_duration_sec=2.0)
            self._publish_stop()
            return

        rx = self._odom.pose.pose.position.x
        ry = self._odom.pose.pose.position.y
        q = self._odom.pose.pose.orientation
        ryaw = self._yaw_from_q(q.x, q.y, q.z, q.w)

        pts = self._path.poses
        nearest_idx = 0
        best_d2 = float('inf')
        for i, ps in enumerate(pts):
            dx = ps.pose.position.x - rx
            dy = ps.pose.position.y - ry
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                nearest_idx = i

        target_idx = min(len(pts) - 1, nearest_idx + max(1, self._lookahead))
        tx = pts[target_idx].pose.position.x
        ty = pts[target_idx].pose.position.y

        heading = math.atan2(ty - ry, tx - rx)
        err = self._wrap(heading - ryaw)
        wz = max(-self._w_max, min(self._w_max, self._k * err))

        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.twist.linear.x = float(self._v)
        out.twist.angular.z = float(wz)
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MidpointTwistTestNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
