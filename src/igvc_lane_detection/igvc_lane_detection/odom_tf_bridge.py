from __future__ import annotations

import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomTfBridgeNode(Node):

    def __init__(self) -> None:
        super().__init__('odom_tf_bridge')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('publish_rate_hz', 500.0)
        self.declare_parameter('use_original_timestamp', False)
        self.declare_parameter('warn_odom_age_sec', 0.5)
        self.declare_parameter('max_odom_age_sec', 0.5)

        self._odom_topic = self.get_parameter('odom_topic').value
        self._odom_frame = self.get_parameter('odom_frame_id').value
        self._base_frame = self.get_parameter('base_frame_id').value
        self._publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self._use_original_timestamp = bool(
            self.get_parameter('use_original_timestamp').value)
        self._warn_odom_age_sec = float(
            self.get_parameter('warn_odom_age_sec').value)
        self._max_odom_age_sec = float(
            self.get_parameter('max_odom_age_sec').value)
        self._latest_pose = Pose()
        self._latest_pose.orientation.w = 1.0
        self._latest_stamp = None

        self._tf_broadcaster = TransformBroadcaster(self)
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self.create_subscription(Odometry, self._odom_topic, self._on_odom, odom_qos)
        self.create_timer(1.0 / max(self._publish_rate_hz, 1.0), self._publish_tf)
        self.get_logger().info(
            f'Bridging {self._odom_topic} to TF {self._odom_frame} -> '
            f'{self._base_frame} (use_original_timestamp={self._use_original_timestamp})')

    def _on_odom(self, msg: Odometry) -> None:
        if self._stamp_age_sec(msg.header.stamp) > self._max_odom_age_sec:
            self.get_logger().warn(
                f'Ignoring unstamped/stale odom older than {self._max_odom_age_sec:.3f}s.',
                throttle_duration_sec=2.0)
            return
        self._latest_pose = msg.pose.pose
        self._latest_stamp = msg.header.stamp
        self._warn_if_stale(msg.header.stamp)

    def _stamp_age_sec(self, stamp) -> float:
        stamp_t = Time.from_msg(stamp)
        if stamp_t.nanoseconds == 0:
            return float('inf')
        return abs((self.get_clock().now() - stamp_t).nanoseconds / 1e9)

    def _publish_tf(self) -> None:
        if self._latest_stamp is None:
            return
        if self._stamp_age_sec(self._latest_stamp) > self._max_odom_age_sec:
            return
        tf = TransformStamped()
        if self._use_original_timestamp:
            tf.header.stamp = self._latest_stamp
        else:
            tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self._odom_frame
        tf.child_frame_id = self._base_frame
        tf.transform.translation.x = self._latest_pose.position.x
        tf.transform.translation.y = self._latest_pose.position.y
        tf.transform.translation.z = self._latest_pose.position.z
        tf.transform.rotation = self._latest_pose.orientation
        self._tf_broadcaster.sendTransform(tf)

    def _warn_if_stale(self, stamp) -> None:
        if self._warn_odom_age_sec <= 0.0:
            return
        age = (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds / 1e9
        if age > self._warn_odom_age_sec:
            self.get_logger().warn(
                f'Odom message on {self._odom_topic} is {age:.2f}s old; '
                'publishing TF with current time to avoid delayed robot pose.',
                throttle_duration_sec=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    rclpy.spin(OdomTfBridgeNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
