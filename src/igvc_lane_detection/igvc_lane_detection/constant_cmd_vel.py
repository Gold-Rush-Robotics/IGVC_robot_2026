import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class ConstantCmdVelPublisher(Node):
    def __init__(self) -> None:
        super().__init__('constant_cmd_vel_publisher')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer_ = self.create_timer(0.1, self.publish_cmd_vel)

        self.linear_velocity = 1.0
        self.angular_velocity = 5.0

        self.get_logger().info(
            f'Publishing /cmd_vel with linear.x={self.linear_velocity}, '
            f'angular.z={self.angular_velocity}'
        )

    def publish_cmd_vel(self) -> None:
        msg = Twist()
        msg.linear.x = self.linear_velocity
        msg.angular.z = self.angular_velocity
        self.publisher_.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConstantCmdVelPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
