import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class ObstacleMapper(Node):
    def __init__(self):
        super().__init__('obstacle_mapper')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        self.subscription
        print("Obstacle Mapper Started. Calculating X, Y coordinates...")

    def listener_callback(self, msg):
        obstacles = []

        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or math.isnan(r):
                continue

            angle = msg.angle_min + (i * msg.angle_increment)

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            if r < 2.0:
                obstacles.append((x, y))

        if obstacles:
            print(f"Detected {len(obstacles)} obstacle points within 2m.")
            print(f"Sample Point -> X: {obstacles[0][0]:.3f}m, Y: {obstacles[0][1]:.3f}m\n")

def main(args=None):
    rclpy.init(args=args)
    mapper = ObstacleMapper()
    try:
        rclpy.spin(mapper)
    except KeyboardInterrupt:
        pass
    finally:
        mapper.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()