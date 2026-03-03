import rclpy
from py_compile import main
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')
        # Subscribe to the '/scan' topic
        self.subscription = self.create_subscription(
        LaserScan,
        '/scan',
        self.listener_callback,
        10)
        self.subscription # prevent unused variable warning
        print("Lidar Subscriber Started. Waiting for data...")
    def listener_callback(self, msg):
        # msg.ranges contains hundreds of distance measurements
        # C1 Lidar scans 360 degrees.
        # Index 0 is usually the front.
        # Calculate indices for Front, Left, Back, Right
        total_readings = len(msg.ranges)
        front_index = 0
        left_index = total_readings // 4
        back_index = total_readings // 2
        right_index = (total_readings * 3) // 4
        # Get the distance at the front
        front_dist = msg.ranges[front_index]
        # 'inf' means infinity (no obstacle detected within range)
        if front_dist == float('inf'):
            print("Front: Clear (No obstacle)")
        else:
        # Print the distance with 3 decimal places
            print(f"Front Distance: {front_dist:.3f} meters")
        # Example logic for competition:
        if front_dist < 0.5:
            print("!!! WARNING: OBSTACLE TOO CLOSE !!!")
def main(args=None):
    rclpy.init(args=args)
    lidar_subscriber = LidarSubscriber()
    try:
        rclpy.spin(lidar_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_subscriber.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()