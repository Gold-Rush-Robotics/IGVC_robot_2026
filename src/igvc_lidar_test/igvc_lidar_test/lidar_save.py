import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import csv
import time
class LidarSaver(Node):
    def __init__(self):
        super().__init__('lidar_saver')
        self.subscription = self.create_subscription(
        LaserScan,
        '/scan',
        self.listener_callback,
        10)
        # Open a CSV file to write data
        self.file = open('lidar_data.csv', 'w', newline='')
        self.writer = csv.writer(self.file)
        # Write the header row
        self.writer.writerow(['Time_Sec', 'Front_Dist', 'Left_Dist', 'Back_Dist', 'Right_Dist'])
        self.start_time = time.time()
        print("Saving data to 'lidar_data.csv'... Press Ctrl+C to stop.")
    def listener_callback(self, msg):
        # Calculate indices
        total_readings = len(msg.ranges)
        front = msg.ranges[0]
        left = msg.ranges[total_readings // 4]
        back = msg.ranges[total_readings // 2]
        right = msg.ranges[(total_readings * 3) // 4]
        # Clean up 'inf' values (infinite distance)
        # If 'inf', we save it as 0.0 or a max value like 10.0
    def clean(self, val):
        return 0.0 if val == float('inf') else val
        # Calculate time elapsed
        current_time = time.time() - self.start_time
        # Write to file
        self.writer.writerow([
        f"{current_time:.2f}",
        f"{clean(front):.3f}",
        f"{clean(left):.3f}",
        f"{clean(back):.3f}",
        f"{clean(right):.3f}"
        ])
def main(args=None):
    rclpy.init(args=args)
    saver = LidarSaver()
    try:
        rclpy.spin(saver)
    except KeyboardInterrupt:
    # Close the file safely when you press Ctrl+C
        saver.file.close()
        print("\nFile closed. Data saved to lidar_data.csv")
    finally:
        saver.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()