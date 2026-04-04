import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import matplotlib.pyplot as plt

class VisualObstacleMapper(Node):
    def __init__(self):
        super().__init__('visual_mapper')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        
        # ---Real-time map settings ---
        plt.ion()  # turn on realtime mode
        self.fig, self.ax = plt.subplots()
        
        # red dot = obstacles 
        self.scat = self.ax.scatter([], [], s=10, c='red') 
        
        # Map size (2 meters front, back, left, and right based on LiDAR)
        self.ax.set_xlim(-2.0, 2.0)
        self.ax.set_ylim(-2.0, 2.0)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Real-time Lidar Obstacle Map')
        self.ax.grid(True) # 격자무늬 표시
        
        # Set the ratio of the X-axis and Y-axis to be the same to prevent distortion
        self.ax.set_aspect('equal', adjustable='box')
        
        print("Visual Mapper Started. Look for the popping up graph window!")

    def listener_callback(self, msg):
        x_data = []
        y_data = []
        
        for i, r in enumerate(msg.ranges):
            # Data or error values beyond 2m are not plotted
            if math.isinf(r) or math.isnan(r) or r > 2.0:
                continue
            
            angle = msg.angle_min + (i * msg.angle_increment)
            
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            
            x_data.append(x)
            y_data.append(y)
        
        # --- Plot the calculated X and Y coordinates as points on the graph---
        if x_data:
            self.scat.set_offsets(list(zip(x_data, y_data)))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    mapper = VisualObstacleMapper()
    try:
        rclpy.spin(mapper)
    except KeyboardInterrupt:
        pass
    finally:
        mapper.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == '__main__':
    main()
