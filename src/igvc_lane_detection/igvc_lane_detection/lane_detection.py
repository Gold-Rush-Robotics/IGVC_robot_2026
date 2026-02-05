import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import numpy as np
from cv_bridge import CvBridge
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.get_logger().info("Lane Detection Node has been started.")
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Camera intrinsics (will be updated from CameraInfo)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Launch argument number of cameras
        self.declare_parameter('use_depth', True)
        use_depth = self.get_parameter('use_depth').get_parameter_value().bool_value
        self.declare_parameter('lane_center_topic', '/lane_center')
        lane_center_topic = self.get_parameter('lane_center_topic').get_parameter_value().string_value
        self.declare_parameter('num_cameras', 1)
        num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value
        self.declare_parameter('camera_topics', ['/camera/image_raw'])
        camera_topics = self.get_parameter('camera_topics').get_parameter_value().string_array_value
        self.declare_parameter('depth_topics', ['/camera/depth'])
        depth_topics = self.get_parameter('depth_topics').get_parameter_value().string_array_value
        self.declare_parameter('camera_info_topics', ['/camera/camera_info'])
        camera_info_topics = self.get_parameter('camera_info_topics').get_parameter_value().string_array_value
        
        # Occupancy grid parameters
        self.declare_parameter('grid_resolution', 0.05)  # meters per cell
        self.declare_parameter('grid_width', 10.0)  # meters
        self.declare_parameter('grid_height', 10.0)  # meters
        self.grid_resolution = self.get_parameter('grid_resolution').get_parameter_value().double_value
        self.grid_width_m = self.get_parameter('grid_width').get_parameter_value().double_value
        self.grid_height_m = self.get_parameter('grid_height').get_parameter_value().double_value
        
        # Subscribe to camera info for intrinsics
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topics[0], self.camera_info_callback, 10)
        
        # Create synchronized subscribers for RGB and depth
        self.sync_subscribers = []
        for i in range(num_cameras):
            rgb_sub = Subscriber(self, Image, camera_topics[i])
            depth_sub = Subscriber(self, Image, depth_topics[i])
            
            sync = ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub], queue_size=10, slop=0.1)
            sync.registerCallback(self.synced_image_callback)
            self.sync_subscribers.append((rgb_sub, depth_sub, sync))
        
        self.lane_center_pub = self.create_publisher(Path, lane_center_topic, 10)
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/lane_occupancy_grid', 10)

    def camera_info_callback(self, msg):
        """Store camera intrinsics from CameraInfo message."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        self.get_logger().info("Received camera intrinsics.", once=True)

    def pixel_to_3d(self, u, v, depth):
        """
        Convert pixel coordinates (u, v) and depth to 3D point in camera frame.
        Returns (x, y, z) in meters.
        """
        if self.camera_matrix is None:
            # Fallback with default intrinsics if not received yet
            fx, fy = 500.0, 500.0
            cx, cy = 320.0, 240.0
        else:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
        
        # Convert to 3D (camera frame: z forward, x right, y down)
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        return x, y, z

    def synced_image_callback(self, rgb_msg, depth_msg):
        """Process synchronized RGB and depth images for lane detection."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            # Handle different depth encodings
            if depth_msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
                depth_image = depth_image.astype(np.float32) / 1000.0  # Convert mm to meters
            elif depth_msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            else:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                depth_image = depth_image.astype(np.float32)
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Convert to grayscale
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Canny edge detection
        low_t = 50
        high_t = 150
        edges = cv2.Canny(blurred, low_t, high_t)
        
        # Apply region of interest mask
        region = self.region_selection(edges)
        
        # Apply Hough transform to find lines
        hough_lines = self.hough_transform(region)
        
        if hough_lines is not None:
            # Get lane lines in pixel coordinates
            left_line, right_line = self.lane_lines(cv_image, hough_lines)
            
            # Convert lane lines to 3D points using depth
            left_3d = self.lane_line_to_3d(left_line, depth_image)
            right_3d = self.lane_line_to_3d(right_line, depth_image)
            
            # Build and publish lane center path in 3D
            lane_center_path = self.build_lane_center_path_3d(left_3d, right_3d, rgb_msg.header)
            self.lane_center_pub.publish(lane_center_path)
            
            # Build and publish occupancy grid using 3D points
            occupancy_grid = self.build_occupancy_grid_3d(left_3d, right_3d, rgb_msg.header)
            self.occupancy_grid_pub.publish(occupancy_grid)
        else:
            # Publish empty messages if no lanes detected
            lane_center_path = Path()
            lane_center_path.header = rgb_msg.header
            self.lane_center_pub.publish(lane_center_path)
            
            occupancy_grid = OccupancyGrid()
            occupancy_grid.header = rgb_msg.header
            self.occupancy_grid_pub.publish(occupancy_grid)

    def lane_line_to_3d(self, line, depth_image):
        """
        Convert a lane line from pixel coordinates to 3D points.
        Samples points along the line and projects them using depth.
        Returns list of (x, y, z) points in camera frame.
        """
        if line is None:
            return None
        
        (x1, y1), (x2, y2) = line
        height, width = depth_image.shape[:2]
        
        points_3d = []
        num_samples = 20
        
        for t in np.linspace(0, 1, num_samples):
            u = int(x1 + t * (x2 - x1))
            v = int(y1 + t * (y2 - y1))
            
            # Bounds check
            if 0 <= u < width and 0 <= v < height:
                depth = depth_image[v, u]
                
                # Skip invalid depth values
                if depth > 0.1 and depth < 20.0:  # Valid range: 0.1m to 20m
                    x, y, z = self.pixel_to_3d(u, v, depth)
                    points_3d.append((x, y, z))
        
        return points_3d if len(points_3d) > 0 else None

    def build_lane_center_path_3d(self, left_3d, right_3d, header):
        """
        Build a Path message representing the lane center using 3D points.
        """
        path = Path()
        path.header = header
        path.header.frame_id = 'camera_frame'
        
        if left_3d is not None and right_3d is not None:
            # Pair up corresponding points from left and right lanes
            min_len = min(len(left_3d), len(right_3d))
            
            for i in range(min_len):
                lx, ly, lz = left_3d[i]
                rx, ry, rz = right_3d[i]
                
                pose = PoseStamped()
                pose.header = header
                pose.header.frame_id = 'camera_frame'
                
                # Center point between left and right
                pose.pose.position.x = (lz + rz) / 2.0  # z is forward in camera frame
                pose.pose.position.y = -(lx + rx) / 2.0  # x is right, negate for standard frame
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                
                path.poses.append(pose)
        
        return path

    def build_occupancy_grid_3d(self, left_3d, right_3d, header):
        """
        Build an OccupancyGrid message using 3D lane points.
        Grid is in robot base frame (x forward, y left).
        """
        grid = OccupancyGrid()
        grid.header = header
        grid.header.frame_id = 'base_link'
        
        resolution = self.grid_resolution
        grid_width_cells = int(self.grid_width_m / resolution)
        grid_height_cells = int(self.grid_height_m / resolution)
        
        grid.info.resolution = resolution
        grid.info.width = grid_width_cells
        grid.info.height = grid_height_cells
        # Origin at robot center, grid extends forward and to sides
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = -self.grid_width_m / 2.0
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        
        # Initialize grid with unknown (-1)
        grid_data = [-1] * (grid_width_cells * grid_height_cells)
        
        if left_3d is not None and right_3d is not None and len(left_3d) > 1 and len(right_3d) > 1:
            # Convert 3D points from camera frame to base_link frame
            # Camera: z forward, x right, y down
            # Base: x forward, y left, z up
            left_base = [(p[2], -p[0]) for p in left_3d]  # (x_base, y_base)
            right_base = [(p[2], -p[0]) for p in right_3d]

            # Sort by x (forward distance)
            left_base = sorted(left_base, key=lambda p: p[0])
            right_base = sorted(right_base, key=lambda p: p[0])
            
            # For each row in the grid (each x distance)
            for grid_row in range(grid_height_cells):
                x_dist = grid_row * resolution  # Forward distance
                
                # Interpolate y positions of left and right lanes at this x
                left_y = self.interpolate_lane_y(left_base, x_dist)
                right_y = self.interpolate_lane_y(right_base, x_dist)
                
                if left_y is not None and right_y is not None:
                    # Ensure left_y > right_y (left is positive y, right is negative)
                    if left_y < right_y:
                        left_y, right_y = right_y, left_y
                    
                    # Convert to grid columns
                    left_col = int((left_y + self.grid_width_m / 2.0) / resolution)
                    right_col = int((right_y + self.grid_width_m / 2.0) / resolution)
                    
                    # Mark cells
                    for col in range(grid_width_cells):
                        idx = grid_row * grid_width_cells + col
                        
                        if right_col < col < left_col:
                            grid_data[idx] = 0  # Free space inside lane
                        elif col == left_col or col == right_col:
                            grid_data[idx] = 100  # Lane boundary
                        else:
                            grid_data[idx] = 100  # Outside lane - occupied
        
        grid.data = grid_data
        return grid

    def interpolate_lane_y(self, lane_points, x_target):
        """
        Interpolate the y position of a lane at a given x distance.
        lane_points: list of (x, y) tuples sorted by x.
        """
        if len(lane_points) < 2:
            return None
        
        # Find bracketing points
        for i in range(len(lane_points) - 1):
            x1, y1 = lane_points[i]
            x2, y2 = lane_points[i + 1]
            
            if x1 <= x_target <= x2:
                # Linear interpolation
                if x2 - x1 < 0.001:
                    return y1
                t = (x_target - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)
        
        # Extrapolate if outside range
        if x_target < lane_points[0][0]:
            return lane_points[0][1]
        if x_target > lane_points[-1][0]:
            return lane_points[-1][1]
        
        return None

    def region_selection(self, image):
        """
        Determine and cut the region of interest in the input image.
        Parameters:
            image: output from canny where we have identified edges in the frame
        """
        mask = np.zeros_like(image)
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        
        rows, cols = image.shape[:2]
        bottom_left = [cols * 0.1, rows * 0.95]
        top_left = [cols * 0.4, rows * 0.6]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right = [cols * 0.6, rows * 0.6]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def hough_transform(self, image):
        """
        Apply Hough transform to detect lines.
        Parameter:
            image: grayscale image which should be an output from the edge detector
        """
        rho = 1
        theta = np.pi / 180
        threshold = 20
        minLineLength = 20
        maxLineGap = 500
        return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                               minLineLength=minLineLength, maxLineGap=maxLineGap)

    def average_slope_intercept(self, lines):
        """
        Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: output from Hough Transform
        """
        left_lines = []
        left_weights = []
        right_lines = []
        right_weights = []
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_weights.append(length)
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append(length)
        
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
        return left_lane, right_lane

    def pixel_points(self, y1, y2, line):
        """
        Converts the slope and intercept of each line into pixel points.
        """
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))

    def lane_lines(self, image, lines):
        """
        Create full length lines from pixel points.
        """
        left_lane, right_lane = self.average_slope_intercept(lines)
        y1 = image.shape[0]
        y2 = y1 * 0.6
        left_line = self.pixel_points(y1, y2, left_lane)
        right_line = self.pixel_points(y1, y2, right_lane)
        return left_line, right_line

    def draw_lane_lines(self, image, lines, color=[255, 0, 0], thickness=12):
        """
        Draw lines onto the input image.
        """
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line, color, thickness)
        return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

