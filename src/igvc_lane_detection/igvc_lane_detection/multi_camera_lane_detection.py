import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid  # Optional: only if you want a custom grid on /map

from geometry_msgs.msg import Point, PointStamped, TransformStamped
import tf2_geometry_msgs 

from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np

import tf2_ros
from sensor_msgs_py import point_cloud2 as pc2

from image_geometry import PinholeCameraModel


class MultiCameraLaneDetector(Node):
    def __init__(self):
        super().__init__('multi_camera_lane_detector')

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ==================== PARAMETERS (customize these in a launch file or via CLI) ====================
        self.declare_parameter('camera_names', ['front', 'left', 'right'])
        self.declare_parameter('target_frame', 'map')  # or 'base_link' if you prefer robot frame
        self.declare_parameter('lane_pc_topic', '/lane_points')
        self.declare_parameter('lane_markers_topic', '/lane_markers')
        self.declare_parameter('debug_stitched_topic', '/debug/stitched_lanes')
        self.declare_parameter('costmap_topic', '/map')

        # NEW: Model path (empty = use fast OpenCV fallback; set to .pt file for YOLOv8-seg)
        self.declare_parameter('model_path', '')

        camera_names = self.get_parameter('camera_names').value
        target_frame = self.get_parameter('target_frame').value

        # Per-camera topics
        self.rgb_topics = {}
        self.depth_topics = {}
        self.info_topics = {}
        for cam in camera_names:
            self.declare_parameter(f'{cam}_rgb_topic', f'/{cam}/camera/color/image_raw')
            self.declare_parameter(f'{cam}_depth_topic', f'/{cam}/camera/depth/image_raw')
            self.declare_parameter(f'{cam}_info_topic', f'/{cam}/camera/color/camera_info')
            self.rgb_topics[cam] = self.get_parameter(f'{cam}_rgb_topic').value
            self.depth_topics[cam] = self.get_parameter(f'{cam}_depth_topic').value
            self.info_topics[cam] = self.get_parameter(f'{cam}_info_topic').value

        # Camera models and latest data caches
        self.camera_models = {cam: PinholeCameraModel() for cam in camera_names}
        self.latest_camera_info = {cam: None for cam in camera_names}
        self.latest_rgb_cv = {cam: None for cam in camera_names}
        self.latest_debug_cv = {cam: None for cam in camera_names}
        self.latest_lane_points = {cam: np.empty((0, 3)) for cam in camera_names}

        # ==================== LOAD SEGMENTATION MODEL (YOLO or CV fallback) ====================
        model_path = self.get_parameter('model_path').value
        self.model = None
        if model_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.get_logger().info(f'Successfully loaded YOLO segmentation model: {model_path}')
            except ImportError:
                self.get_logger().error(
                    'ultralytics not installed. Install with: pip install ultralytics. '
                    'Falling back to traditional OpenCV lane segmentation.'
                )
            except Exception as e:
                self.get_logger().error(f'Failed to load model {model_path}: {e}. Falling back to CV.')
        else:
            self.get_logger().info(
                'No model_path provided. Using traditional OpenCV lane segmentation '
                '(white/yellow markings). Set model_path for a YOLOv8-seg model.'
            )

        # ==================== SUBSCRIBERS (synchronized per camera) ====================
        self.sync_subs = {}
        for cam in camera_names:
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

            rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topics[cam], qos_profile=qos)
            depth_sub = message_filters.Subscriber(self, Image, self.depth_topics[cam], qos_profile=qos)
            info_sub = message_filters.Subscriber(self, CameraInfo, self.info_topics[cam], qos_profile=qos)

            ts = message_filters.ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub, info_sub],
                queue_size=15,
                slop=0.15
            )
            ts.registerCallback(self.per_camera_callback, cam)
            self.get_logger().info(
                f'camera[{cam}] rgb={self.rgb_topics[cam]} depth={self.depth_topics[cam]} info={self.info_topics[cam]}'
            )

        # ==================== PUBLISHERS ====================
        self.lane_pc_pub = self.create_publisher(PointCloud2, self.get_parameter('lane_pc_topic').value, 10)
        self.lane_marker_pub = self.create_publisher(MarkerArray, self.get_parameter('lane_markers_topic').value, 10)
        self.debug_pub = self.create_publisher(Image, self.get_parameter('debug_stitched_topic').value, 10)

        # Timer to fuse data from all cameras, publish PC2 + markers + debug stitched image
        self.fusion_timer = self.create_timer(0.1, self.fusion_callback)  # 10 Hz

        self.get_logger().info('Multi-camera lane detection node started with full TODO implementation. '
                               'Waiting for synchronized RGB + Depth + Info from all 3 cameras...')

    def segment_lanes(self, rgb_cv: np.ndarray) -> np.ndarray:
        """Full implementation: returns uint8 mask (H,W) where >0 = lane pixels.
        - If YOLO model is loaded: uses instance masks (any lane class).
        - Otherwise: traditional CV segmentation for white/yellow lane markings.
        Left/right distinction is handled globally in fusion (RANSAC)."""
        if self.model is not None:
            try:
                results = self.model(rgb_cv, verbose=False, conf=0.3, iou=0.5)
                mask = np.zeros(rgb_cv.shape[:2], dtype=np.uint8)
                if len(results) > 0 and results[0].masks is not None:
                    for seg_mask in results[0].masks.data:
                        seg = (seg_mask.cpu().numpy() * 255).astype(np.uint8)
                        seg = cv2.resize(seg, (rgb_cv.shape[1], rgb_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask = cv2.bitwise_or(mask, seg)
                return mask
            except Exception as e:
                self.get_logger().warning(f'YOLO inference failed: {e}. Falling back to CV.')

        # Traditional CV fallback (no extra dependencies)
        hsv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2HSV)
        # White lane markings
        mask_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
        # Yellow lane markings
        mask_yellow = cv2.inRange(hsv, np.array([20, 60, 100]), np.array([40, 255, 255]))
        lane_mask = cv2.bitwise_or(mask_white, mask_yellow)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        return lane_mask  # 0 or 255

    def _fit_ransac_line_inliers(self, xy: np.ndarray, full_3d: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Helper: RANSAC for a single line. Returns (inliers_3d, inlier_mask) or (None, None)."""
        n_points = len(xy)
        if n_points < 10:
            return None, None

        num_iterations = 200
        distance_threshold = 0.15  # meters - tune to your lane width
        min_inliers_ratio = 0.25

        best_inlier_mask = None
        best_num_inliers = 0

        for _ in range(num_iterations):
            sample_idx = np.random.choice(n_points, 2, replace=False)
            p1 = xy[sample_idx[0]]
            p2 = xy[sample_idx[1]]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if abs(dx) < 1e-8 and abs(dy) < 1e-8:
                continue

            a = dy
            b = -dx
            c = dx * p1[1] - dy * p1[0]
            norm = np.hypot(a, b)
            if norm < 1e-8:
                continue
            a /= norm
            b /= norm
            c /= norm

            dists = np.abs(a * xy[:, 0] + b * xy[:, 1] + c)
            inlier_mask = dists < distance_threshold
            num_inliers = np.count_nonzero(inlier_mask)

            if num_inliers > best_num_inliers and (num_inliers / n_points) >= min_inliers_ratio:
                best_num_inliers = num_inliers
                best_inlier_mask = inlier_mask.copy()

        if best_inlier_mask is None or best_num_inliers < 5:
            return None, None

        return full_3d[best_inlier_mask], best_inlier_mask

    def _order_points_along_line(self, pts_3d: np.ndarray) -> np.ndarray:
        """Sort points along the principal direction of the line for LINE_STRIP."""
        if len(pts_3d) < 2:
            return pts_3d
        xy = pts_3d[:, :2]
        mean = np.mean(xy, axis=0)
        _, _, Vh = np.linalg.svd(xy - mean, full_matrices=False)
        dir_vec = Vh[0]
        projections = np.dot(xy - mean, dir_vec)
        sort_idx = np.argsort(projections)
        return pts_3d[sort_idx]

    def fit_two_lane_lines(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Full TODO implementation: Identify exactly 2 lane lines using sequential RANSAC.
        Returns ordered 3D points for left lane and right lane."""
        if len(points) < 20:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        xy = points[:, :2].astype(np.float32)

        # Fit first line
        inliers1_3d, inlier_mask1 = self._fit_ransac_line_inliers(xy, points)
        if inliers1_3d is None or len(inliers1_3d) < 5:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        # Remaining points for second line
        remaining_mask = ~inlier_mask1
        remaining_xy = xy[remaining_mask]
        remaining_3d = points[remaining_mask]

        if len(remaining_xy) < 10:
            inliers2_3d = np.empty((0, 3), dtype=np.float32)
        else:
            inliers2_3d, _ = self._fit_ransac_line_inliers(remaining_xy, remaining_3d)
            if inliers2_3d is None:
                inliers2_3d = np.empty((0, 3), dtype=np.float32)

        # Order points along each line
        left_pts = self._order_points_along_line(inliers1_3d) if len(inliers1_3d) > 0 else np.empty((0, 3), dtype=np.float32)
        right_pts = self._order_points_along_line(inliers2_3d) if len(inliers2_3d) > 0 else np.empty((0, 3), dtype=np.float32)

        # Heuristic: assign left/right based on mean Y (ROS map convention: +Y = left of travel direction)
        if len(left_pts) > 3 and len(right_pts) > 3:
            mean_y1 = np.mean(left_pts[:, 1])
            mean_y2 = np.mean(right_pts[:, 1])
            if mean_y1 < mean_y2:  # swap so left has larger Y
                left_pts, right_pts = right_pts, left_pts

        return left_pts, right_pts

    def publish_lane_line_markers(self, left_pts: np.ndarray, right_pts: np.ndarray):
        """Full TODO implementation: Publish clean LINE_STRIP markers (green = left, red = right)."""
        ma = MarkerArray()

        # Left lane (green)
        if len(left_pts) >= 2:
            marker = Marker()
            marker.header.frame_id = self.get_parameter('target_frame').value
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'lane_lines'
            marker.id = 0
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.12
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.9
            for p in left_pts:
                pt = Point()
                pt.x = float(p[0])
                pt.y = float(p[1])
                pt.z = float(p[2])
                marker.points.append(pt)
            ma.markers.append(marker)

        # Right lane (red)
        if len(right_pts) >= 2:
            marker = Marker()
            marker.header.frame_id = self.get_parameter('target_frame').value
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'lane_lines'
            marker.id = 1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.12
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.9
            for p in right_pts:
                pt = Point()
                pt.x = float(p[0])
                pt.y = float(p[1])
                pt.z = float(p[2])
                marker.points.append(pt)
            ma.markers.append(marker)

        self.lane_marker_pub.publish(ma)

    def project_mask_to_3d(self,
                           mask: np.ndarray,
                           depth_cv: np.ndarray,
                           cam_model: PinholeCameraModel,
                           camera_frame: str,
                           stamp: Time) -> np.ndarray:
        """Project masked lane pixels (using depth) into 3D points in the target_frame (map/base_link)."""
        if mask.sum() == 0:
            return np.empty((0, 3), dtype=np.float32)

        h, w = mask.shape
        points_3d = []

        target_frame = self.get_parameter('target_frame').value

        ys, xs = np.where(mask > 0)
        for y, x in zip(ys, xs):
            d = depth_cv[y, x]
            if np.isnan(d) or d <= 0.01 or d > 20.0:
                continue

            ray = cam_model.projectPixelTo3dRay((x, y))
            if ray is None:
                continue
            pt_cam = np.array(ray) * d

            point_stamped = PointStamped()
            point_stamped.header.frame_id = camera_frame
            point_stamped.header.stamp = stamp
            point_stamped.point.x = float(pt_cam[0])
            point_stamped.point.y = float(pt_cam[1])
            point_stamped.point.z = float(pt_cam[2])

            try:
                transformed = self.tf_buffer.transform(point_stamped, target_frame, timeout=rclpy.duration.Duration(seconds=0.1))
                points_3d.append([
                    transformed.point.x,
                    transformed.point.y,
                    transformed.point.z
                ])
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().debug(f'TF transform failed for camera {camera_frame}: {e}')
                continue

        return np.array(points_3d, dtype=np.float32)

    def per_camera_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo, cam_name: str):
        """Synchronized callback for one camera."""
        try:
            if self.latest_camera_info[cam_name] is None or \
               info_msg.header.stamp != self.latest_camera_info[cam_name].header.stamp:
                self.camera_models[cam_name].fromCameraInfo(info_msg)
                self.latest_camera_info[cam_name] = info_msg

            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

            # Run segmentation (now fully implemented)
            lane_mask = self.segment_lanes(rgb_cv)

            # Overlay on debug image (cyan for any lane - works for both YOLO and CV)
            debug_cv = rgb_cv.copy()
            debug_cv[lane_mask > 0] = [0, 255, 255]  # cyan

            camera_frame = info_msg.header.frame_id
            lane_points = self.project_mask_to_3d(
                lane_mask, depth_cv, self.camera_models[cam_name], camera_frame, rgb_msg.header.stamp
            )

            self.latest_rgb_cv[cam_name] = rgb_cv
            self.latest_debug_cv[cam_name] = debug_cv
            self.latest_lane_points[cam_name] = lane_points

        except Exception as e:
            self.get_logger().error(f'Error in {cam_name} callback: {e}')

    def publish_points_marker(self, points: np.ndarray):
        """Keep original dense point cloud marker for debugging (yellow)."""
        marker = Marker()
        marker.header.frame_id = self.get_parameter('target_frame').value
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'lane_points'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for p in points:
            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = float(p[2])
            marker.points.append(pt)

        ma = MarkerArray()
        ma.markers.append(marker)
        self.lane_marker_pub.publish(ma)

    def fusion_callback(self):
        """Timer callback: fuse data from all cameras (all TODOs now implemented)."""
        # 1. Publish combined lane point cloud (used directly by costmap)
        all_points = []
        for cam_points in self.latest_lane_points.values():
            if cam_points.shape[0] > 0:
                all_points.append(cam_points)
        if all_points:
            combined_points = np.vstack(all_points)
            if len(combined_points) > 0:
                header = self.get_clock().now().to_msg()
                header.frame_id = self.get_parameter('target_frame').value
                pc_msg = pc2.create_cloud_xyz32(header, combined_points)
                self.lane_pc_pub.publish(pc_msg)

                # Dense points marker (kept for debugging)
                self.publish_points_marker(combined_points)

                # 2. Identify exactly 2 lane lines (RANSAC) + publish clean LINE_STRIP markers
                left_line_points, right_line_points = self.fit_two_lane_lines(combined_points)
                self.publish_lane_line_markers(left_line_points, right_line_points)

        # 3. Publish stitched debug image (with lane overlays)
        debug_images = [self.latest_debug_cv[cam] for cam in self.latest_debug_cv if self.latest_debug_cv[cam] is not None]
        if len(debug_images) == 3:
            try:
                heights = [img.shape[0] for img in debug_images]
                if len(set(heights)) > 1:
                    target_h = min(heights)
                    debug_images = [cv2.resize(img, (int(img.shape[1] * target_h / img.shape[0]), target_h)) for img in debug_images]
                stitched = cv2.hconcat(debug_images)
                cv2.putText(stitched, 'LEFT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(stitched, 'FRONT', (debug_images[0].shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(stitched, 'RIGHT', (debug_images[0].shape[1] * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                debug_msg = self.bridge.cv2_to_imgmsg(stitched, encoding='bgr8')
                debug_msg.header.stamp = self.get_clock().now().to_msg()
                self.debug_pub.publish(debug_msg)
            except Exception as e:
                self.get_logger().debug(f'Stitching failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraLaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()