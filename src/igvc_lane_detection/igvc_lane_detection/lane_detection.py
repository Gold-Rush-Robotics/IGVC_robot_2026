import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2
import numpy as np
from cv_bridge import CvBridge

MIN_ABS_SLOPE = 0.3


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.bridge = CvBridge()
        self.K = {}  # cam_idx -> 3x3 intrinsic matrix

        # ── Parameters ────────────────────────────────────────────────────
        def p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        self.base_frame         = p('base_frame',            'base_link')
        self.assumed_lane_width = p('assumed_lane_width_m',   3.0)
        self.depth_scale        = p('depth_scale',            1.0)
        self.grid_res           = p('grid_resolution',        0.05)
        self.grid_width_m       = p('grid_width',            10.0)
        self.grid_height_m      = p('grid_height',           10.0)
        self.publish_overlay    = p('publish_overlay',        True)

        num_cameras  = p('num_cameras',        1)
        cam_topics   = p('camera_topics',      ['/camera/image_raw'])
        depth_topics = p('depth_topics',       ['/camera/depth/image_raw'])
        info_topics  = p('camera_info_topics', ['/camera/camera_info'])

        num_cameras = min(num_cameras, len(cam_topics), len(depth_topics), len(info_topics))

        # ── Subscribers ───────────────────────────────────────────────────
        self._sync_handles = []
        self.overlay_pubs  = {}

        for i in range(num_cameras):
            self.create_subscription(
                CameraInfo, info_topics[i],
                lambda msg, idx=i: self._on_info(msg, idx), 10)

            rgb_sub   = Subscriber(self, Image, cam_topics[i])
            depth_sub = Subscriber(self, Image, depth_topics[i])
            sync = ApproximateTimeSynchronizer([rgb_sub, depth_sub],
                                               queue_size=5, slop=0.1)
            sync.registerCallback(lambda r, d, idx=i: self._on_images(r, d, idx))
            self._sync_handles.append((rgb_sub, depth_sub, sync))

            if self.publish_overlay:
                self.overlay_pubs[i] = self.create_publisher(
                    Image, f'/lane_debug/cam{i}/overlay', 10)

        # ── Publishers ────────────────────────────────────────────────────
        # Transient local so Nav2's costmap StaticLayer receives it on late-join
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        self.grid_pub    = self.create_publisher(OccupancyGrid, '/lane_costmap', map_qos)
        self.latest_grid = self._empty_grid()

        # Republish last known grid every second so Nav2 costmap stays populated
        # even if the frame rate dips or the camera briefly drops frames.
        self.create_timer(1.0, self._republish_grid)

        self._got_frame = False
        self.create_timer(2.0, self._watchdog)

    # ── Camera info ───────────────────────────────────────────────────────

    def _on_info(self, msg, idx):
        self.K[idx] = np.array(msg.k).reshape(3, 3)
        self.get_logger().info(f'Camera[{idx}] intrinsics received.', once=True)

    # ── Main callback ─────────────────────────────────────────────────────

    def _on_images(self, rgb_msg, depth_msg, cam_idx):
        self._got_frame = True

        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            if depth_msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(
                    depth_msg, '16UC1').astype(np.float32) / 1000.0
            else:
                depth = self.bridge.imgmsg_to_cv2(
                    depth_msg, '32FC1') * self.depth_scale
        except Exception as e:
            self.get_logger().error(f'Decode error: {e}')
            return

        left_px, right_px, dbg_lines = self._detect_lanes(bgr)

        if self.publish_overlay and cam_idx in self.overlay_pubs:
            self._publish_overlay(cam_idx, bgr, dbg_lines, left_px, right_px, rgb_msg)

        left_3d  = self._line_to_3d(left_px,  depth, cam_idx)
        right_3d = self._line_to_3d(right_px, depth, cam_idx)
        left_3d, right_3d = self._fill_missing(left_3d, right_3d)

        self.latest_grid = self._build_grid(left_3d, right_3d, rgb_msg.header.stamp)
        self.grid_pub.publish(self.latest_grid)

        self.get_logger().info(
            f'cam[{cam_idx}] left={left_px is not None} right={right_px is not None} '
            f'left_3d={0 if left_3d is None else len(left_3d)} '
            f'right_3d={0 if right_3d is None else len(right_3d)}',
            throttle_duration_sec=1.0)

    # ── Lane detection ────────────────────────────────────────────────────

    def _detect_lanes(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv,
                                  np.array([0,   0, 180]),
                                  np.array([180, 40, 255]))
        yellow_mask = cv2.inRange(hsv,
                                   np.array([15, 80, 100]),
                                   np.array([35, 255, 255]))
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,  kernel)

        edges = cv2.Canny(color_mask, 50, 150)
        roi   = self._apply_roi(edges)

        hough = cv2.HoughLinesP(roi, 1, np.pi / 180,
                                threshold=30, minLineLength=40, maxLineGap=200)
        if hough is None:
            return None, None, None

        left_px, right_px = self._fit_lines(bgr, hough)
        return left_px, right_px, hough

    def _apply_roi(self, img):
        mask = np.zeros_like(img)
        r, c = img.shape[:2]
        pts  = np.array([[
            [c * 0.05, r * 0.98],
            [c * 0.35, r * 0.55],
            [c * 0.65, r * 0.55],
            [c * 0.95, r * 0.98],
        ]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return cv2.bitwise_and(img, mask)

    def _fit_lines(self, img, hough):
        h, w = img.shape[:2]
        left_segs, left_wts   = [], []
        right_segs, right_wts = [], []

        for seg in hough:
            x1, y1, x2, y2 = seg[0]
            dx = x2 - x1
            if dx == 0:
                continue
            slope = (y2 - y1) / dx
            if not np.isfinite(slope) or abs(slope) < MIN_ABS_SLOPE:
                continue
            intercept = y1 - slope * x1
            x_bottom  = (h - intercept) / slope
            length    = np.hypot(dx, y2 - y1)
            if x_bottom < w * 0.5:
                left_segs.append((slope, intercept));  left_wts.append(length)
            else:
                right_segs.append((slope, intercept)); right_wts.append(length)

        def to_px(segs, wts):
            if not segs:
                return None
            s, b = np.average(segs, axis=0, weights=wts)
            y_lo = int(h * 0.98)
            y_hi = int(h * 0.55)
            return ((int((y_lo - b) / s), y_lo), (int((y_hi - b) / s), y_hi))

        return to_px(left_segs, left_wts), to_px(right_segs, right_wts)

    # ── 3-D projection ────────────────────────────────────────────────────

    def _line_to_3d(self, line, depth, cam_idx, n=20):
        if line is None:
            return None
        (x1, y1), (x2, y2) = line
        h_d, w_d = depth.shape[:2]
        K  = self.K.get(cam_idx)
        fx = K[0, 0] if K is not None else 500.0
        cx = K[0, 2] if K is not None else w_d / 2.0

        pts = []
        for t in np.linspace(0.0, 1.0, n):
            u = int(x1 + t * (x2 - x1))
            v = int(y1 + t * (y2 - y1))
            if not (0 <= u < w_d and 0 <= v < h_d):
                continue
            d = float(depth[v, u])
            if not (0.1 < d < 20.0):
                continue
            pts.append((d, -(u - cx) * d / fx))  # (forward, lateral)
        return pts if pts else None

    def _fill_missing(self, left_3d, right_3d):
        W = self.assumed_lane_width
        if left_3d is not None and right_3d is None:
            right_3d = [(fwd, lat - W) for fwd, lat in left_3d]
        elif right_3d is not None and left_3d is None:
            left_3d  = [(fwd, lat + W) for fwd, lat in right_3d]
        return left_3d, right_3d

    # ── Occupancy grid ────────────────────────────────────────────────────

    def _build_grid(self, left_pts, right_pts, stamp):
        g = self._empty_grid(stamp)
        if left_pts is None or right_pts is None:
            return g

        lpts = sorted(left_pts,  key=lambda p: p[0])
        rpts = sorted(right_pts, key=lambda p: p[0])
        if len(lpts) < 2 or len(rpts) < 2:
            return g

        W, H, res = g.info.width, g.info.height, self.grid_res
        data = list(g.data)

        for row in range(H):
            fwd = row * res
            ll  = self._interp(lpts, fwd)
            rl  = self._interp(rpts, fwd)
            if ll is None or rl is None:
                continue
            left_col  = max(0, min(W - 1, int((ll + self.grid_width_m / 2) / res)))
            right_col = max(0, min(W - 1, int((rl + self.grid_width_m / 2) / res)))
            lo, hi = min(left_col, right_col), max(left_col, right_col)
            for col in range(W):
                if lo < col < hi:
                    data[row * W + col] = 0    # free — driveable interior
                elif col == lo or col == hi:
                    data[row * W + col] = 100  # lethal — lane boundary

        g.data = data
        return g

    def _empty_grid(self, stamp=None):
        g = OccupancyGrid()
        g.header.stamp    = self.get_clock().now().to_msg() if stamp is None else stamp
        g.header.frame_id = self.base_frame
        W = int(self.grid_width_m  / self.grid_res)
        H = int(self.grid_height_m / self.grid_res)
        g.info.resolution = self.grid_res
        g.info.width      = W
        g.info.height     = H
        g.info.origin.position.x   =  0.0
        g.info.origin.position.y   = -self.grid_width_m / 2.0
        g.info.origin.orientation.w = 1.0
        g.data = [-1] * (W * H)
        return g

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _interp(pts, x):
        if len(pts) < 2:
            return None
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            if x1 <= x <= x2:
                t = (x - x1) / (x2 - x1 + 1e-9)
                return y1 + t * (y2 - y1)
        return None

    def _republish_grid(self):
        self.latest_grid.header.stamp = self.get_clock().now().to_msg()
        self.grid_pub.publish(self.latest_grid)

    def _publish_overlay(self, idx, bgr, hough, left_px, right_px, rgb_msg):
        ov = bgr.copy()
        if hough is not None:
            for seg in hough:
                cv2.line(ov, (seg[0][0], seg[0][1]), (seg[0][2], seg[0][3]),
                         (0, 255, 255), 1)
        if left_px:
            cv2.line(ov, left_px[0], left_px[1], (255, 80, 0), 3)
        if right_px:
            cv2.line(ov, right_px[0], right_px[1], (0, 80, 255), 3)
        h, w = ov.shape[:2]
        cv2.line(ov, (w // 2, h), (w // 2, h // 2), (0, 255, 0), 1)
        try:
            msg = self.bridge.cv2_to_imgmsg(ov, 'bgr8')
            msg.header = rgb_msg.header
            self.overlay_pubs[idx].publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Overlay error: {e}', throttle_duration_sec=2.0)

    def _watchdog(self):
        if not self._got_frame:
            self.get_logger().warn(
                'No synced RGB+Depth frames. Check topics and slop.',
                throttle_duration_sec=5.0)
        self._got_frame = False


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LaneDetectionNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()