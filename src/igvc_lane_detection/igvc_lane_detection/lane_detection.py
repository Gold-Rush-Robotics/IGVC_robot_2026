import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import Buffer, TransformListener

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

        self.base_frame             = p('base_frame',               'base_link')
        self.assumed_lane_width     = p('assumed_lane_width_m',      3.0)
        self.depth_scale            = p('depth_scale',               1.0)
        self.grid_res               = p('grid_resolution',           0.05)
        self.grid_width_m           = p('grid_width',               10.0)
        self.grid_height_m          = p('grid_height',              10.0)
        self.publish_overlay        = p('publish_overlay',           True)
        self.fusion_timeout_sec     = p('fusion_timeout_sec',         0.5)
        self.keep_last_grid_on_miss = p('keep_last_grid_on_miss',    True)
        self.min_lane_points        = p('min_lane_points',            6)
        self.occupancy_grid_frame   = p('occupancy_grid_frame',      self.base_frame)

        # ── Persistent map parameters ──────────────────────────────────────
        # The persistent map lives in a fixed frame (default: odom) and
        # accumulates evidence of lane boundaries over time.  Every update
        # the evidence array is multiplied by persist_decay_rate, so cells
        # that are no longer observed fade out gradually.  Cells whose
        # accumulated value exceeds persist_threshold are published as
        # lethal (100) in /lane_map; the rest are published as unknown (-1).
        self.persist_frame      = p('persistent_map_frame',      'odom')
        self.persist_res        = p('persistent_map_resolution',  0.10)   # m/cell – coarser saves RAM
        self.persist_size_m     = p('persistent_map_size_m',     100.0)   # square side length
        self.persist_decay      = p('persistent_map_decay',       0.998)  # multiplied each update
        self.persist_hit_w      = p('persistent_hit_weight',      8.0)    # added per observed point
        self.persist_threshold  = p('persistent_threshold',       20.0)   # publish as boundary
        self.persist_max        = p('persistent_max_value',      200.0)   # clamp to prevent blowup
        self.persist_pub_hz     = p('persistent_publish_hz',       2.0)   # how often to publish map

        self._init_persistent_map()

        # ── Camera subscriptions ──────────────────────────────────────────
        num_cameras  = p('num_cameras',        1)
        cam_topics   = p('camera_topics',      ['/camera/image_raw'])
        depth_topics = p('depth_topics',       ['/camera/depth/image_raw'])
        info_topics  = p('camera_info_topics', ['/camera/camera_info'])

        num_cameras = min(num_cameras, len(cam_topics), len(depth_topics), len(info_topics))

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
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        # Per-frame local costmap (rolling window around robot)
        self.grid_pub    = self.create_publisher(OccupancyGrid, '/lane_costmap', map_qos)
        # Persistent accumulated map in fixed frame
        self.persist_pub = self.create_publisher(OccupancyGrid, '/lane_map', map_qos)

        self.latest_grid = self._empty_grid()
        self._cam_state  = {}

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_timer(1.0, self._republish_grid)
        self.create_timer(1.0 / max(self.persist_pub_hz, 0.1), self._publish_persistent_map)

        self._got_frame = False
        self.create_timer(2.0, self._watchdog)

    # ═══════════════════════════════════════════════════════════════════════
    # Persistent map
    # ═══════════════════════════════════════════════════════════════════════

    def _init_persistent_map(self):
        """Allocate the hit-count array for the persistent fixed-frame map."""
        N = int(self.persist_size_m / self.persist_res)
        self._pN = N
        self._phits = np.zeros((N, N), dtype=np.float32)
        # Grid origin: bottom-left corner in persistent_frame coordinates.
        # We centre the grid on the world origin so the robot starts near
        # the middle and can drive in any direction.
        half = self.persist_size_m / 2.0
        self._p_ox = -half   # world-x of grid column 0
        self._p_oy = -half   # world-y of grid row 0
        self.get_logger().info(
            f'Persistent map: {N}x{N} cells @ {self.persist_res} m/cell '
            f'({self.persist_size_m} m square) in frame "{self.persist_frame}"')

    def _world_to_pgrid(self, wx, wy):
        """Convert world (persistent_frame) coordinates to grid (col, row)."""
        col = int((wx - self._p_ox) / self.persist_res)
        row = int((wy - self._p_oy) / self.persist_res)
        return col, row

    def _update_persistent_map(self, left_pts, right_pts, stamp):
        """
        Project lane boundary points (in base_link) into the persistent frame
        using TF, then increment the hit-count array.  A global decay is
        applied every call so stale evidence fades over time.
        """
        tf = self._lookup_tf(self.persist_frame, self.base_frame, stamp)
        if tf is None:
            return

        tx  = tf.transform.translation.x
        ty  = tf.transform.translation.y
        q   = tf.transform.rotation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        N = self._pN

        # Global evidence decay — keeps the map from accumulating
        # infinite values and allows old detections to be overwritten
        # if the geometry changes (e.g. different track section).
        self._phits *= self.persist_decay

        def mark(pts):
            if pts is None:
                return
            for fwd, lat in pts:
                # Rotate (fwd, lat) from base_link into persistent_frame
                wx = tx + cos_y * fwd - sin_y * lat
                wy = ty + sin_y * fwd + cos_y * lat
                col, row = self._world_to_pgrid(wx, wy)
                if 0 <= col < N and 0 <= row < N:
                    self._phits[row, col] = min(
                        self._phits[row, col] + self.persist_hit_w,
                        self.persist_max)

        mark(left_pts)
        mark(right_pts)

    def _publish_persistent_map(self):
        """Threshold the hit-count array and publish as OccupancyGrid."""
        N = self._pN
        g = OccupancyGrid()
        g.header.stamp              = self.get_clock().now().to_msg()
        g.header.frame_id           = self.persist_frame
        g.info.resolution           = self.persist_res
        g.info.width                = N
        g.info.height               = N
        g.info.origin.position.x    = self._p_ox
        g.info.origin.position.y    = self._p_oy
        g.info.origin.orientation.w = 1.0

        # Cells above threshold → lethal boundary (100); else unknown (-1)
        data = np.where(self._phits >= self.persist_threshold, 100, -1).astype(np.int8)
        g.data = data.flatten().tolist()
        self.persist_pub.publish(g)

    # ═══════════════════════════════════════════════════════════════════════
    # Camera info
    # ═══════════════════════════════════════════════════════════════════════

    def _on_info(self, msg, idx):
        self.K[idx] = np.array(msg.k).reshape(3, 3)
        self.get_logger().info(f'Camera[{idx}] intrinsics received.', once=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Main callback
    # ═══════════════════════════════════════════════════════════════════════

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

        self._cam_state[cam_idx] = {
            'stamp':    rgb_msg.header.stamp,
            'left_3d':  left_3d,
            'right_3d': right_3d,
        }

        fused_left, fused_right = self._fuse_lanes(rgb_msg.header.stamp)

        if fused_left is not None and fused_right is not None:
            new_grid = self._build_grid(fused_left, fused_right, rgb_msg.header.stamp)
            self.latest_grid = new_grid
            self.grid_pub.publish(self.latest_grid)
            # ── Accumulate into the persistent map ─────────────────────
            self._update_persistent_map(fused_left, fused_right, rgb_msg.header.stamp)
        elif self.keep_last_grid_on_miss:
            self.latest_grid.header.stamp = rgb_msg.header.stamp
            self.grid_pub.publish(self.latest_grid)
        else:
            self.latest_grid = self._empty_grid(rgb_msg.header.stamp)
            self.grid_pub.publish(self.latest_grid)

        self.get_logger().info(
            f'cam[{cam_idx}] left={left_px is not None} right={right_px is not None} '
            f'left_3d={0 if left_3d is None else len(left_3d)} '
            f'right_3d={0 if right_3d is None else len(right_3d)} '
            f'active_cams={len(self._active_cam_states(rgb_msg.header.stamp))}',
            throttle_duration_sec=1.0)

    def _active_cam_states(self, stamp):
        now_t = Time.from_msg(stamp)
        active = []
        for state in self._cam_state.values():
            dt = (now_t - Time.from_msg(state['stamp'])).nanoseconds / 1e9
            if dt <= self.fusion_timeout_sec:
                active.append(state)
        return active

    def _fuse_lanes(self, stamp):
        active = self._active_cam_states(stamp)
        if not active:
            return None, None

        left_pts, right_pts = [], []
        for state in active:
            if state['left_3d'] is not None:
                left_pts.extend(state['left_3d'])
            if state['right_3d'] is not None:
                right_pts.extend(state['right_3d'])

        if len(left_pts) < self.min_lane_points or len(right_pts) < self.min_lane_points:
            return None, None

        return left_pts, right_pts

    # ═══════════════════════════════════════════════════════════════════════
    # Lane detection  ── lighting-robust pipeline
    # ═══════════════════════════════════════════════════════════════════════
    #
    # The original pipeline used raw HSV thresholds which fail badly when
    # lighting is uneven, dim, or very bright.  The new pipeline adds two
    # layers of robustness before colour classification:
    #
    #  1. CLAHE (Contrast Limited Adaptive Histogram Equalisation) on the
    #     L channel of LAB.  This stretches contrast locally so that faint
    #     lane markings pop out even in deep shadow or glare.
    #
    #  2. Adaptive Gaussian thresholding directly on the equalised L channel
    #     as a colour-agnostic fallback.  It picks up bright-relative
    #     markings (white, yellow, even retroreflective lines under
    #     headlights) regardless of absolute brightness.
    #
    #  The two masks are OR-ed together before edge detection, which makes
    #  Hough considerably more stable across lighting conditions.

    def _detect_lanes(self, bgr):
        # ── Step 1: CLAHE contrast normalisation ─────────────────────────
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq  = clahe.apply(l)
        bgr_eq = cv2.cvtColor(cv2.merge([l_eq, a, b_ch]), cv2.COLOR_LAB2BGR)

        # ── Step 2: Colour mask on the contrast-equalised image ───────────
        # Loosen the value range slightly versus the original so markings
        # that were clipped by the old thresholds are still caught.
        hsv = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2HSV)
        white_mask  = cv2.inRange(hsv,
                                   np.array([0,   0, 160]),
                                   np.array([180, 55, 255]))
        yellow_mask = cv2.inRange(hsv,
                                   np.array([15, 60,  80]),
                                   np.array([40, 255, 255]))
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # ── Step 3: Adaptive threshold on equalised L channel ─────────────
        # blockSize=15 and C=-5 highlight regions that are locally brighter
        # than their neighbourhood — exactly what lane markings look like.
        adapt = cv2.adaptiveThreshold(
            l_eq, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15, C=-5)

        # ── Step 4: Combine both cues ─────────────────────────────────────
        combined = cv2.bitwise_or(color_mask, adapt)

        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)

        edges = cv2.Canny(combined, 50, 150)
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
        left_segs,  left_wts  = [], []
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

    # ═══════════════════════════════════════════════════════════════════════
    # 3-D projection
    # ═══════════════════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════════════════
    # Local rolling costmap (per-frame)
    # ═══════════════════════════════════════════════════════════════════════

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
        g.header.frame_id = self.occupancy_grid_frame
        W = int(self.grid_width_m  / self.grid_res)
        H = int(self.grid_height_m / self.grid_res)
        g.info.resolution = self.grid_res
        g.info.width      = W
        g.info.height     = H

        if self.occupancy_grid_frame == self.base_frame:
            g.info.origin.position.x = 0.0
            g.info.origin.position.y = -self.grid_width_m / 2.0
            g.info.origin.orientation.w = 1.0
        else:
            tf = self._lookup_tf(self.occupancy_grid_frame, self.base_frame, stamp)
            if tf is None:
                g.info.origin.position.x = 0.0
                g.info.origin.position.y = -self.grid_width_m / 2.0
                g.info.origin.orientation.w = 1.0
            else:
                q = tf.transform.rotation
                yaw = np.arctan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                )
                tx = tf.transform.translation.x
                ty = tf.transform.translation.y
                g.info.origin.position.x = tx + np.sin(yaw) * (self.grid_width_m / 2.0)
                g.info.origin.position.y = ty - np.cos(yaw) * (self.grid_width_m / 2.0)
                g.info.origin.position.z = tf.transform.translation.z
                g.info.origin.orientation = q

        g.data = [-1] * (W * H)
        return g

    # ═══════════════════════════════════════════════════════════════════════
    # TF helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _lookup_tf(self, target_frame, source_frame, stamp):
        try:
            t = Time.from_msg(stamp) if stamp is not None else Time()
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, t,
                timeout=Duration(seconds=0.05))
        except Exception:
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame, source_frame, Time(),
                    timeout=Duration(seconds=0.05))
            except Exception:
                return None

    # ═══════════════════════════════════════════════════════════════════════
    # Misc helpers
    # ═══════════════════════════════════════════════════════════════════════

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