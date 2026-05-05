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
        self.fusion_timeout_sec     = p('fusion_timeout_sec',         2.0)
        self.keep_last_grid_on_miss = p('keep_last_grid_on_miss',    True)
        self.min_lane_points        = p('min_lane_points',            3)
        self.occupancy_grid_frame   = p('occupancy_grid_frame',      self.base_frame)

        # ── Chassis / ROI exclusion ────────────────────────────────────────
        self.chassis_mask_frac  = p('chassis_mask_frac',   0.15)
        self.roi_bottom_frac    = p('roi_bottom_frac',     0.82)
        self.roi_top_frac       = p('roi_top_frac',        0.55)
        # Minimum depth accepted during 3-D projection.  0.5 m keeps a safe
        # margin above the chassis (~0.3 m) while allowing ground points that
        # are physically close to the camera (e.g. mounted at ~0.6 m height).
        self.min_detection_depth_m = p('min_detection_depth_m', 0.5)
        self.depth_search_radius_px = p('depth_search_radius_px', 2)
        self.min_projected_line_points = p('min_projected_line_points', 3)

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
        self.persist_hit_w      = p('persistent_hit_weight',      12.0)    # added per observed point
        self.persist_threshold  = p('persistent_threshold',       15.0)   # publish as boundary
        self.persist_max        = p('persistent_max_value',      200.0)   # clamp to prevent blowup
        self.persist_pub_hz     = p('persistent_publish_hz',       2.0)   # how often to publish map
        self.persist_clear_radius = p('persistent_clear_radius_m', 0.8)

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
        self._clear_persistent_robot_footprint(data)
        g.data = data.flatten().tolist()
        self.persist_pub.publish(g)

    def _clear_persistent_robot_footprint(self, data):
        if self.persist_clear_radius <= 0.0:
            return
        tf = self._lookup_tf(self.persist_frame, self.base_frame, None)
        if tf is None:
            return

        col_c, row_c = self._world_to_pgrid(
            tf.transform.translation.x,
            tf.transform.translation.y,
        )
        radius_cells = max(1, int(np.ceil(self.persist_clear_radius / self.persist_res)))
        row_lo = max(0, row_c - radius_cells)
        row_hi = min(self._pN, row_c + radius_cells + 1)
        col_lo = max(0, col_c - radius_cells)
        col_hi = min(self._pN, col_c + radius_cells + 1)

        if row_lo >= row_hi or col_lo >= col_hi:
            return

        rows, cols = np.ogrid[row_lo:row_hi, col_lo:col_hi]
        mask = (rows - row_c) ** 2 + (cols - col_c) ** 2 <= radius_cells ** 2
        data[row_lo:row_hi, col_lo:col_hi][mask] = 0

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

        left_px, right_px, dbg_lines = self._detect_lanes(bgr, cam_idx)

        cam_frame = depth_msg.header.frame_id or rgb_msg.header.frame_id
        cam_tf = None
        if cam_frame and cam_frame != self.base_frame:
            cam_tf = self._lookup_tf(self.base_frame, cam_frame, rgb_msg.header.stamp)
            if cam_tf is None and (left_px is not None or right_px is not None):
                self.get_logger().warn(
                    f'No TF from {cam_frame} to {self.base_frame}; '
                    f'falling back to pinhole projection for cam[{cam_idx}]',
                    throttle_duration_sec=2.0)

        raw_left_3d = self._line_to_3d(left_px,  depth, cam_idx, cam_tf)
        raw_right_3d = self._line_to_3d(right_px, depth, cam_idx, cam_tf)
        overlay_left_px, overlay_right_px = self._relabel_projected_pixels(
            left_px, raw_left_3d, right_px, raw_right_3d)
        left_3d, right_3d = self._relabel_projected_sides(raw_left_3d, raw_right_3d)

        if self.publish_overlay and cam_idx in self.overlay_pubs:
            self._publish_overlay(
                cam_idx, bgr, dbg_lines, overlay_left_px, overlay_right_px, rgb_msg)

        self._cam_state[cam_idx] = {
            'stamp':    rgb_msg.header.stamp,
            'left_3d':  left_3d,
            'right_3d': right_3d,
        }

        fused_left, fused_right = self._fuse_lanes(rgb_msg.header.stamp)

        fused_left, fused_right = self._fill_missing(fused_left, fused_right)

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

        # Allow updating the map with only one side if the other can be assumed
        if (len(left_pts) >= 2 or len(right_pts) >= 2):   # changed from both >= min
            return left_pts, right_pts
        self.get_logger().warn((f'Not enough lane points for fusion. Left: {len(left_pts)}, Right: {len(right_pts)}'), throttle_duration_sec=2.0)
        return None, None

    # ═══════════════════════════════════════════════════════════════════════
    # Lane detection  ── lean, lighting-robust pipeline
    # ═══════════════════════════════════════════════════════════════════════
    #
    # Design rationale
    # ────────────────
    # The pipeline is colour-gated: Canny only runs on pixels that already
    # passed the white/yellow HSV filter.  This means asphalt texture, dirt,
    # shadows and painted numbers are suppressed before any edge detection.
    #
    # Pipeline:
    #   chassis mask
    #   → CLAHE on L (clipLimit 2.0 — lifts dim markings without amplifying grain)
    #   → HSV colour mask (white + yellow) on CLAHE image
    #   → dilate mask by 1 px to widen thin distant lines
    #   → Canny on (colour_mask * L_eq)  ← colour-gated grayscale edges only
    #   → ROI trapezoid
    #   → HoughLinesP
    #   → midpoint-based left/right classification

    def _detect_lanes(self, bgr, cam_idx):
        h_img, w_img = bgr.shape[:2]

        # ── Step 0: Chassis mask (unchanged) ─────────────────────────────
        if self.chassis_mask_frac > 0.0:
            bgr = bgr.copy()
            cut = int(h_img * (1.0 - self.chassis_mask_frac))
            bgr[cut:, :] = 0

        # ── Step 1: CLAHE on L channel (lighting-robust, surface-agnostic) ──
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)

        # ── Step 2: Colour-gated grayscale edges ─────────────────────────
        # Gate Canny by a white+yellow HSV mask.  The ROI trapezoid
        # already excludes the sky/horizon, so we can be fairly
        # permissive here: we only need to reject the very dark asphalt
        # (low V) and strongly-saturated foliage/dirt (high S non-yellow).
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        white_mask  = cv2.inRange(hsv, (0,   0, 150), (180,  70, 255))
        yellow_mask = cv2.inRange(hsv, (15, 60,  80), ( 40, 255, 255))
        colour_mask = cv2.bitwise_or(white_mask, yellow_mask)
        colour_mask = cv2.dilate(
            colour_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)

        gated = cv2.bitwise_and(l_eq, l_eq, mask=colour_mask)
        blurred = cv2.GaussianBlur(gated, (5, 5), 0)

        edges = cv2.Canny(blurred, 40, 120, apertureSize=3)

        # ROI trapezoid (already very effective)
        roi = self._apply_roi(edges)

        # ── Step 3: Probabilistic Hough ──────────────────────────────────
        hough = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=30,
            maxLineGap=60,
        )
        if hough is None:
            return None, None, None

        left_px, right_px = self._fit_lines(bgr, hough, cam_idx)
        return left_px, right_px, hough


    def _apply_roi(self, img):
        mask = np.zeros_like(img)
        r, c = img.shape[:2]
        bot  = self.roi_bottom_frac
        top  = self.roi_top_frac
        pts  = np.array([[
            [c * 0.05, r * bot],
            [c * 0.30, r * top],
            [c * 0.70, r * top],
            [c * 0.95, r * bot],
        ]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return cv2.bitwise_and(img, mask)

    def _fit_lines(self, img, hough, cam_idx):
        h, w = img.shape[:2]
        y_bot = int(h * self.roi_bottom_frac)
        y_top = int(h * self.roi_top_frac)
        allow_shallow_side_segments = cam_idx != 0

        left_segs,  left_wts  = [], []
        right_segs, right_wts = [], []
        left_relaxed_segs, left_relaxed_wts = [], []
        right_relaxed_segs, right_relaxed_wts = [], []

        min_inward_dx = 5.0
        relaxed_min_inward_dx = 0.5

        for seg in hough:
            x1, y1, x2, y2 = seg[0]

            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = np.hypot(dx, dy)
            slope = dy / dx if abs(dx) > 0.1 else (1e6 if dy > 0 else -1e6)
            intercept = y1 - slope * x1

            # ── Angle filter ─────────────────────────────────────────────
            # Reject only the most horizontal segments (true horizon / roof
            # lines) and near-vertical ones (poles, posts). Side cameras on
            # tight turns can see valid lane boundaries at ~80°+ from
            # vertical, so the upper bound must be loose.
            angle_from_vert = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9))
            if angle_from_vert < 10:
                continue

            if angle_from_vert > 87:
                if allow_shallow_side_segments:
                    if cam_idx == 1:
                        left_relaxed_segs.append((slope, intercept))
                        left_relaxed_wts.append(length)
                    elif cam_idx == 2:
                        right_relaxed_segs.append((slope, intercept))
                        right_relaxed_wts.append(length)
                continue

            # ── Left / right classification by ROI-bottom intercept ──────
            # Midpoint-based classification breaks on sharp turns because a
            # real boundary can cross the image centre while still being the
            # same physical lane edge.  Instead, classify by where the line
            # hits the *bottom* of the ROI (closest / highest-confidence
            # part of the image), then require it to lean inward toward the
            # corridor as it rises.
            x_bot = (y_bot - intercept) / (slope + 1e-9)
            x_top = (y_top - intercept) / (slope + 1e-9)
            inward_dx = x_top - x_bot

            if x_bot < w * 0.5:
                if inward_dx > min_inward_dx:
                    left_segs.append((slope, intercept))
                    left_wts.append(length)
                if inward_dx > relaxed_min_inward_dx:
                    left_relaxed_segs.append((slope, intercept))
                    left_relaxed_wts.append(length)
            else:
                if inward_dx < -min_inward_dx:
                    right_segs.append((slope, intercept))
                    right_wts.append(length)
                if inward_dx < -relaxed_min_inward_dx:
                    right_relaxed_segs.append((slope, intercept))
                    right_relaxed_wts.append(length)

        if not left_segs and left_relaxed_segs:
            left_segs, left_wts = left_relaxed_segs, left_relaxed_wts
        if not right_segs and right_relaxed_segs:
            right_segs, right_wts = right_relaxed_segs, right_relaxed_wts

        def to_px(segs, wts):
            if not segs:
                return None
            s, b = np.average(segs, axis=0, weights=wts)
            if abs(s) < 1e-6:
                return None
            return ((int((y_bot - b) / s), y_bot),
                    (int((y_top - b) / s), y_top))
        left_px = to_px(left_segs, left_wts)
        right_px = to_px(right_segs, right_wts)

        if left_px is not None and right_px is not None:
            lx = left_px[0][0]
            rx = right_px[0][0]
            same_left_half = lx < w * 0.5 and rx < w * 0.5
            same_right_half = lx > w * 0.5 and rx > w * 0.5
            min_sep_px = max(24, int(0.12 * w))
            too_close = abs(rx - lx) < min_sep_px

            if too_close:
                left_strength = float(np.sum(left_wts)) if left_wts else 0.0
                right_strength = float(np.sum(right_wts)) if right_wts else 0.0
                if left_strength >= right_strength:
                    right_px = None
                else:
                    left_px = None
            elif same_left_half or same_right_half:
                # Tight turns can place both real boundaries on one image half.
                # Keep both when they are well separated and let projection/
                # lateral-sign relabeling decide final side assignment.
                pass

        return left_px, right_px

    # ═══════════════════════════════════════════════════════════════════════
    # 3-D projection
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _quat_to_rot(qx, qy, qz, qw):
        return np.array([
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ], dtype=np.float32)

    def _line_to_3d(self, line, depth, cam_idx, cam_tf=None, n=20):
        if line is None:
            return None
        (x1, y1), (x2, y2) = line
        h_d, w_d = depth.shape[:2]
        K  = self.K.get(cam_idx)
        fx = K[0, 0] if K is not None else 500.0
        fy = K[1, 1] if K is not None else fx
        cx = K[0, 2] if K is not None else w_d / 2.0
        cy = K[1, 2] if K is not None else h_d / 2.0

        rot = None
        trans = None
        if cam_tf is not None:
            q = cam_tf.transform.rotation
            rot = self._quat_to_rot(q.x, q.y, q.z, q.w)
            t = cam_tf.transform.translation
            trans = np.array([t.x, t.y, t.z], dtype=np.float32)

        pts = []
        for t in np.linspace(0.0, 1.0, n):
            u = int(x1 + t * (x2 - x1))
            v = int(y1 + t * (y2 - y1))
            if not (0 <= u < w_d and 0 <= v < h_d):
                continue
            d = self._sample_valid_depth(depth, u, v)
            if d is None:
                continue

            if rot is None or trans is None:
                pts.append((d, -(u - cx) * d / fx))  # (forward, lateral)
                continue

            # Registered depth images are typically expressed in the camera
            # optical frame: x=right, y=down, z=forward.  Transform each
            # sampled point into base_link so multi-camera fusion happens in
            # one consistent frame.
            point_cam = np.array([
                (u - cx) * d / fx,
                (v - cy) * d / fy,
                d,
            ], dtype=np.float32)
            point_base = rot @ point_cam + trans
            fwd = float(point_base[0])
            lat = float(point_base[1])
            min_fwd = -0.4 if cam_idx != 0 else 0.0
            if fwd <= min_fwd:
                continue
            if fwd < 0.0:
                fwd = 0.0
            pts.append((fwd, lat))
        if len(pts) < self.min_projected_line_points:
            return None
        return pts

    def _sample_valid_depth(self, depth, u, v):
        radius = max(0, int(self.depth_search_radius_px))
        u0 = max(0, u - radius)
        u1 = min(depth.shape[1], u + radius + 1)
        v0 = max(0, v - radius)
        v1 = min(depth.shape[0], v + radius + 1)
        patch = depth[v0:v1, u0:u1]
        if patch.size == 0:
            return None

        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > self.min_detection_depth_m) & (valid < 20.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _relabel_projected_sides(self, left_3d, right_3d):
        tol = 0.05

        def mean_lat(pts):
            if pts is None or not pts:
                return None
            return float(np.mean([lat for _, lat in pts]))

        left_groups = []
        right_groups = []

        for nominal_side, pts in (("left", left_3d), ("right", right_3d)):
            if pts is None:
                continue
            lat = mean_lat(pts)
            if lat is None:
                continue
            if lat > tol:
                left_groups.append(pts)
            elif lat < -tol:
                right_groups.append(pts)
            elif nominal_side == "left":
                left_groups.append(pts)
            else:
                right_groups.append(pts)

        merged_left = [pt for pts in left_groups for pt in pts] or None
        merged_right = [pt for pts in right_groups for pt in pts] or None
        return merged_left, merged_right

    def _relabel_projected_pixels(self, left_px, left_3d, right_px, right_3d):
        tol = 0.05

        def mean_lat(pts):
            if pts is None or not pts:
                return None
            return float(np.mean([lat for _, lat in pts]))

        left_candidates = []
        right_candidates = []

        for nominal_side, px, pts in (("left", left_px, left_3d), ("right", right_px, right_3d)):
            if px is None:
                continue
            lat = mean_lat(pts)
            if lat is None or abs(lat) <= tol:
                target_side = nominal_side
            else:
                target_side = "left" if lat > 0.0 else "right"

            if target_side == "left":
                left_candidates.append((px, 0 if pts is None else len(pts)))
            else:
                right_candidates.append((px, 0 if pts is None else len(pts)))

        overlay_left = max(left_candidates, key=lambda item: item[1])[0] if left_candidates else None
        overlay_right = max(right_candidates, key=lambda item: item[1])[0] if right_candidates else None
        return overlay_left, overlay_right

    def _fill_missing(self, left_3d, right_3d):
        W = self.assumed_lane_width
        min_valid_sep = max(0.6, 0.35 * W)
        side_tol = 0.08
        min_side_pts = max(3, int(self.min_projected_line_points))

        def mean_lat(pts):
            if pts is None or not pts:
                return None
            return float(np.mean([lat for _, lat in pts]))

        # Repartition fused points by actual lateral sign in base_link.
        # This is the most reliable separator when camera-view heuristics
        # disagree near turn apexes.
        all_pts = []
        if left_3d is not None:
            all_pts.extend(left_3d)
        if right_3d is not None:
            all_pts.extend(right_3d)

        if all_pts:
            pos_pts = [(fwd, lat) for fwd, lat in all_pts if lat > side_tol]
            neg_pts = [(fwd, lat) for fwd, lat in all_pts if lat < -side_tol]
            if len(pos_pts) >= min_side_pts:
                left_3d = pos_pts
            elif left_3d is not None and mean_lat(left_3d) is not None and mean_lat(left_3d) > 0.0:
                left_3d = left_3d
            else:
                left_3d = None

            if len(neg_pts) >= min_side_pts:
                right_3d = neg_pts
            elif right_3d is not None and mean_lat(right_3d) is not None and mean_lat(right_3d) < 0.0:
                right_3d = right_3d
            else:
                right_3d = None

        left_lat = mean_lat(left_3d)
        right_lat = mean_lat(right_3d)

        # Safety net: if a detected boundary lands on the wrong lateral side
        # of the robot, relabel it before synthesising the missing boundary.
        if left_3d is not None and left_lat is not None and left_lat < -0.1:
            if right_3d is None:
                right_3d, left_3d = left_3d, None
            elif right_lat is not None and right_lat > 0.1:
                left_3d, right_3d = right_3d, left_3d
        if right_3d is not None and right_lat is not None and right_lat > 0.1:
            if left_3d is None:
                left_3d, right_3d = right_3d, None
            elif left_lat is not None and left_lat < -0.1:
                left_3d, right_3d = right_3d, left_3d

        left_lat = mean_lat(left_3d)
        right_lat = mean_lat(right_3d)

        if left_3d is not None and right_3d is not None and left_lat is not None and right_lat is not None:
            sep = left_lat - right_lat
            invalid_signs = (left_lat <= 0.0) or (right_lat >= 0.0)
            collapsed = sep < min_valid_sep
            if invalid_signs or collapsed:
                left_count = len(left_3d)
                right_count = len(right_3d)
                if left_count >= right_count:
                    right_3d = [(fwd, lat - W) for fwd, lat in left_3d]
                else:
                    left_3d = [(fwd, lat + W) for fwd, lat in right_3d]

        if left_3d is not None and right_3d is None:
            right_3d = [(fwd, lat - W) for fwd, lat in left_3d]
        elif right_3d is not None and left_3d is None:
            left_3d  = [(fwd, lat + W) for fwd, lat in right_3d]
        # If both missing → return None, None (already handled upstream)
        return left_3d, right_3d

    # ═══════════════════════════════════════════════════════════════════════
    # Local rolling costmap (per-frame)
    # ═══════════════════════════════════════════════════════════════════════

    def _build_grid(self, left_pts, right_pts, stamp):
        g = self._empty_grid(stamp)
        if left_pts is None or right_pts is None:
            return g

        lpts = sorted(left_pts, key=lambda p: p[0])
        rpts = sorted(right_pts, key=lambda p: p[0])

        # === FIXED: only reject if BOTH sides are insufficient ===
        if len(lpts) < 2 and len(rpts) < 2:
            return g

        # Ensure both sides exist (mirrors _fill_missing logic)
        W = self.assumed_lane_width
        if len(lpts) >= 2 and len(rpts) < 2:
            rpts = [(fwd, lat - W) for fwd, lat in lpts]
        elif len(rpts) >= 2 and len(lpts) < 2:
            lpts = [(fwd, lat + W) for fwd, lat in rpts]

        W_grid, H, res = g.info.width, g.info.height, self.grid_res
        data = list(g.data)

        for row in range(H):
            fwd = row * res
            ll = self._interp(lpts, fwd)
            rl = self._interp(rpts, fwd)
            if ll is None or rl is None:
                continue

            left_col = max(0, min(W_grid - 1, int((ll + self.grid_width_m / 2) / res)))
            right_col = max(0, min(W_grid - 1, int((rl + self.grid_width_m / 2) / res)))
            lo, hi = min(left_col, right_col), max(left_col, right_col)

            for col in range(W_grid):
                if lo < col < hi:
                    data[row * W_grid + col] = 0      # free
                elif col == lo or col == hi:
                    data[row * W_grid + col] = 100    # lethal boundary

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