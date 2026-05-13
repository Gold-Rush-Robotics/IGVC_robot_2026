"""ROS 2 lane-detection node backed by the YOLOPv2 segmentation model.

Drop-in replacement for :mod:`igvc_lane_detection.lane_detection`.  Instead
of the Canny / Hough pipeline, every RGB frame is run through a pretrained
YOLOPv2 TorchScript network which returns two binary masks:

* **Drivable-area** → rasterised as *free* (0) cells in the local costmap.
* **Lane-lines**    → rasterised as *lethal* (100) cells and accumulated
  into the persistent map in the fixed frame (default ``map``).

Connected components of the lane-line mask are published as a
``MarkerArray`` on ``/lane_segmentation/lanes`` so RViz can show an
arbitrary number of detected lanes without a forced left/right split.

The topic / QoS / parameter surface deliberately mirrors the Hough node so
downstream consumers (navigator, nav2 costmap fusion) continue to work
unchanged.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from .projection_utils import (
    lookup_tf,
    pixel_to_base,
    sample_valid_depth,
    yaw_from_quat,
)
from .yolopv2_infer import YolopV2


# Cycle-coloured palette for the MarkerArray debug view.  Keeps the first
# few detected lanes visually distinct without drawing the rainbow.
_LANE_COLORS: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 0.9, 0.9),
    (0.9, 0.4, 0.1),
    (0.1, 0.9, 0.3),
    (0.9, 0.1, 0.6),
    (0.9, 0.9, 0.1),
    (0.4, 0.3, 0.9),
)


class LaneSegmentationNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_segmentation_node')
        self.bridge = CvBridge()
        self.K: dict = {}

        # ── Parameters (shared with LaneDetectionNode) ────────────────
        def p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        self.base_frame             = p('base_frame',              'base_link')
        self.depth_scale            = p('depth_scale',              1.0)
        self.grid_res               = p('grid_resolution',          0.05)
        self.grid_width_m           = p('grid_width',              10.0)
        self.grid_height_m          = p('grid_height',             10.0)
        self.publish_overlay        = p('publish_overlay',          True)
        self.max_time_offset_sec    = float(p('max_time_offset_sec', 0.1))
        self.fusion_timeout_sec     = min(
            float(p('fusion_timeout_sec', 0.1)), self.max_time_offset_sec)
        self.keep_last_grid_on_miss = p('keep_last_grid_on_miss',   True)
        self.occupancy_grid_frame   = p('occupancy_grid_frame',    self.base_frame)
        self.sync_queue_size        = max(1, int(p('sync_queue_size', 2)))
        self.sync_slop_sec          = float(p('sync_slop_sec', 0.1))
        self.max_frame_age_sec      = float(p('max_frame_age_sec', 0.5))

        self.chassis_mask_frac      = p('chassis_mask_frac',        0.15)
        # Trapezoidal ROI is off by default — the offline bag-replay
        # experiment showed YOLOPv2 produces noticeably cleaner masks
        # when fed (and kept on) the full frame, because the trapezoid
        # was clipping drivable area on the far sides.  Re-enable only
        # if the wider FOV starts producing spurious detections on the
        # sky / buildings in a specific venue.
        self.roi_enabled            = bool(p('roi_enabled',         False))
        self.roi_bottom_frac        = p('roi_bottom_frac',          0.95)
        self.roi_top_frac           = p('roi_top_frac',             0.55)
        self.min_detection_depth_m  = p('min_detection_depth_m',    0.5)
        self.max_detection_depth_m  = p('max_detection_depth_m',   20.0)
        self.depth_search_radius_px = p('depth_search_radius_px',   2)

        # ── Persistent map parameters ────────────────────────────────
        self.persist_frame        = p('persistent_map_frame',       'odom')
        self.persist_res          = p('persistent_map_resolution',   0.10)
        self.persist_size_m       = p('persistent_map_size_m',     100.0)
        self.persist_decay        = p('persistent_map_decay',        0.998)
        self.persist_hit_w        = p('persistent_hit_weight',       4.0)
        self.persist_threshold    = p('persistent_threshold',       15.0)
        self.persist_max          = p('persistent_max_value',      200.0)
        self.persist_pub_hz       = p('persistent_publish_hz',       2.0)
        self.persist_clear_radius = p('persistent_clear_radius_m',   0.8)
        self.persist_pose_source  = p('persistent_pose_source',      'tf')
        self.odom_topic           = p('odom_topic',                  '/odom')

        # ── Lane-line correlation / map-alignment gate ───────────────
        self.enable_line_correlation = bool(p('enable_line_correlation', True))
        self.line_corr_max_age_sec = float(p('line_correlation_max_age_sec', 0.5))
        self.line_corr_match_dist_m = float(p('line_correlation_match_distance_m', 0.35))
        self.line_corr_match_angle_rad = float(p('line_correlation_match_angle_rad', 0.35))
        self.line_corr_min_overlap = float(p('line_correlation_min_overlap_ratio', 0.25))
        self.line_corr_new_min_length_m = float(p('line_correlation_new_line_min_length_m', 0.6))
        self.line_corr_min_points = int(p('line_correlation_min_points', 4))
        self.line_corr_nudge_alpha = float(p('line_correlation_nudge_alpha', 0.15))
        self.line_corr_reject_between = bool(p('line_correlation_reject_between_parallel_lines', True))
        self.line_corr_debug = bool(p('line_correlation_debug', False))

        # ── Segmentation-specific parameters ─────────────────────────
        self.model_weights          = p('model_weights',            '')
        self.model_device           = p('model_device',             'cuda:0')
        self.model_half             = p('model_half',               True)
        self.model_img_size         = p('model_img_size',           640)
        self.da_subsample_px        = max(1, int(p('da_subsample_px', 6)))
        self.ll_subsample_px        = max(1, int(p('ll_subsample_px', 2)))
        self.min_lane_component_px  = int(p('min_lane_component_px', 150))
        self.max_points_per_frame   = int(p('max_points_per_frame', 4000))
        self.publish_mask_overlay   = p('publish_mask_overlay',     True)
        self.lane_marker_topic      = p('lane_marker_topic',        '/lane_segmentation/lanes')

        if not self.model_weights:
            raise RuntimeError(
                "Parameter 'model_weights' must be set to the path of "
                "yolopv2.pt. On the Jetson target this is typically "
                "$(repo)/models/yolopv2.pt — fetch with "
                "src/igvc_lane_detection/scripts/fetch_yolopv2_weights.sh.")

        # ── Initialise segmentation model ────────────────────────────
        self.get_logger().info(
            f"Loading YOLOPv2 from '{self.model_weights}' on "
            f"'{self.model_device}' (half={self.model_half})…")
        self.model = YolopV2(
            weights_path=self.model_weights,
            device=self.model_device,
            half=self.model_half,
            img_size=self.model_img_size,
        )
        self.model.load()
        if self.model.fallback_warning:
            self.get_logger().warn(self.model.fallback_warning)
        self.get_logger().info(
            f"YOLOPv2 ready on {self.model.device} "
            f"(half={self.model.half}, img_size={self.model_img_size}).")

        self._init_persistent_map()

        # ── Camera subscriptions ─────────────────────────────────────
        num_cameras  = p('num_cameras',        1)
        cam_topics   = p('camera_topics',      ['/camera/image_raw'])
        depth_topics = p('depth_topics',       ['/camera/depth/image_raw'])
        info_topics  = p('camera_info_topics', ['/camera/camera_info'])

        num_cameras = min(
            num_cameras, len(cam_topics), len(depth_topics), len(info_topics))

        self._sync_handles: list = []
        self.overlay_pubs: dict = {}

        for i in range(num_cameras):
            self.create_subscription(
                CameraInfo, info_topics[i],
                lambda msg, idx=i: self._on_info(msg, idx), 10)

            rgb_sub = Subscriber(
                self, Image, cam_topics[i], qos_profile=qos_profile_sensor_data)
            depth_sub = Subscriber(
                self, Image, depth_topics[i], qos_profile=qos_profile_sensor_data)
            sync = ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub],
                queue_size=self.sync_queue_size,
                slop=self.sync_slop_sec)
            sync.registerCallback(
                lambda r, d, idx=i: self._on_images(r, d, idx))
            self._sync_handles.append((rgb_sub, depth_sub, sync))

            if self.publish_overlay:
                self.overlay_pubs[i] = self.create_publisher(
                    Image, f'/lane_debug/cam{i}/overlay', 10)

        # ── Publishers ───────────────────────────────────────────────
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self.grid_pub    = self.create_publisher(
            OccupancyGrid, '/lane_costmap', map_qos)
        self.persist_pub = self.create_publisher(
            OccupancyGrid, '/lane_map', map_qos)
        self.marker_pub  = self.create_publisher(
            MarkerArray, self.lane_marker_topic, 10)

        self.latest_grid = self._empty_grid()
        self._last_persistent_stamp = None
        self._cam_state: dict = {}
        self._latest_odom: Optional[Odometry] = None
        self._line_tracks: dict = {}
        self._next_line_track_id = 0

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if self.persist_pose_source == 'odom':
            odom_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1)
            self.create_subscription(
                Odometry, self.odom_topic, self._on_odom, odom_qos)
            self.get_logger().info(
                f'Persistent map pose source: odom topic {self.odom_topic}')

        self.create_timer(1.0, self._republish_grid)
        self.create_timer(
            1.0 / max(self.persist_pub_hz, 0.1),
            self._publish_persistent_map)

        self._got_frame = False
        self.create_timer(2.0, self._watchdog)

    # ═══════════════════════════════════════════════════════════════════
    # Persistent map
    # ═══════════════════════════════════════════════════════════════════

    def _init_persistent_map(self) -> None:
        n = int(self.persist_size_m / self.persist_res)
        self._pN = n
        self._phits = np.zeros((n, n), dtype=np.float32)
        half = self.persist_size_m / 2.0
        self._p_ox = -half
        self._p_oy = -half
        self.get_logger().info(
            f'Persistent map: {n}x{n} cells @ {self.persist_res} m/cell '
            f'({self.persist_size_m} m square) in frame "{self.persist_frame}"')

    def _world_to_pgrid(self, wx: float, wy: float) -> Tuple[int, int]:
        col = int((wx - self._p_ox) / self.persist_res)
        row = int((wy - self._p_oy) / self.persist_res)
        return col, row

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _persistent_pose(self, stamp=None):
        if self.persist_pose_source == 'odom':
            return self._persistent_pose_from_odom()

        transform = lookup_tf(
            self.tf_buffer, self.persist_frame, self.base_frame, stamp)
        if transform is None:
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            translation.x,
            translation.y,
            translation.z,
            yaw_from_quat(rotation.x, rotation.y, rotation.z, rotation.w),
            rotation,
        )

    def _persistent_pose_from_odom(self):
        if self._latest_odom is None:
            self.get_logger().warn(
                f'Waiting for odometry on {self.odom_topic} before updating persistent map.',
                throttle_duration_sec=2.0)
            return None

        odom = self._latest_odom
        odom_frame = odom.header.frame_id or 'odom'
        pose = odom.pose.pose
        odom_yaw = yaw_from_quat(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w)

        if odom_frame == self.persist_frame:
            return (
                pose.position.x,
                pose.position.y,
                pose.position.z,
                odom_yaw,
                pose.orientation,
            )

        transform = lookup_tf(
            self.tf_buffer, self.persist_frame, odom_frame, None)
        if transform is None:
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        transform_yaw = yaw_from_quat(rotation.x, rotation.y, rotation.z, rotation.w)
        cos_yaw, sin_yaw = np.cos(transform_yaw), np.sin(transform_yaw)
        x = translation.x + cos_yaw * pose.position.x - sin_yaw * pose.position.y
        y = translation.y + sin_yaw * pose.position.x + cos_yaw * pose.position.y
        return (x, y, translation.z + pose.position.z, transform_yaw + odom_yaw, pose.orientation)

    def _update_persistent_map(
        self, lane_pts: Optional[Sequence[Tuple[float, float]]], stamp,
    ) -> None:
        self._last_persistent_stamp = stamp
        if not lane_pts:
            # Still apply the global decay so stale evidence fades even on
            # empty frames.
            self._phits *= self.persist_decay
            return

        pose = self._persistent_pose(stamp)
        if pose is None:
            return

        tx, ty, _tz, yaw, _orientation = pose
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        n = self._pN

        self._phits *= self.persist_decay

        for fwd, lat in lane_pts:
            wx = tx + cos_y * fwd - sin_y * lat
            wy = ty + sin_y * fwd + cos_y * lat
            col, row = self._world_to_pgrid(wx, wy)
            if 0 <= col < n and 0 <= row < n:
                self._phits[row, col] = min(
                    self._phits[row, col] + self.persist_hit_w,
                    self.persist_max)

    def _publish_persistent_map(self) -> None:
        if self._last_persistent_stamp is None:
            return
        n = self._pN
        g = OccupancyGrid()
        g.header.stamp              = self._last_persistent_stamp
        g.header.frame_id           = self.persist_frame
        g.info.resolution           = self.persist_res
        g.info.width                = n
        g.info.height               = n
        g.info.origin.position.x    = self._p_ox
        g.info.origin.position.y    = self._p_oy
        g.info.origin.orientation.w = 1.0

        data = np.where(self._phits >= self.persist_threshold, 100, -1).astype(np.int8)
        self._clear_persistent_robot_footprint(data, self._last_persistent_stamp)
        g.data = data.flatten().tolist()
        self.persist_pub.publish(g)

    def _clear_persistent_robot_footprint(self, data: np.ndarray, stamp) -> None:
        if self.persist_clear_radius <= 0.0:
            return
        pose = self._persistent_pose(None)
        if pose is None:
            return

        tx, ty, _tz, _yaw, _orientation = pose
        col_c, row_c = self._world_to_pgrid(tx, ty)
        radius_cells = max(
            1, int(np.ceil(self.persist_clear_radius / self.persist_res)))
        row_lo = max(0, row_c - radius_cells)
        row_hi = min(self._pN, row_c + radius_cells + 1)
        col_lo = max(0, col_c - radius_cells)
        col_hi = min(self._pN, col_c + radius_cells + 1)
        if row_lo >= row_hi or col_lo >= col_hi:
            return

        rows, cols = np.ogrid[row_lo:row_hi, col_lo:col_hi]
        mask = (rows - row_c) ** 2 + (cols - col_c) ** 2 <= radius_cells ** 2
        data[row_lo:row_hi, col_lo:col_hi][mask] = 0

    # ═══════════════════════════════════════════════════════════════════
    # Camera info
    # ═══════════════════════════════════════════════════════════════════

    def _on_info(self, msg: CameraInfo, idx: int) -> None:
        self.K[idx] = np.array(msg.k).reshape(3, 3)
        self.get_logger().info(
            f'Camera[{idx}] intrinsics received.', once=True)

    # ═══════════════════════════════════════════════════════════════════
    # Main callback
    # ═══════════════════════════════════════════════════════════════════

    def _on_images(self, rgb_msg: Image, depth_msg: Image, cam_idx: int) -> None:
        self._got_frame = True
        process_time = self.get_clock().now()
        process_stamp = process_time.to_msg()

        frame_age = (
            process_time - Time.from_msg(rgb_msg.header.stamp)
        ).nanoseconds / 1e9
        if self.max_frame_age_sec > 0.0 and frame_age > self.max_frame_age_sec:
            self.get_logger().warn(
                f'Dropping stale synced frame from cam[{cam_idx}] '
                f'(age={frame_age:.2f}s > {self.max_frame_age_sec:.2f}s). '
                'YOLO/input is behind camera rate.',
                throttle_duration_sec=2.0)
            return

        if not self._within_time_budget(rgb_msg.header.stamp, depth_msg.header.stamp):
            self.get_logger().warn(
                f'Dropping cam[{cam_idx}] RGB/depth pair: stamp delta exceeds '
                f'{self.max_time_offset_sec:.3f}s.',
                throttle_duration_sec=2.0)
            return

        if self._stamp_age_sec(rgb_msg.header.stamp) > self.max_time_offset_sec:
            self.get_logger().warn(
                f'Dropping cam[{cam_idx}] frame: camera stamp is older than '
                f'{self.max_time_offset_sec:.3f}s.',
                throttle_duration_sec=2.0)
            return

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

        # ── Run segmentation model ──
        try:
            da_mask, ll_mask = self.model.infer(bgr)
        except Exception as e:
            self.get_logger().error(
                f'YOLOPv2 inference error: {e}',
                throttle_duration_sec=2.0)
            return

        # Auto-detect when the lane-line head has flooded into free
        # space (covering more pixels than the drivable-area mask).
        # Real lane paint is thin; anything sprawling wider than the
        # drivable area is almost certainly free space mis-labelled as
        # a line, so drop it to avoid poisoning the lane costmap.
        da_area = int(np.count_nonzero(da_mask)) if da_mask is not None else 0
        ll_area = int(np.count_nonzero(ll_mask)) if ll_mask is not None else 0
        if ll_area > da_area and ll_area > 0:
            self.get_logger().warn(
                f'Lane-line mask ({ll_area}px) exceeds drivable-area '
                f'mask ({da_area}px); discarding as free-space bleed.',
                throttle_duration_sec=5.0)
            ll_mask = np.zeros_like(ll_mask)

        # ── Apply ROI + chassis mask to both seg outputs ──
        da_mask = self._apply_mask_roi(da_mask)
        ll_mask = self._apply_mask_roi(ll_mask)

        # ── Camera → base_link TF (once per frame) ──
        cam_frame = depth_msg.header.frame_id or rgb_msg.header.frame_id
        cam_tf = None
        if cam_frame and cam_frame != self.base_frame:
            cam_tf = lookup_tf(
                self.tf_buffer, self.base_frame, cam_frame, None)
            if cam_tf is None:
                self.get_logger().warn(
                    f'No TF from {cam_frame} to {self.base_frame}; '
                    f'falling back to pinhole projection for cam[{cam_idx}]',
                    throttle_duration_sec=2.0)

        # ── Project masks into base_link ──
        free_pts = self._project_mask_points(
            da_mask, depth, cam_idx, cam_tf,
            stride=self.da_subsample_px)
        lane_pts, lane_components = self._project_lane_mask(
            ll_mask, depth, cam_idx, cam_tf)

        lane_components = self._correlate_lane_components(
            lane_components, rgb_msg.header.stamp)
        lane_pts = [pt for comp in lane_components for pt in comp]

        # ── Publish overlay ──
        if self.publish_overlay and cam_idx in self.overlay_pubs:
            self._publish_overlay(cam_idx, bgr, da_mask, ll_mask, rgb_msg)

        # ── Cache per-camera state for multi-camera fusion ──
        self._cam_state[cam_idx] = {
            'stamp':    process_stamp,
            'free':     free_pts,
            'lane':     lane_pts,
        }

        fused_free, fused_lane = self._fuse_points(process_stamp)

        if fused_free or fused_lane:
            self.latest_grid = self._build_grid(
                fused_free, fused_lane, process_stamp)
            self.grid_pub.publish(self.latest_grid)
            self._update_persistent_map(fused_lane, None)
        elif self.keep_last_grid_on_miss:
            if self._within_time_budget(self.latest_grid.header.stamp, rgb_msg.header.stamp):
                self.grid_pub.publish(self.latest_grid)
            else:
                self.latest_grid = self._empty_grid(rgb_msg.header.stamp)
                self.grid_pub.publish(self.latest_grid)
        else:
            self.latest_grid = self._empty_grid(process_stamp)
            self.grid_pub.publish(self.latest_grid)

        # ── Publish per-component lane markers ──
        self._publish_lane_markers(lane_components, process_stamp)

        self.get_logger().info(
            f'cam[{cam_idx}] free={len(free_pts)} lane={len(lane_pts)} '
            f'components={len(lane_components)} '
            f'active_cams={len(self._active_cam_states(process_stamp))}',
            throttle_duration_sec=1.0)

    def _active_cam_states(self, stamp) -> List[dict]:
        now_t = Time.from_msg(stamp)
        active = []
        for state in self._cam_state.values():
            dt = abs((now_t - Time.from_msg(state['stamp'])).nanoseconds / 1e9)
            if dt <= self.fusion_timeout_sec:
                active.append(state)
        return active

    def _within_time_budget(self, a, b) -> bool:
        dt = abs((Time.from_msg(a) - Time.from_msg(b)).nanoseconds / 1e9)
        return dt <= self.max_time_offset_sec

    def _stamp_age_sec(self, stamp) -> float:
        stamp_t = Time.from_msg(stamp)
        if stamp_t.nanoseconds == 0:
            return 0.0
        return abs((self.get_clock().now() - stamp_t).nanoseconds / 1e9)

    def _fuse_points(self, stamp) -> Tuple[List, List]:
        free_pts: List[Tuple[float, float]] = []
        lane_pts: List[Tuple[float, float]] = []
        for state in self._active_cam_states(stamp):
            free_pts.extend(state['free'])
            lane_pts.extend(state['lane'])
        return free_pts, lane_pts

    # ═══════════════════════════════════════════════════════════════════
    # Lane-line correlation
    # ═══════════════════════════════════════════════════════════════════

    def _correlate_lane_components(
        self,
        components: Sequence[Sequence[Tuple[float, float]]],
        stamp,
    ) -> List[List[Tuple[float, float]]]:
        if not self.enable_line_correlation or not components:
            return [list(comp) for comp in components]

        tf = lookup_tf(self.tf_buffer, self.persist_frame, self.base_frame, stamp)
        if tf is None:
            self.get_logger().debug(
                'Line correlation skipped: no stamped TF to persistent frame.',
                throttle_duration_sec=2.0)
            return [list(comp) for comp in components]

        self._prune_line_tracks(stamp)

        accepted: List[List[Tuple[float, float]]] = []
        for comp in components:
            desc = self._fit_component_line(comp, tf)
            if desc is None:
                self._debug_line_corr('reject: too few or incoherent points')
                continue

            best_track, match = self._best_line_track_match(desc)
            persistent_score = self._persistent_hit_score(desc['world_points'])
            matched = (
                best_track is not None
                and match is not None
                and match['distance'] <= self.line_corr_match_dist_m
                and match['angle'] <= self.line_corr_match_angle_rad
                and match['overlap'] >= self.line_corr_min_overlap
            )

            if matched:
                world_points = self._nudge_world_points_to_track(
                    desc['world_points'], best_track)
                base_points = self._world_points_to_base(world_points, tf)
                self._update_line_track(best_track['id'], world_points, stamp)
                accepted.append(base_points)
                self._debug_line_corr(
                    f"match track={best_track['id']} dist={match['distance']:.2f} "
                    f"angle={match['angle']:.2f} overlap={match['overlap']:.2f} "
                    f"map={persistent_score:.2f}")
                continue

            if desc['length'] < self.line_corr_new_min_length_m:
                self._debug_line_corr(f"reject: short new line len={desc['length']:.2f}")
                continue

            if self._is_between_parallel_tracks(desc) and persistent_score < 0.25:
                self._debug_line_corr('reject: between stable parallel tracks')
                continue

            track_id = self._create_line_track(desc['world_points'], stamp)
            accepted.append(list(comp))
            self._debug_line_corr(
                f"new track={track_id} len={desc['length']:.2f} map={persistent_score:.2f}")

        return accepted

    def _fit_component_line(self, comp, tf):
        if len(comp) < self.line_corr_min_points:
            return None

        base_points = np.asarray(comp, dtype=np.float32)
        if base_points.ndim != 2 or base_points.shape[1] != 2:
            return None

        world_points = self._base_points_to_world(base_points, tf)
        return self._line_descriptor(world_points)

    def _base_points_to_world(self, base_points: np.ndarray, tf) -> np.ndarray:
        q = tf.transform.rotation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y

        fwd = base_points[:, 0]
        lat = base_points[:, 1]
        world = np.empty_like(base_points, dtype=np.float32)
        world[:, 0] = tx + cos_y * fwd - sin_y * lat
        world[:, 1] = ty + sin_y * fwd + cos_y * lat
        return world

    def _world_points_to_base(self, world_points: np.ndarray, tf) -> List[Tuple[float, float]]:
        q = tf.transform.rotation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y

        dx = world_points[:, 0] - tx
        dy = world_points[:, 1] - ty
        fwd = cos_y * dx + sin_y * dy
        lat = -sin_y * dx + cos_y * dy
        return [(float(x), float(y)) for x, y in zip(fwd.tolist(), lat.tolist())]

    def _line_descriptor(self, world_points: np.ndarray):
        if world_points.shape[0] < self.line_corr_min_points:
            return None

        centroid = np.mean(world_points, axis=0)
        centered = world_points - centroid
        try:
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        direction = vh[0].astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        direction /= norm

        projections = centered @ direction
        p_min = float(np.min(projections))
        p_max = float(np.max(projections))
        length = p_max - p_min
        if length < 1e-3:
            return None

        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        lateral_scatter = float(np.std(centered @ normal))
        longitudinal_scatter = float(np.std(projections))
        if lateral_scatter > max(0.15, self.line_corr_match_dist_m * 1.5):
            return None

        return {
            'world_points': world_points,
            'centroid': centroid.astype(np.float32),
            'direction': direction,
            'normal': normal,
            'p_min': p_min,
            'p_max': p_max,
            'length': float(length),
            'lateral_scatter': lateral_scatter,
            'longitudinal_scatter': longitudinal_scatter,
        }

    def _best_line_track_match(self, desc):
        best_track = None
        best_match = None
        best_score = None
        for track in self._line_tracks.values():
            match = self._line_match(desc, track)
            score = match['distance'] + match['angle'] - match['overlap']
            if best_score is None or score < best_score:
                best_score = score
                best_track = track
                best_match = match
        return best_track, best_match

    def _line_match(self, desc, track):
        direction = desc['direction']
        track_direction = track['direction']
        dot = float(np.clip(abs(np.dot(direction, track_direction)), 0.0, 1.0))
        angle = float(np.arccos(dot))

        delta = desc['centroid'] - track['centroid']
        distance = float(abs(np.dot(delta, track['normal'])))

        projections = (desc['world_points'] - track['centroid']) @ track_direction
        new_min = float(np.min(projections))
        new_max = float(np.max(projections))
        overlap = max(0.0, min(new_max, track['p_max']) - max(new_min, track['p_min']))
        denom = max(0.05, min(new_max - new_min, track['p_max'] - track['p_min']))
        overlap_ratio = float(np.clip(overlap / denom, 0.0, 1.0))
        return {'distance': distance, 'angle': angle, 'overlap': overlap_ratio}

    def _nudge_world_points_to_track(self, world_points: np.ndarray, track) -> np.ndarray:
        alpha = float(np.clip(self.line_corr_nudge_alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return world_points

        projections = (world_points - track['centroid']) @ track['direction']
        closest = track['centroid'] + projections[:, None] * track['direction']
        return ((1.0 - alpha) * world_points + alpha * closest).astype(np.float32)

    def _persistent_hit_score(self, world_points: np.ndarray) -> float:
        if world_points.size == 0:
            return 0.0

        radius_cells = max(1, int(np.ceil(self.line_corr_match_dist_m / self.persist_res)))
        scores = []
        for wx, wy in world_points[::max(1, len(world_points) // 20)]:
            col, row = self._world_to_pgrid(float(wx), float(wy))
            row_lo = max(0, row - radius_cells)
            row_hi = min(self._pN, row + radius_cells + 1)
            col_lo = max(0, col - radius_cells)
            col_hi = min(self._pN, col + radius_cells + 1)
            if row_lo >= row_hi or col_lo >= col_hi:
                continue
            scores.append(float(np.max(self._phits[row_lo:row_hi, col_lo:col_hi])))
        if not scores:
            return 0.0
        return float(np.clip(np.mean(scores) / max(self.persist_threshold, 1e-6), 0.0, 1.0))

    def _is_between_parallel_tracks(self, desc) -> bool:
        if not self.line_corr_reject_between or len(self._line_tracks) < 2:
            return False

        parallel = []
        for track in self._line_tracks.values():
            match = self._line_match(desc, track)
            if match['angle'] <= self.line_corr_match_angle_rad:
                signed_distance = float(np.dot(track['centroid'] - desc['centroid'], desc['normal']))
                parallel.append((signed_distance, match['distance']))

        if len(parallel) < 2:
            return False

        left = [p for p in parallel if p[0] > self.line_corr_match_dist_m]
        right = [p for p in parallel if p[0] < -self.line_corr_match_dist_m]
        if not left or not right:
            return False

        nearest_left = min(abs(p[0]) for p in left)
        nearest_right = min(abs(p[0]) for p in right)
        return nearest_left < 2.0 and nearest_right < 2.0

    def _create_line_track(self, world_points: np.ndarray, stamp) -> int:
        desc = self._line_descriptor(world_points)
        if desc is None:
            return -1
        track_id = self._next_line_track_id
        self._next_line_track_id += 1
        self._line_tracks[track_id] = {
            'id': track_id,
            'stamp': stamp,
            'confidence': 1.0,
            **{key: value for key, value in desc.items() if key != 'world_points'},
            'world_points': world_points,
        }
        return track_id

    def _update_line_track(self, track_id: int, world_points: np.ndarray, stamp) -> None:
        if track_id not in self._line_tracks:
            self._create_line_track(world_points, stamp)
            return
        desc = self._line_descriptor(world_points)
        if desc is None:
            return
        confidence = min(10.0, self._line_tracks[track_id].get('confidence', 1.0) + 1.0)
        self._line_tracks[track_id].update({
            'stamp': stamp,
            'confidence': confidence,
            **{key: value for key, value in desc.items() if key != 'world_points'},
            'world_points': world_points,
        })

    def _prune_line_tracks(self, stamp) -> None:
        now_t = Time.from_msg(stamp)
        stale = []
        for track_id, track in self._line_tracks.items():
            age = abs((now_t - Time.from_msg(track['stamp'])).nanoseconds / 1e9)
            if age > self.line_corr_max_age_sec:
                stale.append(track_id)
        for track_id in stale:
            del self._line_tracks[track_id]

    def _debug_line_corr(self, msg: str) -> None:
        if self.line_corr_debug:
            self.get_logger().info(f'line_correlation: {msg}', throttle_duration_sec=1.0)

    # ═══════════════════════════════════════════════════════════════════
    # Mask → base_link projection
    # ═══════════════════════════════════════════════════════════════════

    def _apply_mask_roi(self, mask: np.ndarray) -> np.ndarray:
        """Zero-out chassis + out-of-ROI regions of a binary mask."""
        if mask.size == 0:
            return mask
        out = mask.copy()
        h, w = out.shape[:2]

        # Chassis (bottom strip)
        if self.chassis_mask_frac > 0.0:
            cut = int(h * (1.0 - self.chassis_mask_frac))
            out[cut:, :] = 0

        # Trapezoidal ROI — optional.  Off by default: the offline
        # overlay test showed the full-frame mask is tighter to the
        # actual road than the trapezoid, which was clipping useful
        # drivable area at the sides.
        if not self.roi_enabled:
            return out

        roi = np.zeros((h, w), dtype=np.uint8)
        bot = self.roi_bottom_frac
        top = self.roi_top_frac
        pts = np.array([[
            [w * 0.05, h * bot],
            [w * 0.30, h * top],
            [w * 0.70, h * top],
            [w * 0.95, h * bot],
        ]], dtype=np.int32)
        cv2.fillPoly(roi, pts, 255)
        out = cv2.bitwise_and(out, out, mask=roi)
        return out

    def _intrinsics(self, cam_idx: int, depth_shape: Tuple[int, int]):
        h_d, w_d = depth_shape[:2]
        K = self.K.get(cam_idx)
        fx = K[0, 0] if K is not None else 500.0
        fy = K[1, 1] if K is not None else fx
        cx = K[0, 2] if K is not None else w_d / 2.0
        cy = K[1, 2] if K is not None else h_d / 2.0
        return fx, fy, cx, cy

    def _cam_tf_components(self, cam_tf):
        if cam_tf is None:
            return None, None
        q = cam_tf.transform.rotation
        rot = np.array([
            [1.0 - 2.0 * (q.y * q.y + q.z * q.z), 2.0 * (q.x * q.y - q.z * q.w), 2.0 * (q.x * q.z + q.y * q.w)],
            [2.0 * (q.x * q.y + q.z * q.w), 1.0 - 2.0 * (q.x * q.x + q.z * q.z), 2.0 * (q.y * q.z - q.x * q.w)],
            [2.0 * (q.x * q.z - q.y * q.w), 2.0 * (q.y * q.z + q.x * q.w), 1.0 - 2.0 * (q.x * q.x + q.y * q.y)],
        ], dtype=np.float32)
        t = cam_tf.transform.translation
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        return rot, trans

    def _project_mask_points(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        cam_idx: int,
        cam_tf,
        stride: int,
    ) -> List[Tuple[float, float]]:
        """Return a list of ``(fwd, lat)`` in ``base_link`` for mask pixels."""
        if mask is None or mask.size == 0:
            return []

        # Align mask size to depth — seg model output is already resized to
        # the RGB frame; depth may be a different resolution.  If so, scale
        # pixel coords accordingly when sampling depth.
        h_m, w_m = mask.shape[:2]
        h_d, w_d = depth.shape[:2]
        sx = w_d / float(w_m) if w_m > 0 else 1.0
        sy = h_d / float(h_m) if h_m > 0 else 1.0

        fx, fy, cx, cy = self._intrinsics(cam_idx, depth.shape)
        rot, trans = self._cam_tf_components(cam_tf)

        sub = mask[::stride, ::stride]
        ys, xs = np.nonzero(sub)
        if ys.size == 0:
            return []

        # Random subsample to cap cost.
        if ys.size > self.max_points_per_frame:
            idx = np.random.default_rng().choice(
                ys.size, size=self.max_points_per_frame, replace=False)
            ys = ys[idx]
            xs = xs[idx]

        pts: List[Tuple[float, float]] = []
        max_d = self.max_detection_depth_m
        min_d = self.min_detection_depth_m
        for v_sub, u_sub in zip(ys.tolist(), xs.tolist()):
            v = v_sub * stride
            u = u_sub * stride
            # Map mask (u, v) → depth (u, v)
            vd = int(v * sy)
            ud = int(u * sx)
            if not (0 <= vd < h_d and 0 <= ud < w_d):
                continue
            d = sample_valid_depth(
                depth, ud, vd,
                radius=self.depth_search_radius_px,
                min_d=min_d, max_d=max_d)
            if d is None:
                continue
            fwd, lat, _ = pixel_to_base(
                ud, vd, d, fx, fy, cx, cy, rot, trans)
            if fwd <= 0.0:
                continue
            pts.append((fwd, lat))
        return pts

    def _project_lane_mask(
        self,
        ll_mask: np.ndarray,
        depth: np.ndarray,
        cam_idx: int,
        cam_tf,
    ) -> Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]:
        """Project the lane-line mask and split by connected component.

        Returns
        -------
        all_pts
            Flat list of every projected lane-line point in ``base_link``.
        components
            List of per-component point lists, sorted by average lateral
            offset (most-positive = left-most) so the marker colouring is
            stable frame-to-frame.
        """
        if ll_mask is None or ll_mask.size == 0:
            return [], []

        # Connected components on the full-resolution mask so small blobs
        # can be filtered by real pixel area.
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            ll_mask.astype(np.uint8), connectivity=8)

        fx, fy, cx, cy = self._intrinsics(cam_idx, depth.shape)
        rot, trans = self._cam_tf_components(cam_tf)
        h_d, w_d = depth.shape[:2]
        h_m, w_m = ll_mask.shape[:2]
        sx = w_d / float(w_m) if w_m > 0 else 1.0
        sy = h_d / float(h_m) if h_m > 0 else 1.0
        stride = self.ll_subsample_px
        min_d = self.min_detection_depth_m
        max_d = self.max_detection_depth_m

        components: List[List[Tuple[float, float]]] = []
        all_pts: List[Tuple[float, float]] = []

        # Skip label 0 (background).
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_lane_component_px:
                continue

            ys, xs = np.where(labels == label)
            if ys.size == 0:
                continue

            # Stride-subsample within the component.
            if stride > 1:
                keep = np.arange(0, ys.size, stride)
                ys = ys[keep]
                xs = xs[keep]

            if ys.size > self.max_points_per_frame:
                rng = np.random.default_rng()
                idx = rng.choice(
                    ys.size, size=self.max_points_per_frame, replace=False)
                ys = ys[idx]
                xs = xs[idx]

            comp_pts: List[Tuple[float, float]] = []
            for v, u in zip(ys.tolist(), xs.tolist()):
                vd = int(v * sy)
                ud = int(u * sx)
                if not (0 <= vd < h_d and 0 <= ud < w_d):
                    continue
                d = sample_valid_depth(
                    depth, ud, vd,
                    radius=self.depth_search_radius_px,
                    min_d=min_d, max_d=max_d)
                if d is None:
                    continue
                fwd, lat, _ = pixel_to_base(
                    ud, vd, d, fx, fy, cx, cy, rot, trans)
                if fwd <= 0.0:
                    continue
                comp_pts.append((fwd, lat))

            if comp_pts:
                components.append(comp_pts)
                all_pts.extend(comp_pts)

        # Sort left-to-right by mean lateral offset (most positive first).
        components.sort(
            key=lambda pts: -float(np.mean([p[1] for p in pts])))
        return all_pts, components

    # ═══════════════════════════════════════════════════════════════════
    # Local rolling costmap
    # ═══════════════════════════════════════════════════════════════════

    def _build_grid(
        self,
        free_pts: Iterable[Tuple[float, float]],
        lane_pts: Iterable[Tuple[float, float]],
        stamp,
    ) -> OccupancyGrid:
        g = self._empty_grid(stamp)
        w_g, h_g, res = g.info.width, g.info.height, self.grid_res
        data = np.full((h_g, w_g), -1, dtype=np.int8)

        def to_cell(fwd: float, lat: float) -> Optional[Tuple[int, int]]:
            row = int(fwd / res)
            col = int((lat + self.grid_width_m / 2.0) / res)
            if 0 <= row < h_g and 0 <= col < w_g:
                return row, col
            return None

        # Free first so lethal overrides it.
        for fwd, lat in free_pts:
            cell = to_cell(fwd, lat)
            if cell is not None:
                data[cell] = 0

        for fwd, lat in lane_pts:
            cell = to_cell(fwd, lat)
            if cell is not None:
                data[cell] = 100

        g.data = data.flatten().tolist()
        return g

    def _empty_grid(self, stamp=None) -> OccupancyGrid:
        g = OccupancyGrid()
        g.header.stamp = (
            self.get_clock().now().to_msg() if stamp is None else stamp)
        g.header.frame_id = self.occupancy_grid_frame
        w_g = int(self.grid_width_m / self.grid_res)
        h_g = int(self.grid_height_m / self.grid_res)
        g.info.resolution = self.grid_res
        g.info.width = w_g
        g.info.height = h_g

        if self.occupancy_grid_frame == self.base_frame:
            g.info.origin.position.x = 0.0
            g.info.origin.position.y = -self.grid_width_m / 2.0
            g.info.origin.orientation.w = 1.0
        else:
            tf = lookup_tf(
                self.tf_buffer, self.occupancy_grid_frame,
                self.base_frame, stamp)
            if tf is None:
                g.info.origin.position.x = 0.0
                g.info.origin.position.y = -self.grid_width_m / 2.0
                g.info.origin.orientation.w = 1.0
            else:
                q = tf.transform.rotation
                yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
                tx = tf.transform.translation.x
                ty = tf.transform.translation.y
                g.info.origin.position.x = tx + np.sin(yaw) * (self.grid_width_m / 2.0)
                g.info.origin.position.y = ty - np.cos(yaw) * (self.grid_width_m / 2.0)
                g.info.origin.position.z = tf.transform.translation.z
                g.info.origin.orientation = q

        g.data = [-1] * (w_g * h_g)
        return g

    # ═══════════════════════════════════════════════════════════════════
    # Debug
    # ═══════════════════════════════════════════════════════════════════

    def _publish_overlay(
        self, idx: int, bgr: np.ndarray,
        da_mask: np.ndarray, ll_mask: np.ndarray, rgb_msg: Image,
    ) -> None:
        ov = bgr.copy()
        colour = np.zeros_like(ov)
        colour[da_mask > 0] = (0, 255, 0)     # green  = drivable area
        colour[ll_mask > 0] = (0, 0, 255)     # red    = lane lines
        mask_any = (da_mask > 0) | (ll_mask > 0)
        ov[mask_any] = cv2.addWeighted(
            ov, 0.5, colour, 0.5, 0.0)[mask_any]

        try:
            msg = self.bridge.cv2_to_imgmsg(ov, 'bgr8')
            msg.header = rgb_msg.header
            self.overlay_pubs[idx].publish(msg)
        except Exception as e:
            self.get_logger().warn(
                f'Overlay error: {e}', throttle_duration_sec=2.0)

    def _publish_lane_markers(
        self,
        components: Sequence[Sequence[Tuple[float, float]]],
        stamp,
    ) -> None:
        if self.marker_pub.get_subscription_count() == 0:
            return
        arr = MarkerArray()
        # Clear previous frame so removed lanes disappear.
        clear = Marker()
        clear.header.frame_id = self.base_frame
        clear.header.stamp = stamp
        clear.ns = 'lane_segmentation'
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for i, pts in enumerate(components):
            m = Marker()
            m.header.frame_id = self.base_frame
            m.header.stamp = stamp
            m.ns = 'lane_segmentation'
            m.id = i
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.08
            m.scale.y = 0.08
            r, g, b = _LANE_COLORS[i % len(_LANE_COLORS)]
            m.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0)
            m.lifetime = Duration(seconds=1).to_msg()
            for fwd, lat in pts:
                p = Point()
                p.x = float(fwd)
                p.y = float(lat)
                p.z = 0.0
                m.points.append(p)
            arr.markers.append(m)

        self.marker_pub.publish(arr)

    # ═══════════════════════════════════════════════════════════════════
    # Misc
    # ═══════════════════════════════════════════════════════════════════

    def _republish_grid(self) -> None:
        if self._stamp_age_sec(self.latest_grid.header.stamp) > self.max_time_offset_sec:
            return
        self.grid_pub.publish(self.latest_grid)

    def _watchdog(self) -> None:
        if not self._got_frame:
            self.get_logger().warn(
                'No synced RGB+Depth frames. Check topics and slop.',
                throttle_duration_sec=5.0)
        self._got_frame = False


def main(args=None):
    rclpy.init(args=args)
    node = LaneSegmentationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
