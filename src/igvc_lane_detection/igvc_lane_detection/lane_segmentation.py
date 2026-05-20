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

import math
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
        self.camera_info_size: dict = {}

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
        self.sync_queue_size        = max(1, int(p('sync_queue_size', 1)))
        self.sync_slop_sec          = float(p('sync_slop_sec', 0.1))
        self.max_frame_age_sec      = float(p('max_frame_age_sec', 1.2))

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
        self.persist_free_hit_w   = p('persistent_free_hit_weight',  1.0)
        self.persist_threshold    = p('persistent_threshold',       15.0)
        self.persist_free_threshold = p('persistent_free_threshold',  3.0)
        self.persist_max          = p('persistent_max_value',      200.0)
        self.persist_pub_hz       = p('persistent_publish_hz',       2.0)
        self.persist_clear_radius = p('persistent_clear_radius_m',   0.8)
        self.persist_publish_clear_robot = bool(
            p('persistent_publish_clear_robot', True))
        self.persist_skip_no_subscribers = bool(
            p('persistent_skip_publish_without_subscribers', True))
        self.persist_pose_source  = p('persistent_pose_source',      'tf')
        self.odom_topic           = p('odom_topic',                  '/front_zed_camera_x/zed_node/odom')
        self.local_from_persistent = bool(p('local_costmap_from_persistent', True))
        self.local_publish_hz      = float(p('local_costmap_publish_hz', 10.0))
        self.local_back_nogo_buffer_m = max(
            0.0, float(p('local_back_nogo_buffer_m', 0.30)))

        # ── Pose deduplication — skip persistent write if not moved ────
        self.min_pose_change_m   = float(p('min_pose_change_m',   0.05))
        self.min_pose_change_rad = float(p('min_pose_change_rad', 0.02))

        # ── Yaw-rate gate ────────────────────────────────────────────
        # During fast in-place turns the depth-projected lane points
        # smear in the world frame (camera/odom sync error grows with
        # angular velocity, and ZED depth at oblique angles is noisy).
        # Skip persistent-map writes when |yaw_rate| exceeds this
        # threshold — decay still applies so stale cells fade out.
        self.max_yaw_rate_persist = float(
            p('max_yaw_rate_for_persist_update_rad_s', 0.6))

        # ── Segmentation-specific parameters ─────────────────────────
        self.model_weights          = p('model_weights',            '')
        self.model_device           = p('model_device',             'cuda:0')
        self.model_half             = p('model_half',               True)
        self.model_img_size         = p('model_img_size',           640)
        self.da_subsample_px        = max(1, int(p('da_subsample_px', 6)))
        self.ll_subsample_px        = max(1, int(p('ll_subsample_px', 2)))
        self.min_lane_component_px  = int(p('min_lane_component_px', 150))
        self.min_da_component_px    = int(p('min_da_component_px', 0))
        self.max_points_per_frame   = int(p('max_points_per_frame', 4000))

        # ── Lane raster thickening / corridor inflation ─────────────
        # ``ll_dilation_px`` widens the raw lane-line mask *before* it is
        # projected into the persistent map.  Each detected lane pixel
        # therefore covers a slightly larger neighbourhood in world
        # coordinates, which fills small mask gaps (dashed markings,
        # far-field thin strokes) and prevents sparse evidence from
        # failing the persistent-threshold accumulator.
        # ``local_lane_inflation_m`` inflates the *output* /lane_costmap
        # lane (100) cells by this radius before publishing.  Acts as a
        # soft corridor margin (~1 m wide drivable swath given a 0.5 m
        # default) so navigator paths cannot squeeze through gaps in
        # the segmentation mask.
        self.ll_dilation_px         = max(0, int(p('ll_dilation_px', 0)))
        if self.ll_dilation_px > 0:
            _kll = self.ll_dilation_px | 1  # ensure odd
            self._ll_dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (_kll, _kll))
        else:
            self._ll_dilate_kernel = None
        self.local_lane_inflation_m = float(p('local_lane_inflation_m', 0.0))
        self.publish_mask_overlay   = p('publish_mask_overlay',     True)
        self.lane_marker_topic      = p('lane_marker_topic',        '/lane_segmentation/lanes')

        # ── Preprocessor / mask cleanup (tunable for sim domain gap) ─
        self.model_preprocess       = bool(p('model_preprocess', True))
        self.model_clahe_clip       = float(p('model_clahe_clip', 2.0))
        self.model_clahe_tile       = [int(x) for x in p('model_clahe_tile', [8, 8])]
        self.model_blur_ksize       = [int(x) for x in p('model_blur_ksize', [5, 5])]
        self.model_blur_sigma       = float(p('model_blur_sigma', 0.0))
        self.da_morph_kernel_px     = int(p('da_morph_kernel_px', 0))
        # Pre-allocate morph kernel once — avoids heap allocation inside the 30 Hz inference callback
        _k = self.da_morph_kernel_px | 1  # ensure odd
        self._da_morph_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_k, _k))
            if self.da_morph_kernel_px > 1 else None
        )

        # ── Depth-based obstacle masking ─────────────────────────────
        self.obstacle_mask_enabled      = bool(p('obstacle_mask_enabled', False))
        self.obstacle_z_min_m           = float(p('obstacle_z_min_m', 0.15))
        self.obstacle_z_max_m           = float(p('obstacle_z_max_m', 2.5))
        self.obstacle_depth_min_m       = float(p('obstacle_depth_min_m', 0.3))
        self.obstacle_depth_max_m       = float(p('obstacle_depth_max_m', 8.0))
        self.obstacle_dilation_px       = int(p('obstacle_dilation_px', 25))
        self.camera_height_fallback_m   = float(p('camera_height_fallback_m', 0.45))

        if not self.model_weights:
            raise RuntimeError(
                "Parameter 'model_weights' must be set to the path of "
                "yolopv2.pt. On the Jetson target this is typically "
                "$(repo)/models/yolopv2.pt — fetch with "
                "src/igvc_lane_detection/scripts/fetch_yolopv2_weights.sh.")

        # ── Initialise segmentation model ────────────────────────────
        # Backend dispatch: ``.engine`` → TensorRT runtime; anything else
        # → TorchScript via the YolopV2 wrapper.
        weights_lower = str(self.model_weights).lower()
        use_trt = weights_lower.endswith('.engine') or weights_lower.endswith('.trt')
        self.get_logger().info(
            f"Loading YOLOPv2 from '{self.model_weights}' "
            f"(backend={'tensorrt' if use_trt else 'torchscript'}, "
            f"half={self.model_half})…")
        if use_trt:
            from .yolopv2_trt import YolopV2TRT
            self.model = YolopV2TRT(
                engine_path=self.model_weights,
                img_size=self.model_img_size,
                preprocess=self.model_preprocess,
                clahe_clip=self.model_clahe_clip,
                clahe_tile=tuple(self.model_clahe_tile),
                blur_ksize=tuple(self.model_blur_ksize),
                blur_sigma=self.model_blur_sigma,
            )
        else:
            self.model = YolopV2(
                weights_path=self.model_weights,
                device=self.model_device,
                half=self.model_half,
                img_size=self.model_img_size,
                preprocess=self.model_preprocess,
                clahe_clip=self.model_clahe_clip,
                clahe_tile=tuple(self.model_clahe_tile),
                blur_ksize=tuple(self.model_blur_ksize),
                blur_sigma=self.model_blur_sigma,
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
        self._last_persist_pose: Optional[Tuple] = None

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if self.persist_pose_source == 'odom' or self.local_from_persistent:
            odom_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1)
            self.create_subscription(
                Odometry, self.odom_topic, self._on_odom, odom_qos)
            self.get_logger().info(
                f'Lane map/local crop pose source: odom topic {self.odom_topic}')

        self.create_timer(
            1.0 / max(self.local_publish_hz, 0.1),
            self._republish_grid)
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
        self._pfree = np.zeros((n, n), dtype=np.float32)
        self._persistent_dirty = True
        self._persistent_data_cache: Optional[np.ndarray] = None
        self._persistent_msg: Optional[OccupancyGrid] = None
        self._persistent_msg_dirty = True
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
        self,
        free_pts: Optional[Sequence[Tuple[float, float]]],
        lane_pts: Optional[Sequence[Tuple[float, float]]],
        stamp,
    ) -> None:
        self._last_persistent_stamp = stamp
        if not free_pts and not lane_pts:
            # Still apply the global decay so stale evidence fades even on
            # empty frames.
            self._phits *= self.persist_decay
            self._pfree *= self.persist_decay
            self._persistent_dirty = True
            self._persistent_msg_dirty = True
            return

        # Yaw-rate gate: during fast in-place turns, depth-projected
        # lane points smear in odom (camera/odom timing skew + oblique
        # depth noise).  Apply decay only and skip the stamping pass so
        # the persistent map fades stale evidence instead of stacking
        # new bad evidence on top of it.
        if (
            self.max_yaw_rate_persist > 0.0
            and self._latest_odom is not None
        ):
            yaw_rate = abs(self._latest_odom.twist.twist.angular.z)
            if yaw_rate > self.max_yaw_rate_persist:
                self._phits *= self.persist_decay
                self._pfree *= self.persist_decay
                self._persistent_dirty = True
                self._persistent_msg_dirty = True
                self.get_logger().debug(
                    f'Skipping persistent-map write: |yaw_rate|='
                    f'{yaw_rate:.2f} rad/s > '
                    f'{self.max_yaw_rate_persist:.2f}')
                return

        pose = self._persistent_pose(stamp)
        if pose is None:
            return

        tx, ty, _tz, yaw, _orientation = pose

        # Dedup: skip write if robot hasn't moved enough since last update.
        # Prevents lane cells stacking on top of each other when YOLO frames
        # share the same effective pose (e.g. slow inference, stationary robot).
        if self._last_persist_pose is not None:
            lx, ly, lyaw = self._last_persist_pose
            dist = np.hypot(tx - lx, ty - ly)
            dang = abs(((yaw - lyaw + np.pi) % (2.0 * np.pi)) - np.pi)
            if dist < self.min_pose_change_m and dang < self.min_pose_change_rad:
                return
        self._last_persist_pose = (tx, ty, yaw)

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        self._phits *= self.persist_decay
        self._pfree *= self.persist_decay

        self._stamp_persistent_points(
            free_pts, tx, ty, cos_y, sin_y,
            self._pfree, self.persist_free_hit_w)
        self._stamp_persistent_points(
            lane_pts, tx, ty, cos_y, sin_y,
            self._phits, self.persist_hit_w)
        self._persistent_dirty = True
        self._persistent_msg_dirty = True

    def _stamp_persistent_points(
        self,
        points: Optional[Sequence[Tuple[float, float]]],
        tx: float,
        ty: float,
        cos_y: float,
        sin_y: float,
        grid: np.ndarray,
        weight: float,
    ) -> None:
        if points is None or len(points) == 0:
            return

        pts = np.asarray(points, dtype=np.float32)
        fwd = pts[:, 0]
        lat = pts[:, 1]
        cols = (
            (tx + cos_y * fwd - sin_y * lat - self._p_ox)
            / self.persist_res
        ).astype(np.int32)
        rows = (
            (ty + sin_y * fwd + cos_y * lat - self._p_oy)
            / self.persist_res
        ).astype(np.int32)
        valid = (
            (cols >= 0) & (cols < self._pN) &
            (rows >= 0) & (rows < self._pN)
        )
        if not np.any(valid):
            return

        rows = rows[valid]
        cols = cols[valid]
        np.add.at(grid, (rows, cols), weight)
        grid[rows, cols] = np.minimum(grid[rows, cols], self.persist_max)

    def _persistent_grid_data(self, stamp=None, clear_robot: bool = True) -> np.ndarray:
        if self._persistent_data_cache is None or self._persistent_dirty:
            data = np.full((self._pN, self._pN), -1, dtype=np.int8)
            data[self._pfree >= self.persist_free_threshold] = 0
            data[self._phits >= self.persist_threshold] = 100
            self._persistent_data_cache = data
            self._persistent_dirty = False

        data = self._persistent_data_cache
        if clear_robot:
            data = data.copy()
            self._clear_persistent_robot_footprint(data, stamp)
        return data

    def _publish_persistent_map(self) -> None:
        if (
            self.persist_skip_no_subscribers
            and self.persist_pub.get_subscription_count() == 0
        ):
            return

        stamp = self._last_persistent_stamp
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        rebuild_msg = (
            self._persistent_msg is None
            or self._persistent_msg_dirty
            or self.persist_publish_clear_robot
        )
        if rebuild_msg:
            n = self._pN
            g = OccupancyGrid()
            g.header.frame_id           = self.persist_frame
            g.info.resolution           = self.persist_res
            g.info.width                = n
            g.info.height               = n
            g.info.origin.position.x    = self._p_ox
            g.info.origin.position.y    = self._p_oy
            g.info.origin.orientation.w = 1.0

            data = self._persistent_grid_data(
                stamp, clear_robot=self.persist_publish_clear_robot)
            g.data = data.ravel().tolist()
            self._persistent_msg = g
            self._persistent_msg_dirty = False
        else:
            g = self._persistent_msg

        g.header.stamp = stamp
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
        self.camera_info_size[idx] = (int(msg.width), int(msg.height))
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

        # ── Morphological cleanup of drivable-area mask (removes sim noise) ──
        if self._da_morph_kernel is not None:
            da_mask = cv2.morphologyEx(
                da_mask.astype(np.uint8), cv2.MORPH_OPEN, self._da_morph_kernel)

        # ── Minimum component size filter on drivable-area mask ──
        if self.min_da_component_px > 0 and np.any(da_mask):
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                da_mask.astype(np.uint8), connectivity=8)
            clean = np.zeros_like(da_mask)
            for lbl in range(1, n_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= self.min_da_component_px:
                    clean[labels == lbl] = 1
            da_mask = clean

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

        # ── Depth-based obstacle masking ──
        # Zeros out pixels where 3-D base_link height lands in the obstacle
        # band — suppresses YOLOPv2 hallucinations over barrel/cone geometry.
        if self.obstacle_mask_enabled and cam_idx in self.K:
            obs_mask = self._build_obstacle_mask(depth, cam_tf, cam_idx)
            if obs_mask is not None:
                # Resize obs_mask to match seg-head resolution if needed
                if obs_mask.shape != da_mask.shape:
                    obs_mask = cv2.resize(
                        obs_mask.astype(np.uint8),
                        (da_mask.shape[1], da_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
                da_mask[obs_mask] = 0
                ll_mask[obs_mask] = 0

        # ── Lane mask dilation (pre-projection) ──
        # Thicken the raw lane mask so each detected lane pixel projects
        # to a small neighbourhood instead of a single ray.  This fills
        # small gaps in the persistent map without changing the model.
        if self._ll_dilate_kernel is not None and ll_mask is not None and ll_mask.size:
            ll_mask = cv2.dilate(
                ll_mask.astype(np.uint8), self._ll_dilate_kernel, iterations=1)

        # ── Project masks into base_link ──
        free_pts = self._project_mask_points(
            da_mask, depth, cam_idx, cam_tf,
            stride=self.da_subsample_px)
        lane_pts, lane_components = self._project_lane_mask(
            ll_mask, depth, cam_idx, cam_tf)

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
            self._update_persistent_map(
                fused_free, fused_lane, rgb_msg.header.stamp)
            if self.local_from_persistent:
                self._publish_local_costmap_from_persistent()
            else:
                self.grid_pub.publish(self.latest_grid)
        elif self.keep_last_grid_on_miss:
            self._republish_grid()
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


    def _fuse_points(self, stamp) -> Tuple[List, List]:
        free_pts: List[Tuple[float, float]] = []
        lane_pts: List[Tuple[float, float]] = []
        for state in self._active_cam_states(stamp):
            free_pts.extend(state['free'])
            lane_pts.extend(state['lane'])
        return free_pts, lane_pts

    # ═══════════════════════════════════════════════════════════════════
    # Mask → base_link projection
    # ═══════════════════════════════════════════════════════════════════

    def _build_obstacle_mask(self, depth: np.ndarray, cam_tf, cam_idx: int) -> Optional[np.ndarray]:
        """Return a bool mask (same H×W as depth) where pixels are occupied by obstacles.

        Uses the ZED depth to project every pixel into base_link 3-D space.  Any
        pixel whose base_link Z height lands between ``obstacle_z_min_m`` and
        ``obstacle_z_max_m`` is flagged as an obstacle and masked out of the
        segmentation heads, preventing YOLOPv2 from hallucinating lane lines over
        barrel / cone geometry.

        The computation is fully vectorised — no Python pixel loops.
        """
        K = self.K.get(cam_idx)
        if K is None:
            return None

        dh, dw = depth.shape[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Scale intrinsics if depth image resolution differs from CameraInfo.
        # Use the actual calibrated image size instead of approximating from
        # principal point; this keeps obstacle-height projection stable.
        info_size = self.camera_info_size.get(cam_idx)
        if info_size is not None and info_size[0] > 0 and info_size[1] > 0:
            sx = dw / float(info_size[0])
            sy = dh / float(info_size[1])
            fx, fy = fx * sx, fy * sy
            cx, cy = cx * sx, cy * sy

        # Build pixel coordinate grids
        us = np.arange(dw, dtype=np.float32)
        vs = np.arange(dh, dtype=np.float32)
        ug, vg = np.meshgrid(us, vs)  # (dh, dw)

        d = depth.astype(np.float32)

        # Valid depth gate
        valid = np.isfinite(d) & (d > self.obstacle_depth_min_m) & (d < self.obstacle_depth_max_m)

        # Camera-frame 3-D coords
        xc = (ug - cx) * d / fx
        yc = (vg - cy) * d / fy
        zc = d  # z forward in camera optical frame

        if cam_tf is not None:
            t = cam_tf.transform.translation
            r = cam_tf.transform.rotation
            R = np.array([
                [1 - 2*(r.y*r.y + r.z*r.z),   2*(r.x*r.y - r.z*r.w),   2*(r.x*r.z + r.y*r.w)],
                [2*(r.x*r.y + r.z*r.w),   1 - 2*(r.x*r.x + r.z*r.z),   2*(r.y*r.z - r.x*r.w)],
                [2*(r.x*r.z - r.y*r.w),   2*(r.y*r.z + r.x*r.w),   1 - 2*(r.x*r.x + r.y*r.y)],
            ], dtype=np.float64)
            # Stack into (3, N), transform, then reshape back
            pts = np.stack([xc.ravel(), yc.ravel(), zc.ravel()], axis=0)  # (3, N)
            pts_base = R @ pts  # (3, N)
            bz = pts_base[2].reshape(dh, dw).astype(np.float32) + float(t.z)
        else:
            # Approximate: camera points straight forward, height from param
            # bz ≈ camera_height - Y_camera_frame (Y down in optical frame)
            bz = (self.camera_height_fallback_m - yc).astype(np.float32)

        # Obstacle: height in [z_min, z_max] with valid depth
        obs = valid & (bz > self.obstacle_z_min_m) & (bz < self.obstacle_z_max_m)

        # Dilate to cover silhouette / shadow halo edges
        if self.obstacle_dilation_px > 1:
            k = self.obstacle_dilation_px | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            obs = cv2.dilate(obs.astype(np.uint8), kernel).astype(bool)

        return obs

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
        # ROS OccupancyGrid convention: data[row=y_idx, col=x_idx].
        # info.width  = forward cells (+x), info.height = lateral cells (+y).
        nx = g.info.width
        ny = g.info.height
        res = self.grid_res
        data = np.full((ny, nx), -1, dtype=np.int8)

        def to_cell(fwd: float, lat: float) -> Optional[Tuple[int, int]]:
            col = int(fwd / res)
            row = int((lat + self.grid_width_m / 2.0) / res)
            if 0 <= row < ny and 0 <= col < nx:
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

        # ── Corridor fill from lane boundaries ──────────────────────────────
        # Use reliably-detected lane lines to infer the drivable corridor.
        # For each forward column that has lane cells, find the innermost
        # lane-boundary row on each side of the robot centreline and fill
        # unknown (-1) cells between them with free (0).  Compensates for
        # sparse depth-projection of the drivable-area YOLO head.
        center_row = ny // 2  # row index for lateral=0 (robot centreline)
        half_fill  = max(4, int(round(1.2 / res)))  # 1.2 m half-width in cells
        lane_cols  = np.where(np.any(data == 100, axis=0))[0]
        for col in lane_cols:
            lane_in_col = np.where(data[:, col] == 100)[0]
            # +y is left of robot → higher row index = left side.
            right_side = lane_in_col[lane_in_col < center_row]
            left_side  = lane_in_col[lane_in_col > center_row]
            lo = int(right_side.max()) + 1 if len(right_side) > 0 else max(0, center_row - half_fill)
            hi = int(left_side.min())      if len(left_side)  > 0 else min(ny, center_row + half_fill)
            if lo >= hi:
                continue
            seg = data[lo:hi, col]
            data[lo:hi, col] = np.where(seg == np.int8(-1), np.int8(0), seg)
        # ────────────────────────────────────────────────────────────────────

        g.data = data.flatten().tolist()
        return g

    def _empty_grid(self, stamp=None) -> OccupancyGrid:
        g = OccupancyGrid()
        g.header.stamp = (
            self.get_clock().now().to_msg() if stamp is None else stamp)
        g.header.frame_id = self.occupancy_grid_frame
        # ROS convention: info.width = #cells along +x (forward),
        # info.height = #cells along +y (lateral).
        nx = int(self.grid_height_m / self.grid_res)  # forward cells
        ny = int(self.grid_width_m / self.grid_res)   # lateral cells
        g.info.resolution = self.grid_res
        g.info.width = nx
        g.info.height = ny

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
                # Origin = world position of cell (0,0). Lateral extent is
                # centred on the robot, so the (0,0) corner is offset by
                # -grid_width_m/2 along the rotated +y axis from the robot.
                g.info.origin.position.x = tx + np.sin(yaw) * (self.grid_width_m / 2.0)
                g.info.origin.position.y = ty - np.cos(yaw) * (self.grid_width_m / 2.0)
                g.info.origin.position.z = tf.transform.translation.z
                g.info.origin.orientation = q

        g.data = [-1] * (nx * ny)
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
        if self.local_from_persistent:
            if self._publish_local_costmap_from_persistent():
                return
        self.grid_pub.publish(self.latest_grid)

    def _publish_local_costmap_from_persistent(self) -> bool:
        pose = self._persistent_pose(None)
        if pose is None:
            return False

        tx, ty, _tz, yaw, _orientation = pose
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        if self.grid_res <= 0.0 or self.persist_res <= 0.0:
            return False

        nx = max(1, int(round(self.grid_height_m / self.grid_res)))
        ny = max(1, int(round(self.grid_width_m / self.grid_res)))
        local_oy = -0.5 * self.grid_width_m
        local_data = np.full((ny, nx), -1, dtype=np.int8)
        persistent_data = self._persistent_grid_data(None, clear_robot=False)

        col_idx = np.arange(nx, dtype=np.float32)
        row_idx = np.arange(ny, dtype=np.float32)
        local_x = col_idx[None, :] * self.grid_res
        local_y = local_oy + row_idx[:, None] * self.grid_res

        world_x = tx + cos_yaw * local_x - sin_yaw * local_y
        world_y = ty + sin_yaw * local_x + cos_yaw * local_y

        src_cols = np.rint((world_x - self._p_ox) / self.persist_res).astype(np.int32)
        src_rows = np.rint((world_y - self._p_oy) / self.persist_res).astype(np.int32)
        valid = (
            (src_cols >= 0) & (src_cols < self._pN) &
            (src_rows >= 0) & (src_rows < self._pN)
        )
        local_data[valid] = persistent_data[src_rows[valid], src_cols[valid]]

        if not np.any((local_data == 0) | (local_data == 100)):
            return False

        # ── Lane corridor inflation ──
        # Dilate lane (100) cells by ``local_lane_inflation_m`` so the
        # navigator sees a continuous lane swath even when the raw mask
        # has small gaps.  Equivalent to enforcing a minimum corridor
        # margin on either side of the path without changing the
        # extractor.  Only the lane class is inflated — free/unknown
        # cells are left alone so the corridor interior stays open.
        if self.local_lane_inflation_m > 0.0 and self.grid_res > 0.0:
            radius_cells = int(math.ceil(
                self.local_lane_inflation_m / self.grid_res))
            if radius_cells > 0:
                k = 2 * radius_cells + 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (k, k))
                lane_mask = (local_data == 100).astype(np.uint8)
                inflated = cv2.dilate(lane_mask, kernel, iterations=1)
                # Only overwrite cells that are currently free (0) or
                # unknown (-1).  Existing 100 stays 100.
                grow = (inflated > 0) & (local_data != 100)
                local_data[grow] = 100

        back_buf_cells = int(math.ceil(self.local_back_nogo_buffer_m / self.grid_res))
        if back_buf_cells > 0:
            local_data[:, :min(back_buf_cells, nx)] = 100

        local = OccupancyGrid()
        local.header.stamp = self.get_clock().now().to_msg()
        local.header.frame_id = self.base_frame
        local.info.resolution = self.grid_res
        local.info.width = nx
        local.info.height = ny
        local.info.origin.position.x = 0.0
        local.info.origin.position.y = local_oy
        local.info.origin.position.z = 0.0
        local.info.origin.orientation.w = 1.0
        local.data = local_data.ravel().tolist()
        self.latest_grid = local
        self.grid_pub.publish(local)
        return True

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
        try:
            from rclpy.experimental import EventsExecutor
            executor = EventsExecutor()
        except ImportError:
            from rclpy.executors import SingleThreadedExecutor
            node.get_logger().warn(
                'EventsExecutor is not available in this rclpy install; '
                'falling back to SingleThreadedExecutor.')
            executor = SingleThreadedExecutor()

        executor.add_node(node)
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
