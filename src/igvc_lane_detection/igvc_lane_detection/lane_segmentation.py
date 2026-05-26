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

import array
import math
import threading
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
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


# Cycle-colored palette for the MarkerArray debug view.  Keeps the first
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
        self.roi_left_frac          = float(p('roi_left_frac',      0.05))
        self.roi_right_frac         = float(p('roi_right_frac',     0.95))
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

        # ── Lane-line refinement (paint-only filter) ─────────────────
        # YOLOPv2's lane-line head fires on more than just painted lines —
        # in particular it tags road boundaries / sidewalk edges where
        # the asphalt ends.  IGVC lanes are *only* painted white, so we
        # filter the raw lane mask through:
        #   1. drivable-area gate     → keep only paint that lies inside
        #      (or within ``lane_da_dilate_px`` of) the drivable area, so
        #      curbs / off-road edges are dropped.
        #   2. white-paint color gate → keep only pixels whose RGB looks
        #      like white paint (high V, low S in HSV).  This is what
        #      separates a true lane line from the road-edge texture.
        # Both filters can be disabled independently.
        self.lane_threshold             = float(p('lane_threshold', 0.5))
        self.lane_in_drivable_only      = bool(p('lane_in_drivable_only', True))
        self.lane_da_dilate_px          = int(p('lane_da_dilate_px', 30))
        self.lane_color_filter_enabled  = bool(p('lane_color_filter_enabled', True))
        self.lane_color_v_min           = int(p('lane_color_v_min', 170))
        self.lane_color_s_max           = int(p('lane_color_s_max', 70))
        self.lane_morph_close_px        = int(p('lane_morph_close_px', 3))
        # Pre-allocate kernels.
        _kc = self.lane_morph_close_px | 1
        self._lane_close_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_kc, _kc))
            if self.lane_morph_close_px > 1 else None)
        _kd = self.lane_da_dilate_px | 1
        self._lane_da_dilate_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_kd, _kd))
            if self.lane_da_dilate_px > 0 else None)

        # ── Classical white-paint augmentation (for thin IRL lines) ──
        # The model is trained on real driving data with lane lines
        # several inches wide.  IGVC IRL paint is 0.5–2 inches wide and
        # often too thin to trigger the head reliably even at lower
        # ``lane_threshold``.  When this flag is on we add white-paint
        # pixels (gated by drivable-area + color) directly to the lane
        # mask, supplementing the model's response.  The augmentation is
        # restricted to the drivable area so road edges and white sky
        # regions cannot contribute.
        self.lane_color_augment_enabled = bool(
            p('lane_color_augment_enabled', False))
        self.lane_color_augment_v_min   = int(p('lane_color_augment_v_min', 200))
        self.lane_color_augment_s_max   = int(p('lane_color_augment_s_max', 50))
        self.lane_color_augment_min_area_px = int(
            p('lane_color_augment_min_area_px', 25))

        # ── Canny edge augmentation (white-edge recovery) ─────────────
        # Runs Canny on the grayscale image inside the drivable area,
        # then keeps only edges that also pass the white-paint HSV gate.
        # The surviving edge pixels are OR-ed into the lane mask so thin
        # lines that YOLOPv2 missed but have a clear brightness edge are
        # still sent to the costmap.
        # `lane_canny_dilate_px` thickens the detected edges before
        # merging so individual Canny pixels project reliably.
        self.lane_canny_augment_enabled = bool(
            p('lane_canny_augment_enabled', False))
        self.lane_canny_low             = int(p('lane_canny_low',  30))
        self.lane_canny_high            = int(p('lane_canny_high', 100))
        self.lane_canny_v_min           = int(p('lane_canny_v_min', 170))
        self.lane_canny_s_max           = int(p('lane_canny_s_max',  70))
        self.lane_canny_dilate_px       = int(p('lane_canny_dilate_px', 5))
        _kce = self.lane_canny_dilate_px | 1
        self._lane_canny_dilate_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_kce, _kce))
            if self.lane_canny_dilate_px > 0 else None)

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
                lane_threshold=self.lane_threshold,
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
                lane_threshold=self.lane_threshold,
            )
        self.model.load()
        if self.model.fallback_warning:
            self.get_logger().warn(self.model.fallback_warning)
        self.get_logger().info(
            f"YOLOPv2 ready on {self.model.device} "
            f"(half={self.model.half}, img_size={self.model_img_size}).")

        self._init_persistent_map()

        # ── Multi-camera threading ────────────────────────────────────
        # YOLOPv2 GPU inference is not thread-safe for concurrent calls, so
        # a lock serialises inference while allowing all camera callbacks to
        # be scheduled in parallel by the MultiThreadedExecutor.
        self._model_lock = threading.Lock()
        # _state_lock guards _cam_state, _phits, _pfree, latest_grid and all
        # derived publish calls so concurrent post-inference writes are safe.
        self._state_lock = threading.Lock()
        # ReentrantCallbackGroup lets the executor run camera callbacks
        # concurrently instead of forcing a single-active-callback policy.
        self._cam_cb_group = ReentrantCallbackGroup()

        # ── Camera subscriptions ─────────────────────────────────────
        num_cameras  = p('num_cameras',        1)
        cam_topics   = p('camera_topics',      ['/camera/image_raw'])
        depth_topics = p('depth_topics',       ['/camera/depth/image_raw'])
        info_topics  = p('camera_info_topics', ['/camera/camera_info'])

        num_cameras = min(
            num_cameras, len(cam_topics), len(depth_topics), len(info_topics))

        self._camera_topic_pairs = {
            i: (str(cam_topics[i]), str(depth_topics[i]))
            for i in range(num_cameras)
        }
        self._got_frame = False
        self._got_frame_by_cam = {i: False for i in range(num_cameras)}

        self._sync_handles: list = []
        self.overlay_pubs: dict = {}

        for i in range(num_cameras):
            self.create_subscription(
                CameraInfo, info_topics[i],
                lambda msg, idx=i: self._on_info(msg, idx), 10,
                callback_group=self._cam_cb_group)

            rgb_sub = Subscriber(
                self, Image, cam_topics[i], qos_profile=qos_profile_sensor_data,
                callback_group=self._cam_cb_group)
            depth_sub = Subscriber(
                self, Image, depth_topics[i], qos_profile=qos_profile_sensor_data,
                callback_group=self._cam_cb_group)
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

            self.get_logger().info(
                f'Configured cam[{i}]: rgb={cam_topics[i]} depth={depth_topics[i]} '
                f'info={info_topics[i]}')

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
        # Per-shape pixel meshgrid cache (built lazily, reused across frames).
        self._pixel_grid_cache: dict = {}

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
        # Prefer a stamp-aligned TF lookup whenever a stamp is provided.
        # With multiple cameras, inference is serialised so left/right
        # frames are processed several model-latency periods after they
        # were captured.  Using the *latest* odom for those frames
        # projects their lane points to a world pose that no longer
        # matches when the photons hit the sensor — the lane lines then
        # appear shifted in the persistent map.  A TF lookup at the
        # frame stamp uses odom_tf_bridge_node's history to recover the
        # robot pose at capture time, eliminating the shift.
        if stamp is not None:
            transform = lookup_tf(
                self.tf_buffer, self.persist_frame, self.base_frame, stamp)
            if transform is not None:
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                return (
                    translation.x,
                    translation.y,
                    translation.z,
                    yaw_from_quat(rotation.x, rotation.y, rotation.z, rotation.w),
                    rotation,
                )

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
        free_empty = free_pts is None or len(free_pts) == 0
        lane_empty = lane_pts is None or len(lane_pts) == 0
        if free_empty and lane_empty:
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

        with self._state_lock:
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
                g.data = array.array('b', data.tobytes())
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
        with self._state_lock:
            self._got_frame = True
            self._got_frame_by_cam[cam_idx] = True

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
            with self._model_lock:
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

        # ── Lane refinement: drop road edges, recover thin paint ────
        # See ``_refine_lane_mask`` for details.  Runs before the ROI /
        # chassis crop so the color gate sees the original RGB.
        ll_mask = self._refine_lane_mask(bgr, da_mask, ll_mask)

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

        # ── Cache per-camera state + fuse + publish (serialised) ──
        with self._state_lock:
            self._cam_state[cam_idx] = {
                'stamp':    process_stamp,
                'free':     free_pts,
                'lane':     lane_pts,
            }

            fused_free, fused_lane = self._fuse_points(process_stamp)

            if fused_free.shape[0] > 0 or fused_lane.shape[0] > 0:
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
            active_cam_count = len(self._active_cam_states(process_stamp))

        # ── Publish per-component lane markers ──
        self._publish_lane_markers(lane_components, process_stamp)

        self.get_logger().info(
            f'cam[{cam_idx}] free={free_pts.shape[0]} lane={lane_pts.shape[0]} '
            f'components={len(lane_components)} '
            f'active_cams={active_cam_count}',
            throttle_duration_sec=1.0)

    def _active_cam_states(self, stamp) -> List[dict]:
        now_t = Time.from_msg(stamp)
        active = []
        for state in self._cam_state.values():
            dt = abs((now_t - Time.from_msg(state['stamp'])).nanoseconds / 1e9)
            if dt <= self.fusion_timeout_sec:
                active.append(state)
        return active


    def _fuse_points(self, stamp) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate per-camera ``(N, 2)`` arrays into one fused array each."""
        free_chunks: List[np.ndarray] = []
        lane_chunks: List[np.ndarray] = []
        for state in self._active_cam_states(stamp):
            f = state['free']
            l = state['lane']
            if f is not None and f.shape[0] > 0:
                free_chunks.append(f)
            if l is not None and l.shape[0] > 0:
                lane_chunks.append(l)
        free_arr = (
            np.concatenate(free_chunks, axis=0)
            if free_chunks else np.empty((0, 2), dtype=np.float32))
        lane_arr = (
            np.concatenate(lane_chunks, axis=0)
            if lane_chunks else np.empty((0, 2), dtype=np.float32))
        return free_arr, lane_arr

    # ═══════════════════════════════════════════════════════════════════
    # Mask → base_link projection
    # ═══════════════════════════════════════════════════════════════════

    def _pixel_grid(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return cached ``(ug, vg)`` pixel-coordinate grids for shape (h, w).

        Building these via ``np.meshgrid`` every frame on Jetson is a
        non-trivial cost at depth-image resolution (1280×720 → ~1 M
        floats × 2).  Cache by shape; we typically only see one or two
        sizes per node lifetime.
        """
        key = (h, w)
        cached = self._pixel_grid_cache.get(key)
        if cached is not None:
            return cached
        us = np.arange(w, dtype=np.float32)
        vs = np.arange(h, dtype=np.float32)
        ug, vg = np.meshgrid(us, vs)
        self._pixel_grid_cache[key] = (ug, vg)
        return ug, vg

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
        ug, vg = self._pixel_grid(dh, dw)

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
            ], dtype=np.float32)
            # Z-row only — the obstacle gate is on base_link Z so we
            # don't need the X / Y rows.  Saves ~2/3 of the matmul cost.
            bz = (
                R[2, 0] * xc + R[2, 1] * yc + R[2, 2] * zc + float(t.z)
            ).astype(np.float32)
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

    def _refine_lane_mask(
        self,
        bgr: np.ndarray,
        da_mask: np.ndarray,
        ll_mask: np.ndarray,
    ) -> np.ndarray:
        """Filter & augment the raw YOLOPv2 lane-line mask.

        Steps (all individually toggleable via parameters):
          1. Restrict to the drivable area (optionally dilated) — the
             head fires on road-edge texture which is *not* painted; we
             only want lines that lie on or near the road surface.
          2. color-gate: keep only pixels that look like white paint
             (high V, low S in HSV).  Removes asphalt-vs-grass edges
             that the model still labels as "lane line".
          3. Optional white-paint augmentation: ADD bright white-paint
             pixels (within the drivable area) that the model may have
             missed because IRL paint is too thin (0.5–2 inches) to
             trigger the head reliably.
          4. Morphological close: bridges 1–2 pixel gaps along thin
             lines so connected-component filtering does not throw away
             dashed / broken paint.
        """
        if ll_mask is None or ll_mask.size == 0:
            return ll_mask

        # Make sure we work in uint8 {0,1}.
        ll = (ll_mask > 0).astype(np.uint8)

        # Resize masks to BGR shape if a model returns at a different size.
        h, w = bgr.shape[:2]
        if ll.shape[:2] != (h, w):
            ll = cv2.resize(ll, (w, h), interpolation=cv2.INTER_NEAREST)
        if da_mask is not None and da_mask.shape[:2] != (h, w):
            da = cv2.resize(
                (da_mask > 0).astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST)
        else:
            da = (da_mask > 0).astype(np.uint8) if da_mask is not None else None

        # Compute drivable-area gate (DA optionally dilated outward so
        # paint just outside the predicted DA still survives).
        da_gate: Optional[np.ndarray] = None
        if da is not None:
            if self._lane_da_dilate_kernel is not None:
                da_gate = cv2.dilate(da, self._lane_da_dilate_kernel)
            else:
                da_gate = da

        # 1) Keep only lane mask cells inside the (dilated) drivable area.
        if self.lane_in_drivable_only and da_gate is not None:
            ll = ll & da_gate

        # Pre-compute HSV once if we need it for color filter / augment.
        need_hsv = (
            self.lane_color_filter_enabled
            or self.lane_color_augment_enabled
        )
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) if need_hsv else None

        # 2) White-paint color gate on the model's lane mask.
        if self.lane_color_filter_enabled and hsv is not None:
            v = hsv[:, :, 2]
            s = hsv[:, :, 1]
            white = ((v >= self.lane_color_v_min) &
                     (s <= self.lane_color_s_max)).astype(np.uint8)
            ll = ll & white

        # 3) White-paint augmentation: classical detector for thin lines.
        if self.lane_color_augment_enabled and hsv is not None and da_gate is not None:
            v = hsv[:, :, 2]
            s = hsv[:, :, 1]
            paint = ((v >= self.lane_color_augment_v_min) &
                     (s <= self.lane_color_augment_s_max)).astype(np.uint8)
            paint &= da_gate
            # Drop tiny specular highlights / pebbles by area filter.
            if self.lane_color_augment_min_area_px > 0 and np.any(paint):
                n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(
                    paint, connectivity=8)
                keep = np.zeros_like(paint)
                for lbl in range(1, n_lbl):
                    if stats[lbl, cv2.CC_STAT_AREA] >= \
                            self.lane_color_augment_min_area_px:
                        keep[lbls == lbl] = 1
                paint = keep
            ll = ((ll | paint) > 0).astype(np.uint8)

        # 4) Canny white-edge augmentation: recover thin lines the model missed.
        if self.lane_canny_augment_enabled and da_gate is not None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Restrict Canny to the drivable area so road boundaries/walls
            # outside it cannot contribute edge pixels.
            gray_gated = gray.copy()
            gray_gated[da_gate == 0] = 0
            edges = cv2.Canny(gray_gated,
                              self.lane_canny_low, self.lane_canny_high,
                              apertureSize=3, L2gradient=True)
            # Keep only edges that also look like white paint in HSV.
            if hsv is None:
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            white_gate = ((hsv[:, :, 2] >= self.lane_canny_v_min) &
                          (hsv[:, :, 1] <= self.lane_canny_s_max)).astype(np.uint8)
            white_edges = cv2.bitwise_and(edges, edges, mask=white_gate)
            # Thicken edges before merging so single-pixel Canny responses
            # project as small regions rather than individual points.
            if self._lane_canny_dilate_kernel is not None and np.any(white_edges):
                white_edges = cv2.dilate(white_edges, self._lane_canny_dilate_kernel)
            ll = ((ll | (white_edges > 0)) > 0).astype(np.uint8)

        # 5) Morphological close to bridge thin-line gaps.
        if self._lane_close_kernel is not None and np.any(ll):
            ll = cv2.morphologyEx(ll, cv2.MORPH_CLOSE, self._lane_close_kernel)

        return ll

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

        left = float(np.clip(self.roi_left_frac, 0.0, 1.0))
        right = float(np.clip(self.roi_right_frac, 0.0, 1.0))
        if right <= left:
            self.get_logger().warn(
                'Invalid ROI bounds: roi_right_frac must be greater than '
                'roi_left_frac. Falling back to defaults (0.95, 0.05).',
                throttle_duration_sec=2.0)
            left, right = 0.05, 0.95

        # Keep the historical top-width proportion (0.4 of full image when
        # defaults are used) while allowing left/right ROI bounds to shift.
        top_width_ratio = 0.4 / 0.9
        center = 0.5 * (left + right)
        half_span = 0.5 * (right - left)
        top_half_span = half_span * top_width_ratio
        top_left = center - top_half_span
        top_right = center + top_half_span

        pts = np.array([[
            [w * left, h * bot],
            [w * top_left, h * top],
            [w * top_right, h * top],
            [w * right, h * bot],
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
        info_size = self.camera_info_size.get(cam_idx)
        if info_size is not None and info_size[0] > 0 and info_size[1] > 0:
            sx = w_d / float(info_size[0])
            sy = h_d / float(info_size[1])
            fx, fy = fx * sx, fy * sy
            cx, cy = cx * sx, cy * sy
        return fx, fy, cx, cy

    def _sample_depth_with_fallback(
        self,
        depth: np.ndarray,
        yd: np.ndarray,
        xd: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d = depth[yd, xd].astype(np.float32, copy=False)
        direct_valid = (
            np.isfinite(d)
            & (d > self.min_detection_depth_m)
            & (d < self.max_detection_depth_m)
        )

        radius = max(0, int(self.depth_search_radius_px))
        if radius == 0 or np.all(direct_valid):
            return d, direct_valid

        valid_depth = (
            np.isfinite(depth)
            & (depth > self.min_detection_depth_m)
            & (depth < self.max_detection_depth_m)
        )
        if not np.any(valid_depth):
            return d, direct_valid

        k = 2 * radius + 1
        depth32 = depth.astype(np.float32, copy=False)
        depth_sum = cv2.boxFilter(
            np.where(valid_depth, depth32, 0.0), cv2.CV_32F, (k, k),
            normalize=False, borderType=cv2.BORDER_CONSTANT)
        depth_count = cv2.boxFilter(
            valid_depth.astype(np.float32), cv2.CV_32F, (k, k),
            normalize=False, borderType=cv2.BORDER_CONSTANT)

        fallback = np.zeros_like(d, dtype=np.float32)
        sample_count = depth_count[yd, xd]
        has_fallback = sample_count > 0.0
        fallback[has_fallback] = (
            depth_sum[yd[has_fallback], xd[has_fallback]]
            / sample_count[has_fallback]
        )

        use_fallback = ~direct_valid & has_fallback
        if np.any(use_fallback):
            d = d.copy()
            d[use_fallback] = fallback[use_fallback]
        return d, direct_valid | has_fallback

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

    def _project_pixel_indices(
        self,
        ys_mask: np.ndarray,
        xs_mask: np.ndarray,
        mask_shape: Tuple[int, int],
        depth: np.ndarray,
        cam_idx: int,
        cam_tf,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised mask-pixel → ``base_link`` projection.

        Parameters
        ----------
        ys_mask, xs_mask
            Pixel coords in *mask* resolution.
        mask_shape
            ``(h, w)`` of the source mask, used to scale to depth coords.

        Returns
        -------
        pts
            ``(N, 2) float32`` ``(fwd, lat)`` array of valid points.
        valid_idx
            ``(N,) int`` indices into the input arrays that survived all
            gates (NaN/range/forward).  Lets the caller carry side-data
            (e.g. component labels) through the projection without a
            Python loop.
        """
        if ys_mask.size == 0:
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0,), dtype=np.int64))

        h_d, w_d = depth.shape[:2]
        h_m, w_m = mask_shape
        sx = w_d / float(w_m) if w_m > 0 else 1.0
        sy = h_d / float(h_m) if h_m > 0 else 1.0

        # Mask → depth coords.  Round to nearest int.
        if sx == 1.0 and sy == 1.0:
            xd = xs_mask.astype(np.int32, copy=False)
            yd = ys_mask.astype(np.int32, copy=False)
        else:
            xd = (xs_mask.astype(np.float32) * sx).astype(np.int32)
            yd = (ys_mask.astype(np.float32) * sy).astype(np.int32)

        in_bounds = (xd >= 0) & (xd < w_d) & (yd >= 0) & (yd < h_d)
        if not np.any(in_bounds):
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0,), dtype=np.int64))

        # Carry the surviving original indices through each gate.
        idx0 = np.flatnonzero(in_bounds)
        xd = xd[idx0]
        yd = yd[idx0]

        d, gate = self._sample_depth_with_fallback(depth, yd, xd)
        if not np.any(gate):
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0,), dtype=np.int64))

        idx1 = idx0[gate]
        xd = xd[gate].astype(np.float32, copy=False)
        yd = yd[gate].astype(np.float32, copy=False)
        d = d[gate]

        fx, fy, cx, cy = self._intrinsics(cam_idx, depth.shape)
        rot, trans = self._cam_tf_components(cam_tf)

        if rot is None or trans is None:
            fwd = d
            lat = -(xd - float(cx)) * d / float(fx)
        else:
            xc = (xd - float(cx)) * d / float(fx)
            yc = (yd - float(cy)) * d / float(fy)
            zc = d
            # rot: (3, 3) float32; pts_cam: (3, N).
            pts_cam = np.stack([xc, yc, zc], axis=0)
            pts_base = rot @ pts_cam
            pts_base[0] += float(trans[0])
            pts_base[1] += float(trans[1])
            fwd = pts_base[0]
            lat = pts_base[1]

        forward_gate = fwd > 0.0
        if not np.any(forward_gate):
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0,), dtype=np.int64))

        idx_final = idx1[forward_gate]
        pts = np.stack([fwd[forward_gate], lat[forward_gate]], axis=1).astype(
            np.float32, copy=False)
        return pts, idx_final

    def _project_mask_points(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        cam_idx: int,
        cam_tf,
        stride: int,
    ) -> np.ndarray:
        """Vectorised drivable-area projection.

        Returns an ``(N, 2) float32`` array of ``(fwd, lat)`` points in
        ``base_link``.  The list-of-tuples API of the previous
        implementation is replaced with a numpy array — downstream code
        (``_fuse_points`` / ``_build_grid`` / ``_stamp_persistent_points``)
        already accepts both.
        """
        if mask is None or mask.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        # Stride decimation on the mask itself — cheaper than indexing
        # after the fact because the mask is mostly zeros.
        sub = mask[::stride, ::stride] if stride > 1 else mask
        ys_sub, xs_sub = np.nonzero(sub)
        if ys_sub.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        # Cap.  Use a deterministic stride-decimation rather than RNG
        # so frame-to-frame the same pixels are kept (less flicker in
        # the persistent map).
        if ys_sub.size > self.max_points_per_frame:
            keep_every = int(math.ceil(ys_sub.size / self.max_points_per_frame))
            ys_sub = ys_sub[::keep_every]
            xs_sub = xs_sub[::keep_every]

        # Map back to mask coords.
        ys_mask = ys_sub * stride if stride > 1 else ys_sub
        xs_mask = xs_sub * stride if stride > 1 else xs_sub

        pts, _ = self._project_pixel_indices(
            ys_mask, xs_mask, mask.shape[:2], depth, cam_idx, cam_tf)
        return pts

    def _project_lane_mask(
        self,
        ll_mask: np.ndarray,
        depth: np.ndarray,
        cam_idx: int,
        cam_tf,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Vectorised lane-line projection, partitioned by component.

        Returns
        -------
        all_pts
            ``(N, 2) float32`` array of every projected lane point.
        components
            List of per-component ``(M_i, 2) float32`` arrays, sorted by
            mean lateral offset (most-positive first) so colors are
            stable.
        """
        if ll_mask is None or ll_mask.size == 0:
            return np.empty((0, 2), dtype=np.float32), []

        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            ll_mask.astype(np.uint8), connectivity=8)
        if num <= 1:
            return np.empty((0, 2), dtype=np.float32), []

        # Build a label-keep mask in one pass (avoids building a python
        # list and per-label np.where).
        keep_label = np.zeros(num, dtype=bool)
        keep_label[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_lane_component_px
        if not np.any(keep_label):
            return np.empty((0, 2), dtype=np.float32), []

        # Pixels that survive the area filter, full-resolution.
        survivors = keep_label[labels]
        ys_all, xs_all = np.nonzero(survivors)
        if ys_all.size == 0:
            return np.empty((0, 2), dtype=np.float32), []

        # Stride subsample.
        stride = max(1, self.ll_subsample_px)
        if stride > 1:
            ys_all = ys_all[::stride]
            xs_all = xs_all[::stride]

        # Cap.
        if ys_all.size > self.max_points_per_frame:
            keep_every = int(math.ceil(ys_all.size / self.max_points_per_frame))
            ys_all = ys_all[::keep_every]
            xs_all = xs_all[::keep_every]

        # Carry the per-pixel component label through projection.
        lbl_per_px = labels[ys_all, xs_all]

        all_pts, valid_idx = self._project_pixel_indices(
            ys_all, xs_all, ll_mask.shape[:2], depth, cam_idx, cam_tf)
        if all_pts.shape[0] == 0:
            return all_pts, []

        lbl_per_pt = lbl_per_px[valid_idx]

        # Group by component.  np.unique + np.argsort runs in C; we then
        # slice with searchsorted boundaries — no per-pixel python loop.
        order = np.argsort(lbl_per_pt, kind='stable')
        pts_sorted = all_pts[order]
        lbl_sorted = lbl_per_pt[order]
        unique_lbls, starts = np.unique(lbl_sorted, return_index=True)
        ends = np.append(starts[1:], lbl_sorted.size)

        components: List[np.ndarray] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            comp_pts = pts_sorted[s:e]
            if comp_pts.shape[0] > 0:
                components.append(comp_pts)

        # Sort components left-to-right by mean lateral offset.  +y is
        # left of the robot, so most-positive first.
        components.sort(key=lambda a: -float(a[:, 1].mean()))
        return all_pts, components

    # ═══════════════════════════════════════════════════════════════════
    # Local rolling costmap
    # ═══════════════════════════════════════════════════════════════════

    def _build_grid(
        self,
        free_pts,
        lane_pts,
        stamp,
    ) -> OccupancyGrid:
        g = self._empty_grid(stamp)
        # ROS OccupancyGrid convention: data[row=y_idx, col=x_idx].
        # info.width  = forward cells (+x), info.height = lateral cells (+y).
        nx = g.info.width
        ny = g.info.height
        res = self.grid_res
        data = np.full((ny, nx), -1, dtype=np.int8)
        half_w = self.grid_width_m / 2.0

        def _stamp(pts, value: int) -> None:
            arr = pts if isinstance(pts, np.ndarray) else np.asarray(
                pts, dtype=np.float32)
            if arr.size == 0:
                return
            cols = (arr[:, 0] / res).astype(np.int32)
            rows = ((arr[:, 1] + half_w) / res).astype(np.int32)
            ok = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
            if np.any(ok):
                data[rows[ok], cols[ok]] = np.int8(value)

        # Free first so lethal overrides it.
        _stamp(free_pts, 0)
        _stamp(lane_pts, 100)

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

        g.data = array.array('b', data.tobytes())
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

        empty = np.full(nx * ny, -1, dtype=np.int8)
        g.data = array.array('b', empty.tobytes())
        return g

    # ═══════════════════════════════════════════════════════════════════
    # Debug
    # ═══════════════════════════════════════════════════════════════════

    def _publish_overlay(
        self, idx: int, bgr: np.ndarray,
        da_mask: np.ndarray, ll_mask: np.ndarray, rgb_msg: Image,
    ) -> None:
        ov = bgr.copy()
        color = np.zeros_like(ov)
        color[da_mask > 0] = (0, 255, 0)     # green  = drivable area
        color[ll_mask > 0] = (0, 0, 255)     # red    = lane lines
        mask_any = (da_mask > 0) | (ll_mask > 0)
        ov[mask_any] = cv2.addWeighted(
            ov, 0.5, color, 0.5, 0.0)[mask_any]

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
        local.data = array.array('b', local_data.tobytes())
        self.latest_grid = local
        self.grid_pub.publish(local)
        return True

    def _watchdog(self) -> None:
        with self._state_lock:
            got_any = self._got_frame
            missing = [
                idx for idx, got_frame in self._got_frame_by_cam.items()
                if not got_frame
            ]
            self._got_frame = False
            for idx in self._got_frame_by_cam:
                self._got_frame_by_cam[idx] = False

        if not got_any:
            self.get_logger().warn(
                'No synced RGB+Depth frames. Check topics and slop.',
                throttle_duration_sec=5.0)
        elif missing:
            missing_text = ', '.join(
                f'cam[{idx}] rgb={self._camera_topic_pairs[idx][0]} '
                f'depth={self._camera_topic_pairs[idx][1]}'
                for idx in missing)
            self.get_logger().warn(
                f'No synced RGB+Depth frames for configured camera(s): '
                f'{missing_text}. Check that these topics are publishing.',
                throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = LaneSegmentationNode()
    try:
        from rclpy.executors import MultiThreadedExecutor
        # Allocate enough threads for all camera callbacks to be in flight
        # simultaneously (each camera needs one thread for its callback, plus
        # headroom for timers and service callbacks).
        executor = MultiThreadedExecutor(num_threads=8)
        executor.add_node(node)
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
