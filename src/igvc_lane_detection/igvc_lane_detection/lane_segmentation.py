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
import os
import threading
import time
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
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3Stamped

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
        # Temporal lane filtering for /lane_costmap publication.
        # The filter tracks per-cell lane occupancy confidence over time
        # and applies hysteresis to suppress one-frame flicker.
        self.temporal_filter_enabled = bool(p('temporal_filter_enabled', True))
        self.temporal_lane_rise_alpha = float(
            np.clip(float(p('temporal_lane_rise_alpha', 0.60)), 0.0, 1.0))
        self.temporal_lane_fall_alpha = float(
            np.clip(float(p('temporal_lane_fall_alpha', 0.35)), 0.0, 1.0))
        self.temporal_lane_unknown_decay = float(
            np.clip(float(p('temporal_lane_unknown_decay', 0.08)), 0.0, 1.0))
        self.temporal_lane_on_threshold = float(
            np.clip(float(p('temporal_lane_on_threshold', 0.60)), 0.0, 1.0))
        self.temporal_lane_off_threshold = float(
            np.clip(float(p('temporal_lane_off_threshold', 0.45)), 0.0, 1.0))
        if self.temporal_lane_off_threshold > self.temporal_lane_on_threshold:
            self.temporal_lane_off_threshold = self.temporal_lane_on_threshold
        # If no frame applies the filter for this many seconds, zero the
        # EMA state on the next call so stale evidence does not re-emerge
        # after a long perception gap (e.g. dropped frames, mode switch).
        # <= 0 disables the gap reset.
        self.temporal_reset_gap_sec = float(
            p('temporal_reset_gap_sec', 1.0))

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
        self.ufldv2_root            = p('ufldv2_root',              '')
        self.ufldv2_config          = p('ufldv2_config',            '')
        self.ufldv2_lane_width_px   = max(1, int(p('ufldv2_lane_width_px', 8)))
        self.ufldv2_drivable_fill   = bool(p('ufldv2_drivable_fill', True))
        self.ufldv2_drivable_lane_dilation_px = max(
            0, int(p('ufldv2_drivable_lane_dilation_px', 45)))
        self.ufldv2_min_points_per_lane = max(
            1, int(p('ufldv2_min_points_per_lane', 4)))
        self.ufldv2_refine_lane_mask = bool(p('ufldv2_refine_lane_mask', False))
        self.ufldv2_paint_refine_enabled = bool(
            p('ufldv2_paint_refine_enabled', True))
        self.ufldv2_paint_refine_dilation_px = max(
            0, int(p('ufldv2_paint_refine_dilation_px', 15)))
        self.ufldv2_paint_refine_ground_gate = bool(
            p('ufldv2_paint_refine_ground_gate', True))
        self.ufldv2_paint_refine_min_component_px = max(
            1, int(p('ufldv2_paint_refine_min_component_px', 35)))
        self.da_subsample_px        = max(1, int(p('da_subsample_px', 6)))
        self.ll_subsample_px        = max(1, int(p('ll_subsample_px', 2)))
        self.min_lane_component_px  = int(p('min_lane_component_px', 150))
        self.min_da_component_px    = int(p('min_da_component_px', 0))
        self.max_points_per_frame   = int(p('max_points_per_frame', 4000))

        # Detection mode:
        #   yolopv2          -> model masks + PC2 projection (legacy path)
        #   ufldv2           -> Ultra-Fast-Lane-Detection-v2 lane coordinates
        #   pc2_color_ground -> white/yellow ground-plane PC2 points only
        self.detection_mode = str(p('detection_mode', 'yolopv2')).lower()
        self._use_yolo = self.detection_mode in ('yolopv2', 'yolo', 'segmentation')
        self._use_ufldv2 = self.detection_mode in (
            'ufldv2', 'ultrafast', 'ultrafast_lane_detection_v2')
        self._use_model = self._use_yolo or self._use_ufldv2
        if not self._use_model and self.detection_mode != 'pc2_color_ground':
            raise RuntimeError(
                "Unsupported detection_mode. Use 'yolopv2', 'ufldv2', or "
                "'pc2_color_ground'.")
        if self._use_ufldv2 and not self.model_weights:
            self.model_weights = os.environ.get('UFLDV2_WEIGHTS', '')
        elif self._use_yolo and not self.model_weights:
            self.model_weights = os.environ.get('YOLOPV2_WEIGHTS', '')
        self.model_pc2_fallback_enabled = bool(
            p('model_pc2_fallback_enabled', self._use_ufldv2))
        self.model_debug_stats = bool(p('model_debug_stats', True))

        # Validate the parameters we need before any expensive setup
        # (model deserialization, persistent-map allocation, subscriptions).
        # Catches typos / bad combinations in seconds instead of after a
        # multi-second TensorRT load.
        self._validate_params()

        # ── PC2 color-ground detector ───────────────────────────────
        # Lane points are ground-plane cloud pixels whose packed RGB is
        # white or yellow.  Free space is sampled from non-lane ground
        # points that fall laterally between the detected lane boundaries.
        self.pc2_ground_z_min_m = float(p('pc2_ground_z_min_m', -0.12))
        self.pc2_ground_z_max_m = float(p('pc2_ground_z_max_m', 0.12))
        self.pc2_lane_z_min_m = float(p('pc2_lane_z_min_m', self.pc2_ground_z_min_m))
        self.pc2_lane_z_max_m = float(p('pc2_lane_z_max_m', self.pc2_ground_z_max_m))
        self.pc2_obstacle_exclusion_enabled = bool(
            p('pc2_obstacle_exclusion_enabled', True))
        self.pc2_obstacle_z_min_m = float(p('pc2_obstacle_z_min_m', 0.12))
        self.pc2_obstacle_z_max_m = float(p('pc2_obstacle_z_max_m', 2.5))
        self.pc2_obstacle_depth_min_m = float(p('pc2_obstacle_depth_min_m', 0.3))
        self.pc2_obstacle_depth_max_m = float(p('pc2_obstacle_depth_max_m', 8.0))
        self.pc2_obstacle_dilation_px = max(0, int(p('pc2_obstacle_dilation_px', 25)))
        self.pc2_lane_subsample_px = max(1, int(p('pc2_lane_subsample_px', 1)))
        self.pc2_free_subsample_px = max(1, int(p('pc2_free_subsample_px', 4)))
        self.pc2_lane_morph_close_px = int(p('pc2_lane_morph_close_px', 3))
        _kpc2 = self.pc2_lane_morph_close_px | 1
        self._pc2_lane_close_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_kpc2, _kpc2))
            if self.pc2_lane_morph_close_px > 1 else None)
        self.pc2_white_v_min = int(p('pc2_white_v_min', 185))
        self.pc2_white_s_max = int(p('pc2_white_s_max', 80))
        self.pc2_yellow_h_min = int(p('pc2_yellow_h_min', 15))
        self.pc2_yellow_h_max = int(p('pc2_yellow_h_max', 40))
        self.pc2_yellow_s_min = int(p('pc2_yellow_s_min', 60))
        self.pc2_yellow_v_min = int(p('pc2_yellow_v_min', 120))
        self.pc2_free_bin_size_m = max(0.05, float(p('pc2_free_bin_size_m', 0.25)))
        self.pc2_free_boundary_percentile = float(
            p('pc2_free_boundary_percentile', 10.0))
        self.pc2_free_min_lane_points_per_bin = int(
            p('pc2_free_min_lane_points_per_bin', 4))
        self.pc2_free_min_lane_width_m = float(p('pc2_free_min_lane_width_m', 0.6))
        self.pc2_free_max_lane_width_m = float(p('pc2_free_max_lane_width_m', 6.0))
        self.pc2_free_lane_margin_m = float(p('pc2_free_lane_margin_m', 0.10))
        self.pc2_free_max_gap_m = float(p('pc2_free_max_gap_m', 1.0))
        self.pc2_free_single_boundary_enabled = bool(
            p('pc2_free_single_boundary_enabled', True))
        self.pc2_free_nominal_lane_width_m = float(
            p('pc2_free_nominal_lane_width_m', 2.4))
        self.pc2_free_single_boundary_min_abs_lat_m = float(
            p('pc2_free_single_boundary_min_abs_lat_m', 0.10))

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

        # ── Ground-plane RANSAC projection ────────────────────────────
        # Robustness experiment: instead of per-pixel PC2 z (which is
        # very noisy on white lane paint due to low texture + specular
        # returns), fit a single ground plane each frame in base_link
        # using RANSAC on the *good* PC2 points (mostly asphalt/grass),
        # then back-project lane-mask pixels by intersecting the camera
        # ray with that plane.
        #
        # Modes:
        #   'off'      : disabled (default, current behaviour)
        #   'fallback' : use PC2 first; only fill NaN pixels via plane
        #   'force'    : always use ray-plane (ignore per-pixel PC2 z)
        # All other parameters are no-ops when mode == 'off'.
        self.ground_plane_projection_mode = str(
            p('ground_plane_projection_mode', 'off')).lower()
        self.ground_plane_ransac_iters     = max(1, int(p('ground_plane_ransac_iters', 60)))
        self.ground_plane_inlier_thresh_m  = float(p('ground_plane_inlier_thresh_m', 0.05))
        self.ground_plane_roi_fwd_min_m    = float(p('ground_plane_roi_fwd_min_m', 1.0))
        self.ground_plane_roi_fwd_max_m    = float(p('ground_plane_roi_fwd_max_m', 12.0))
        self.ground_plane_roi_lat_abs_m    = float(p('ground_plane_roi_lat_abs_m', 3.0))
        self.ground_plane_roi_z_abs_m      = float(p('ground_plane_roi_z_abs_m', 0.30))
        self.ground_plane_min_inliers      = max(3, int(p('ground_plane_min_inliers', 200)))
        self.ground_plane_min_z_normal     = float(p('ground_plane_min_z_normal', 0.7))
        self.ground_plane_ema_alpha        = float(np.clip(
            float(p('ground_plane_ema_alpha', 0.3)), 0.0, 1.0))
        self.ground_plane_max_sample_pts   = max(100, int(p('ground_plane_max_sample_pts', 5000)))
        # Per-camera plane cache (n_hat, d) in base_link. Persists across
        # frames so a single-frame RANSAC failure does not lose the plane.
        self._plane_by_cam: dict = {}

        if self.ground_plane_projection_mode not in ('off', 'fallback', 'force'):
            self.get_logger().warn(
                f"ground_plane_projection_mode='{self.ground_plane_projection_mode}' "
                "not recognised; falling back to 'off'.")
            self.ground_plane_projection_mode = 'off'

        # ── Ramp / slope detection ───────────────────────────────────
        # RANSAC-fits the ground plane ahead of the robot (independent of
        # ground_plane_projection_mode) and publishes the forward terrain
        # slope plus the in-plane heading to the steepest-ascent ("fall")
        # line on ``ramp_status_topic`` as a geometry_msgs/Vector3Stamped:
        #   vector.x = slope angle ahead (degrees)
        #   vector.y = heading error to the fall line (radians, base_link;
        #              the yaw the robot must turn to face straight up-slope)
        #   vector.z = detection confidence in [0, 1]
        # mission_planner_node consumes this (fused with IMU pitch) to align
        # the robot parallel to the IGVC ramp before climbing it.
        self.ramp_detect_enabled  = bool(p('ramp_detect_enabled', False))
        self.ramp_status_topic    = str(p('ramp_status_topic', '/ramp/state'))
        self.ramp_min_period_sec  = float(p('ramp_min_period_sec', 0.2))
        self._ramp_last_pub_mono: Optional[float] = None

        # ── Plane-based height obstacle gate ─────────────────────────
        # Once we have a RANSAC plane, any mask pixel whose true 3-D
        # height above the plane exceeds the threshold is almost
        # certainly an obstacle rather than ground paint.  This is a
        # stronger version of the fixed-Z obstacle_mask gate because it
        # adapts to terrain pitch and camera mounting drift.  Requires
        # ground_plane_projection_mode != 'off' (we need the plane).
        self.plane_height_gate_enabled    = bool(p('plane_height_gate_enabled', False))
        self.plane_height_gate_thresh_m   = float(p('plane_height_gate_thresh_m', 0.10))
        self.plane_height_gate_dilate_px  = max(0, int(p('plane_height_gate_dilate_px', 15)))

        # ── Per-component lane curve fitting ─────────────────────────
        # Fit a polynomial (lat = f(fwd)) to each lane component in
        # base_link and replace the raw projected points with samples
        # along the smoothed curve.  Removes scatter from depth noise
        # and bridges short gaps in dashed paint.  Can also extrapolate
        # the curve forward to the nearest range gate so the corridor
        # is continuous even when the model only sees the far end.
        self.lane_curve_fit_enabled        = bool(p('lane_curve_fit_enabled', False))
        self.lane_curve_poly_order         = max(1, min(3, int(p('lane_curve_poly_order', 2))))
        self.lane_curve_min_points         = max(3, int(p('lane_curve_min_points', 10)))
        self.lane_curve_sample_step_m      = max(0.02, float(p('lane_curve_sample_step_m', 0.10)))
        self.lane_curve_max_residual_m     = float(p('lane_curve_max_residual_m', 0.30))
        self.lane_curve_extend_to_robot    = bool(p('lane_curve_extend_to_robot', False))
        self.lane_curve_extend_forward_m   = float(p('lane_curve_extend_forward_m', 0.0))

        # ── Confidence-weighted persistent-map writes ────────────────
        # Per-point write weight = base_weight * w(point), where w is
        # derived from the size of the lane component that produced the
        # point (large connected components are more likely to be real
        # lane paint than small specks).  Disabled by default so the
        # persistent map behaves exactly as before.
        self.confidence_weighted_writes_enabled = bool(
            p('confidence_weighted_writes_enabled', False))
        self.confidence_min_weight              = float(np.clip(
            float(p('confidence_min_weight', 0.3)), 0.05, 1.0))
        self.confidence_component_full_px       = max(1, int(
            p('confidence_component_full_px', 400)))

        if self._use_model and not self.model_weights:
            raise RuntimeError(
                "Parameter 'model_weights' must be set to the path of "
                "the selected lane model checkpoint. For UFLDv2, run "
                "scripts/fetch_ufldv2_weights.sh; for YOLOPv2, run "
                "src/igvc_lane_detection/scripts/fetch_yolopv2_weights.sh.")
        if self._use_ufldv2 and not self.ufldv2_root:
            raise RuntimeError(
                "Parameter 'ufldv2_root' must point to a cloned "
                "Ultra-Fast-Lane-Detection-v2 repo. Run scripts/setup_ufldv2.sh.")
        if self._use_ufldv2 and not self.ufldv2_config:
            raise RuntimeError(
                "Parameter 'ufldv2_config' must point to an upstream config, "
                "for example configs/culane_res18.py.")

        # ── Read camera count now so we can build one model per camera ────
        # These are also used later for the subscription setup; storing them
        # as instance variables avoids a second declare_parameter call.
        self._num_cameras_param  = p('num_cameras',        1)
        self._cam_topics_param   = p('camera_topics',      ['/camera/image_raw'])
        self._pc2_topics_param   = p('pc2_topics',         ['/camera/points'])
        self._info_topics_param  = p('camera_info_topics', ['/camera/camera_info'])
        _num_cams = min(
            self._num_cameras_param,
            len(self._cam_topics_param),
            len(self._pc2_topics_param),
            len(self._info_topics_param))

        # ── Initialise segmentation models (one per camera, truly parallel) ─
        # YOLOPv2 dispatch: ``.engine`` -> TensorRT; anything else -> TorchScript.
        # The PC2 color-ground mode does not need the neural model at all.
        if self._use_model:
            weights_lower = str(self.model_weights).lower()
            use_trt = (
                self._use_yolo
                and (weights_lower.endswith('.engine') or weights_lower.endswith('.trt'))
            )
            if self._use_ufldv2:
                from .ufldv2_infer import UltraFastLaneDetectionV2
                self.get_logger().info(
                    f"Loading UFLDv2 from '{self.model_weights}' "
                    f"(root={self.ufldv2_root}, config={self.ufldv2_config}, "
                    f"half={self.model_half}, instances={_num_cams})...")
                self._models = []
                for _i in range(_num_cams):
                    _m = UltraFastLaneDetectionV2(
                        root_path=self.ufldv2_root,
                        config_path=self.ufldv2_config,
                        weights_path=self.model_weights,
                        device=self.model_device,
                        half=self.model_half,
                        lane_width_px=self.ufldv2_lane_width_px,
                        drivable_fill=self.ufldv2_drivable_fill,
                        drivable_lane_dilation_px=self.ufldv2_drivable_lane_dilation_px,
                        min_points_per_lane=self.ufldv2_min_points_per_lane,
                    )
                    _m.load()
                    if _m.fallback_warning:
                        self.get_logger().warn(_m.fallback_warning)
                    self._models.append(_m)
            elif use_trt:
                self.get_logger().info(
                    f"Loading YOLOPv2 from '{self.model_weights}' "
                    f"(backend=tensorrt, half={self.model_half}, instances={_num_cams})...")
                from .yolopv2_trt import YolopV2TRT
                # Deserialise once - weights live in GPU memory once regardless
                # of the number of cameras.
                self.get_logger().info("Deserialising TRT engine (shared across cameras)...")
                _shared_engine = YolopV2TRT.deserialize_engine(self.model_weights)
                self._models = []
                for _i in range(_num_cams):
                    _m = YolopV2TRT(
                        engine_path=self.model_weights,
                        img_size=self.model_img_size,
                        preprocess=self.model_preprocess,
                        clahe_clip=self.model_clahe_clip,
                        clahe_tile=tuple(self.model_clahe_tile),
                        blur_ksize=tuple(self.model_blur_ksize),
                        blur_sigma=self.model_blur_sigma,
                        lane_threshold=self.lane_threshold,
                        _shared_engine=_shared_engine,
                    )
                    _m.load()
                    self._models.append(_m)
            else:
                self.get_logger().info(
                    f"Loading YOLOPv2 from '{self.model_weights}' "
                    f"(backend=torchscript, half={self.model_half}, instances={_num_cams})...")
                self._models = []
                for _i in range(_num_cams):
                    _m = YolopV2(
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
                    _m.load()
                    if _m.fallback_warning:
                        self.get_logger().warn(_m.fallback_warning)
                    self._models.append(_m)
            _m0 = self._models[0]
            model_name = 'UFLDv2' if self._use_ufldv2 else 'YOLOPv2'
            self.get_logger().info(
                f"{model_name} ready: {_num_cams} instance(s) on {_m0.device} "
                f"(half={_m0.half}).")
        else:
            self._models = []
            self.get_logger().info(
                f'Lane detection mode "{self.detection_mode}": skipping neural model load.')

        self._init_persistent_map()

        # ── Multi-camera threading ────────────────────────────────────
        # Each camera has its own YOLOPv2 instance + CUDA stream, so
        # inference runs concurrently with no serialisation lock needed.
        # _state_lock still guards the shared map / grid state.
        # RLock so handlers that already hold the lock can safely call
        # methods (e.g. _republish_grid) that also acquire it.
        self._state_lock = threading.RLock()
        # ReentrantCallbackGroup lets the executor run camera callbacks
        # concurrently instead of forcing a single-active-callback policy.
        self._cam_cb_group = ReentrantCallbackGroup()

        # ── Camera subscriptions ─────────────────────────────────────
        # Topic lists were already read during model init above.
        num_cameras = _num_cams
        cam_topics  = self._cam_topics_param
        pc2_topics  = self._pc2_topics_param
        info_topics = self._info_topics_param

        self._camera_topic_pairs = {
            i: (str(cam_topics[i]), str(pc2_topics[i]))
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
            pc2_sub = Subscriber(
                self, PointCloud2, pc2_topics[i], qos_profile=qos_profile_sensor_data,
                callback_group=self._cam_cb_group)
            sync = ApproximateTimeSynchronizer(
                [rgb_sub, pc2_sub],
                queue_size=self.sync_queue_size,
                slop=self.sync_slop_sec)
            sync.registerCallback(
                lambda r, p, idx=i: self._on_images(r, p, idx))
            self._sync_handles.append((rgb_sub, pc2_sub, sync))

            if self.publish_overlay:
                self.overlay_pubs[i] = self.create_publisher(
                    Image, f'/lane_debug/cam{i}/overlay', 10)

            self.get_logger().info(
                f'Configured cam[{i}]: rgb={cam_topics[i]} pc2={pc2_topics[i]} '
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

        # Ramp/slope status for mission_planner (see ramp params above).
        self.ramp_pub = (
            self.create_publisher(Vector3Stamped, self.ramp_status_topic, 10)
            if self.ramp_detect_enabled else None)
        self.latest_grid = self._empty_grid()
        self._last_persistent_stamp = None
        self._cam_state: dict = {}
        self._latest_odom: Optional[Odometry] = None
        self._last_persist_pose: Optional[Tuple] = None
        self._temporal_lane_score: Optional[np.ndarray] = None
        self._temporal_lane_mask: Optional[np.ndarray] = None
        self._temporal_last_update_monotonic: Optional[float] = None
        # _temporal_lane_score / _temporal_lane_mask are always mutated
        # under _state_lock (both the camera-callback path and the
        # _republish_grid timer path).  No separate lock is needed.
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

    def _validate_params(self) -> None:
        """Fail loudly at startup for invalid parameter combinations.

        Only checks combinations that previously failed silently or with
        obscure runtime errors.  Values that are already clipped/clamped
        at parse time are not re-checked here.
        """
        errors: List[str] = []
        if self.grid_res <= 0.0:
            errors.append(f'grid_resolution must be > 0 (got {self.grid_res})')
        if self.grid_width_m <= 0.0:
            errors.append(f'grid_width must be > 0 (got {self.grid_width_m})')
        if self.grid_height_m <= 0.0:
            errors.append(f'grid_height must be > 0 (got {self.grid_height_m})')
        if self.persist_res <= 0.0:
            errors.append(f'persistent_map_resolution must be > 0 (got {self.persist_res})')
        if self.persist_size_m <= 0.0:
            errors.append(f'persistent_map_size_m must be > 0 (got {self.persist_size_m})')
        if not (0.0 < self.persist_decay <= 1.0):
            errors.append(
                f'persistent_map_decay must be in (0, 1] (got {self.persist_decay})')
        if self.local_from_persistent and not self.odom_topic:
            errors.append(
                'local_costmap_from_persistent=true requires a non-empty odom_topic')
        if self.min_detection_depth_m >= self.max_detection_depth_m:
            errors.append(
                f'min_detection_depth_m ({self.min_detection_depth_m}) must be < '
                f'max_detection_depth_m ({self.max_detection_depth_m})')
        mode = getattr(self, 'detection_mode', '')
        if mode not in ('pc2_color_ground', 'yolopv2', 'ufldv2'):
            errors.append(
                f"detection_mode must be one of 'pc2_color_ground'|'yolopv2'|'ufldv2' "
                f'(got {mode!r})')
        if errors:
            for e in errors:
                self.get_logger().error(f'Invalid parameter: {e}')
            raise ValueError(
                f'lane_segmentation_node parameter validation failed: '
                + '; '.join(errors))

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
        free_weights: Optional[np.ndarray] = None,
        lane_weights: Optional[np.ndarray] = None,
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
        odom = self._latest_odom  # single-ref snapshot avoids attribute-chain race
        if (
            self.max_yaw_rate_persist > 0.0
            and odom is not None
        ):
            yaw_rate = abs(odom.twist.twist.angular.z)
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
            self.get_logger().warn(
                f'Skipping persistent-map write: no pose from {self.persist_frame} '
                f'to {self.base_frame}.',
                throttle_duration_sec=2.0)
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
                self.get_logger().debug(
                    f'Skipping persistent-map write: pose change '
                    f'{dist:.3f} m / {dang:.3f} rad below thresholds '
                    f'{self.min_pose_change_m:.3f} m / '
                    f'{self.min_pose_change_rad:.3f} rad')
                return
        self._last_persist_pose = (tx, ty, yaw)

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        self._phits *= self.persist_decay
        self._pfree *= self.persist_decay

        self._stamp_persistent_points(
            free_pts, tx, ty, cos_y, sin_y,
            self._pfree, self.persist_free_hit_w,
            point_weights=free_weights)
        self._stamp_persistent_points(
            lane_pts, tx, ty, cos_y, sin_y,
            self._phits, self.persist_hit_w,
            point_weights=lane_weights)
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
        point_weights: Optional[np.ndarray] = None,
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

        # Confidence-weighted writes: per-point weight scales the base.
        # Caller passes None when CWW is disabled (current behaviour).
        if (
            point_weights is not None
            and len(point_weights) == pts.shape[0]
        ):
            pw = np.asarray(point_weights, dtype=np.float32)[valid]
            np.add.at(grid, (rows, cols), weight * pw)
        else:
            np.add.at(grid, (rows, cols), weight)
        # Clamp the entire grid in-place.  Using np.minimum on the full
        # array avoids the fancy-index last-write-wins issue for duplicate
        # (row, col) pairs that the original per-index write had.
        np.minimum(grid, self.persist_max, out=grid)

    def _persistent_grid_data(self, stamp=None, clear_robot: bool = True) -> np.ndarray:
        if self._persistent_data_cache is None or self._persistent_dirty:
            data = np.full((self._pN, self._pN), -1, dtype=np.int8)
            data[self._pfree >= self.persist_free_threshold] = 0
            data[self._phits >= self.persist_threshold] = 100
            self._persistent_data_cache = data
            self._persistent_dirty = False

        # Always return a copy so callers cannot mutate the cache array.
        data = self._persistent_data_cache.copy()
        if clear_robot:
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
        pose = self._persistent_pose(stamp)  # use frame-aligned stamp, not latest TF
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

    def _pc2_to_xyzrgb(
        self, pc2_msg: PointCloud2,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Parse organized PointCloud2 into XYZ and optional RGB arrays.

        The ZED publishes an organized cloud whose (row, col) maps directly to
        the corresponding image pixel.  Invalid points have NaN xyz.  Color is
        returned as RGB uint8 when the cloud includes a packed ``rgb``/``rgba``
        field, or ``None`` if the field is unavailable.

        Returns ``(None, None)`` if the message cannot be parsed.
        """
        if pc2_msg.height == 0 or pc2_msg.width == 0:
            return None, None
        try:
            h = int(pc2_msg.height)
            w = int(pc2_msg.width)
            field_map = {f.name: f.offset for f in pc2_msg.fields}
            x_off = field_map.get('x', 0)
            y_off = field_map.get('y', 4)
            z_off = field_map.get('z', 8)
            ps = pc2_msg.point_step
            n = h * w
            raw = np.frombuffer(pc2_msg.data, dtype=np.uint8).reshape(n, ps)
            float_dtype = np.dtype('>f4' if pc2_msg.is_bigendian else '<f4')
            x = raw[:, x_off:x_off + 4].copy().view(float_dtype).reshape(
                h, w).astype(np.float32, copy=False)
            y = raw[:, y_off:y_off + 4].copy().view(float_dtype).reshape(
                h, w).astype(np.float32, copy=False)
            z = raw[:, z_off:z_off + 4].copy().view(float_dtype).reshape(
                h, w).astype(np.float32, copy=False)
            xyz = np.stack([x, y, z], axis=-1)  # (H, W, 3)

            rgb = None
            rgb_field = 'rgb' if 'rgb' in field_map else 'rgba' if 'rgba' in field_map else None
            if rgb_field is not None:
                off = field_map[rgb_field]
                packed = raw[:, off:off + 4]
                if pc2_msg.is_bigendian:
                    r = packed[:, 1]
                    g = packed[:, 2]
                    b = packed[:, 3]
                else:
                    b = packed[:, 0]
                    g = packed[:, 1]
                    r = packed[:, 2]
                rgb = np.stack([r, g, b], axis=1).reshape(h, w, 3)
            elif all(name in field_map for name in ('r', 'g', 'b')):
                r = raw[:, field_map['r']]
                g = raw[:, field_map['g']]
                b = raw[:, field_map['b']]
                rgb = np.stack([r, g, b], axis=1).reshape(h, w, 3)

            return xyz, rgb
        except Exception as e:
            self.get_logger().error(f'PC2 parse error: {e}', throttle_duration_sec=2.0)
            return None, None

    def _pc2_to_xyz(self, pc2_msg: PointCloud2) -> Optional[np.ndarray]:
        xyz, _rgb = self._pc2_to_xyzrgb(pc2_msg)
        return xyz

    # ═══════════════════════════════════════════════════════════════════
    # Main callback
    # ═══════════════════════════════════════════════════════════════════

    def _on_images(self, rgb_msg: Image, pc2_msg: PointCloud2, cam_idx: int) -> None:
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
                'lane model/input is behind camera rate.',
                throttle_duration_sec=2.0)
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            xyz, cloud_rgb = self._pc2_to_xyzrgb(pc2_msg)
            if xyz is None:
                self.get_logger().error(
                    f'cam[{cam_idx}] PC2 decode returned None — skipping frame.',
                    throttle_duration_sec=2.0)
                return
        except Exception as e:
            self.get_logger().error(f'Decode error: {e}')
            return

        # ── Camera → base_link TF (once per frame) ──
        cam_frame = pc2_msg.header.frame_id or rgb_msg.header.frame_id
        cam_tf = None
        cloud_in_base_frame = bool(cam_frame and cam_frame == self.base_frame)
        if cam_frame and cam_frame != self.base_frame:
            cam_tf = lookup_tf(
                self.tf_buffer, self.base_frame, cam_frame, None)
            if cam_tf is None:
                self.get_logger().warn(
                    f'No TF from {cam_frame} to {self.base_frame}; '
                    f'falling back to camera-frame PC2 projection for cam[{cam_idx}]',
                    throttle_duration_sec=2.0)

        # ── Ramp / slope detection (independent of projection mode) ──
        if self.ramp_pub is not None and cam_idx == 0:
            self._detect_and_publish_ramp(
                cam_idx, xyz, cam_tf, cloud_in_base_frame)

        # ── Optional ground-plane RANSAC + optical-frame TF lookup ──
        # Only paid for when ground-plane projection is enabled.  The
        # optical TF is separate from cam_tf above because the ZED
        # publishes the cloud in the body (FLU) frame while the RGB
        # message is in the optical frame (x=right, y=down, z=forward),
        # which is what K is expressed in.
        plane: Optional[Tuple[np.ndarray, float]] = None
        optical_tf = None
        if self.ground_plane_projection_mode != 'off':
            plane = self._fit_ground_plane_base(
                cam_idx, xyz, cam_tf, cloud_in_base_frame)
            opt_frame = rgb_msg.header.frame_id
            if opt_frame and opt_frame != self.base_frame:
                optical_tf = lookup_tf(
                    self.tf_buffer, self.base_frame, opt_frame, None)
                if optical_tf is None:
                    self.get_logger().warn(
                        f'No TF from optical frame {opt_frame} to '
                        f'{self.base_frame}; ground-plane projection '
                        f'disabled this frame for cam[{cam_idx}].',
                        throttle_duration_sec=2.0)

        if not self._use_model:
            free_pts, lane_pts, lane_components, da_mask, ll_mask = (
                self._detect_pc2_color_ground(
                    bgr, xyz, cloud_rgb, cam_tf, cloud_in_base_frame))
            self._finish_projected_frame(
                cam_idx, process_stamp, rgb_msg, bgr,
                da_mask, ll_mask, free_pts, lane_pts, lane_components)
            return

        # ── Run segmentation model (this camera's dedicated instance + stream) ──
        try:
            da_mask, ll_mask = self._models[cam_idx].infer(bgr)
        except Exception as e:
            model_name = 'UFLDv2' if self._use_ufldv2 else 'YOLOPv2'
            self.get_logger().error(
                f'{model_name} inference error: {e}',
                throttle_duration_sec=2.0)
            return

        raw_da_area = int(np.count_nonzero(da_mask)) if da_mask is not None else 0
        raw_ll_area = int(np.count_nonzero(ll_mask)) if ll_mask is not None else 0
        model_lane_counts = ''
        if self._use_ufldv2:
            point_counts = getattr(
                self._models[cam_idx], 'last_lane_point_counts', [])
            model_lane_counts = ','.join(str(v) for v in point_counts)

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
        if self._use_yolo or self.ufldv2_refine_lane_mask:
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

        if self._use_ufldv2 and self.ufldv2_paint_refine_enabled:
            ll_mask = self._refine_ufldv2_lane_mask_with_pc2_paint(
                bgr, ll_mask, xyz, cloud_rgb, cam_tf, cloud_in_base_frame)

        # ── Depth-based obstacle masking ──
        # Zeros out pixels where 3-D base_link height lands in the obstacle
        # band — suppresses YOLOPv2 hallucinations over barrel/cone geometry.
        if self.obstacle_mask_enabled:
            obs_mask = self._build_obstacle_mask(xyz, cam_tf)
            if obs_mask is not None:
                # Resize obs_mask to match seg-head resolution if needed
                if obs_mask.shape != da_mask.shape:
                    obs_mask = cv2.resize(
                        obs_mask.astype(np.uint8),
                        (da_mask.shape[1], da_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
                da_mask[obs_mask] = 0
                ll_mask[obs_mask] = 0

        # ── Plane-based obstacle gate ──
        # Stronger than the fixed-Z obstacle band above: any cloud pixel
        # whose height above the RANSAC ground plane exceeds the threshold
        # is treated as an obstacle.  Adapts to terrain pitch and slow
        # camera-mount drift since the plane is re-fit every frame.
        if self.plane_height_gate_enabled and plane is not None:
            ph_mask = self._plane_height_obstacle_mask(
                xyz, cam_tf, cloud_in_base_frame, plane)
            if ph_mask is not None and np.any(ph_mask):
                if ph_mask.shape != da_mask.shape:
                    ph_mask = cv2.resize(
                        ph_mask.astype(np.uint8),
                        (da_mask.shape[1], da_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
                da_mask[ph_mask] = 0
                ll_mask[ph_mask] = 0

        # ── Lane mask dilation (pre-projection) ──
        # Thicken the raw lane mask so each detected lane pixel projects
        # to a small neighbourhood instead of a single ray.  This fills
        # small gaps in the persistent map without changing the model.
        if self._ll_dilate_kernel is not None and ll_mask is not None and ll_mask.size:
            ll_mask = cv2.dilate(
                ll_mask.astype(np.uint8), self._ll_dilate_kernel, iterations=1)

        post_da_area = int(np.count_nonzero(da_mask)) if da_mask is not None else 0
        post_ll_area = int(np.count_nonzero(ll_mask)) if ll_mask is not None else 0

        # ── Project masks into base_link ──
        free_pts = self._project_mask_points(
            da_mask, xyz, cam_tf,
            stride=self.da_subsample_px,
            cam_idx=cam_idx, optical_tf=optical_tf, plane=plane)
        lane_pts, lane_components = self._project_lane_mask(
            ll_mask, xyz, cam_tf,
            cam_idx=cam_idx, optical_tf=optical_tf, plane=plane)

        mask_stats = (
            f'mask_free_px={post_da_area} mask_lane_px={post_ll_area}'
        )
        if self._use_ufldv2:
            mask_stats += (
                f' raw_mask_free_px={raw_da_area} raw_mask_lane_px={raw_ll_area}'
                f' model_lane_points=[{model_lane_counts}]'
            )

        if (
            self._use_ufldv2
            and self.model_pc2_fallback_enabled
            and (free_pts.shape[0] == 0 or lane_pts.shape[0] == 0)
        ):
            fallback_free, fallback_lane, fallback_components, fallback_da, fallback_ll = (
                self._detect_pc2_color_ground(
                    bgr, xyz, cloud_rgb, cam_tf, cloud_in_base_frame))
            use_fallback_free = free_pts.shape[0] == 0 and fallback_free.shape[0] > 0
            use_fallback_lane = lane_pts.shape[0] == 0 and fallback_lane.shape[0] > 0
            if use_fallback_free or use_fallback_lane:
                used_parts = []
                if use_fallback_free:
                    free_pts = fallback_free
                    da_mask = fallback_da
                    used_parts.append('free')
                if use_fallback_lane:
                    lane_pts = fallback_lane
                    lane_components = fallback_components
                    ll_mask = fallback_ll
                    used_parts.append('lane')
                self.get_logger().warn(
                    f'UFLDv2 missing {"/".join(used_parts)} projection for cam[{cam_idx}] '
                    f'(mask_free_px={post_da_area}, mask_lane_px={post_ll_area}, '
                    f'model_lane_points=[{model_lane_counts}]); using PC2 '
                    f'color-ground fallback free={fallback_free.shape[0]} '
                    f'lane={fallback_lane.shape[0]} '
                    f'components={len(fallback_components)}.',
                    throttle_duration_sec=2.0)
                mask_stats = (
                    f'mask_free_px={int(np.count_nonzero(da_mask))} '
                    f'mask_lane_px={int(np.count_nonzero(ll_mask))} '
                    f'fallback=pc2_color_ground:{"/".join(used_parts)} '
                    f'ufldv2_mask_free_px={post_da_area} '
                    f'ufldv2_mask_lane_px={post_ll_area} '
                    f'ufldv2_model_lane_points=[{model_lane_counts}]'
                )
            elif free_pts.shape[0] == 0 and lane_pts.shape[0] == 0:
                self._log_empty_model_projection(
                    cam_idx, cam_frame, xyz, cam_tf, cloud_in_base_frame,
                    raw_da_area, raw_ll_area, post_da_area, post_ll_area,
                    model_lane_counts, fallback_attempted=True)
        elif free_pts.shape[0] == 0 and lane_pts.shape[0] == 0:
            self._log_empty_model_projection(
                cam_idx, cam_frame, xyz, cam_tf, cloud_in_base_frame,
                raw_da_area, raw_ll_area, post_da_area, post_ll_area,
                model_lane_counts, fallback_attempted=False)

        self._finish_projected_frame(
            cam_idx, process_stamp, rgb_msg, bgr,
            da_mask, ll_mask, free_pts, lane_pts, lane_components,
            mask_stats=mask_stats)

    def _finish_projected_frame(
        self,
        cam_idx: int,
        process_stamp,
        rgb_msg: Image,
        bgr: np.ndarray,
        da_mask: np.ndarray,
        ll_mask: np.ndarray,
        free_pts: np.ndarray,
        lane_pts: np.ndarray,
        lane_components: Sequence[np.ndarray],
        mask_stats: str = '',
    ) -> None:
        # ── Publish overlay ──
        if self.publish_overlay and cam_idx in self.overlay_pubs:
            self._publish_overlay(cam_idx, bgr, da_mask, ll_mask, rgb_msg)

        # ── Cache per-camera state + fuse + publish ──
        # _build_grid is pure on its inputs (fused_free, fused_lane,
        # process_stamp) and does no shared-state I/O, so we run it
        # outside the lock to shorten the critical section.  All shared
        # state reads/writes (cam_state, latest_grid, persistent map,
        # temporal filter, grid publish) stay inside _state_lock.
        with self._state_lock:
            self._cam_state[cam_idx] = {
                'stamp':    process_stamp,
                'free':     free_pts,
                'lane':     lane_pts,
            }
            fused_free, fused_lane = self._fuse_points(process_stamp)
            has_points = (
                fused_free.shape[0] > 0 or fused_lane.shape[0] > 0)

        new_grid: Optional[OccupancyGrid] = None
        if has_points:
            new_grid = self._build_grid(
                fused_free, fused_lane, process_stamp)

        # Optional per-point confidence weights for the persistent map.
        # Sized & ordered to match the points actually written below.
        lane_weights: Optional[np.ndarray] = None
        lane_pts_for_persist = lane_pts
        if self.confidence_weighted_writes_enabled and lane_components:
            # Component-derived weights are aligned with concat(components),
            # which may re-order the points relative to ``lane_pts`` (when
            # curve-fit is off).  Use concat(components) as the source of
            # truth so weights[i] matches point[i].
            lane_pts_for_persist = np.concatenate(
                lane_components, axis=0).astype(np.float32, copy=False)
            lane_weights = self._component_weights(lane_components)

        with self._state_lock:
            if has_points and new_grid is not None:
                self.latest_grid = new_grid
                # Write only THIS camera's freshly-projected points to the
                # persistent map.  Neural models use the original capture stamp
                # to compensate for inference latency; PC2 color-ground mode is
                # cheap and should use the freshest odom pose.
                persist_stamp = rgb_msg.header.stamp if self._use_model else None
                self._update_persistent_map(
                    free_pts, lane_pts_for_persist, persist_stamp,
                    lane_weights=lane_weights)
                if self.local_from_persistent:
                    self._publish_local_costmap_from_persistent()
                else:
                    # Temporal filter lives in _publish_local_costmap_from_persistent
                    # for the persistent path; apply it here for the direct path so
                    # every published frame is filtered exactly once.
                    _h = self.latest_grid.info.height
                    _w = self.latest_grid.info.width
                    _gd = np.frombuffer(
                        bytes(self.latest_grid.data), dtype=np.int8,
                    ).reshape(_h, _w).copy()
                    _gd = self._apply_temporal_lane_filter(_gd)
                    self.latest_grid.data = array.array('b', _gd.tobytes())
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
            f'cam[{cam_idx}] mode={self.detection_mode} '
            f'free={free_pts.shape[0]} lane={lane_pts.shape[0]} '
            f'components={len(lane_components)} '
            f'active_cams={active_cam_count}'
            f'{" " + mask_stats if mask_stats else ""}',
            throttle_duration_sec=1.0)

    def _log_empty_model_projection(
        self,
        cam_idx: int,
        cam_frame: str,
        xyz: np.ndarray,
        cam_tf,
        cloud_in_base_frame: bool,
        raw_da_area: int,
        raw_ll_area: int,
        post_da_area: int,
        post_ll_area: int,
        model_lane_counts: str,
        fallback_attempted: bool,
    ) -> None:
        if not self.model_debug_stats:
            return

        total = int(xyz.shape[0] * xyz.shape[1]) if xyz.ndim >= 2 else int(xyz.size)
        if xyz.ndim >= 3 and xyz.shape[2] >= 3:
            finite = int(np.count_nonzero(
                np.isfinite(xyz[:, :, 0])
                & np.isfinite(xyz[:, :, 1])
                & np.isfinite(xyz[:, :, 2])))
        else:
            finite = 0
        tf_state = 'base_frame' if cloud_in_base_frame else 'ok' if cam_tf is not None else 'missing'
        fallback_text = 'pc2 fallback also empty' if fallback_attempted else 'pc2 fallback not attempted'
        self.get_logger().warn(
            f'{self.detection_mode} produced 0 projected points for cam[{cam_idx}]; '
            f'raw_mask_px free={raw_da_area} lane={raw_ll_area}, '
            f'post_mask_px free={post_da_area} lane={post_ll_area}, '
            f'model_lane_points=[{model_lane_counts}], '
            f'pc2_shape={xyz.shape[:2]}, pc2_finite={finite}/{total}, '
            f'pc2_frame={cam_frame or "<empty>"}, tf={tf_state}, '
            f'{fallback_text}.',
            throttle_duration_sec=2.0)

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
        if len(self._pixel_grid_cache) >= 8:
            # Evict oldest entry; dict preserves insertion order (Python 3.7+).
            self._pixel_grid_cache.pop(next(iter(self._pixel_grid_cache)))
        self._pixel_grid_cache[key] = (ug, vg)
        return ug, vg

    def _cloud_base_arrays(
        self, xyz: np.ndarray, cam_tf, cloud_in_base_frame: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return organized base_link forward/lateral/height arrays."""
        xc = xyz[:, :, 0]
        yc = xyz[:, :, 1]
        zc = xyz[:, :, 2]
        valid = np.isfinite(xc) & np.isfinite(yc) & np.isfinite(zc)

        rot, trans = self._cam_tf_components(cam_tf)
        if rot is None or trans is None:
            fwd = xc.astype(np.float32)
            lat = yc.astype(np.float32)
            if cloud_in_base_frame:
                height = zc.astype(np.float32)
            else:
                height = (self.camera_height_fallback_m + zc).astype(np.float32)
            return fwd, lat, height, valid

        with np.errstate(invalid='ignore'):
            fwd = (
                rot[0, 0] * xc + rot[0, 1] * yc + rot[0, 2] * zc
                + float(trans[0])
            ).astype(np.float32)
            lat = (
                rot[1, 0] * xc + rot[1, 1] * yc + rot[1, 2] * zc
                + float(trans[1])
            ).astype(np.float32)
            height = (
                rot[2, 0] * xc + rot[2, 1] * yc + rot[2, 2] * zc
                + float(trans[2])
            ).astype(np.float32)
        return fwd, lat, height, valid

    def _pc2_color_mask(
        self,
        cloud_rgb: Optional[np.ndarray],
        bgr: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Return white/yellow mask from PC2 RGB, falling back to RGB image."""
        h, w = target_shape
        if cloud_rgb is None:
            self.get_logger().warn(
                'PointCloud2 has no rgb/rgba field; using synchronized RGB image '
                'as color fallback.',
                throttle_duration_sec=5.0)
            color_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            color_rgb = cloud_rgb

        if color_rgb.shape[:2] != (h, w):
            color_rgb = cv2.resize(
                color_rgb.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            color_rgb = color_rgb.astype(np.uint8, copy=False)

        hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        white = (val >= self.pc2_white_v_min) & (sat <= self.pc2_white_s_max)
        if self.pc2_yellow_h_min <= self.pc2_yellow_h_max:
            yellow_h = (
                (hue >= self.pc2_yellow_h_min)
                & (hue <= self.pc2_yellow_h_max)
            )
        else:
            yellow_h = (
                (hue >= self.pc2_yellow_h_min)
                | (hue <= self.pc2_yellow_h_max)
            )
        yellow = (
            yellow_h
            & (sat >= self.pc2_yellow_s_min)
            & (val >= self.pc2_yellow_v_min)
        )
        return white | yellow

    def _pc2_obstacle_exclusion_mask(
        self,
        valid: np.ndarray,
        fwd: np.ndarray,
        height: np.ndarray,
    ) -> np.ndarray:
        """Return image-space obstacle silhouettes to remove from PC2 lanes/free."""
        if not self.pc2_obstacle_exclusion_enabled:
            return np.zeros_like(valid, dtype=bool)

        obs = (
            valid
            & (fwd > self.pc2_obstacle_depth_min_m)
            & (fwd < self.pc2_obstacle_depth_max_m)
            & (height > self.pc2_obstacle_z_min_m)
            & (height < self.pc2_obstacle_z_max_m)
        )
        if self.pc2_obstacle_dilation_px > 1 and np.any(obs):
            k = self.pc2_obstacle_dilation_px | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            obs = cv2.dilate(obs.astype(np.uint8), kernel).astype(bool)
        return obs

    def _refine_ufldv2_lane_mask_with_pc2_paint(
        self,
        bgr: np.ndarray,
        lane_mask: np.ndarray,
        xyz: np.ndarray,
        cloud_rgb: Optional[np.ndarray],
        cam_tf,
        cloud_in_base_frame: bool = False,
    ) -> np.ndarray:
        """Keep UFLDv2 lane pixels only where PC2/RGB sees lane paint."""
        if lane_mask is None or lane_mask.size == 0 or not np.any(lane_mask):
            return lane_mask

        target_shape = lane_mask.shape[:2]
        paint_gate = self._pc2_color_mask(cloud_rgb, bgr, target_shape)
        if self.ufldv2_paint_refine_dilation_px > 1 and np.any(paint_gate):
            k = self.ufldv2_paint_refine_dilation_px | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            paint_gate = cv2.dilate(
                paint_gate.astype(np.uint8), kernel, iterations=1).astype(bool)

        gate = paint_gate
        if self.ufldv2_paint_refine_ground_gate:
            fwd, _lat, height, valid = self._cloud_base_arrays(
                xyz, cam_tf, cloud_in_base_frame)
            ground_gate = (
                valid
                & (fwd > self.min_detection_depth_m)
                & (fwd < self.max_detection_depth_m)
                & (height >= self.pc2_lane_z_min_m)
                & (height <= self.pc2_lane_z_max_m)
            )
            obstacle_mask = self._pc2_obstacle_exclusion_mask(valid, fwd, height)
            if np.any(obstacle_mask):
                ground_gate &= ~obstacle_mask
            if ground_gate.shape != target_shape:
                ground_gate = cv2.resize(
                    ground_gate.astype(np.uint8),
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
            gate &= ground_gate

        refined = (lane_mask > 0) & gate
        if (
            self.ufldv2_paint_refine_min_component_px > 1
            and np.any(refined)
        ):
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                refined.astype(np.uint8), connectivity=8)
            keep = np.zeros(n_labels, dtype=bool)
            keep[1:] = (
                stats[1:, cv2.CC_STAT_AREA]
                >= self.ufldv2_paint_refine_min_component_px
            )
            refined = keep[labels]
        if not np.any(refined) and np.any(lane_mask):
            self.get_logger().warn(
                'UFLDv2 paint refinement removed all lane pixels; check RGB/PC2 '
                'color thresholds or set ufldv2_paint_refine_enabled=false.',
                throttle_duration_sec=2.0)
        return refined.astype(np.uint8)

    def _points_from_bool_mask(
        self,
        mask: np.ndarray,
        fwd: np.ndarray,
        lat: np.ndarray,
        stride: int,
        limit: int,
    ) -> np.ndarray:
        """Sample organized bool mask into ``(fwd, lat)`` point array."""
        if mask is None or mask.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        stride = max(1, int(stride))
        if stride > 1:
            ys_sub, xs_sub = np.nonzero(mask[::stride, ::stride])
            if ys_sub.size == 0:
                return np.empty((0, 2), dtype=np.float32)
            ys = ys_sub * stride
            xs = xs_sub * stride
        else:
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                return np.empty((0, 2), dtype=np.float32)

        if limit > 0 and ys.size > limit:
            keep_every = int(math.ceil(ys.size / limit))
            ys = ys[::keep_every]
            xs = xs[::keep_every]

        pts = np.stack([fwd[ys, xs], lat[ys, xs]], axis=1)
        finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
        return pts[finite].astype(np.float32, copy=False)

    def _pc2_lane_points_and_components(
        self,
        lane_mask: np.ndarray,
        fwd: np.ndarray,
        lat: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Filter lane mask components and return projected lane points."""
        if lane_mask is None or lane_mask.size == 0 or not np.any(lane_mask):
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, [], np.zeros_like(lane_mask, dtype=bool)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            lane_mask.astype(np.uint8), connectivity=8)
        if num <= 1:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, [], np.zeros_like(lane_mask, dtype=bool)

        keep_label = np.zeros(num, dtype=bool)
        keep_label[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_lane_component_px
        clean_mask = keep_label[labels]
        if not np.any(clean_mask):
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, [], clean_mask

        all_pts = self._points_from_bool_mask(
            clean_mask, fwd, lat, self.pc2_lane_subsample_px,
            self.max_points_per_frame)

        components: List[np.ndarray] = []
        for lbl in np.flatnonzero(keep_label):
            comp_pts = self._points_from_bool_mask(
                labels == lbl, fwd, lat, self.pc2_lane_subsample_px,
                self.max_points_per_frame)
            if comp_pts.shape[0] > 0:
                components.append(comp_pts)
        components.sort(key=lambda a: -float(a[:, 1].mean()))
        return all_pts, components, clean_mask

    def _pc2_corridor_free_mask(
        self,
        ground_mask: np.ndarray,
        lane_mask: np.ndarray,
        fwd: np.ndarray,
        lat: np.ndarray,
    ) -> np.ndarray:
        """Free space is non-lane ground laterally between lane boundaries."""
        free_mask = np.zeros_like(ground_mask, dtype=bool)
        ys_lane, xs_lane = np.nonzero(lane_mask)
        if ys_lane.size == 0:
            return free_mask

        min_d = float(self.min_detection_depth_m)
        max_d = float(self.max_detection_depth_m)
        bin_size = self.pc2_free_bin_size_m
        n_bins = max(1, int(math.ceil((max_d - min_d) / bin_size)))
        lane_fwd = fwd[ys_lane, xs_lane]
        lane_lat = lat[ys_lane, xs_lane]
        lane_bins = ((lane_fwd - min_d) / bin_size).astype(np.int32)
        in_bins = (lane_bins >= 0) & (lane_bins < n_bins)
        if not np.any(in_bins):
            return free_mask

        lane_bins = lane_bins[in_bins]
        lane_lat = lane_lat[in_bins]
        left = np.full(n_bins, np.nan, dtype=np.float32)
        right = np.full(n_bins, np.nan, dtype=np.float32)
        pct = float(np.clip(self.pc2_free_boundary_percentile, 0.0, 49.0))

        # Sort once so each bin is a contiguous slice — avoids O(n) boolean
        # mask per unique bin and makes percentile calls cache-friendly.
        _sort_order = np.argsort(lane_bins, kind='stable')
        _bins_sorted = lane_bins[_sort_order]
        _lat_sorted = lane_lat[_sort_order]
        _unique_bins, _bin_starts = np.unique(_bins_sorted, return_index=True)
        _bin_ends = np.append(_bin_starts[1:], _bins_sorted.size)
        for bin_idx, _s, _e in zip(
            _unique_bins.tolist(), _bin_starts.tolist(), _bin_ends.tolist()
        ):
            values = _lat_sorted[_s:_e]
            if values.size < self.pc2_free_min_lane_points_per_bin:
                continue
            lo = float(np.percentile(values, pct))
            hi = float(np.percentile(values, 100.0 - pct))
            width = hi - lo
            if (
                width >= self.pc2_free_min_lane_width_m
                and width <= self.pc2_free_max_lane_width_m
            ):
                right[bin_idx] = lo
                left[bin_idx] = hi
            elif self.pc2_free_single_boundary_enabled:
                boundary = float(np.median(values))
                if abs(boundary) < self.pc2_free_single_boundary_min_abs_lat_m:
                    continue
                nominal_width = float(np.clip(
                    self.pc2_free_nominal_lane_width_m,
                    self.pc2_free_min_lane_width_m,
                    self.pc2_free_max_lane_width_m))
                if boundary > 0.0:
                    left[bin_idx] = boundary
                    right[bin_idx] = boundary - nominal_width
                else:
                    right[bin_idx] = boundary
                    left[bin_idx] = boundary + nominal_width

        valid_bins = np.isfinite(left) & np.isfinite(right)
        observed = np.flatnonzero(valid_bins)
        if observed.size == 0:
            return free_mask

        all_bins = np.arange(n_bins, dtype=np.int32)
        left_interp = np.interp(all_bins, observed, left[observed]).astype(np.float32)
        right_interp = np.interp(all_bins, observed, right[observed]).astype(np.float32)

        max_gap_bins = max(0, int(math.ceil(self.pc2_free_max_gap_m / bin_size)))
        pos = np.searchsorted(observed, all_bins)
        dist_prev = np.full(n_bins, n_bins, dtype=np.int32)
        has_prev = pos > 0
        dist_prev[has_prev] = all_bins[has_prev] - observed[pos[has_prev] - 1]
        dist_next = np.full(n_bins, n_bins, dtype=np.int32)
        has_next = pos < observed.size
        dist_next[has_next] = observed[pos[has_next]] - all_bins[has_next]
        supported = np.minimum(dist_prev, dist_next) <= max_gap_bins

        candidate = ground_mask & ~lane_mask
        ys, xs = np.nonzero(candidate)
        if ys.size == 0:
            return free_mask
        point_fwd = fwd[ys, xs]
        point_lat = lat[ys, xs]
        point_bins = ((point_fwd - min_d) / bin_size).astype(np.int32)
        in_range = (point_bins >= 0) & (point_bins < n_bins)
        if not np.any(in_range):
            return free_mask

        ys = ys[in_range]
        xs = xs[in_range]
        point_lat = point_lat[in_range]
        point_bins = point_bins[in_range]
        margin = self.pc2_free_lane_margin_m
        inside = (
            supported[point_bins]
            & (point_lat > (right_interp[point_bins] + margin))
            & (point_lat < (left_interp[point_bins] - margin))
        )
        if np.any(inside):
            free_mask[ys[inside], xs[inside]] = True
        return free_mask

    def _detect_pc2_color_ground(
        self,
        bgr: np.ndarray,
        xyz: np.ndarray,
        cloud_rgb: Optional[np.ndarray],
        cam_tf,
        cloud_in_base_frame: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
        """Detect lanes/free space from cloud color and ground-plane geometry."""
        fwd, lat, height, valid = self._cloud_base_arrays(
            xyz, cam_tf, cloud_in_base_frame)
        ground_mask = (
            valid
            & (fwd > self.min_detection_depth_m)
            & (fwd < self.max_detection_depth_m)
            & (height >= self.pc2_ground_z_min_m)
            & (height <= self.pc2_ground_z_max_m)
        )

        obstacle_mask = self._pc2_obstacle_exclusion_mask(valid, fwd, height)
        if np.any(obstacle_mask):
            ground_mask &= ~obstacle_mask

        lane_ground_mask = (
            ground_mask
            & (height >= self.pc2_lane_z_min_m)
            & (height <= self.pc2_lane_z_max_m)
        )

        color_mask = self._pc2_color_mask(cloud_rgb, bgr, xyz.shape[:2])
        if np.any(obstacle_mask):
            color_mask &= ~obstacle_mask
        lane_mask = lane_ground_mask & color_mask
        if self._pc2_lane_close_kernel is not None and np.any(lane_mask):
            lane_mask = cv2.morphologyEx(
                lane_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                self._pc2_lane_close_kernel).astype(bool)
            lane_mask &= lane_ground_mask

        lane_pts, lane_components, lane_mask = self._pc2_lane_points_and_components(
            lane_mask, fwd, lat)
        free_mask = self._pc2_corridor_free_mask(
            ground_mask, lane_mask, fwd, lat)
        free_pts = self._points_from_bool_mask(
            free_mask, fwd, lat, self.pc2_free_subsample_px,
            self.max_points_per_frame)

        if lane_pts.shape[0] == 0 and np.any(color_mask & valid):
            color_valid = color_mask & valid
            color_ground = color_valid & ground_mask
            color_lane_ground = color_valid & lane_ground_mask
            heights = height[color_valid]
            forwards = fwd[color_valid]
            height_range = (
                f'height=[{float(np.nanmin(heights)):.2f}, '
                f'{float(np.nanmax(heights)):.2f}] m'
                if heights.size > 0 else 'height=[n/a]')
            fwd_range = (
                f'fwd=[{float(np.nanmin(forwards)):.2f}, '
                f'{float(np.nanmax(forwards)):.2f}] m'
                if forwards.size > 0 else 'fwd=[n/a]')
            self.get_logger().warn(
                f'PC2 color detector saw {int(np.count_nonzero(color_valid))} '
                f'finite white/yellow cloud pixels, but only '
                f'{int(np.count_nonzero(color_ground))} survived obstacle+ground/range '
                f'gates, {int(np.count_nonzero(color_lane_ground))} survived '
                f'lane-height gates, and none survived component filtering. '
                f'{height_range}, {fwd_range}',
                throttle_duration_sec=2.0)
        elif lane_pts.shape[0] > 0 and free_pts.shape[0] == 0:
            self.get_logger().warn(
                'PC2 color detector found lane points but no free corridor; '
                'check lane width, ground z, and boundary-bin parameters.',
                throttle_duration_sec=2.0)

        return (
            free_pts,
            lane_pts,
            lane_components,
            free_mask.astype(np.uint8),
            lane_mask.astype(np.uint8),
        )

    def _build_obstacle_mask(self, xyz: np.ndarray, cam_tf) -> Optional[np.ndarray]:
        """Return a bool mask (same H×W as xyz) where pixels are occupied by obstacles.

        Uses the ZED PointCloud2 XYZ array to project every pixel into base_link
        3-D space directly — no pinhole math or intrinsics scaling needed.  Any
        pixel whose base_link Z height lands between ``obstacle_z_min_m`` and
        ``obstacle_z_max_m`` is flagged as an obstacle and masked out of the
        segmentation heads.

        ZED publishes ``point_cloud/cloud_registered`` in the camera body
        frame (FLU: x=forward, y=left, z=up), not the optical frame.
        """
        xc = xyz[:, :, 0]
        yc = xyz[:, :, 1]
        zc = xyz[:, :, 2]

        # Validity gate — drop NaN/Inf points; range gate is applied in
        # base_link below where the forward axis is unambiguous.
        valid = np.isfinite(xc) & np.isfinite(yc) & np.isfinite(zc)

        if cam_tf is not None:
            t = cam_tf.transform.translation
            r = cam_tf.transform.rotation
            R = np.array([
                [1 - 2*(r.y*r.y + r.z*r.z),   2*(r.x*r.y - r.z*r.w),   2*(r.x*r.z + r.y*r.w)],
                [2*(r.x*r.y + r.z*r.w),   1 - 2*(r.x*r.x + r.z*r.z),   2*(r.y*r.z - r.x*r.w)],
                [2*(r.x*r.z - r.y*r.w),   2*(r.y*r.z + r.x*r.w),   1 - 2*(r.x*r.x + r.y*r.y)],
            ], dtype=np.float32)
            # Full transform — we need both forward (X) for the range gate
            # and Z for the height gate.  NaN points (no return) propagate
            # through the multiply; the ``valid`` mask drops them below.
            with np.errstate(invalid='ignore'):
                bx = (R[0, 0] * xc + R[0, 1] * yc + R[0, 2] * zc + float(t.x)).astype(np.float32)
                bz = (R[2, 0] * xc + R[2, 1] * yc + R[2, 2] * zc + float(t.z)).astype(np.float32)
        else:
            # Fallback assumes camera body frame ≈ base_link with a fixed
            # vertical offset.  zc is already "up" in FLU.
            bx = xc.astype(np.float32)
            bz = (self.camera_height_fallback_m + zc).astype(np.float32)

        # Obstacle: forward range gate + height gate, on valid points only.
        obs = (
            valid
            & (bx > self.obstacle_depth_min_m)
            & (bx < self.obstacle_depth_max_m)
            & (bz > self.obstacle_z_min_m)
            & (bz < self.obstacle_z_max_m)
        )

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
        # Nothing to do — skip the full-image copy.
        if self.chassis_mask_frac <= 0.0 and not self.roi_enabled:
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

    # ═══════════════════════════════════════════════════════════════════
    # Ground-plane RANSAC (optional, see ground_plane_projection_mode)
    # ═══════════════════════════════════════════════════════════════════

    def _detect_and_publish_ramp(
        self,
        cam_idx: int,
        xyz: np.ndarray,
        cam_tf,
        cloud_in_base_frame: bool,
    ) -> None:
        """Estimate forward terrain slope and publish ``/ramp/state``.

        Reuses the RANSAC ground-plane fit (in ``base_link``) to derive the
        slope angle ahead of the robot and the heading to the steepest-ascent
        line, so mission_planner can align the robot parallel to the IGVC ramp
        before climbing it.  Runs independently of
        ``ground_plane_projection_mode`` and is throttled to
        ``ramp_min_period_sec``.

        Published as geometry_msgs/Vector3Stamped on ``ramp_status_topic``:
          vector.x = slope angle ahead (degrees)
          vector.y = heading error to the fall line (radians, base_link)
          vector.z = detection confidence in [0, 1]
        """
        if self.ramp_pub is None:
            return

        now_mono = time.monotonic()
        if (self._ramp_last_pub_mono is not None
                and (now_mono - self._ramp_last_pub_mono) < self.ramp_min_period_sec):
            return
        self._ramp_last_pub_mono = now_mono

        plane = self._fit_ground_plane_base(
            cam_idx, xyz, cam_tf, cloud_in_base_frame)
        if plane is None:
            return
        n_hat, _d = plane
        nx, ny, nz = float(n_hat[0]), float(n_hat[1]), float(n_hat[2])

        # Slope = angle between the up-pointing plane normal and +z.
        horiz = math.hypot(nx, ny)
        slope_rad = math.atan2(horiz, max(abs(nz), 1e-6))
        slope_deg = math.degrees(slope_rad)

        # Steepest-ascent (fall-line) direction in base_link is (-nx, -ny);
        # this is the yaw the robot must turn to face straight up-slope.
        fall_line_yaw = math.atan2(-ny, -nx) if horiz > 1e-6 else 0.0

        # Confidence proxy: normal verticality (a degenerate near-horizontal
        # normal would have been rejected by the RANSAC z-normal gate).
        confidence = float(np.clip(nz, 0.0, 1.0))

        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.vector.x = slope_deg
        msg.vector.y = fall_line_yaw
        msg.vector.z = confidence
        self.ramp_pub.publish(msg)

    def _fit_ground_plane_base(
        self,
        cam_idx: int,
        xyz: np.ndarray,
        cam_tf,
        cloud_in_base_frame: bool,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """RANSAC-fit a ground plane in ``base_link`` from the PC2.

        Returns ``(n_hat, d)`` with ``n_hat·p + d = 0`` on the plane and
        ``n_hat`` pointing roughly +z (up).  EMA-smoothed across frames
        and cached per-camera so a single failed RANSAC pass keeps the
        last good plane available to downstream code.
        """
        fwd, lat, height, valid = self._cloud_base_arrays(
            xyz, cam_tf, cloud_in_base_frame)
        roi = (
            valid
            & (fwd >= self.ground_plane_roi_fwd_min_m)
            & (fwd <= self.ground_plane_roi_fwd_max_m)
            & (np.abs(lat) <= self.ground_plane_roi_lat_abs_m)
            & (np.abs(height) <= self.ground_plane_roi_z_abs_m)
        )
        ys, xs = np.nonzero(roi)
        n_total = int(ys.size)
        if n_total < self.ground_plane_min_inliers:
            return self._plane_by_cam.get(cam_idx)

        if n_total > self.ground_plane_max_sample_pts:
            step = int(math.ceil(n_total / self.ground_plane_max_sample_pts))
            ys = ys[::step]
            xs = xs[::step]

        pts = np.stack(
            [fwd[ys, xs], lat[ys, xs], height[ys, xs]],
            axis=1,
        ).astype(np.float32, copy=False)
        n = pts.shape[0]
        if n < 3:
            return self._plane_by_cam.get(cam_idx)

        rng = np.random.default_rng(0xA1CE)
        thresh = self.ground_plane_inlier_thresh_m
        best_count = 0
        best_n: Optional[np.ndarray] = None
        best_d: Optional[float] = None

        for _ in range(self.ground_plane_ransac_iters):
            idx = rng.choice(n, size=3, replace=False)
            p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
            nrm = np.cross(p1 - p0, p2 - p0)
            nl = float(np.linalg.norm(nrm))
            if nl < 1e-6:
                continue
            nrm = (nrm / nl).astype(np.float32)
            if nrm[2] < 0.0:
                nrm = -nrm
            if nrm[2] < self.ground_plane_min_z_normal:
                continue
            d = -float(np.dot(nrm, p0))
            dist = np.abs(pts @ nrm + d)
            count = int(np.count_nonzero(dist < thresh))
            if count > best_count:
                best_count = count
                best_n = nrm
                best_d = d

        if (
            best_n is None
            or best_d is None
            or best_count < self.ground_plane_min_inliers
        ):
            return self._plane_by_cam.get(cam_idx)

        # Least-squares refit on inliers (SVD of centred points).
        dist = np.abs(pts @ best_n + best_d)
        inliers = pts[dist < thresh]
        if inliers.shape[0] >= 3:
            centroid = inliers.mean(axis=0)
            _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
            refit_n = vh[-1].astype(np.float32)
            if refit_n[2] < 0.0:
                refit_n = -refit_n
            if refit_n[2] >= self.ground_plane_min_z_normal:
                best_n = refit_n
                best_d = -float(np.dot(best_n, centroid))

        prev = self._plane_by_cam.get(cam_idx)
        if prev is not None and self.ground_plane_ema_alpha < 1.0:
            a = self.ground_plane_ema_alpha
            blended_n = a * best_n + (1.0 - a) * prev[0]
            bln = float(np.linalg.norm(blended_n))
            if bln > 1e-6:
                best_n = (blended_n / bln).astype(np.float32)
                best_d = float(a * best_d + (1.0 - a) * prev[1])

        self._plane_by_cam[cam_idx] = (best_n, float(best_d))
        return self._plane_by_cam[cam_idx]

    def _ray_plane_intersect(
        self,
        ys_mask: np.ndarray,
        xs_mask: np.ndarray,
        mask_shape: Tuple[int, int],
        cam_idx: int,
        optical_tf,
        plane: Optional[Tuple[np.ndarray, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Intersect camera rays (built from K) with a base_link plane.

        Returns ``(pts (M,2) [fwd,lat], local_idx (M,))`` where
        ``local_idx`` indexes the input ``ys_mask/xs_mask`` arrays for the
        rays that produced a valid (in-front, in-range) intersection.
        """
        empty_pts = np.empty((0, 2), dtype=np.float32)
        empty_idx = np.empty((0,), dtype=np.int64)
        if (
            ys_mask.size == 0
            or plane is None
            or optical_tf is None
            or cam_idx not in self.K
        ):
            return empty_pts, empty_idx

        K = self.K[cam_idx]
        if K is None or K.size != 9:
            return empty_pts, empty_idx
        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            return empty_pts, empty_idx

        img_size = self.camera_info_size.get(cam_idx)
        if img_size is None:
            return empty_pts, empty_idx
        img_w, img_h = img_size
        h_m, w_m = mask_shape
        sx = float(img_w) / float(w_m) if w_m > 0 else 1.0
        sy = float(img_h) / float(h_m) if h_m > 0 else 1.0

        u = xs_mask.astype(np.float32) * sx
        v = ys_mask.astype(np.float32) * sy

        # Rays in the camera *optical* frame (x=right, y=down, z=forward).
        rx = (u - cx) / fx
        ry = (v - cy) / fy
        rz = np.ones_like(u)
        rays_cam = np.stack([rx, ry, rz], axis=0)  # (3, N)

        rot, trans = self._cam_tf_components(optical_tf)
        if rot is None or trans is None:
            return empty_pts, empty_idx
        rays_base = rot @ rays_cam                   # (3, N)
        origin = trans                               # (3,)

        n_hat, d_plane = plane
        n_hat = n_hat.astype(np.float32, copy=False)
        n_dot_dir = (n_hat @ rays_base).astype(np.float32)  # (N,)
        n_dot_o_plus_d = float(n_hat @ origin) + float(d_plane)

        valid_dir = np.abs(n_dot_dir) > 1e-6
        if not np.any(valid_dir):
            return empty_pts, empty_idx

        t = np.full(n_dot_dir.shape, np.nan, dtype=np.float32)
        t[valid_dir] = -n_dot_o_plus_d / n_dot_dir[valid_dir]

        pts_base = origin[:, None] + rays_base * t[None, :]
        fwd = pts_base[0]
        lat = pts_base[1]

        ok = (
            np.isfinite(fwd) & np.isfinite(lat) & (t > 0.0)
            & (fwd > self.min_detection_depth_m)
            & (fwd < self.max_detection_depth_m)
        )
        if not np.any(ok):
            return empty_pts, empty_idx
        pts = np.stack([fwd[ok], lat[ok]], axis=1).astype(np.float32, copy=False)
        idx_local = np.flatnonzero(ok).astype(np.int64, copy=False)
        return pts, idx_local

    def _plane_height_obstacle_mask(
        self,
        xyz: np.ndarray,
        cam_tf,
        cloud_in_base_frame: bool,
        plane: Optional[Tuple[np.ndarray, float]],
    ) -> Optional[np.ndarray]:
        """Bool mask of pixels whose 3-D base_link point lies above the plane.

        Returns ``None`` when the plane is unavailable.  Pixels with NaN
        cloud values are *not* flagged — they cannot be classified.
        """
        if plane is None or xyz is None or xyz.size == 0:
            return None
        fwd, lat, height, valid = self._cloud_base_arrays(
            xyz, cam_tf, cloud_in_base_frame)
        n_hat, d_plane = plane
        # Signed height above plane: n·p + d (positive = above for an
        # upward-pointing normal, which we enforce in the RANSAC step).
        with np.errstate(invalid='ignore'):
            sign_h = (
                n_hat[0] * fwd + n_hat[1] * lat + n_hat[2] * height + d_plane
            )
        obs = valid & np.isfinite(sign_h) & (sign_h > self.plane_height_gate_thresh_m)
        if not np.any(obs):
            return obs
        if self.plane_height_gate_dilate_px > 1:
            k = self.plane_height_gate_dilate_px | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            obs = cv2.dilate(obs.astype(np.uint8), kernel).astype(bool)
        return obs

    # ═══════════════════════════════════════════════════════════════════
    # Lane curve fitting (optional, see lane_curve_fit_enabled)
    # ═══════════════════════════════════════════════════════════════════

    def _fit_lane_curves(
        self, components: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Fit lat = poly(fwd) per component; replace with smoothed samples.

        Components with too few points or excessive fit residuals are
        kept as-is (so curve fit can only improve, never delete, lane
        data).  Returns ``(all_pts, new_components)``.
        """
        new_components: List[np.ndarray] = []
        for comp in components:
            if comp is None or comp.shape[0] < self.lane_curve_min_points:
                new_components.append(comp)
                continue
            fwd = comp[:, 0]
            lat = comp[:, 1]
            order = min(
                self.lane_curve_poly_order,
                max(1, comp.shape[0] - 1),
            )
            try:
                coeffs = np.polyfit(fwd, lat, order)
                pred = np.polyval(coeffs, fwd)
                residual = float(np.sqrt(np.mean((lat - pred) ** 2)))
            except (np.linalg.LinAlgError, ValueError):
                new_components.append(comp)
                continue
            if (
                self.lane_curve_max_residual_m > 0.0
                and residual > self.lane_curve_max_residual_m
            ):
                new_components.append(comp)
                continue
            f_lo = float(np.min(fwd))
            f_hi = float(np.max(fwd))
            if self.lane_curve_extend_to_robot:
                f_lo = float(self.min_detection_depth_m)
            if self.lane_curve_extend_forward_m > 0.0:
                f_hi = min(
                    float(self.max_detection_depth_m),
                    f_hi + self.lane_curve_extend_forward_m,
                )
            if f_hi <= f_lo:
                new_components.append(comp)
                continue
            n_samples = max(
                2,
                int(math.ceil((f_hi - f_lo) / self.lane_curve_sample_step_m)) + 1,
            )
            f_samples = np.linspace(f_lo, f_hi, n_samples, dtype=np.float32)
            l_samples = np.polyval(coeffs, f_samples).astype(np.float32)
            ok = (
                np.isfinite(f_samples) & np.isfinite(l_samples)
                & (f_samples > self.min_detection_depth_m)
                & (f_samples < self.max_detection_depth_m)
            )
            if not np.any(ok):
                new_components.append(comp)
                continue
            new_components.append(
                np.stack([f_samples[ok], l_samples[ok]], axis=1).astype(
                    np.float32, copy=False))

        new_components = [c for c in new_components if c is not None and c.shape[0] > 0]
        if not new_components:
            return np.empty((0, 2), dtype=np.float32), []
        all_pts = np.concatenate(new_components, axis=0).astype(np.float32, copy=False)
        return all_pts, new_components

    def _component_weights(
        self, components: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Per-point write weights derived from component size.

        Returns a float32 array aligned with ``np.concatenate(components)``.
        Each point inherits its parent component's weight in
        ``[confidence_min_weight, 1.0]``.
        """
        if not components:
            return np.empty((0,), dtype=np.float32)
        full = float(self.confidence_component_full_px)
        chunks: List[np.ndarray] = []
        for comp in components:
            if comp is None or comp.shape[0] == 0:
                continue
            w = float(np.clip(
                comp.shape[0] / max(1.0, full),
                self.confidence_min_weight, 1.0,
            ))
            chunks.append(np.full(comp.shape[0], w, dtype=np.float32))
        if not chunks:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def _project_pixel_indices(
        self,
        ys_mask: np.ndarray,
        xs_mask: np.ndarray,
        mask_shape: Tuple[int, int],
        xyz: np.ndarray,
        cam_tf,
        cam_idx: Optional[int] = None,
        optical_tf=None,
        plane: Optional[Tuple[np.ndarray, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised mask-pixel → ``base_link`` projection.

        When ``ground_plane_projection_mode`` is set, the optional
        ``cam_idx``, ``optical_tf`` and ``plane`` arguments enable a
        ray-plane back-projection that supplements (``'fallback'``) or
        replaces (``'force'``) the per-pixel PC2 lookup.

        Parameters
        ----------
        ys_mask, xs_mask
            Pixel coords in *mask* resolution.
        mask_shape
            ``(h, w)`` of the source mask, used to scale to xyz coords.
        xyz
            ``(H, W, 3) float32`` XYZ array from the ZED PointCloud2.
            Invalid/occluded points carry NaN; no pinhole math needed.

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

        mode = self.ground_plane_projection_mode
        plane_ok = (
            mode != 'off'
            and plane is not None
            and optical_tf is not None
            and cam_idx is not None
        )

        # ── Force mode: skip PC2 entirely, ray-plane only ─────────────
        if mode == 'force' and plane_ok:
            return self._ray_plane_intersect(
                ys_mask, xs_mask, mask_shape, cam_idx, optical_tf, plane)

        h_d, w_d = xyz.shape[:2]
        h_m, w_m = mask_shape
        sx = w_d / float(w_m) if w_m > 0 else 1.0
        sy = h_d / float(h_m) if h_m > 0 else 1.0

        # Mask → xyz coords.  Round to nearest int.
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

        # Direct XYZ lookup — no pinhole unprojection needed.
        # NOTE: the ZED publishes ``point_cloud/cloud_registered`` in the
        # camera *body* frame (FLU: x=forward, y=left, z=up), not the
        # optical frame.  We therefore can't gate on a single axis ahead
        # of the TF transform — instead we keep any finite point and
        # apply the forward-distance gate in ``base_link`` below.
        pts_xyz = xyz[yd, xd]                               # (N, 3)
        xc = pts_xyz[:, 0].astype(np.float32, copy=False)
        yc = pts_xyz[:, 1].astype(np.float32, copy=False)
        zc = pts_xyz[:, 2].astype(np.float32, copy=False)

        n_in = int(xc.size)
        gate = np.isfinite(xc) & np.isfinite(yc) & np.isfinite(zc)
        n_finite = int(np.count_nonzero(gate))
        if n_finite == 0:
            # If plane fallback is enabled, try to recover ALL pixels via
            # ray-plane instead of giving up.
            if mode == 'fallback' and plane_ok:
                return self._ray_plane_intersect(
                    ys_mask, xs_mask, mask_shape,
                    cam_idx, optical_tf, plane)
            self.get_logger().warn(
                f'_project_pixel_indices: all {n_in} sampled PC2 points are '
                f'NaN/Inf — projection produced 0 pts',
                throttle_duration_sec=2.0)
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0,), dtype=np.int64))

        idx1 = idx0[gate]
        xc = xc[gate]
        yc = yc[gate]
        zc = zc[gate]

        rot, trans = self._cam_tf_components(cam_tf)

        if rot is None or trans is None:
            # Fallback for missing TF: assume camera frame is FLU (matches
            # the ZED point_cloud publisher's default).  If the upstream
            # source happens to be in optical convention, the TF should
            # exist and the rotation matrix will fix it up.
            fwd = xc
            lat = yc
        else:
            # rot: (3, 3) float32; pts_cam: (3, N).
            pts_cam = np.stack([xc, yc, zc], axis=0)
            pts_base = rot @ pts_cam
            pts_base[0] += float(trans[0])
            pts_base[1] += float(trans[1])
            fwd = pts_base[0]
            lat = pts_base[1]

        forward_gate = (
            (fwd > self.min_detection_depth_m)
            & (fwd < self.max_detection_depth_m)
        )
        n_fwd = int(np.count_nonzero(forward_gate))
        if n_fwd == 0:
            # Help the user see why projection produced 0 points: report
            # the actual base_link forward range of the finite samples so
            # they can sanity-check min/max_detection_depth_m vs the data.
            if fwd.size > 0:
                self.get_logger().warn(
                    f'_project_pixel_indices: 0/{n_finite} finite points '
                    f'passed the forward gate '
                    f'[{self.min_detection_depth_m:.2f}, '
                    f'{self.max_detection_depth_m:.2f}] m. '
                    f'fwd range was [{float(fwd.min()):.2f}, '
                    f'{float(fwd.max()):.2f}] m',
                    throttle_duration_sec=2.0)
            if mode == 'fallback' and plane_ok:
                return self._ray_plane_intersect(
                    ys_mask, xs_mask, mask_shape,
                    cam_idx, optical_tf, plane)
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0,), dtype=np.int64))

        idx_final = idx1[forward_gate]
        pts = np.stack([fwd[forward_gate], lat[forward_gate]], axis=1).astype(
            np.float32, copy=False)

        # ── Fallback mode: fill NaN/dropped pixels via ray-plane ─────
        if mode == 'fallback' and plane_ok:
            # Indices the PC2 path lost (NaN or out-of-range).
            survived = np.zeros(ys_mask.shape[0], dtype=bool)
            survived[idx_final] = True
            missing = np.flatnonzero(~survived)
            if missing.size > 0:
                plane_pts, plane_local_idx = self._ray_plane_intersect(
                    ys_mask[missing], xs_mask[missing], mask_shape,
                    cam_idx, optical_tf, plane)
                if plane_pts.shape[0] > 0:
                    pts = np.concatenate([pts, plane_pts], axis=0)
                    idx_final = np.concatenate(
                        [idx_final, missing[plane_local_idx]], axis=0)
        return pts, idx_final

    def _project_mask_points(
        self,
        mask: np.ndarray,
        xyz: np.ndarray,
        cam_tf,
        stride: int,
        cam_idx: Optional[int] = None,
        optical_tf=None,
        plane: Optional[Tuple[np.ndarray, float]] = None,
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
            ys_mask, xs_mask, mask.shape[:2], xyz, cam_tf,
            cam_idx=cam_idx, optical_tf=optical_tf, plane=plane)
        return pts

    def _project_lane_mask(
        self,
        ll_mask: np.ndarray,
        xyz: np.ndarray,
        cam_tf,
        cam_idx: Optional[int] = None,
        optical_tf=None,
        plane: Optional[Tuple[np.ndarray, float]] = None,
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
            ys_all, xs_all, ll_mask.shape[:2], xyz, cam_tf,
            cam_idx=cam_idx, optical_tf=optical_tf, plane=plane)
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

        # Optional per-component polynomial smoothing / extrapolation.
        # Replaces both ``components`` and ``all_pts`` so downstream code
        # (markers, costmap, persistent map) sees the smoothed curves.
        if self.lane_curve_fit_enabled and components:
            all_pts, components = self._fit_lane_curves(components)
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

        # Temporal filtering is applied in the publish path (_finish_projected_frame
        # for the direct route, _publish_local_costmap_from_persistent for the
        # persistent route) so it runs exactly once per published frame.

        g.data = array.array('b', data.tobytes())
        return g

    def _apply_temporal_lane_filter(self, data: np.ndarray) -> np.ndarray:
        if not self.temporal_filter_enabled:
            return data
        if data.ndim != 2:
            return data

        # Must be called under _state_lock so the EMA state arrays are not
        # modified concurrently by the camera-callback and timer paths.
        ny, nx = data.shape

        # Gap reset: if the filter has not run for a while, the stored EMA
        # state is stale (robot has likely moved and the grid corresponds
        # to a different patch of the world).  Zero it so old evidence
        # cannot re-emerge through hysteresis.
        now_mono = time.monotonic()
        if (
            self.temporal_reset_gap_sec > 0.0
            and self._temporal_last_update_monotonic is not None
            and self._temporal_lane_score is not None
            and (now_mono - self._temporal_last_update_monotonic)
                > self.temporal_reset_gap_sec
        ):
            self._temporal_lane_score.fill(0.0)
            if self._temporal_lane_mask is not None:
                self._temporal_lane_mask.fill(False)
        self._temporal_last_update_monotonic = now_mono

        if (
            self._temporal_lane_score is None
            or self._temporal_lane_mask is None
            or self._temporal_lane_score.shape != (ny, nx)
        ):
            self._temporal_lane_score = np.zeros((ny, nx), dtype=np.float32)
            self._temporal_lane_mask = np.zeros((ny, nx), dtype=bool)

        score = self._temporal_lane_score
        lane_mask = self._temporal_lane_mask

        lane_obs = (data == 100)
        free_obs = (data == 0)
        unknown_obs = ~(lane_obs | free_obs)

        if np.any(lane_obs):
            score[lane_obs] += self.temporal_lane_rise_alpha * (1.0 - score[lane_obs])
        if np.any(free_obs):
            score[free_obs] *= (1.0 - self.temporal_lane_fall_alpha)
        if np.any(unknown_obs):
            score[unknown_obs] *= (1.0 - self.temporal_lane_unknown_decay)

        np.clip(score, 0.0, 1.0, out=score)

        lane_mask[score >= self.temporal_lane_on_threshold] = True
        lane_mask[score <= self.temporal_lane_off_threshold] = False

        filtered = data.copy()
        filtered[filtered == 100] = -1
        filtered[lane_mask] = 100
        return filtered

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
        if da_mask.shape[:2] != bgr.shape[:2]:
            da_mask = cv2.resize(
                da_mask.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        if ll_mask.shape[:2] != bgr.shape[:2]:
            ll_mask = cv2.resize(
                ll_mask.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST)
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
            m.points = [Point(x=float(fwd), y=float(lat), z=0.0) for fwd, lat in pts]
            arr.markers.append(m)

        self.marker_pub.publish(arr)

    # ═══════════════════════════════════════════════════════════════════
    # Misc
    # ═══════════════════════════════════════════════════════════════════

    def _republish_grid(self) -> None:
        # Must hold _state_lock: _publish_local_costmap_from_persistent reads
        # _phits/_pfree which are written by _update_persistent_map (also
        # under _state_lock), and _apply_temporal_lane_filter mutates shared
        # EMA state that is protected by the same lock.
        with self._state_lock:
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

        # Apply temporal filter after inflation (and regardless of whether
        # inflation is enabled) so the filter always runs once per published
        # frame for the persistent path.  Must be called under _state_lock.
        local_data = self._apply_temporal_lane_filter(local_data)

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
