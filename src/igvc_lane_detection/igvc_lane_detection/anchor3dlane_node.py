"""ROS 2 lane-detection node backed by Anchor3DLane++ (CVPR 2023 / TPAMI 2024).

Replaces the per-pixel depth-projection pipeline of ``LaneSegmentationNode``
with a model that produces 3D lane polylines **directly** — no depth image
subscription required.  Each detected lane is a list of
``(x, y, z)`` points in the camera optical frame; a single TF lookup
transforms them into base_link ``(fwd, lat)`` pairs that feed into the
same persistent-map and local-costmap machinery already proven with YOLOPv2.

Output topics are intentionally **identical** to ``lane_segmentation_node``:
  /lane_costmap    — rolling OccupancyGrid (lethal lane cells, free corridor)
  /lane_map        — persistent OccupancyGrid in ``odom``
  /lane_segmentation/lanes  — MarkerArray (one POINTS marker per polyline)

Prerequisites
-------------
See ``anchor3dlane_infer.py`` for the full setup guide.  In short:

    git clone -b anchor3dlane++ https://github.com/tusen-ai/Anchor3DLane
    pip install mmcv-full
    cd Anchor3DLane && python setup.py develop
    cd mmseg/models/utils/ops && sh make.sh

    # R-18 camera-only weights (360×480, OpenLane F1=57.9):
    huggingface-cli download nowherespyfly/anchor3dlane \\
        anchor3dlane_plusplus_r18_360x480.pth \\
        --local-dir ~/anchor3dlane_weights/
"""

from __future__ import annotations

import array
import math
import threading
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from .anchor3dlane_infer import Anchor3DLane
from .projection_utils import lookup_tf, yaw_from_quat

# ── Debug palette ─────────────────────────────────────────────────────────
_LANE_COLORS: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 0.9, 0.9),
    (0.9, 0.4, 0.1),
    (0.1, 0.9, 0.3),
    (0.9, 0.1, 0.6),
    (0.9, 0.9, 0.1),
    (0.4, 0.3, 0.9),
)


def _quat_to_rot(q) -> np.ndarray:
    """geometry_msgs Quaternion → 3×3 float32 rotation matrix."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [    2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


class Anchor3DLaneNode(Node):
    """ROS 2 wrapper around Anchor3DLane++ for IGVC lane detection."""

    def __init__(self) -> None:
        super().__init__('anchor3dlane_node')
        self.bridge = CvBridge()

        # Per-camera intrinsics (populated by CameraInfo callbacks)
        self.K: dict = {}                   # cam_idx → (3,3) float32
        self._cam_frame: dict = {}          # cam_idx → TF frame string

        # ── Parameter helper ─────────────────────────────────────────
        def p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        # ── Frames / costmap geometry ─────────────────────────────────
        self.base_frame          = p('base_frame',             'base_link')
        self.occupancy_grid_frame = p('occupancy_grid_frame', self.base_frame)
        self.grid_res            = p('grid_resolution',         0.05)
        self.grid_width_m        = p('grid_width',             10.0)
        self.grid_height_m       = p('grid_height',            10.0)

        # ── Misc subscriber / callback settings ──────────────────────
        self.max_frame_age_sec   = float(p('max_frame_age_sec',   1.5))
        self.keep_last_grid      = p('keep_last_grid_on_miss',    True)

        # ── Depth range filter (applied after TF to drop far/implausible pts) ──
        self.min_fwd_m           = float(p('min_fwd_m',  0.3))
        self.max_fwd_m           = float(p('max_fwd_m', 30.0))
        self.max_lat_m           = float(p('max_lat_m', 12.0))

        # ── Persistent map ────────────────────────────────────────────
        self.persist_frame       = p('persistent_map_frame',    'odom')
        self.persist_res         = p('persistent_map_resolution', 0.20)
        self.persist_size_m      = p('persistent_map_size_m',  100.0)
        self.persist_decay       = p('persistent_map_decay',     0.998)
        self.persist_hit_w       = p('persistent_hit_weight',    5.0)
        self.persist_free_hit_w  = p('persistent_free_hit_weight', 0.0)
        self.persist_threshold   = p('persistent_threshold',     6.0)
        self.persist_free_threshold = p('persistent_free_threshold', 3.0)
        self.persist_max         = p('persistent_max_value',   200.0)
        self.persist_pub_hz      = p('persistent_publish_hz',    2.0)
        self.persist_clear_radius = p('persistent_clear_radius_m', 0.8)
        self.persist_pub_clear   = bool(p('persistent_publish_clear_robot', False))
        self.persist_skip_nosubs = bool(
            p('persistent_skip_publish_without_subscribers', True))
        self.persist_pose_source = p('persistent_pose_source',  'tf')
        self.odom_topic          = p('odom_topic', '/front_zed_camera_x/zed_node/odom')
        self.local_from_persist  = bool(p('local_costmap_from_persistent', True))
        self.local_pub_hz        = float(p('local_costmap_publish_hz', 10.0))
        self.local_back_buf_m    = max(0.0, float(p('local_back_nogo_buffer_m', 0.30)))
        self.local_lane_infl_m   = float(p('local_lane_inflation_m', 0.0))
        self.min_pose_change_m   = float(p('min_pose_change_m',   0.05))
        self.min_pose_change_rad = float(p('min_pose_change_rad', 0.02))
        self.max_yaw_rate_persist = float(
            p('max_yaw_rate_for_persist_update_rad_s', 0.6))

        # ── Lane marker topic ─────────────────────────────────────────
        self.lane_marker_topic   = p('lane_marker_topic', '/lane_segmentation/lanes')
        self.publish_overlay     = bool(p('publish_overlay', True))

        # ── Model parameters ──────────────────────────────────────────
        self.config_path         = p('config_path',    '')
        self.checkpoint_path     = p('checkpoint_path', '')
        self.anchor3dlane_root   = p('anchor3dlane_root', '')
        self.model_device        = p('model_device',   'cuda:0')
        self.model_half          = bool(p('model_half', True))
        self.input_h             = int(p('input_h', 360))
        self.input_w             = int(p('input_w', 480))
        self.score_threshold     = float(p('score_threshold', 0.4))

        if not self.config_path or not self.checkpoint_path:
            self.get_logger().error(
                "Parameters 'config_path' and 'checkpoint_path' must be set. "
                "See anchor3dlane_infer.py for setup instructions.")
            raise RuntimeError("anchor3dlane_node: config_path / checkpoint_path not set.")

        # ── Camera topics ─────────────────────────────────────────────
        num_req           = p('num_cameras',         1)
        cam_topics        = p('camera_topics',       ['/camera/image_raw'])
        info_topics       = p('camera_info_topics',  ['/camera/camera_info'])
        cam_frame_params  = p('camera_frames',       [''])  # optional: TF frames

        _num_cams = min(num_req, len(cam_topics), len(info_topics))

        # ── Load model ────────────────────────────────────────────────
        import os
        self.get_logger().info(
            f"Loading Anchor3DLane++ from '{self.checkpoint_path}' "
            f"(device={self.model_device}, half={self.model_half}, "
            f"res={self.input_h}×{self.input_w})…")
        self._model = Anchor3DLane(
            config_path=os.path.expanduser(self.config_path),
            checkpoint_path=os.path.expanduser(self.checkpoint_path),
            device=self.model_device,
            input_h=self.input_h,
            input_w=self.input_w,
            score_threshold=self.score_threshold,
            half=self.model_half,
        )
        # Insert anchor3dlane_root into sys.path before loading
        if self.anchor3dlane_root:
            import sys
            root = os.path.expanduser(self.anchor3dlane_root)
            if root not in sys.path:
                sys.path.insert(0, root)
        self._model.load()
        if self._model.fallback_warning:
            self.get_logger().warn(self._model.fallback_warning)
        self.get_logger().info("Anchor3DLane++ loaded and ready.")

        # ── Shared state ──────────────────────────────────────────────
        self._state_lock = threading.Lock()
        self._got_frame  = False
        self._got_frame_by_cam = {i: False for i in range(_num_cams)}
        self._latest_odom: Optional[Odometry] = None
        self._last_persist_pose: Optional[Tuple] = None
        self._cam_topics = {i: str(cam_topics[i]) for i in range(_num_cams)}

        self._init_persistent_map()

        # ── TF ────────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Callback group ────────────────────────────────────────────
        self._cam_cb_group = ReentrantCallbackGroup()

        _latest_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Subscriptions ─────────────────────────────────────────────
        self._sync_handles: list = []
        self.overlay_pubs: dict = {}

        for i in range(_num_cams):
            self.create_subscription(
                CameraInfo, info_topics[i],
                lambda msg, idx=i: self._on_info(msg, idx), 10,
                callback_group=self._cam_cb_group)

            self.create_subscription(
                Image, str(cam_topics[i]),
                lambda msg, idx=i: self._on_rgb(msg, idx),
                _latest_qos,
                callback_group=self._cam_cb_group)

            # Override TF frame from explicit parameter list if provided
            if len(cam_frame_params) > i and cam_frame_params[i]:
                self._cam_frame[i] = str(cam_frame_params[i])

            if self.publish_overlay:
                self.overlay_pubs[i] = self.create_publisher(
                    Image, f'/anchor3dlane/cam{i}/overlay', 10)

            self.get_logger().info(
                f'Configured cam[{i}]: rgb={cam_topics[i]} info={info_topics[i]}')

        # ── Publishers ────────────────────────────────────────────────
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        self.grid_pub    = self.create_publisher(OccupancyGrid, '/lane_costmap', map_qos)
        self.persist_pub = self.create_publisher(OccupancyGrid, '/lane_map',     map_qos)
        self.marker_pub  = self.create_publisher(MarkerArray, self.lane_marker_topic, 10)

        self.latest_grid = self._empty_grid()
        self._last_persistent_stamp = None
        self._persistent_msg: Optional[OccupancyGrid] = None
        self._persistent_msg_dirty = True
        self._persistent_data_cache: Optional[np.ndarray] = None

        # ── Odom subscriber (for persistent map pose source) ──────────
        if self.persist_pose_source == 'odom' or self.local_from_persist:
            odom_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1)
            self.create_subscription(Odometry, self.odom_topic, self._on_odom, odom_qos)
            self.get_logger().info(
                f'Pose source: odom topic {self.odom_topic}')

        # ── Timers ────────────────────────────────────────────────────
        self.create_timer(1.0 / max(self.local_pub_hz,   0.1), self._republish_grid)
        self.create_timer(1.0 / max(self.persist_pub_hz, 0.1), self._publish_persistent_map)
        self.create_timer(2.0, self._watchdog)

    # ═══════════════════════════════════════════════════════════════════
    # Camera info
    # ═══════════════════════════════════════════════════════════════════

    def _on_info(self, msg: CameraInfo, idx: int) -> None:
        self.K[idx] = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        # Use the header frame_id from CameraInfo as a fallback TF frame
        if idx not in self._cam_frame and msg.header.frame_id:
            self._cam_frame[idx] = msg.header.frame_id
        self.get_logger().info(
            f'Camera[{idx}] intrinsics received (frame={self._cam_frame.get(idx,"?")})',
            once=True)

    # ═══════════════════════════════════════════════════════════════════
    # Main callback
    # ═══════════════════════════════════════════════════════════════════

    def _on_rgb(self, rgb_msg: Image, cam_idx: int) -> None:
        with self._state_lock:
            self._got_frame = True
            self._got_frame_by_cam[cam_idx] = True

        # Age gate
        now = self.get_clock().now()
        frame_age = (now - Time.from_msg(rgb_msg.header.stamp)).nanoseconds / 1e9
        if self.max_frame_age_sec > 0.0 and frame_age > self.max_frame_age_sec:
            self.get_logger().warn(
                f'Dropping stale frame from cam[{cam_idx}] '
                f'(age={frame_age:.2f}s > {self.max_frame_age_sec:.2f}s).',
                throttle_duration_sec=2.0)
            return

        # Wait for intrinsics
        K = self.K.get(cam_idx)
        if K is None:
            self.get_logger().warn(
                f'cam[{cam_idx}]: waiting for CameraInfo…',
                throttle_duration_sec=5.0)
            return

        # Decode image
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        except Exception as exc:
            self.get_logger().error(f'cam[{cam_idx}] decode error: {exc}')
            return

        # Run 3D lane detection
        try:
            lanes_cam = self._model.infer(bgr, K)
        except Exception as exc:
            self.get_logger().error(
                f'cam[{cam_idx}] inference error: {exc}',
                throttle_duration_sec=2.0)
            return

        stamp = rgb_msg.header.stamp

        # Get camera → base_link TF
        cam_frame = self._cam_frame.get(cam_idx) or rgb_msg.header.frame_id
        cam_tf = None
        if cam_frame and cam_frame != self.base_frame:
            cam_tf = lookup_tf(self.tf_buffer, self.base_frame, cam_frame, None)
            if cam_tf is None:
                self.get_logger().warn(
                    f'No TF from {cam_frame} to {self.base_frame}; '
                    'using approximate pinhole fallback.',
                    throttle_duration_sec=5.0)

        # Project 3D polylines from camera optical frame → base_link (fwd, lat)
        lane_pts = self._lanes_to_base_link(lanes_cam, cam_tf)

        # Optionally publish a debug overlay
        if self.publish_overlay and cam_idx in self.overlay_pubs:
            self._publish_overlay(cam_idx, bgr, lanes_cam, K, rgb_msg)

        # Publish lane markers for RViz
        self._publish_lane_markers(lanes_cam, cam_tf, stamp)

        with self._state_lock:
            if lane_pts.shape[0] > 0:
                grid = self._build_grid(None, lane_pts, stamp)
                self.latest_grid = grid
                self._update_persistent_map(None, lane_pts, stamp)

                if self.local_from_persist:
                    self._publish_local_costmap_from_persistent()
                else:
                    self.grid_pub.publish(self.latest_grid)
            elif self.keep_last_grid:
                self._republish_grid()
            else:
                self.latest_grid = self._empty_grid(stamp)
                self.grid_pub.publish(self.latest_grid)

        self.get_logger().info(
            f'cam[{cam_idx}] lanes={len(lanes_cam)} '
            f'pts={lane_pts.shape[0]}',
            throttle_duration_sec=1.0)

    # ═══════════════════════════════════════════════════════════════════
    # Camera-frame 3D lanes → base_link (fwd, lat)
    # ═══════════════════════════════════════════════════════════════════

    def _lanes_to_base_link(
        self,
        lanes: List[List[Tuple[float, float, float]]],
        cam_tf,
    ) -> np.ndarray:
        """Convert anchor3dlane camera-optical-frame polylines to base_link.

        anchor3dlane_infer.py outputs (x, y, z) in camera optical frame:
            x = right, y = down, z = forward (standard optical convention)

        base_link convention:
            x = forward, y = left, z = up

        Returns
        -------
        np.ndarray  shape (N, 2)  columns [fwd, lat] in base_link meters.
        """
        if not lanes:
            return np.empty((0, 2), dtype=np.float32)

        # Flatten all lane points into one (N, 3) array
        all_pts = []
        for lane in lanes:
            for pt in lane:
                all_pts.append(pt)

        if not all_pts:
            return np.empty((0, 2), dtype=np.float32)

        pts_cam = np.array(all_pts, dtype=np.float32)   # (N, 3)

        if cam_tf is not None:
            R = _quat_to_rot(cam_tf.transform.rotation)
            t_vec = cam_tf.transform.translation
            t = np.array([t_vec.x, t_vec.y, t_vec.z], dtype=np.float32)
            pts_base = pts_cam @ R.T + t                 # (N, 3)
        else:
            # Approximate: camera optical → base_link for forward-facing camera.
            # Camera optical: x=right, y=down, z=forward
            # base_link:      x=forward, y=left, z=up
            # fwd = z_cam, lat = -x_cam, up = -y_cam
            pts_base = np.column_stack([
                pts_cam[:, 2],    # x_base = z_cam  (forward)
                -pts_cam[:, 0],   # y_base = -x_cam (left)
                -pts_cam[:, 1],   # z_base = -y_cam (up)
            ]).astype(np.float32)

        fwd = pts_base[:, 0]
        lat = pts_base[:, 1]

        # Discard points behind the robot, too far, or too far to the side
        mask = (
            (fwd > self.min_fwd_m) &
            (fwd < self.max_fwd_m) &
            (np.abs(lat) < self.max_lat_m)
        )
        if not np.any(mask):
            return np.empty((0, 2), dtype=np.float32)

        return np.column_stack([fwd[mask], lat[mask]]).astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════
    # Persistent map  (mirrored from LaneSegmentationNode)
    # ═══════════════════════════════════════════════════════════════════

    def _init_persistent_map(self) -> None:
        n = int(self.persist_size_m / self.persist_res)
        self._pN = n
        self._phits = np.zeros((n, n), dtype=np.float32)
        self._pfree = np.zeros((n, n), dtype=np.float32)
        self._persistent_dirty = True
        half = self.persist_size_m / 2.0
        self._p_ox = -half
        self._p_oy = -half
        self.get_logger().info(
            f'Persistent map: {n}×{n} cells @ {self.persist_res} m/cell '
            f'({self.persist_size_m} m square) in frame "{self.persist_frame}"')

    def _world_to_pgrid(self, wx: float, wy: float) -> Tuple[int, int]:
        col = int((wx - self._p_ox) / self.persist_res)
        row = int((wy - self._p_oy) / self.persist_res)
        return col, row

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _persistent_pose(self, stamp=None):
        if stamp is not None:
            transform = lookup_tf(
                self.tf_buffer, self.persist_frame, self.base_frame, stamp)
            if transform is not None:
                q  = transform.transform.rotation
                tr = transform.transform.translation
                yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
                return (tr.x, tr.y, tr.z, yaw, transform.transform.rotation)

        if self.persist_pose_source == 'odom' and self._latest_odom is not None:
            p = self._latest_odom.pose.pose
            q = p.orientation
            yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
            return (p.position.x, p.position.y, p.position.z, yaw, q)

        transform = lookup_tf(
            self.tf_buffer, self.persist_frame, self.base_frame, None)
        if transform is None:
            return None
        q  = transform.transform.rotation
        tr = transform.transform.translation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        return (tr.x, tr.y, tr.z, yaw, q)

    def _update_persistent_map(
        self,
        free_pts,
        lane_pts,
        stamp,
    ) -> None:
        self._last_persistent_stamp = stamp
        free_empty = free_pts is None or (hasattr(free_pts, '__len__') and len(free_pts) == 0)
        lane_empty = lane_pts is None or (hasattr(lane_pts, '__len__') and len(lane_pts) == 0)
        if free_empty and lane_empty:
            self._phits *= self.persist_decay
            self._pfree *= self.persist_decay
            self._persistent_dirty = True
            self._persistent_msg_dirty = True
            return

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
                return

        pose = self._persistent_pose(stamp)
        if pose is None:
            return

        tx, ty, _tz, yaw, _ = pose

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
            free_pts, tx, ty, cos_y, sin_y, self._pfree, self.persist_free_hit_w)
        self._stamp_persistent_points(
            lane_pts, tx, ty, cos_y, sin_y, self._phits, self.persist_hit_w)
        self._persistent_dirty = True
        self._persistent_msg_dirty = True

    def _stamp_persistent_points(
        self,
        points,
        tx: float, ty: float, cos_y: float, sin_y: float,
        grid: np.ndarray,
        weight: float,
    ) -> None:
        if points is None or len(points) == 0:
            return
        pts = np.asarray(points, dtype=np.float32)
        fwd = pts[:, 0]
        lat = pts[:, 1]
        cols = (
            (tx + cos_y * fwd - sin_y * lat - self._p_ox) / self.persist_res
        ).astype(np.int32)
        rows = (
            (ty + sin_y * fwd + cos_y * lat - self._p_oy) / self.persist_res
        ).astype(np.int32)
        valid = (cols >= 0) & (cols < self._pN) & (rows >= 0) & (rows < self._pN)
        if not np.any(valid):
            return
        rows, cols = rows[valid], cols[valid]
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
        if self.persist_skip_nosubs and self.persist_pub.get_subscription_count() == 0:
            return
        with self._state_lock:
            stamp = (self._last_persistent_stamp
                     or self.get_clock().now().to_msg())
            rebuild = (
                self._persistent_msg is None
                or self._persistent_msg_dirty
                or self.persist_pub_clear
            )
            if rebuild:
                n = self._pN
                g = OccupancyGrid()
                g.header.frame_id           = self.persist_frame
                g.info.resolution           = self.persist_res
                g.info.width                = n
                g.info.height               = n
                g.info.origin.position.x    = self._p_ox
                g.info.origin.position.y    = self._p_oy
                g.info.origin.orientation.w = 1.0
                data = self._persistent_grid_data(stamp, clear_robot=self.persist_pub_clear)
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
        tx, ty, *_ = pose
        col_c, row_c = self._world_to_pgrid(tx, ty)
        r_cells = max(1, int(np.ceil(self.persist_clear_radius / self.persist_res)))
        row_lo = max(0, row_c - r_cells)
        row_hi = min(self._pN, row_c + r_cells + 1)
        col_lo = max(0, col_c - r_cells)
        col_hi = min(self._pN, col_c + r_cells + 1)
        if row_lo >= row_hi or col_lo >= col_hi:
            return
        rows, cols = np.ogrid[row_lo:row_hi, col_lo:col_hi]
        mask = (rows - row_c) ** 2 + (cols - col_c) ** 2 <= r_cells ** 2
        data[row_lo:row_hi, col_lo:col_hi][mask] = 0

    # ═══════════════════════════════════════════════════════════════════
    # Local rolling costmap  (mirrored from LaneSegmentationNode)
    # ═══════════════════════════════════════════════════════════════════

    def _build_grid(self, free_pts, lane_pts, stamp) -> OccupancyGrid:
        g = self._empty_grid(stamp)
        nx, ny = g.info.width, g.info.height
        res = self.grid_res
        data = np.full((ny, nx), -1, dtype=np.int8)
        half_w = self.grid_width_m / 2.0

        def _stamp(pts, value: int) -> None:
            if pts is None:
                return
            arr = pts if isinstance(pts, np.ndarray) else np.asarray(pts, dtype=np.float32)
            if arr.size == 0:
                return
            cols = (arr[:, 0] / res).astype(np.int32)
            rows = ((arr[:, 1] + half_w) / res).astype(np.int32)
            ok = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
            if np.any(ok):
                data[rows[ok], cols[ok]] = np.int8(value)

        _stamp(free_pts, 0)
        _stamp(lane_pts, 100)

        # Corridor fill between detected lane boundaries
        center_row = ny // 2
        half_fill = max(4, int(round(1.2 / res)))
        for col in np.where(np.any(data == 100, axis=0))[0]:
            lane_rows = np.where(data[:, col] == 100)[0]
            right = lane_rows[lane_rows < center_row]
            left  = lane_rows[lane_rows > center_row]
            lo = int(right.max()) + 1 if len(right) > 0 else max(0, center_row - half_fill)
            hi = int(left.min())       if len(left)  > 0 else min(ny, center_row + half_fill)
            if lo < hi:
                seg = data[lo:hi, col]
                data[lo:hi, col] = np.where(seg == np.int8(-1), np.int8(0), seg)

        g.data = array.array('b', data.tobytes())
        return g

    def _empty_grid(self, stamp=None) -> OccupancyGrid:
        g = OccupancyGrid()
        g.header.stamp = (self.get_clock().now().to_msg()
                          if stamp is None else stamp)
        g.header.frame_id = self.occupancy_grid_frame
        nx = int(self.grid_height_m / self.grid_res)
        ny = int(self.grid_width_m  / self.grid_res)
        g.info.resolution = self.grid_res
        g.info.width      = nx
        g.info.height     = ny
        if self.occupancy_grid_frame == self.base_frame:
            g.info.origin.position.x    = 0.0
            g.info.origin.position.y    = -self.grid_width_m / 2.0
            g.info.origin.orientation.w = 1.0
        else:
            tf = lookup_tf(self.tf_buffer, self.occupancy_grid_frame,
                           self.base_frame, stamp)
            if tf is None:
                g.info.origin.position.x    = 0.0
                g.info.origin.position.y    = -self.grid_width_m / 2.0
                g.info.origin.orientation.w = 1.0
            else:
                q   = tf.transform.rotation
                yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
                tx  = tf.transform.translation.x
                ty  = tf.transform.translation.y
                g.info.origin.position.x    = tx + np.sin(yaw) * (self.grid_width_m / 2.0)
                g.info.origin.position.y    = ty - np.cos(yaw) * (self.grid_width_m / 2.0)
                g.info.origin.position.z    = tf.transform.translation.z
                g.info.origin.orientation   = q
        empty = np.full(nx * ny, -1, dtype=np.int8)
        g.data = array.array('b', empty.tobytes())
        return g

    def _republish_grid(self) -> None:
        if self.local_from_persist:
            if self._publish_local_costmap_from_persistent():
                return
        self.grid_pub.publish(self.latest_grid)

    def _publish_local_costmap_from_persistent(self) -> bool:
        pose = self._persistent_pose(None)
        if pose is None:
            return False

        tx, ty, _tz, yaw, _ = pose
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        if self.grid_res <= 0.0 or self.persist_res <= 0.0:
            return False

        nx = max(1, int(round(self.grid_height_m / self.grid_res)))
        ny = max(1, int(round(self.grid_width_m  / self.grid_res)))
        local_oy   = -0.5 * self.grid_width_m
        local_data = np.full((ny, nx), -1, dtype=np.int8)
        pdata = self._persistent_grid_data(None, clear_robot=False)

        col_idx = np.arange(nx, dtype=np.float32)
        row_idx = np.arange(ny, dtype=np.float32)
        local_x = col_idx[None, :] * self.grid_res
        local_y = local_oy + row_idx[:, None] * self.grid_res

        world_x = tx + cos_yaw * local_x - sin_yaw * local_y
        world_y = ty + sin_yaw * local_x + cos_yaw * local_y

        src_cols = np.rint((world_x - self._p_ox) / self.persist_res).astype(np.int32)
        src_rows = np.rint((world_y - self._p_oy) / self.persist_res).astype(np.int32)
        valid = ((src_cols >= 0) & (src_cols < self._pN) &
                 (src_rows >= 0) & (src_rows < self._pN))
        local_data[valid] = pdata[src_rows[valid], src_cols[valid]]

        if not np.any((local_data == 0) | (local_data == 100)):
            return False

        # Lane inflation
        if self.local_lane_infl_m > 0.0 and self.grid_res > 0.0:
            r_cells = int(math.ceil(self.local_lane_infl_m / self.grid_res))
            if r_cells > 0:
                k = 2 * r_cells + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                inflated = cv2.dilate(
                    (local_data == 100).astype(np.uint8), kernel, iterations=1)
                grow = (inflated > 0) & (local_data != 100)
                local_data[grow] = 100

        # Backward no-go buffer
        buf = int(math.ceil(self.local_back_buf_m / self.grid_res))
        if buf > 0:
            local_data[:, :min(buf, nx)] = 100

        local = OccupancyGrid()
        local.header.stamp           = self.get_clock().now().to_msg()
        local.header.frame_id        = self.base_frame
        local.info.resolution        = self.grid_res
        local.info.width             = nx
        local.info.height            = ny
        local.info.origin.position.x = 0.0
        local.info.origin.position.y = local_oy
        local.info.origin.orientation.w = 1.0
        local.data = array.array('b', local_data.tobytes())
        self.latest_grid = local
        self.grid_pub.publish(local)
        return True

    # ═══════════════════════════════════════════════════════════════════
    # Visualisation
    # ═══════════════════════════════════════════════════════════════════

    def _publish_overlay(
        self,
        idx: int,
        bgr: np.ndarray,
        lanes: List[List[Tuple[float, float, float]]],
        K: np.ndarray,
        rgb_msg: Image,
    ) -> None:
        """Project 3D polyline endpoints back onto the image and draw them."""
        ov = bgr.copy()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Scale intrinsics from original resolution to model input resolution
        sx = self._model.input_w / float(bgr.shape[1])
        sy = self._model.input_h / float(bgr.shape[0])

        for li, lane in enumerate(lanes):
            color_f = _LANE_COLORS[li % len(_LANE_COLORS)]
            color = (int(color_f[2] * 255), int(color_f[1] * 255), int(color_f[0] * 255))
            pts_2d = []
            for x_cam, y_cam, z_cam in lane:
                if z_cam <= 0:
                    continue
                # Project: u = fx*x/z + cx  (at original resolution)
                u = int(round((fx * x_cam / z_cam + cx)))
                v = int(round((fy * y_cam / z_cam + cy)))
                if 0 <= u < bgr.shape[1] and 0 <= v < bgr.shape[0]:
                    pts_2d.append((u, v))

            for j in range(len(pts_2d) - 1):
                cv2.line(ov, pts_2d[j], pts_2d[j + 1], color, 2, cv2.LINE_AA)
            for pt in pts_2d:
                cv2.circle(ov, pt, 3, color, -1)

        try:
            msg = self.bridge.cv2_to_imgmsg(ov, 'bgr8')
            msg.header = rgb_msg.header
            self.overlay_pubs[idx].publish(msg)
        except Exception as exc:
            self.get_logger().warn(f'Overlay error: {exc}', throttle_duration_sec=2.0)

    def _publish_lane_markers(
        self,
        lanes_cam: List[List[Tuple[float, float, float]]],
        cam_tf,
        stamp,
    ) -> None:
        if self.marker_pub.get_subscription_count() == 0:
            return
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.base_frame
        clear.header.stamp    = stamp
        clear.ns     = 'anchor3dlane'
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for li, lane in enumerate(lanes_cam):
            if not lane:
                continue
            m = Marker()
            m.header.frame_id = self.base_frame
            m.header.stamp    = stamp
            m.ns              = 'anchor3dlane'
            m.id              = li
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.06
            r, g, b = _LANE_COLORS[li % len(_LANE_COLORS)]
            m.color           = ColorRGBA(r=float(r), g=float(g), b=float(b), a=0.9)
            m.lifetime        = Duration(seconds=1).to_msg()

            for x_cam, y_cam, z_cam in lane:
                pt_cam = np.array([[x_cam, y_cam, z_cam]], dtype=np.float32)
                if cam_tf is not None:
                    R = _quat_to_rot(cam_tf.transform.rotation)
                    t_vec = cam_tf.transform.translation
                    t = np.array([t_vec.x, t_vec.y, t_vec.z], dtype=np.float32)
                    pt_base = (pt_cam @ R.T + t)[0]
                else:
                    pt_base = np.array(
                        [z_cam, -x_cam, -y_cam], dtype=np.float32)
                p = Point()
                p.x = float(pt_base[0])
                p.y = float(pt_base[1])
                p.z = float(pt_base[2])
                m.points.append(p)

            if m.points:
                arr.markers.append(m)

        self.marker_pub.publish(arr)

    # ═══════════════════════════════════════════════════════════════════
    # Watchdog
    # ═══════════════════════════════════════════════════════════════════

    def _watchdog(self) -> None:
        with self._state_lock:
            got_any = self._got_frame
            missing = [idx for idx, ok in self._got_frame_by_cam.items() if not ok]
            self._got_frame = False
            for idx in self._got_frame_by_cam:
                self._got_frame_by_cam[idx] = False

        if not got_any:
            self.get_logger().warn(
                'No RGB frames received.  Check camera topics and QoS.',
                throttle_duration_sec=5.0)
        elif missing:
            msg = ', '.join(
                f'cam[{idx}]={self._cam_topics.get(idx,"?")}' for idx in missing)
            self.get_logger().warn(
                f'No frames from: {msg}',
                throttle_duration_sec=5.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    from rclpy.executors import MultiThreadedExecutor
    node = Anchor3DLaneNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
