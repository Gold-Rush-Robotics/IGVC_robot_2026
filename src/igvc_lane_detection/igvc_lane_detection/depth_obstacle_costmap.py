"""depth_obstacle_costmap.py
===========================
Builds a persistent 2-D occupancy grid from the front ZED camera depth image
and publishes it as a ``nav_msgs/OccupancyGrid`` on ``/depth_obstacle_map``
(TRANSIENT_LOCAL, same format as ``/lidar_obstacle_map``).

Only depth pixels whose reconstructed 3-D point falls within a configurable
height band above the ground are treated as obstacle evidence.  Ground returns
(below ``min_height_m``) and sky / far-field noise (above ``max_height_m``)
are ignored.  Free-space rays are *not* cast (unlike the LiDAR node) because
ZED depth images contain many NaN / inf pixels that make free-space
reconstruction noisy; instead, evidence is allowed to decay naturally.

Parameters
----------
frame_id              (str,   'odom')                   Fixed frame for output grid.
depth_topic           (str,   '/front_zed_camera_x/zed_node/depth/depth_registered')
info_topic            (str,   '/front_zed_camera_x/zed_node/depth/camera_info')
output_topic          (str,   '/depth_obstacle_map')
resolution            (float, 0.10)  Cell size in meters.
width_m               (float, 60.0)
height_m              (float, 60.0)
origin_x              (float, -30.0)
origin_y              (float, -30.0)
min_depth_m           (float, 0.40)  Ignore closer depth returns (noisy).
max_depth_m           (float, 8.0)   Ignore farther depth returns.
min_height_m          (float, 0.10)  Min obstacle height above ground [m].
max_height_m          (float, 2.20)  Max obstacle height above ground [m].
stride                (int,   4)     Sample every Nth pixel in u and v.
hit_weight            (float, 2.0)   Evidence added per valid hit cell.
decay                 (float, 0.995) Per-frame multiplicative decay.
decay_every_n_frames  (int,   3)     Apply decay once every N depth frames.
hit_threshold         (float, 8.0)   Accumulated evidence for lethal.
min_points_per_cell   (int,   1)     Per-frame returns required in a grid cell.
min_component_cells   (int,   1)     Remove lethal blobs smaller than this.
max_value             (float, 200.0) Clamp for accumulator grid.
inflate_radius_m      (float, 0.20)  Inflate lethal cells at publish time.
publish_hz            (float, 5.0)   OccupancyGrid publish frequency.
tf_timeout_sec        (float, 0.10)  TF lookup timeout.
use_latest_tf         (bool,  True)  Use latest TF instead of depth stamp.
max_frame_age_sec     (float, 2.0)   Drop stale depth frames.
depth_frame_convention (str, 'auto') 'optical', 'flu', or 'auto'.
use_odom_pose         (bool,  True)  Stamp base-frame points using odom topic.
odom_topic            (str,   '/front_zed_camera_x/zed_node/odom')
camera_x/y/z_m        (float) Camera body frame origin in base_link.
camera_roll/pitch/yaw_rad (float) Camera body frame rotation in base_link.
"""

from __future__ import annotations

import array
import math
from typing import Optional

import cv2
import numpy as np

import rclpy
import rclpy.time
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover
    CvBridge = None  # type: ignore[assignment,misc]


class DepthObstacleCostmapNode(Node):

    @staticmethod
    def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        return np.array([
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ], dtype=np.float64)

    @staticmethod
    def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
        return math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz))

    @staticmethod
    def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
        rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
        rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return rot_z @ rot_y @ rot_x

    def __init__(self) -> None:
        super().__init__('depth_obstacle_costmap_node')

        def p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        # ── Parameters ────────────────────────────────────────────────
        self._frame        = p('frame_id',  'odom')
        self._depth_topic  = p('depth_topic',
                               '/front_zed_camera_x/zed_node/depth/depth_registered')
        self._info_topic   = p('info_topic',
                               '/front_zed_camera_x/zed_node/depth/camera_info')
        self._output_topic = p('output_topic', '/depth_obstacle_map')
        self._res          = float(p('resolution',   0.10))
        self._width_m      = float(p('width_m',     60.0))
        self._height_m     = float(p('height_m',    60.0))
        self._origin_x     = float(p('origin_x',   -30.0))
        self._origin_y     = float(p('origin_y',   -30.0))
        self._min_depth    = float(p('min_depth_m',  0.40))
        self._max_depth    = float(p('max_depth_m',  8.0))
        self._min_height   = float(p('min_height_m', 0.10))
        self._max_height   = float(p('max_height_m', 2.20))
        self._stride       = max(1, int(p('stride',  4)))
        self._hit_weight   = float(p('hit_weight',   2.0))
        self._decay        = float(p('decay',        0.995))
        self._decay_n      = max(1, int(p('decay_every_n_frames', 3)))
        self._hit_thresh   = float(p('hit_threshold', 8.0))
        self._min_points_per_cell = max(1, int(p('min_points_per_cell', 1)))
        self._min_component_cells = max(1, int(p('min_component_cells', 1)))
        self._max_value    = float(p('max_value',   200.0))
        self._inflate_r    = float(p('inflate_radius_m', 0.20))
        self._publish_hz   = float(p('publish_hz',   5.0))
        self._tf_timeout   = float(p('tf_timeout_sec', 0.10))
        self._use_latest_tf = bool(p('use_latest_tf', True))
        self._max_frame_age_sec = float(p('max_frame_age_sec', 2.0))
        self._use_odom_pose = bool(p('use_odom_pose', True))
        self._odom_topic = p('odom_topic', '/front_zed_camera_x/zed_node/odom')
        self._max_odom_age_sec = float(p('max_odom_age_sec', 2.0))
        # Defaults match front_zed_camera_x_left_camera_frame from the robot URDF:
        # base_link -> camera_link (0.45, 0, 0.194), camera_center z +0.016,
        # left_camera_frame (-0.01, +0.06, 0).
        self._camera_xyz = np.array([
            float(p('camera_x_m', 0.44)),
            float(p('camera_y_m', 0.06)),
            float(p('camera_z_m', 0.21)),
        ], dtype=np.float64)
        self._camera_rpy = np.array([
            float(p('camera_roll_rad', 0.0)),
            float(p('camera_pitch_rad', 0.0)),
            float(p('camera_yaw_rad', 0.0)),
        ], dtype=np.float64)
        self._camera_rot = self._rpy_to_rot(*self._camera_rpy)
        self._depth_frame_convention = str(
            p('depth_frame_convention', 'auto')).lower()
        if self._depth_frame_convention not in ('auto', 'optical', 'flu'):
            raise ValueError(
                "depth_frame_convention must be one of: 'auto', 'optical', 'flu'")
        self._logged_depth_frame_convention = False
        self._logged_projection_source = False

        # ── Grid state ────────────────────────────────────────────────
        self._nx = int(round(self._width_m  / self._res))
        self._ny = int(round(self._height_m / self._res))
        if self._nx <= 0 or self._ny <= 0:
            raise ValueError('depth_obstacle_costmap: width/height must be > 0')

        self._hits = np.zeros((self._ny, self._nx), dtype=np.float32)
        self._frame_count = 0
        self._latest_odom: Optional[Odometry] = None

        # ── Camera intrinsics (set on first CameraInfo message) ───────
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._cam_frame: Optional[str] = None

        # ── cv_bridge ─────────────────────────────────────────────────
        if CvBridge is None:
            raise RuntimeError('cv_bridge is not available')
        self._bridge = CvBridge()

        # ── Inflation kernel ──────────────────────────────────────────
        r_cells = max(0, int(math.ceil(self._inflate_r / self._res)))
        if r_cells > 0:
            k = 2 * r_cells + 1
            self._dilate_kernel: Optional[np.ndarray] = (
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        else:
            self._dilate_kernel = None

        # ── QoS / I-O ─────────────────────────────────────────────────
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._map_pub = self.create_publisher(OccupancyGrid, self._output_topic, map_qos)

        self.create_subscription(
            CameraInfo, self._info_topic,
            self._on_camera_info, qos_profile_sensor_data)
        self.create_subscription(
            Image, self._depth_topic,
            self._on_depth, qos_profile_sensor_data)
        if self._use_odom_pose:
            odom_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1)
            self.create_subscription(
                Odometry, self._odom_topic, self._on_odom, odom_qos)
        else:
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

        period = 1.0 / max(self._publish_hz, 0.5)
        self.create_timer(period, self._publish)

        self.get_logger().info(
            f'depth_obstacle_costmap: {self._nx}×{self._ny} cells @ '
            f'{self._res:.2f} m, frame={self._frame}, '
            f'in={self._depth_topic}, out={self._output_topic}')

    # ── Camera info callback ──────────────────────────────────────────────

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._fx is not None:
            return  # Only need to set once
        self._fx = float(msg.k[0])
        self._fy = float(msg.k[4])
        self._cx = float(msg.k[2])
        self._cy = float(msg.k[5])
        self._cam_frame = msg.header.frame_id
        self.get_logger().info(
            f'depth_obstacle_costmap: camera intrinsics set '
            f'fx={self._fx:.1f} fy={self._fy:.1f} cx={self._cx:.1f} cy={self._cy:.1f} '
            f'frame={self._cam_frame}')

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _resolve_depth_frame_convention(self, frame_id: str) -> str:
        if self._depth_frame_convention != 'auto':
            return self._depth_frame_convention
        return 'optical' if 'optical' in frame_id.lower() else 'flu'

    # ── Depth callback ────────────────────────────────────────────────────

    def _on_depth(self, msg: Image) -> None:
        if self._fx is None:
            return  # Waiting for camera_info

        now = self.get_clock().now()
        frame_age = (now - rclpy.time.Time.from_msg(msg.header.stamp)).nanoseconds / 1e9
        if self._max_frame_age_sec > 0.0 and frame_age > self._max_frame_age_sec:
            self.get_logger().warn(
                f'Dropping stale depth frame (age={frame_age:.2f}s > '
                f'{self._max_frame_age_sec:.2f}s)',
                throttle_duration_sec=2.0)
            return

        src_frame = msg.header.frame_id or self._cam_frame or 'base_link'
        stamp = msg.header.stamp

        if self._use_odom_pose:
            odom = self._latest_odom
            if odom is None:
                self.get_logger().warn(
                    f'Waiting for odometry on {self._odom_topic} before updating depth obstacle map.',
                    throttle_duration_sec=2.0)
                return
            odom_frame = odom.header.frame_id or self._frame
            if odom_frame != self._frame:
                self.get_logger().warn(
                    f'Odometry frame "{odom_frame}" does not match depth map frame "{self._frame}". '
                    'Set frame_id to the odom message frame or disable use_odom_pose.',
                    throttle_duration_sec=2.0)
                return
            odom_age = (now - rclpy.time.Time.from_msg(odom.header.stamp)).nanoseconds / 1e9
            if self._max_odom_age_sec > 0.0 and odom_age > self._max_odom_age_sec:
                self.get_logger().warn(
                    f'Dropping depth frame: latest odom age={odom_age:.2f}s > '
                    f'{self._max_odom_age_sec:.2f}s',
                    throttle_duration_sec=2.0)
                return

            pose = odom.pose.pose
            odom_yaw = self._yaw_from_quat(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w)
            cos_yaw = math.cos(odom_yaw)
            sin_yaw = math.sin(odom_yaw)
            R_odom_base = np.array([
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            t_odom_base = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ], dtype=np.float64)
            if not self._logged_projection_source:
                self.get_logger().info(
                    f'depth_obstacle_costmap: stamping depth points with odom topic '
                    f'{self._odom_topic}; camera_xyz={self._camera_xyz.tolist()}')
                self._logged_projection_source = True
        else:
            # Legacy fallback: project with TF camera→fixed frame.
            if self._use_latest_tf:
                try:
                    tf = self._tf_buffer.lookup_transform(
                        self._frame, src_frame, rclpy.time.Time(),
                        timeout=Duration(seconds=self._tf_timeout))
                except TransformException as ex:
                    self.get_logger().warn(
                        f'TF {src_frame}->{self._frame} failed: {ex}',
                        throttle_duration_sec=2.0)
                    return
            else:
                try:
                    tf = self._tf_buffer.lookup_transform(
                        self._frame, src_frame, stamp,
                        timeout=Duration(seconds=self._tf_timeout))
                except TransformException:
                    try:
                        tf = self._tf_buffer.lookup_transform(
                            self._frame, src_frame, rclpy.time.Time(),
                            timeout=Duration(seconds=self._tf_timeout))
                    except TransformException as ex:
                        self.get_logger().warn(
                            f'TF {src_frame}->{self._frame} failed: {ex}',
                            throttle_duration_sec=2.0)
                        return

            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            tz = tf.transform.translation.z
            q = tf.transform.rotation
            R_tf = self._quat_to_rot(q.x, q.y, q.z, q.w)
            t_tf = np.array([tx, ty, tz], dtype=np.float64)

        # Convert depth image to float32 array
        try:
            depth_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as ex:
            self.get_logger().warn(f'cv_bridge depth conversion failed: {ex}',
                                   throttle_duration_sec=2.0)
            return

        depth_img = np.asarray(depth_img, dtype=np.float32)

        # ── Periodic decay ────────────────────────────────────────────
        self._frame_count += 1
        if self._frame_count % self._decay_n == 0:
            self._hits *= self._decay

        # ── Subsample pixel grid ──────────────────────────────────────
        rows = np.arange(0, depth_img.shape[0], self._stride, dtype=np.int32)
        cols = np.arange(0, depth_img.shape[1], self._stride, dtype=np.int32)
        vc, uc = np.meshgrid(rows, cols, indexing='ij')  # shape (R, C)
        d = depth_img[vc, uc]  # shape (R, C)

        # Flatten
        d_flat  = d.ravel()
        u_flat  = uc.ravel().astype(np.float32)
        v_flat  = vc.ravel().astype(np.float32)

        # Filter by valid depth range
        valid = (
            np.isfinite(d_flat)
            & (d_flat >= self._min_depth)
            & (d_flat <= self._max_depth)
        )
        if not np.any(valid):
            return

        d_v  = d_flat[valid].astype(np.float64)
        u_v  = u_flat[valid].astype(np.float64)
        v_v  = v_flat[valid].astype(np.float64)

        # ── Back-project pixels to 3-D in camera frame ───────────────
        # Pinhole math naturally produces optical-frame points:
        # x=right, y=down, z=forward.  The odom path converts those to
        # camera-body FLU points before applying the configured mount.
        x_opt = (u_v - self._cx) * d_v / self._fx
        y_opt = (v_v - self._cy) * d_v / self._fy
        z_opt = d_v

        pts_cam_flu = np.stack([z_opt, -x_opt, -y_opt], axis=1)  # (N, 3)
        if self._use_odom_pose:
            pts_base = (self._camera_rot @ pts_cam_flu.T).T + self._camera_xyz
            pts_world = (R_odom_base @ pts_base.T).T + t_odom_base
        else:
            convention = self._resolve_depth_frame_convention(src_frame)
            if not self._logged_depth_frame_convention:
                self.get_logger().info(
                    f'depth_obstacle_costmap: interpreting depth points as '
                    f'{convention} frame coordinates for source frame "{src_frame}"',
                    once=True)
                self._logged_depth_frame_convention = True
            if convention == 'flu':
                pts_cam = pts_cam_flu
            else:
                pts_cam = np.stack([x_opt, y_opt, z_opt], axis=1)  # (N, 3)
            pts_world = (R_tf @ pts_cam.T).T + t_tf

        # ── Filter by height above ground ─────────────────────────────
        # pts_world[:,2] is the Z coordinate in odom (approximately height)
        height = pts_world[:, 2]
        in_band = (height >= self._min_height) & (height <= self._max_height)
        if not np.any(in_band):
            return

        wx = pts_world[in_band, 0]
        wy = pts_world[in_band, 1]

        # ── Mark hit cells ────────────────────────────────────────────
        ci = np.floor((wx - self._origin_x) / self._res).astype(np.int32)
        cj = np.floor((wy - self._origin_y) / self._res).astype(np.int32)
        in_g = (ci >= 0) & (ci < self._nx) & (cj >= 0) & (cj < self._ny)
        if np.any(in_g):
            hit_rows = cj[in_g]
            hit_cols = ci[in_g]
            if self._min_points_per_cell > 1:
                frame_counts = np.zeros_like(self._hits, dtype=np.uint16)
                np.add.at(frame_counts, (hit_rows, hit_cols), 1)
                supported = frame_counts[hit_rows, hit_cols] >= self._min_points_per_cell
                if not np.any(supported):
                    return
                hit_rows = hit_rows[supported]
                hit_cols = hit_cols[supported]
            cell_ids = hit_rows.astype(np.int64) * self._nx + hit_cols.astype(np.int64)
            unique_cells = np.unique(cell_ids)
            hit_rows = (unique_cells // self._nx).astype(np.int32)
            hit_cols = (unique_cells % self._nx).astype(np.int32)
            self._hits[hit_rows, hit_cols] += self._hit_weight
            np.minimum(self._hits, self._max_value, out=self._hits)

    # ── Publish ───────────────────────────────────────────────────────────

    def _publish(self) -> None:
        data = np.full((self._ny, self._nx), np.int8(-1), dtype=np.int8)

        hit_mask = (self._hits >= self._hit_thresh).astype(np.uint8)
        if self._min_component_cells > 1 and np.any(hit_mask):
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                hit_mask, connectivity=8)
            keep = np.zeros(n_labels, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= self._min_component_cells
            hit_mask = keep[labels].astype(np.uint8)
        if self._dilate_kernel is not None:
            hit_mask = cv2.dilate(hit_mask, self._dilate_kernel, iterations=1)
        data[hit_mask > 0] = np.int8(100)

        grid = OccupancyGrid()
        grid.header.stamp             = self.get_clock().now().to_msg()
        grid.header.frame_id          = self._frame
        grid.info.resolution          = self._res
        grid.info.width               = self._nx
        grid.info.height              = self._ny
        grid.info.origin.position.x   = self._origin_x
        grid.info.origin.position.y   = self._origin_y
        grid.info.origin.position.z   = 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = array.array('b', data.tobytes())
        self._map_pub.publish(grid)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = DepthObstacleCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
