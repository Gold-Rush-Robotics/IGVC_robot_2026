"""lidar_obstacle_costmap.py
===========================
Builds a persistent 2-D occupancy grid from raw ``sensor_msgs/LaserScan``
data and publishes it as a ``nav_msgs/OccupancyGrid`` on
``/lidar_obstacle_map`` (TRANSIENT_LOCAL, same format as ``/obstacle_map``).

Unlike Nav2's built-in ObstacleLayer this node:

* Accumulates probabilistic evidence across many scans — a single stray
  hit does not immediately declare a cell lethal.
* Decays evidence over time so cleared space is eventually forgotten.
* Uses free-space ray-casting (beam model) to actively clear cells between
  the sensor and the first valid return.
* Inflates lethal cells at publish time by ``inflate_radius_m`` using an
  elliptic kernel so physical obstacles wider than one cell are padded.

Parameters
----------
frame_id              (str,   'odom')         — Fixed frame for the output grid.
scan_topic            (str,   '/scan')         — Input LaserScan topic.
output_topic          (str,   '/lidar_obstacle_map') — Output OccupancyGrid topic.
resolution            (float, 0.10)            — Cell size in meters.
width_m               (float, 60.0)            — Grid width in meters.
height_m              (float, 60.0)            — Grid height in meters.
origin_x              (float, -30.0)           — Grid origin X (bottom-left).
origin_y              (float, -30.0)           — Grid origin Y (bottom-left).
min_range_m           (float, 0.15)            — Ignore ranges shorter than this.
max_range_m           (float, 10.0)            — Ignore ranges longer than this.
hit_weight            (float, 3.0)             — Evidence added per valid hit.
free_weight           (float, 0.5)             — Evidence added per free-ray step.
decay                 (float, 0.998)           — Multiplicative decay applied every
                                                 decay_every_n_scans scans.
decay_every_n_scans   (int,   5)               — Apply decay once per N scans.
hit_threshold         (float, 10.0)            — Accumulated hits needed for lethal.
free_threshold        (float, 3.0)             — Accumulated free needed to clear.
max_value             (float, 200.0)           — Clamp for both accumulator grids.
inflate_radius_m      (float, 0.15)            — Inflate lethal cells by this radius.
publish_hz            (float, 5.0)             — OccupancyGrid publish frequency.
tf_timeout_sec        (float, 0.10)            — TF lookup timeout.
scan_step             (int,   1)               — Use every Nth beam (1 = all beams).
free_ray_step_m       (float, 0.20)            — Distance between free-ray samples.
use_odom_pose         (bool,  True)            — Project scans using odometry topic.
odom_topic            (str,   '/front_zed_camera_x/zed_node/odom')
max_odom_age_sec      (float, 2.0)             — Drop scans when odom is stale.
lidar_x/y_m           (float)                  — LiDAR origin in base_link.
lidar_yaw_rad         (float)                  — LiDAR yaw relative to base_link.
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
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


class LidarObstacleCostmapNode(Node):

    @staticmethod
    def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
        return math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz))

    def __init__(self) -> None:
        super().__init__('lidar_obstacle_costmap_node')

        def p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        # ── Parameters ────────────────────────────────────────────────
        self._frame         = p('frame_id',            'odom')
        self._scan_topic    = p('scan_topic',           '/scan')
        self._output_topic  = p('output_topic',         '/lidar_obstacle_map')
        self._res           = float(p('resolution',      0.10))
        self._width_m       = float(p('width_m',        60.0))
        self._height_m      = float(p('height_m',       60.0))
        self._origin_x      = float(p('origin_x',      -30.0))
        self._origin_y      = float(p('origin_y',      -30.0))
        self._min_range_m   = float(p('min_range_m',    0.15))
        self._max_range_m   = float(p('max_range_m',   10.0))
        self._hit_weight    = float(p('hit_weight',     3.0))
        self._free_weight   = float(p('free_weight',    0.5))
        self._decay         = float(p('decay',          0.998))
        self._decay_n       = max(1, int(p('decay_every_n_scans', 5)))
        self._hit_thresh    = float(p('hit_threshold',  10.0))
        self._free_thresh   = float(p('free_threshold',  3.0))
        self._max_value     = float(p('max_value',     200.0))
        self._inflate_r     = float(p('inflate_radius_m', 0.15))
        self._publish_hz    = float(p('publish_hz',      5.0))
        self._tf_timeout    = float(p('tf_timeout_sec',  0.10))
        self._scan_step     = max(1, int(p('scan_step',  1)))
        self._free_step_m   = max(self._res, float(p('free_ray_step_m', 0.20)))
        self._use_odom_pose = bool(p('use_odom_pose', True))
        self._odom_topic    = p('odom_topic', '/front_zed_camera_x/zed_node/odom')
        self._max_odom_age_sec = float(p('max_odom_age_sec', 2.0))
        self._lidar_x       = float(p('lidar_x_m', 0.225))
        self._lidar_y       = float(p('lidar_y_m', 0.0))
        self._lidar_yaw     = float(p('lidar_yaw_rad', 0.0))

        # ── Grid state ────────────────────────────────────────────────
        self._nx = int(round(self._width_m  / self._res))
        self._ny = int(round(self._height_m / self._res))
        if self._nx <= 0 or self._ny <= 0:
            raise ValueError('lidar_obstacle_costmap: width/height must be > 0')

        # Floating-point accumulators (evidence model)
        self._hits = np.zeros((self._ny, self._nx), dtype=np.float32)
        self._free = np.zeros((self._ny, self._nx), dtype=np.float32)

        self._scan_count = 0
        self._latest_odom: Optional[Odometry] = None
        self._logged_pose_source = False

        # ── Pose source ───────────────────────────────────────────────
        if self._use_odom_pose:
            odom_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(
                Odometry, self._odom_topic, self._on_odom, odom_qos)
        else:
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

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

        self._map_pub = self.create_publisher(
            OccupancyGrid, self._output_topic, map_qos)

        self.create_subscription(
            LaserScan, self._scan_topic,
            self._on_scan, qos_profile_sensor_data)

        period = 1.0 / max(self._publish_hz, 0.5)
        self.create_timer(period, self._publish)

        self.get_logger().info(
            f'lidar_obstacle_costmap: {self._nx}×{self._ny} cells @ '
            f'{self._res:.2f} m, frame={self._frame}, '
            f'in={self._scan_topic}, out={self._output_topic}')

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg

    # ── Scan callback ─────────────────────────────────────────────────────

    def _on_scan(self, msg: LaserScan) -> None:
        src = msg.header.frame_id or 'base_link'
        stamp = msg.header.stamp

        if self._use_odom_pose:
            odom = self._latest_odom
            if odom is None:
                self.get_logger().warn(
                    f'Waiting for odometry on {self._odom_topic} before updating lidar obstacle map.',
                    throttle_duration_sec=2.0)
                return
            odom_frame = odom.header.frame_id or self._frame
            if odom_frame != self._frame:
                self.get_logger().warn(
                    f'Odometry frame "{odom_frame}" does not match lidar map frame "{self._frame}". '
                    'Set frame_id to the odom message frame or disable use_odom_pose.',
                    throttle_duration_sec=2.0)
                return
            now = self.get_clock().now()
            odom_age = (now - rclpy.time.Time.from_msg(odom.header.stamp)).nanoseconds / 1e9
            if self._max_odom_age_sec > 0.0 and odom_age > self._max_odom_age_sec:
                self.get_logger().warn(
                    f'Dropping lidar scan: latest odom age={odom_age:.2f}s > '
                    f'{self._max_odom_age_sec:.2f}s',
                    throttle_duration_sec=2.0)
                return

            pose = odom.pose.pose
            base_yaw = self._yaw_from_quat(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w)
            cos_yaw = math.cos(base_yaw)
            sin_yaw = math.sin(base_yaw)
            tx = (pose.position.x +
                  cos_yaw * self._lidar_x - sin_yaw * self._lidar_y)
            ty = (pose.position.y +
                  sin_yaw * self._lidar_x + cos_yaw * self._lidar_y)
            yaw = base_yaw + self._lidar_yaw
            if not self._logged_pose_source:
                self.get_logger().info(
                    f'lidar_obstacle_costmap: projecting scans with odom topic '
                    f'{self._odom_topic}; lidar_xy=({self._lidar_x:.3f}, '
                    f'{self._lidar_y:.3f}), lidar_yaw={self._lidar_yaw:.3f} rad')
                self._logged_pose_source = True
        else:
            # Legacy fallback: look up sensor→odom transform at the scan stamp;
            # fall back to latest.
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._frame, src, stamp,
                    timeout=Duration(seconds=self._tf_timeout))
            except TransformException as ex:
                try:
                    tf = self._tf_buffer.lookup_transform(
                        self._frame, src, rclpy.time.Time(),
                        timeout=Duration(seconds=self._tf_timeout))
                except TransformException:
                    self.get_logger().warn(
                        f'TF {src}->{self._frame} failed: {ex}',
                        throttle_duration_sec=2.0)
                    return

            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            q = tf.transform.rotation
            yaw = self._yaw_from_quat(q.x, q.y, q.z, q.w)

        # ── Periodic decay ────────────────────────────────────────────
        self._scan_count += 1
        if self._scan_count % self._decay_n == 0:
            self._hits *= self._decay
            self._free *= self._decay

        # ── Build vectorised angle/range arrays ───────────────────────
        ranges_all = np.asarray(msg.ranges, dtype=np.float32)
        if self._scan_step > 1:
            ranges_all = ranges_all[::self._scan_step]
            n = len(ranges_all)
            angles = (np.arange(n, dtype=np.float32)
                      * (self._scan_step * msg.angle_increment)
                      + msg.angle_min)
        else:
            n = len(ranges_all)
            angles = (np.arange(n, dtype=np.float32)
                      * msg.angle_increment + msg.angle_min)

        world_angles = angles + yaw
        cos_a = np.cos(world_angles)
        sin_a = np.sin(world_angles)

        valid = (
            np.isfinite(ranges_all)
            & (ranges_all >= self._min_range_m)
            & (ranges_all <= self._max_range_m)
        )

        # ── Mark hit cells ────────────────────────────────────────────
        if np.any(valid):
            hx = tx + ranges_all[valid] * cos_a[valid]
            hy = ty + ranges_all[valid] * sin_a[valid]
            ci = np.floor((hx - self._origin_x) / self._res).astype(np.int32)
            cj = np.floor((hy - self._origin_y) / self._res).astype(np.int32)
            in_g = (ci >= 0) & (ci < self._nx) & (cj >= 0) & (cj < self._ny)
            if np.any(in_g):
                np.add.at(self._hits, (cj[in_g], ci[in_g]), self._hit_weight)
                np.minimum(self._hits, self._max_value, out=self._hits)

        # ── Clear free-space rays ─────────────────────────────────────
        # For each beam, step from the sensor toward the endpoint (hit or
        # max_range) and accumulate free evidence along the way.
        free_r = np.where(
            valid, ranges_all,
            np.minimum(
                np.where(np.isfinite(ranges_all), ranges_all, self._max_range_m),
                self._max_range_m,
            ),
        )

        n_steps = max(1, int(math.ceil(self._max_range_m / self._free_step_m)))
        for step in range(1, n_steps + 1):
            d = step * self._free_step_m
            mask = d < free_r
            if not np.any(mask):
                break
            fx = (tx + d * cos_a[mask]).astype(np.float32)
            fy = (ty + d * sin_a[mask]).astype(np.float32)
            fci = np.floor((fx - self._origin_x) / self._res).astype(np.int32)
            fcj = np.floor((fy - self._origin_y) / self._res).astype(np.int32)
            in_g = (fci >= 0) & (fci < self._nx) & (fcj >= 0) & (fcj < self._ny)
            if np.any(in_g):
                np.add.at(self._free, (fcj[in_g], fci[in_g]), self._free_weight)

        np.minimum(self._free, self._max_value, out=self._free)

    # ── Publish ───────────────────────────────────────────────────────────

    def _publish(self) -> None:
        # Start with unknown (-1), apply free (0), then lethal (100).
        data = np.full((self._ny, self._nx), np.int8(-1), dtype=np.int8)
        data[self._free >= self._free_thresh] = np.int8(0)

        hit_mask = (self._hits >= self._hit_thresh).astype(np.uint8)
        if self._dilate_kernel is not None:
            hit_mask = cv2.dilate(hit_mask, self._dilate_kernel, iterations=1)
        data[hit_mask > 0] = np.int8(100)

        grid = OccupancyGrid()
        grid.header.stamp            = self.get_clock().now().to_msg()
        grid.header.frame_id         = self._frame
        grid.info.resolution         = self._res
        grid.info.width              = self._nx
        grid.info.height             = self._ny
        grid.info.origin.position.x  = self._origin_x
        grid.info.origin.position.y  = self._origin_y
        grid.info.origin.position.z  = 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = array.array('b', data.tobytes())
        self._map_pub.publish(grid)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = LidarObstacleCostmapNode()
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
