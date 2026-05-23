"""
obstacle_costmap.py
===================

Builds a persistent 2-D occupancy grid in the ``odom`` frame from
``yolo_msgs/DetectionArray`` 3-D detections produced by ``yolo_ros``
(and, when active, virtual obstacles injected by the mission planner).

This node replaces the Nav2 ``ObstacleLayer`` that used to consume the
ZED point clouds.  Both the local and global Nav2 costmaps now read
``/obstacle_map`` via a ``StaticLayer`` — mirroring how the lane layer
is fed by ``/lane_map``.

Inputs
------
* ``/yolo/detections_3d``       ``yolo_msgs/DetectionArray``  (sensor QoS)
* ``/mission/virtual_obstacles`` ``geometry_msgs/PoseArray``  (latched)
    Optional.  Used by ``mission_planner_node`` to stamp the two GPS
    waypoints we are asked to *avoid* (they are obstacles, not goals).
* ``/mission/clear_virtual_obstacles`` ``std_msgs/Empty``
    Optional.  Clears any latched virtual obstacles.

Outputs
-------
* ``/obstacle_map``  ``nav_msgs/OccupancyGrid``  (TRANSIENT_LOCAL)
    Persistent grid in the ``odom`` frame.  Lethal cells (100) where an
    obstacle has been observed within ``obstacle_lifetime_sec``;
    everything else is free (0).
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time  # noqa: F401  (used by buffer.transform internally)

from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Empty

from tf2_ros import Buffer, TransformException, TransformListener
import tf2_geometry_msgs  # noqa: F401  (registers PoseStamped transform)

try:
    from yolo_msgs.msg import DetectionArray
except ImportError:  # pragma: no cover - dev convenience
    DetectionArray = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Obstacle:
    """Footprint of a single obstacle expressed in the ``odom`` frame."""
    stamp: float          # node-clock seconds when accepted
    x: float              # centre, odom frame [m]
    y: float
    yaw: float            # box yaw in odom frame [rad]
    size_x: float         # full extent along box +x [m]
    size_y: float         # full extent along box +y [m]
    persistent: bool = False  # mission-injected obstacles never decay


class ObstacleCostmapNode(Node):

    def __init__(self) -> None:
        super().__init__('obstacle_costmap_node')

        # ── Parameters ────────────────────────────────────────────────
        p = self.declare_parameter
        self._frame              = p('frame_id',           'odom').value
        self._res                = float(p('resolution',   0.10).value)
        self._width_m            = float(p('width_m',     60.0).value)
        self._height_m           = float(p('height_m',    60.0).value)
        self._origin_x           = float(p('origin_x',   -30.0).value)
        self._origin_y           = float(p('origin_y',   -30.0).value)
        self._lifetime           = float(p('obstacle_lifetime_sec', 5.0).value)
        self._publish_rate_hz    = float(p('publish_rate_hz', 5.0).value)
        self._detections_topic   = p('detections_topic',
                                     '/yolo/detections_3d').value
        self._virtual_topic      = p('virtual_obstacles_topic',
                                     '/mission/virtual_obstacles').value
        self._clear_topic        = p('clear_virtual_obstacles_topic',
                                     '/mission/clear_virtual_obstacles').value
        self._output_topic       = p('output_topic',      '/obstacle_map').value
        self._virtual_radius_m   = float(p('virtual_obstacle_radius_m',
                                           0.75).value)
        self._min_box_size       = float(p('min_box_size_m', 0.20).value)
        self._max_box_size       = float(p('max_box_size_m', 4.00).value)
        self._tf_timeout         = float(p('tf_timeout_sec', 0.10).value)

        # ── Grid state ────────────────────────────────────────────────
        self._nx = int(round(self._width_m  / self._res))
        self._ny = int(round(self._height_m / self._res))
        if self._nx <= 0 or self._ny <= 0:
            raise ValueError('obstacle_costmap: width/height must be > 0')

        self._obstacles: List[_Obstacle]      = []
        self._virtual:   List[_Obstacle]      = []

        # ── TF ────────────────────────────────────────────────────────
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── QoS / I-O ─────────────────────────────────────────────────
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._map_pub = self.create_publisher(
            OccupancyGrid, self._output_topic, map_qos)

        if DetectionArray is not None:
            self.create_subscription(
                DetectionArray, self._detections_topic,
                self._on_detections, qos_profile_sensor_data)
        else:
            self.get_logger().error(
                'yolo_msgs not available — running without YOLO input. '
                'Only mission-injected virtual obstacles will be stamped.')

        self.create_subscription(
            PoseArray, self._virtual_topic,
            self._on_virtual, latched)
        self.create_subscription(
            Empty, self._clear_topic, self._on_clear, 10)

        period = 1.0 / max(self._publish_rate_hz, 0.5)
        self._timer = self.create_timer(period, self._publish)

        self.get_logger().info(
            f'obstacle_costmap_node: {self._nx}x{self._ny} cells @ '
            f'{self._res:.2f} m, frame={self._frame}, '
            f'lifetime={self._lifetime:.1f}s, in={self._detections_topic}, '
            f'out={self._output_topic}')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_detections(self, msg) -> None:  # type: ignore[no-untyped-def]
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        added = 0
        for det in msg.detections:
            bb = det.bbox3d
            sx = float(bb.size.x)
            sy = float(bb.size.y)
            # Reject malformed / tiny / absurd boxes
            if (not math.isfinite(sx) or not math.isfinite(sy)
                    or sx < self._min_box_size or sy < self._min_box_size
                    or sx > self._max_box_size or sy > self._max_box_size):
                continue
            src_frame = bb.frame_id or msg.header.frame_id or 'base_link'
            pose = PoseStamped()
            pose.header.frame_id = src_frame
            pose.header.stamp = msg.header.stamp
            pose.pose = bb.center
            obs = self._pose_to_obstacle(pose, sx, sy, now_sec, persistent=False)
            if obs is not None:
                self._obstacles.append(obs)
                added += 1
        if added > 0:
            self.get_logger().debug(
                f'obstacle_costmap: +{added} detections '
                f'(total active {len(self._obstacles)})')

    def _on_virtual(self, msg: PoseArray) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        new_list: List[_Obstacle] = []
        for pose in msg.poses:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose = pose
            d = 2.0 * self._virtual_radius_m
            obs = self._pose_to_obstacle(ps, d, d, now_sec, persistent=True)
            if obs is not None:
                new_list.append(obs)
        self._virtual = new_list
        self.get_logger().info(
            f'obstacle_costmap: virtual obstacles set ({len(new_list)})')

    def _on_clear(self, _msg: Empty) -> None:
        if self._virtual:
            self.get_logger().info('obstacle_costmap: virtual obstacles cleared')
        self._virtual = []

    # ── Helpers ───────────────────────────────────────────────────────────

    def _pose_to_obstacle(self,
                          pose: PoseStamped,
                          size_x: float,
                          size_y: float,
                          now_sec: float,
                          persistent: bool) -> Optional[_Obstacle]:
        """Transform a pose to ``self._frame`` and convert to an _Obstacle."""
        if pose.header.frame_id == self._frame:
            transformed = pose
        else:
            try:
                transformed = self._tf_buffer.transform(
                    pose,
                    self._frame,
                    timeout=Duration(seconds=self._tf_timeout),
                )
            except TransformException as ex:
                self.get_logger().warn(
                    f'TF {pose.header.frame_id}->{self._frame} failed: {ex}',
                    throttle_duration_sec=2.0)
                return None

        x = transformed.pose.position.x
        y = transformed.pose.position.y
        q = transformed.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return _Obstacle(
            stamp=now_sec, x=x, y=y, yaw=yaw,
            size_x=size_x, size_y=size_y, persistent=persistent)

    def _expire(self, now_sec: float) -> None:
        if not self._obstacles:
            return
        cutoff = now_sec - self._lifetime
        self._obstacles = [o for o in self._obstacles
                           if o.persistent or o.stamp >= cutoff]

    # ── Publish ───────────────────────────────────────────────────────────

    def _publish(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self._expire(now_sec)

        data = np.zeros((self._ny, self._nx), dtype=np.int8)
        for obs in self._obstacles:
            self._stamp(data, obs)
        for obs in self._virtual:
            self._stamp(data, obs)

        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = self._frame
        grid.info.resolution = self._res
        grid.info.width  = self._nx
        grid.info.height = self._ny
        grid.info.origin.position.x = self._origin_x
        grid.info.origin.position.y = self._origin_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = array.array('b', data.tobytes())
        self._map_pub.publish(grid)

    def _stamp(self, data: np.ndarray, obs: _Obstacle) -> None:
        """Rasterise an oriented rectangular footprint as lethal (100)."""
        cos_y = math.cos(obs.yaw)
        sin_y = math.sin(obs.yaw)
        hx = 0.5 * obs.size_x
        hy = 0.5 * obs.size_y

        # AABB enclosing the rotated rectangle, clipped to the grid
        extent = abs(hx * cos_y) + abs(hy * sin_y), \
                 abs(hx * sin_y) + abs(hy * cos_y)
        min_x = obs.x - extent[0]
        max_x = obs.x + extent[0]
        min_y = obs.y - extent[1]
        max_y = obs.y + extent[1]

        i0 = max(0,           int(math.floor((min_x - self._origin_x) / self._res)))
        i1 = min(self._nx - 1, int(math.floor((max_x - self._origin_x) / self._res)))
        j0 = max(0,           int(math.floor((min_y - self._origin_y) / self._res)))
        j1 = min(self._ny - 1, int(math.floor((max_y - self._origin_y) / self._res)))
        if i0 > i1 or j0 > j1:
            return

        # Per-cell rotated-rect test for the AABB slice
        ii = np.arange(i0, i1 + 1)
        jj = np.arange(j0, j1 + 1)
        xs = self._origin_x + (ii + 0.5) * self._res - obs.x
        ys = self._origin_y + (jj + 0.5) * self._res - obs.y
        # local coords in the box frame
        XS, YS = np.meshgrid(xs, ys)            # shape (len(jj), len(ii))
        lx =  cos_y * XS + sin_y * YS
        ly = -sin_y * XS + cos_y * YS
        mask = (np.abs(lx) <= hx) & (np.abs(ly) <= hy)
        data[j0:j1 + 1, i0:i1 + 1][mask] = np.int8(100)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = ObstacleCostmapNode()
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
