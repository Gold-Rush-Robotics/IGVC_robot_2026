"""
gt_nav_bridge_node.py

Bridges the ground-truth occupancy grid (published by track_ground_truth_node)
to the two topics that the navigator and nav2 costmap layers consume:

    /lane_map      – nav2 StaticLayer (TRANSIENT_LOCAL, keeps the whole track
                     visible to the global costmap)
    /lane_costmap  – navigator's local rolling-window topic (TRANSIENT_LOCAL)

Obstacle positions are loaded from the ``obstacles_m`` array in the track JSON
(written by the IGVC track generator when ``--generate-usd`` is used) and
stamped into both grids as lethal cells (value = 100) so the nav2 obstacle
inflation layer and the planner can route around them.

This node is the replacement for the YOLOPv2 lane segmentation pipeline when
running a pure navigation test.  It has zero dependency on any camera or model.

Parameters
    track_file              str     ''          Path to track_points.json
    ground_truth_topic      str     /lane_ground_truth
    obstacle_inflate_radius_m float 0.3         Extra radius per obstacle beyond
                                                the footprint in the JSON
    use_sim_time            bool    false
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry


class GTNavBridgeNode(Node):
    """Republish ground-truth grid as /lane_map and /lane_costmap, with obstacles."""

    def __init__(self) -> None:
        super().__init__('gt_nav_bridge_node')

        self.declare_parameter('track_file', '')
        self.declare_parameter('ground_truth_topic', '/lane_ground_truth')
        self.declare_parameter('obstacle_inflate_radius_m', 0.3)
        self.declare_parameter('local_costmap_width_m', 6.0)
        self.declare_parameter('local_costmap_height_m', 6.0)
        self.declare_parameter('local_costmap_resolution_m', 0.1)
        self.declare_parameter('local_costmap_publish_hz', 5.0)
        self.declare_parameter('local_costmap_back_nogo_buffer_m', 0.3)
        self.declare_parameter('lane_boundary_thickness_cells', 1)

        self._track_file: str = str(self.get_parameter('track_file').value)
        self._gt_topic: str = str(self.get_parameter('ground_truth_topic').value)
        self._inflate_r: float = float(
            self.get_parameter('obstacle_inflate_radius_m').value)
        self._local_width_m: float = float(
            self.get_parameter('local_costmap_width_m').value)
        self._local_height_m: float = float(
            self.get_parameter('local_costmap_height_m').value)
        self._local_res: float = float(
            self.get_parameter('local_costmap_resolution_m').value)
        self._local_publish_hz: float = float(
            self.get_parameter('local_costmap_publish_hz').value)
        self._local_back_nogo_buffer_m: float = max(
            0.0,
            float(self.get_parameter('local_costmap_back_nogo_buffer_m').value),
        )
        self._lane_boundary_thickness_cells: int = max(
            0, int(self.get_parameter('lane_boundary_thickness_cells').value))

        # Load obstacle list from JSON once at startup.
        self._obstacles: List[Tuple[float, float, float]] = []  # (x_m, y_m, r_m) raw coords
        self._origin_offset_x: float = 0.0
        self._origin_offset_y: float = 0.0
        self._origin_yaw: float = 0.0
        self._latest_map: Optional[OccupancyGrid] = None
        self._latest_odom: Optional[Odometry] = None
        self._load_obstacles()

        # QoS: TRANSIENT_LOCAL so late-joining nodes (nav2 StaticLayer, navigator)
        # receive the last message immediately on subscription.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Ground-truth publisher uses standard QoS (not latched) — subscribe accordingly.
        gt_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._lane_map_pub = self.create_publisher(OccupancyGrid, '/lane_map', latched_qos)
        self._lane_costmap_pub = self.create_publisher(
            OccupancyGrid, '/lane_costmap', latched_qos)

        self.create_subscription(OccupancyGrid, self._gt_topic, self._on_grid, gt_qos)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)

        period = 1.0 / max(self._local_publish_hz, 0.1)
        self.create_timer(period, self._publish_local_costmap)

        self.get_logger().info(
            f'GT nav bridge ready — listening on {self._gt_topic}, '
            f'{len(self._obstacles)} obstacle(s) loaded'
            + (f' (inflate +{self._inflate_r:.2f} m)' if self._obstacles else ''))

    # ──────────────────────────────────────────────────────────────────────────

    def _load_obstacles(self) -> None:
        """Load obstacles_m from track JSON if available."""
        if not self._track_file:
            return
        path = Path(self._track_file)
        if not path.exists():
            self.get_logger().warn(f'track_file not found: {self._track_file}')
            return
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            self.get_logger().error(f'Failed to parse track_file: {exc}')
            return

        # Read origin offset so obstacle world coords match the GT grid frame.
        pose = payload.get('robot_start_pose', {})
        pos = pose.get('position_m', {})
        self._origin_offset_x = float(pos.get('x', 0.0))
        self._origin_offset_y = float(pos.get('y', 0.0))
        self._origin_yaw = float(pose.get('yaw_rad', 0.0))
        self.get_logger().info(
            f'Origin offset from robot_start_pose: '
            f'({self._origin_offset_x:.3f}, {self._origin_offset_y:.3f}), '
            f'yaw {self._origin_yaw:.3f} rad')

        raw = payload.get('obstacles_m', [])
        for obs in raw:
            try:
                self._obstacles.append((
                    float(obs['x_m']),
                    float(obs['y_m']),
                    float(obs['radius_m']),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                self.get_logger().warn(f'Skipping malformed obstacle entry: {obs!r} — {exc}')

        if raw and not self._obstacles:
            self.get_logger().warn(
                'obstacles_m present in JSON but all entries failed to parse.')
        elif not raw:
            self.get_logger().info(
                'No obstacles_m in track JSON — bridge will not stamp obstacle cells. '
                'Run the track generator with --generate-usd to add obstacles.')

    # ──────────────────────────────────────────────────────────────────────────

    def _stamp_obstacles(self, grid: OccupancyGrid) -> None:
        """Stamp lethal cells (100) for each obstacle in-place on *grid*.

        Obstacle positions in the JSON are in raw track coordinates (same origin
        as centerline_m). The GT node shifts and rotates the grid so that
        robot_start maps to odom/map (0, 0, 0). We apply the same transform here.

        The grid's ``info.origin`` is then used to convert from the shifted world
        frame to grid cell indices, so this works regardless of grid dimensions.
        """
        if not self._obstacles:
            return

        res = grid.info.resolution
        if res <= 0.0:
            return
        w = grid.info.width
        h = grid.info.height
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y

        data = np.array(grid.data, dtype=np.int8).reshape((h, w))

        for obs_x_m, obs_y_m, obs_r_m in self._obstacles:
            # Shift from raw track coords to start-translated map frame.
            # (no rotation — odom inherits world axes from the spawn pose).
            world_x = obs_x_m - self._origin_offset_x
            world_y = obs_y_m - self._origin_offset_y

            # Obstacle footprint radius + extra inflation
            total_r_m = obs_r_m + self._inflate_r
            total_r_cells = total_r_m / res

            # Grid center cell of this obstacle
            cx = (world_x - ox) / res
            cy = (world_y - oy) / res

            # Bounding box in cell coords
            imin = max(0, int(math.floor(cy - total_r_cells)))
            imax = min(h - 1, int(math.ceil(cy + total_r_cells)))
            jmin = max(0, int(math.floor(cx - total_r_cells)))
            jmax = min(w - 1, int(math.ceil(cx + total_r_cells)))

            if imax < imin or jmax < jmin:
                continue  # obstacle entirely outside grid

            # Vectorised circle stamp
            js = np.arange(jmin, jmax + 1)
            is_ = np.arange(imin, imax + 1)
            jg, ig = np.meshgrid(js, is_)
            dist2 = (jg - cx) ** 2 + (ig - cy) ** 2
            in_circle = dist2 <= total_r_cells ** 2
            data[imin:imax + 1, jmin:jmax + 1][in_circle] = 100

        grid.data = data.ravel().tolist()

    def _stamp_lane_boundaries(self, grid: OccupancyGrid) -> None:
        """Stamp lane boundaries as lethal cells where FREE meets UNKNOWN.

        This creates explicit black lane lines in RViz and gives downstream
        planners/controllers a crisp no-go edge at corridor borders.
        """
        thickness = self._lane_boundary_thickness_cells
        if thickness <= 0:
            return

        w = int(grid.info.width)
        h = int(grid.info.height)
        if w <= 0 or h <= 0:
            return

        data = np.array(grid.data, dtype=np.int8).reshape((h, w))
        free = data == 0
        unknown = data < 0

        # A boundary cell is FREE and touches UNKNOWN (8-neighborhood).
        up = np.pad(unknown[:-1, :], ((1, 0), (0, 0)), constant_values=True)
        down = np.pad(unknown[1:, :], ((0, 1), (0, 0)), constant_values=True)
        left = np.pad(unknown[:, :-1], ((0, 0), (1, 0)), constant_values=True)
        right = np.pad(unknown[:, 1:], ((0, 0), (0, 1)), constant_values=True)
        up_left = np.pad(unknown[:-1, :-1], ((1, 0), (1, 0)), constant_values=True)
        up_right = np.pad(unknown[:-1, 1:], ((1, 0), (0, 1)), constant_values=True)
        down_left = np.pad(unknown[1:, :-1], ((0, 1), (1, 0)), constant_values=True)
        down_right = np.pad(unknown[1:, 1:], ((0, 1), (0, 1)), constant_values=True)
        boundary = free & (
            up | down | left | right |
            up_left | up_right | down_left | down_right
        )

        # Optional extra thickness grown inward through FREE cells.
        mask = boundary.copy()
        for _ in range(1, thickness):
            m_up = np.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)
            m_down = np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
            m_left = np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
            m_right = np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)
            m_up_left = np.pad(mask[:-1, :-1], ((1, 0), (1, 0)), constant_values=False)
            m_up_right = np.pad(mask[:-1, 1:], ((1, 0), (0, 1)), constant_values=False)
            m_down_left = np.pad(mask[1:, :-1], ((0, 1), (1, 0)), constant_values=False)
            m_down_right = np.pad(mask[1:, 1:], ((0, 1), (0, 1)), constant_values=False)
            grown = (
                m_up | m_down | m_left | m_right |
                m_up_left | m_up_right | m_down_left | m_down_right
            )
            mask = mask | (free & grown)

        data[mask] = 100
        grid.data = data.ravel().tolist()

    # ──────────────────────────────────────────────────────────────────────────

    def _on_grid(self, msg: OccupancyGrid) -> None:
        """Receive ground-truth grid, merge obstacles, republish."""
        # Deep-copy so we don't mutate the received message.
        merged = deepcopy(msg)
        self._stamp_obstacles(merged)
        self._stamp_lane_boundaries(merged)
        merged.header.stamp = self.get_clock().now().to_msg()
        self._latest_map = merged
        self._lane_map_pub.publish(merged)

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _publish_local_costmap(self) -> None:
        """Publish a base_link-frame crop of the fixed GT map for navigator.py."""
        fixed = self._latest_map
        odom = self._latest_odom
        if fixed is None or odom is None:
            return
        if fixed.info.resolution <= 0.0 or self._local_res <= 0.0:
            return

        fixed_w = int(fixed.info.width)
        fixed_h = int(fixed.info.height)
        # ROS OccupancyGrid convention: data[row=y_idx, col=x_idx].
        # info.width  = #cells along +x (forward in base_link)
        # info.height = #cells along +y (lateral in base_link)
        # _local_height_m is the forward extent; _local_width_m is the lateral
        # extent (matches existing parameter naming in the codebase).
        nx = max(1, int(round(self._local_height_m / self._local_res)))
        ny = max(1, int(round(self._local_width_m / self._local_res)))

        fixed_data = np.frombuffer(bytes(fixed.data), dtype=np.int8).reshape(
            fixed_h, fixed_w)
        local_data = np.full((ny, nx), -1, dtype=np.int8)

        pose = odom.pose.pose
        robot_x = pose.position.x
        robot_y = pose.position.y
        q = pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        fixed_ox = fixed.info.origin.position.x
        fixed_oy = fixed.info.origin.position.y
        local_oy = -0.5 * self._local_width_m

        # cols → +x (forward), rows → +y (lateral) in base_link.
        col_idx = np.arange(nx, dtype=np.float32)
        row_idx = np.arange(ny, dtype=np.float32)
        local_x = col_idx[None, :] * self._local_res
        local_y = local_oy + row_idx[:, None] * self._local_res

        world_x = robot_x + cos_yaw * local_x - sin_yaw * local_y
        world_y = robot_y + sin_yaw * local_x + cos_yaw * local_y

        src_cols = np.rint((world_x - fixed_ox) / fixed.info.resolution).astype(np.int32)
        src_rows = np.rint((world_y - fixed_oy) / fixed.info.resolution).astype(np.int32)
        valid = (
            (src_cols >= 0) & (src_cols < fixed_w) &
            (src_rows >= 0) & (src_rows < fixed_h)
        )
        local_data[valid] = fixed_data[src_rows[valid], src_cols[valid]]

        # Lethal strip at the back edge (x ∈ [0, back_buf_m)): first columns.
        back_buf_cells = int(math.ceil(self._local_back_nogo_buffer_m / self._local_res))
        if back_buf_cells > 0:
            back_buf_cells = min(back_buf_cells, nx)
            local_data[:, :back_buf_cells] = 100

        local = OccupancyGrid()
        local.header.stamp = self.get_clock().now().to_msg()
        local.header.frame_id = 'base_link'
        local.info.resolution = self._local_res
        local.info.width = nx
        local.info.height = ny
        local.info.origin.position.x = 0.0
        local.info.origin.position.y = local_oy
        local.info.origin.position.z = 0.0
        local.info.origin.orientation.w = 1.0
        local.data = local_data.ravel().tolist()
        self._lane_costmap_pub.publish(local)


# ─────────────────────────────────────────────────────────────────────────────


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GTNavBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
