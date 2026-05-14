from __future__ import annotations

import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path as NavPath


def _as_xy_pairs(data: Iterable) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for item in data:
        if isinstance(item, dict):
            x = item.get('x')
            y = item.get('y')
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            continue
        if x is None or y is None:
            continue
        pts.append((float(x), float(y)))
    return pts


def _load_track_points(path_str: str) -> List[Tuple[float, float]]:
    path = Path(path_str)
    if not path.exists():
        return []

    if path.suffix.lower() == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, dict):
            for key in ('centerline_m', 'midpoint_m'):
                if key in payload:
                    return _as_xy_pairs(payload[key])
            for key in ('midpoint', 'midpoints', 'centerline', 'points', 'track'):
                if key in payload:
                    return _as_xy_pairs(payload[key])
            return []
        if isinstance(payload, list):
            return _as_xy_pairs(payload)
        return []

    if path.suffix.lower() in ('.yaml', '.yml'):
        try:
            import yaml  # type: ignore
        except Exception:
            return []
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
        if isinstance(payload, dict):
            for key in ('midpoint', 'midpoints', 'centerline', 'points', 'track'):
                if key in payload:
                    return _as_xy_pairs(payload[key])
            return []
        if isinstance(payload, list):
            return _as_xy_pairs(payload)
        return []

    if path.suffix.lower() == '.csv':
        pts: List[Tuple[float, float]] = []
        with path.open('r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    pts.append((float(row[0]), float(row[1])))
                except ValueError:
                    continue
        return pts

    return []


def _fallback_circle(radius_m: float, n: int) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for i in range(max(16, n)):
        t = (2.0 * math.pi * i) / float(max(16, n))
        pts.append((radius_m * math.cos(t), radius_m * math.sin(t)))
    return pts


def _get_lane_polygon_from_image(image_path: str) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """Extract outer and inner boundary contours from track image as pixel coordinates.

    The track image has two thin white boundary lines.  findContours returns 4
    contours (outer/inner edge of each white line).  Sorted by area descending:
      [0] outer edge of outer line  → encloses everything
      [1] inner edge of outer line  → encloses track area (no outer line itself)
      [2] outer edge of inner line  → encloses center island
      [3] inner edge of inner line  → encloses center island minus line

    Filling with [1] and cutting with [2] gives the lane corridor between the lines.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    _, boundary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(boundary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours or len(contours) < 4:
        return None, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # [1] = inner edge of outer boundary line (fill up to here)
    # [2] = outer edge of inner boundary line (cut from here inward)
    c_fill = contours[1].reshape(-1, 2).astype(np.float32)
    c_cut  = contours[2].reshape(-1, 2).astype(np.float32)
    if c_fill.shape[0] < 20 or c_cut.shape[0] < 20:
        return None, None

    return c_fill, c_cut


def _track_points_from_image(image_path: str, pixels_per_meter: float) -> List[Tuple[float, float]]:
    """Infer a centerline loop from a generated track image with white lane boundaries."""
    c_outer, c_inner = _get_lane_polygon_from_image(image_path)
    if c_outer is None or c_inner is None:
        return []

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sample_step = max(1, c_outer.shape[0] // 1200)
    sampled_outer = c_outer[::sample_step]
    center_px = []
    for p in sampled_outer:
        deltas = c_inner - p
        j = int(np.argmin(np.einsum('ij,ij->i', deltas, deltas)))
        q = c_inner[j]
        center_px.append(0.5 * (p + q))

    if len(center_px) < 16:
        return []

    center_px = np.asarray(center_px, dtype=np.float32)
    cx = img.shape[1] / 2.0
    cy = img.shape[0] / 2.0
    ppm = max(1e-6, float(pixels_per_meter))

    points_m: List[Tuple[float, float]] = []
    for x, y in center_px:
        # X is negated to match USD floor placement (pygame_to_usd_xy negates X).
        wx = float(-(x - cx) / ppm)
        # JSON uses y-down (Pygame/image convention, matching USD floor placement).
        # Do NOT flip y here so the centreline matches track_points.json exactly.
        wy = float((y - cy) / ppm)
        points_m.append((wx, wy))
    return points_m


class TrackGroundTruthNode(Node):
    def __init__(self) -> None:
        super().__init__('track_ground_truth_node')

        self.declare_parameter('track_file', '')
        self.declare_parameter('track_image_file', '')
        self.declare_parameter('track_image_pixels_per_meter', 52.5)
        self.declare_parameter('debug_png_path', '')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('publish_hz', 5.0)
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('width_m', 40.0)
        self.declare_parameter('height_m', 40.0)
        self.declare_parameter('lane_half_width_m', 0.25)
        self.declare_parameter('fallback_circle_radius_m', 8.0)
        self.declare_parameter('fallback_circle_points', 200)
        self.declare_parameter('occupancy_topic', '/lane_ground_truth')
        self.declare_parameter('midpoint_path_topic', '/lane_ground_truth_midpoint_path')

        self._frame_id = str(self.get_parameter('frame_id').value)
        self._pub_hz = float(self.get_parameter('publish_hz').value)
        self._res = float(self.get_parameter('resolution').value)
        self._w_m = float(self.get_parameter('width_m').value)
        self._h_m = float(self.get_parameter('height_m').value)
        self._half_lane = float(self.get_parameter('lane_half_width_m').value)
        self._debug_png_path = str(self.get_parameter('debug_png_path').value)

        track_file = str(self.get_parameter('track_file').value)
        track_image_file = str(self.get_parameter('track_image_file').value)
        self._track_image_file = track_image_file
        image_ppm = float(self.get_parameter('track_image_pixels_per_meter').value)

        # Prefer pixels_per_meter from the track JSON when available, so the
        # image-derived lane polygon scales consistently with centerline_m
        # (which uses the same ppm). The parameter is only a fallback.
        if track_file:
            try:
                _ppm_payload = json.loads(Path(track_file).read_text(encoding='utf-8'))
                _json_ppm = _ppm_payload.get('pixels_per_meter')
                if _json_ppm is not None and float(_json_ppm) > 0.0:
                    if abs(float(_json_ppm) - image_ppm) > 1e-3:
                        self.get_logger().info(
                            f'Overriding track_image_pixels_per_meter '
                            f'({image_ppm:.3f}) with JSON pixels_per_meter '
                            f'({float(_json_ppm):.3f}).')
                    image_ppm = float(_json_ppm)
            except Exception as exc:
                self.get_logger().warn(
                    f'Could not read pixels_per_meter from track_file: {exc}')

        pts = _load_track_points(track_file) if track_file else []
        if pts:
            self.get_logger().info(
                f'Loaded {len(pts)} centerline points from track file: {track_file}')
        elif track_file:
            self.get_logger().warn(
                f'No valid points in track file: {track_file}',
                throttle_duration_sec=5.0,
            )
        
        # Always try to extract lane boundaries from the image when available (for variable-width GT)
        lane_polygon = None
        if track_image_file:
            c_outer, c_inner = _get_lane_polygon_from_image(track_image_file)
            if c_outer is not None and c_inner is not None:
                lane_polygon = (c_outer, c_inner, image_ppm, track_image_file)
                self.get_logger().info(
                    f'Extracted lane boundaries from track image: {c_outer.shape[0]} outer, {c_inner.shape[0]} inner points')

        if not pts and track_image_file:
            pts = _track_points_from_image(track_image_file, image_ppm)
            if pts:
                self.get_logger().info(
                    f'Inferred {len(pts)} centerline points from track image: {track_image_file}')
        if not pts:
            pts = _fallback_circle(
                float(self.get_parameter('fallback_circle_radius_m').value),
                int(self.get_parameter('fallback_circle_points').value),
            )
            self.get_logger().warn(
                'No valid track file/image points found; using fallback circular test track.',
                throttle_duration_sec=5.0,
            )

        # Read robot_start_pose from JSON track file to offset grid so start = (0,0) in odom
        self._origin_offset_x = 0.0
        self._origin_offset_y = 0.0
        self._origin_yaw = 0.0
        if track_file:
            try:
                payload = json.loads(Path(track_file).read_text(encoding='utf-8'))
                pose = payload.get('robot_start_pose', {})
                pos = pose.get('position_m', {})
                self._origin_offset_x = float(pos.get('x', 0.0))
                self._origin_offset_y = float(pos.get('y', 0.0))
                self._origin_yaw = float(pose.get('yaw_rad', 0.0))
                self.get_logger().info(
                    f'Grid origin offset to robot start: '
                    f'({self._origin_offset_x:.3f}, {self._origin_offset_y:.3f}), '
                    f'yaw {self._origin_yaw:.3f} rad')
            except Exception as exc:
                self.get_logger().warn(f'Could not read robot_start_pose for grid offset: {exc}')

        self._points = pts
        self._grid = self._build_ground_truth_grid(pts, lane_polygon)
        self._path = self._build_midpoint_path(pts)
        self._save_debug_png(self._grid, self._debug_png_path)

        occ_topic = str(self.get_parameter('occupancy_topic').value)
        path_topic = str(self.get_parameter('midpoint_path_topic').value)
        self._occ_pub = self.create_publisher(OccupancyGrid, occ_topic, 10)
        self._path_pub = self.create_publisher(NavPath, path_topic, 10)

        self.create_timer(1.0 / max(0.1, self._pub_hz), self._publish)
        self.get_logger().info(
            f'Ground-truth ready: {len(self._points)} points, publishing {occ_topic} and {path_topic}.')

    @staticmethod
    def _point_in_polygon(point: Tuple[float, float], polygon: list) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        if not polygon:
            return False
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _track_to_map(self, x: float, y: float) -> Tuple[float, float]:
        # Odom/map frame is world-axis aligned with the spawn point at (0, 0).
        # Only translate by -origin_offset; Isaac Sim's odom inherits world axes
        # from the spawn pose, so the map must NOT be rotated by start yaw.
        return (
            float(x) - self._origin_offset_x,
            float(y) - self._origin_offset_y,
        )

    def _build_midpoint_path(self, points: Sequence[Tuple[float, float]]) -> NavPath:
        msg = NavPath()
        msg.header.frame_id = self._frame_id
        for x, y in points:
            map_x, map_y = self._track_to_map(x, y)
            pose = PoseStamped()
            pose.header.frame_id = self._frame_id
            pose.pose.position.x = map_x
            pose.pose.position.y = map_y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def _build_ground_truth_grid(self, points: Sequence[Tuple[float, float]], lane_polygon=None) -> OccupancyGrid:
        w = max(1, int(round(self._w_m / self._res)))
        h = max(1, int(round(self._h_m / self._res)))
        # Grid is published in the odom/map frame, centered on the robot start.
        # Track coords are shifted by origin_offset so robot_start_pose becomes
        # (0, 0) in the grid. No rotation — odom inherits world axes.
        ox = -self._w_m / 2.0
        oy = -self._h_m / 2.0

        img = np.full((h, w), -1, dtype=np.int8)

        def world_to_cell(x: float, y: float) -> Tuple[int, int]:
            # Convert track coords → start-aligned map, then map → cell index.
            map_x, map_y = self._track_to_map(x, y)
            cx = int((map_x - ox) / self._res)
            cy = int((map_y - oy) / self._res)
            return cx, cy

        # If we have lane boundary polygon from image, use it to fill the actual lane area
        if lane_polygon is not None:
            c_outer, c_inner, image_ppm, image_file = lane_polygon

            track_img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if track_img is not None:
                img_h, img_w = track_img.shape

                # Build a binary mask in image-pixel space (outer filled, inner cut out)
                lane_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.drawContours(lane_mask, [np.int32(c_outer)], -1, 255, -1)
                cv2.drawContours(lane_mask, [np.int32(c_inner)], -1, 0,   -1)

                # Vectorised: get all occupied pixel coords then map to grid cells
                pys, pxs = np.nonzero(lane_mask)
                if len(pxs) > 0:
                    cx_img = img_w / 2.0
                    cy_img = img_h / 2.0
                    ppm = float(image_ppm)

                    # X is negated to match USD floor placement (pygame_to_usd_xy negates X).
                    wx = -(pxs.astype(np.float64) - cx_img) / ppm
                    # y-down to match JSON / USD convention — no flip.
                    wy = (pys.astype(np.float64) - cy_img) / ppm
                    # Convert track coords → start-translated map (no rotation).
                    map_wx = wx - self._origin_offset_x
                    map_wy = wy - self._origin_offset_y
                    cxs = ((map_wx - ox) / self._res).astype(int)
                    cys = ((map_wy - oy) / self._res).astype(int)

                    in_bounds = (cxs >= 0) & (cxs < w) & (cys >= 0) & (cys < h)
                    # Corridor cells are FREE (0). Lane-line / outside cells
                    # remain unknown (-1). Nav2 lethal_cost_threshold >= 90
                    # on /lane_map stays out of the corridor; navigator
                    # _extract_centreline looks for data == 0 (free cells).
                    img[cys[in_bounds], cxs[in_bounds]] = 0

                msg = OccupancyGrid()
                msg.header.frame_id = self._frame_id
                msg.info.resolution = self._res
                msg.info.width = w
                msg.info.height = h
                msg.info.origin.position.x = ox
                msg.info.origin.position.y = oy
                msg.info.origin.orientation.w = 1.0
                msg.data = img.flatten().tolist()
                return msg

        # Fallback: draw constant-width lane centered on centerline
        pts = list(points)
        if len(pts) >= 2:
            rad_cells = max(1, int(math.ceil(self._half_lane / self._res)))
            for i in range(len(pts)):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % len(pts)]
                c0x, c0y = world_to_cell(x0, y0)
                c1x, c1y = world_to_cell(x1, y1)
                steps = max(abs(c1x - c0x), abs(c1y - c0y), 1)
                for k in range(steps + 1):
                    a = k / float(steps)
                    cx = int(round((1.0 - a) * c0x + a * c1x))
                    cy = int(round((1.0 - a) * c0y + a * c1y))
                    for dy in range(-rad_cells, rad_cells + 1):
                        for dx in range(-rad_cells, rad_cells + 1):
                            if dx * dx + dy * dy > rad_cells * rad_cells:
                                continue
                            xx = cx + dx
                            yy = cy + dy
                            if 0 <= xx < w and 0 <= yy < h:
                                img[yy, xx] = 0

        msg = OccupancyGrid()
        msg.header.frame_id = self._frame_id
        msg.info.resolution = self._res
        msg.info.width = w
        msg.info.height = h
        msg.info.origin.position.x = ox
        msg.info.origin.position.y = oy
        msg.info.origin.orientation.w = 1.0
        msg.data = img.flatten().tolist()
        return msg

    def _save_debug_png(self, grid: OccupancyGrid, path_str: str) -> None:
        if not path_str:
            return

        try:
            path = Path(path_str).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)

            arr = np.array(grid.data, dtype=np.int16).reshape((grid.info.height, grid.info.width))
            img = np.zeros((grid.info.height, grid.info.width), dtype=np.uint8)
            img[arr < 0] = 0
            img[np.logical_and(arr >= 0, arr < 50)] = 128
            img[arr >= 50] = 255

            if cv2.imwrite(str(path), img):
                self.get_logger().info(f'Saved ground-truth PNG to {path}')
            else:
                self.get_logger().warn(f'Failed to save ground-truth PNG to {path}')
        except Exception as exc:
            self.get_logger().warn(f'Could not save ground-truth PNG: {exc}')

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._grid.header.stamp = stamp
        self._path.header.stamp = stamp
        for pose in self._path.poses:
            pose.header.stamp = stamp
        self._occ_pub.publish(self._grid)
        self._path_pub.publish(self._path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackGroundTruthNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
