"""Deterministic OccupancyGrid fixtures for lane-navigation tests."""

from __future__ import annotations

from collections.abc import Callable

from nav_msgs.msg import OccupancyGrid

import numpy as np


FREE = 0
LETHAL = 100
UNKNOWN = -1


def make_empty_costmap(
    *,
    width_m: float = 10.0,
    height_m: float = 5.0,
    resolution: float = 0.05,
    frame_id: str = 'base_link',
) -> OccupancyGrid:
    """Create an unknown rolling costmap matching lane_detection output."""
    width_cells = int(round(width_m / resolution))
    height_cells = int(round(height_m / resolution))

    grid = OccupancyGrid()
    grid.header.frame_id = frame_id
    grid.info.resolution = resolution
    grid.info.width = width_cells
    grid.info.height = height_cells
    grid.info.origin.position.x = 0.0
    grid.info.origin.position.y = -width_m / 2.0
    grid.info.origin.orientation.w = 1.0
    grid.data = [UNKNOWN] * (width_cells * height_cells)
    return grid


def lateral_to_col(grid: OccupancyGrid, lateral_m: float) -> int:
    """Convert base_link lateral metres to a costmap column."""
    resolution = grid.info.resolution
    origin_y = grid.info.origin.position.y
    return int(round((lateral_m - origin_y) / resolution))


def row_to_forward(grid: OccupancyGrid, row: int) -> float:
    """Convert a costmap row to forward metres in base_link."""
    return row * grid.info.resolution


def costmap_to_array(grid: OccupancyGrid) -> np.ndarray:
    """Return OccupancyGrid data as a height x width int8 array."""
    return np.asarray(grid.data, dtype=np.int8).reshape(
        grid.info.height, grid.info.width)


def make_corridor_costmap(
    *,
    centerline: Callable[[float], float] | float = 0.0,
    lane_width_m: Callable[[float], float] | float = 2.4,
    sensed_start_m: float = 0.25,
    sensed_end_m: float = 4.0,
    width_m: float = 10.0,
    height_m: float = 5.0,
    resolution: float = 0.05,
) -> OccupancyGrid:
    """Build a local lane corridor with lethal boundaries and free interior."""
    grid = make_empty_costmap(
        width_m=width_m,
        height_m=height_m,
        resolution=resolution,
    )
    data = costmap_to_array(grid)

    for row in range(grid.info.height):
        forward = row_to_forward(grid, row)
        if forward < sensed_start_m or forward > sensed_end_m:
            continue

        center = centerline(forward) if callable(centerline) else centerline
        lane_width = (
            lane_width_m(forward) if callable(lane_width_m) else lane_width_m)
        left_col = lateral_to_col(grid, center + lane_width / 2.0)
        right_col = lateral_to_col(grid, center - lane_width / 2.0)
        lo = max(0, min(left_col, right_col))
        hi = min(grid.info.width - 1, max(left_col, right_col))
        if lo >= hi:
            continue

        data[row, lo] = LETHAL
        data[row, hi] = LETHAL
        data[row, lo + 1:hi] = FREE

    grid.data = data.reshape(-1).tolist()
    return grid


def erase_rows(
    grid: OccupancyGrid,
    start_m: float,
    end_m: float,
) -> OccupancyGrid:
    """Replace a forward span with unknown cells."""
    data = costmap_to_array(grid).copy()
    start_row = max(0, int(round(start_m / grid.info.resolution)))
    end_row = min(grid.info.height, int(round(end_m / grid.info.resolution)))
    data[start_row:end_row, :] = UNKNOWN
    grid.data = data.reshape(-1).tolist()
    return grid
