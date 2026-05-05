"""Scale checks for track-generator-derived costmap fixtures."""

from __future__ import annotations

import sys
from pathlib import Path


TRACK_GENERATOR_DIR = (
    Path(__file__).resolve().parents[3] / 'IGVC_track_generator')
sys.path.append(str(TRACK_GENERATOR_DIR))

from constants import COURSE_AREA_WIDTH_FT  # noqa: E402
from constants import PIXELS_PER_FOOT, WIDTH  # noqa: E402

from costmap_fixtures import make_empty_costmap  # noqa: E402


def test_track_generator_pixel_scale_matches_igvc_width():
    """The generator pixel scale should match the declared IGVC field."""
    assert WIDTH == 1920
    assert COURSE_AREA_WIDTH_FT == 120
    assert PIXELS_PER_FOOT == WIDTH / COURSE_AREA_WIDTH_FT
    assert PIXELS_PER_FOOT == 16.0


def test_costmap_fixture_matches_lane_detection_origin_convention():
    """The local fixture should match lane_detection costmap placement."""
    grid = make_empty_costmap(width_m=10.0, height_m=5.0, resolution=0.05)

    assert grid.header.frame_id == 'base_link'
    assert grid.info.width == 200
    assert grid.info.height == 100
    assert grid.info.origin.position.x == 0.0
    assert grid.info.origin.position.y == -5.0


def test_track_pixels_convert_to_expected_metres():
    """Track widths in pixels should convert to expected metric widths."""
    metres_per_pixel = 0.3048 / PIXELS_PER_FOOT

    assert abs(160 * metres_per_pixel - 3.048) < 1e-9
    assert abs(320 * metres_per_pixel - 6.096) < 1e-9
