"""Shared geometry / TF / depth-sampling helpers.

These were originally private methods of
``igvc_lane_detection.lane_detection.LaneDetectionNode``.  They are lifted
into a module-level form so that the new segmentation-based node can reuse
the same projection math without duplicating ~200 lines or importing a
large ROS-coupled class.  ``lane_detection.py`` is intentionally left
unchanged; this module is additive.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit-quaternion → 3×3 rotation matrix (float32)."""
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=np.float32)


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw component of a unit quaternion (rad)."""
    return float(np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ))


def sample_valid_depth(
    depth: np.ndarray,
    u: int,
    v: int,
    radius: int = 2,
    min_d: float = 0.5,
    max_d: float = 20.0,
) -> Optional[float]:
    """Median of finite, in-range depth samples in a ``(2r+1)×(2r+1)`` patch.

    Mirrors ``LaneDetectionNode._sample_valid_depth`` exactly so behaviour
    stays identical between the two nodes.
    """
    r = max(0, int(radius))
    u0 = max(0, u - r)
    u1 = min(depth.shape[1], u + r + 1)
    v0 = max(0, v - r)
    v1 = min(depth.shape[0], v + r + 1)
    patch = depth[v0:v1, u0:u1]
    if patch.size == 0:
        return None

    valid = patch[np.isfinite(patch)]
    valid = valid[(valid > min_d) & (valid < max_d)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def pixel_to_base(
    u: float,
    v: float,
    d: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rot: Optional[np.ndarray] = None,
    trans: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Project a single image pixel + depth into ``base_link`` (x,y,z).

    Registered depth images are in the camera optical frame
    (x=right, y=down, z=forward).  When ``rot`` and ``trans`` are provided
    they must express ``base_link ← camera`` so the returned tuple is in
    ``base_link``.  If either is ``None`` a pinhole-only fallback is used
    and ``(forward, lateral, 0.0)`` is returned in the camera's forward
    frame — matching ``LaneDetectionNode._line_to_3d`` behaviour.
    """
    if rot is None or trans is None:
        fwd = float(d)
        lat = float(-(u - cx) * d / fx)
        return fwd, lat, 0.0

    point_cam = np.array([
        (u - cx) * d / fx,
        (v - cy) * d / fy,
        d,
    ], dtype=np.float32)
    point_base = rot @ point_cam + trans
    return float(point_base[0]), float(point_base[1]), float(point_base[2])


def lookup_tf(tf_buffer, target_frame: str, source_frame: str, stamp):
    """TF lookup at the requested timestamp.

    Camera-derived occupancy data must not silently use an arbitrary latest
    transform; doing so can mix image/depth data with robot poses outside the
    timing budget.  ``stamp=None`` is still treated as "latest" for callers
    that are explicitly not tied to a sensor sample.
    """
    try:
        t = Time.from_msg(stamp) if stamp is not None else Time()
        return tf_buffer.lookup_transform(
            target_frame, source_frame, t,
            timeout=Duration(seconds=0.05))
    except Exception:
        return None
