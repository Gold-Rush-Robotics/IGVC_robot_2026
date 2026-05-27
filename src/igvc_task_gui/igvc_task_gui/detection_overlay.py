"""OpenCV bounding box overlay utilities for YOLO detections."""

from __future__ import annotations

import cv2
import numpy as np

# Colour table (BGR) for IGVC-specific detection classes.
# Unknown classes fall back to WHITE.
_CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    'stop_sign': (0, 0, 220),        # red
    'pedestrian': (220, 100, 0),     # blue
    'person': (220, 100, 0),         # blue
    'pothole': (0, 140, 255),        # orange
    'tire': (180, 0, 180),           # purple
    'barrel': (0, 220, 220),         # yellow
    'cone': (0, 200, 255),           # amber
    'obstacle': (50, 50, 200),       # dark red
    'lane_line': (255, 255, 255),    # white
    'white_lane': (255, 255, 255),   # white
    'curved_lane': (200, 255, 200),  # light green
    'parking_space': (100, 255, 100), # green
    'intersection': (255, 200, 100), # light blue
}

_DEFAULT_COLOUR: tuple[int, int, int] = (200, 200, 200)  # light grey

_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1


def _colour_for(class_name: str) -> tuple[int, int, int]:
    return _CLASS_COLOURS.get(class_name.lower(), _DEFAULT_COLOUR)


def draw_detections(
    img_bgr: np.ndarray,
    detections: list,
) -> np.ndarray:
    """Draw YOLO bounding boxes and labels onto *img_bgr* (in-place copy).

    Parameters
    ----------
    img_bgr:
        BGR image as a NumPy array (H×W×3).
    detections:
        List of ``yolo_msgs/Detection`` message objects.  Each must expose
        ``class_name``, ``score``, and ``bbox`` (a ``BoundingBox2D`` with
        ``center.position.{x,y}`` and ``size.{x,y}``).

    Returns
    -------
    np.ndarray
        A new BGR array with boxes and labels rendered.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    for det in detections:
        class_name: str = det.class_name.lower()
        score: float = float(det.score)
        colour = _colour_for(class_name)

        # BoundingBox2D uses centre + half-size in pixel coordinates.
        cx = float(det.bbox.center.position.x)
        cy = float(det.bbox.center.position.y)
        bw = float(det.bbox.size.x)
        bh = float(det.bbox.size.y)

        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(w - 1, int(cx + bw / 2))
        y2 = min(h - 1, int(cy + bh / 2))

        cv2.rectangle(out, (x1, y1), (x2, y2), colour, _BOX_THICKNESS)

        label = f'{class_name} {score:.2f}'
        (tw, th), baseline = cv2.getTextSize(
            label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        # Draw a filled background rectangle behind the label for readability.
        lx1, ly1 = x1, max(0, y1 - th - baseline - 4)
        lx2, ly2 = x1 + tw + 4, y1
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), colour, cv2.FILLED)
        text_colour = (0, 0, 0) if sum(colour) > 400 else (255, 255, 255)
        cv2.putText(
            out, label,
            (x1 + 2, y1 - baseline - 2),
            _FONT, _FONT_SCALE, text_colour, _FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return out
