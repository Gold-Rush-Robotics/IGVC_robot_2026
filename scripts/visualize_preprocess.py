#!/usr/bin/env python3
"""Visualize the YOLOPv2 preprocessing pipeline stage-by-stage.

Usage:
    python3 scripts/visualize_preprocess.py <image_path> [options]

    # Use a live ROS topic instead of a file:
    python3 scripts/visualize_preprocess.py --ros-topic /front_zed_camera_x/zed_node/rgb/color/rect/image

Options:
    --clahe-clip FLOAT     CLAHE clip limit          (default: 2.0)
    --clahe-tile W H       CLAHE tile grid size       (default: 8 8)
    --blur-ksize W H       Gaussian kernel size       (default: 5 5)
    --blur-sigma FLOAT     Gaussian sigma (0=auto)    (default: 0.0)
    --img-size INT         Letterbox side length (px) (default: 384)
    --no-preprocess        Skip CLAHE+blur stage
    --ros-topic TOPIC      Subscribe to a ROS2 image topic (requires rclpy)
    --save PATH            Save the figure instead of displaying it
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers (mirrors yolopv2_infer.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _letterbox(img, new_shape=640, color=(114, 114, 114),
               auto=True, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2.0
    dh /= 2.0
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
    return img, (dw, dh)


def run_pipeline(bgr: np.ndarray, args) -> dict[str, np.ndarray]:
    """Return an ordered dict of {stage_name: bgr_image} for every step."""
    stages: dict[str, np.ndarray] = {}

    stages["1  Original"] = bgr.copy()

    # Step 1 – YOLOPv2 demo resize (1280 × 720)
    resized = cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)
    stages["2  Resized\n(1280×720)"] = resized.copy()

    # Step 2 – CLAHE + Gaussian blur
    if args.preprocess:
        clahe = cv2.createCLAHE(
            clipLimit=args.clahe_clip,
            tileGridSize=tuple(args.clahe_tile))

        yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        after_clahe = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        stages["3  CLAHE\n(clip={:.1f}, tile={}×{})".format(
            args.clahe_clip, *args.clahe_tile)] = after_clahe.copy()

        kw, kh = args.blur_ksize
        if kw > 1 and kh > 1:
            after_blur = cv2.GaussianBlur(
                after_clahe, (kw, kh), args.blur_sigma)
            stages["4  Gaussian Blur\n(k={}×{}, σ={})".format(
                kw, kh, args.blur_sigma)] = after_blur.copy()
            pre_lb = after_blur
        else:
            pre_lb = after_clahe
    else:
        pre_lb = resized

    # Step 3 – Letterbox
    lb, (dw, dh) = _letterbox(pre_lb, new_shape=args.img_size,
                               auto=True, stride=32)
    n = len(stages) + 1
    pad_label = "{}  Letterbox\n({}×{}, pad={},{})" .format(
        n, lb.shape[1], lb.shape[0],
        int(dw * 2), int(dh * 2))
    stages[pad_label] = lb.copy()

    # Step 4 – White-region mask (HSV threshold, floor ROI only)
    h_img = pre_lb.shape[0]
    roi_top    = int(h_img * args.roi_top_frac)
    roi_bottom = int(h_img * args.roi_bottom_frac)
    hsv = cv2.cvtColor(pre_lb, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        hsv,
        np.array([0,   0,   args.white_v_min]),
        np.array([180, args.white_s_max, 255]))
    # Zero out everything outside the floor ROI
    roi_gate = np.zeros_like(white_mask)
    roi_gate[roi_top:roi_bottom, :] = 255
    white_mask = cv2.bitwise_and(white_mask, roi_gate)
    n = len(stages) + 1
    stages[f"{n}  White Mask\n(V≥{args.white_v_min}, S≤{args.white_s_max}, "
           f"ROI {args.roi_top_frac:.0%}–{args.roi_bottom_frac:.0%})"] = \
        cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)

    # Step 5 – Canny edge detection (within ROI on grayscale)
    gray = cv2.cvtColor(pre_lb, cv2.COLOR_BGR2GRAY)
    gray_roi = gray.copy()
    gray_roi[:roi_top, :]  = 0
    gray_roi[roi_bottom:, :] = 0
    canny = cv2.Canny(gray_roi, args.canny_low, args.canny_high,
                      apertureSize=3, L2gradient=True)
    n = len(stages) + 1
    stages[f"{n}  Canny Edges\n(lo={args.canny_low}, hi={args.canny_high})"] = \
        cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    # Step 6 – White edges: Canny ∩ white mask (drops non-white edges)
    white_edges = cv2.bitwise_and(canny, white_mask)
    n = len(stages) + 1
    stages[f"{n}  White Edges\n(Canny ∩ white mask)"] = \
        cv2.cvtColor(white_edges, cv2.COLOR_GRAY2BGR)

    # Step 7 – Probabilistic Hough lines on white edges
    hough_vis = pre_lb.copy()
    lines = cv2.HoughLinesP(
        white_edges,
        rho=1, theta=np.pi / 180,
        threshold=args.hough_thresh,
        minLineLength=args.hough_min_len,
        maxLineGap=args.hough_max_gap)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    n = len(stages) + 1
    stages[f"{n}  Hough Lines\n({0 if lines is None else len(lines)} lines, "
           f"thresh={args.hough_thresh}, gap={args.hough_max_gap})"] = hough_vis

    # Step 8 – Overlay: white edges thickened and highlighted on preprocessed image
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thick_edges = cv2.dilate(white_edges, thick_kernel, iterations=2)
    overlay = pre_lb.copy()
    overlay[thick_edges > 0] = (255, 255, 255)
    n = len(stages) + 1
    stages[f"{n}  Overlay"] = overlay

    return stages


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def _show_stages(stages: dict[str, np.ndarray], save_path: str | None = None):
    """Tile all stages into one figure and show/save it."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n = len(stages)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 6, rows * 4.5))
    fig.suptitle("YOLOPv2 Preprocessing Pipeline", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(rows, cols, figure=fig,
                           hspace=0.35, wspace=0.05)

    for idx, (title, img) in enumerate(stages.items()):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{img.shape[1]}×{img.shape[0]} px", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused cells
    for idx in range(n, rows * cols):
        fig.add_subplot(gs[idx // cols, idx % cols]).set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 subscriber mode
# ─────────────────────────────────────────────────────────────────────────────

def _ros_mode(args):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    import cv_bridge

    class PreviewNode(Node):
        def __init__(self):
            super().__init__("preprocess_visualizer")
            self._bridge = cv_bridge.CvBridge()
            self._sub = self.create_subscription(
                Image, args.ros_topic, self._cb, 1)
            self._received = False
            self.get_logger().info(
                f"Waiting for a frame on '{args.ros_topic}'…")

        def _cb(self, msg: Image):
            if self._received:
                return
            self._received = True
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            stages = run_pipeline(bgr, args)
            _show_stages(stages, args.save)

    rclpy.init()
    node = PreviewNode()
    while rclpy.ok() and not node._received:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize the YOLOPv2 preprocessing pipeline.")
    p.add_argument("image", nargs="?",
                   help="Path to an input image (BGR).")
    p.add_argument("--ros-topic", metavar="TOPIC",
                   help="Subscribe to this ROS2 sensor_msgs/Image topic "
                        "instead of reading a file.")
    p.add_argument("--clahe-clip", type=float, default=2.0)
    p.add_argument("--clahe-tile", type=int, nargs=2, default=[8, 8],
                   metavar=("W", "H"))
    p.add_argument("--blur-ksize", type=int, nargs=2, default=[5, 5],
                   metavar=("W", "H"))
    p.add_argument("--blur-sigma", type=float, default=0.0)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--no-preprocess", dest="preprocess",
                   action="store_false", default=True,
                   help="Skip CLAHE + blur stages.")
    p.add_argument("--white-v-min", type=int, default=120,
                   help="HSV V lower bound for white detection (default: 100).")
    p.add_argument("--white-s-max", type=int, default=50,
                   help="HSV S upper bound for white detection (default: 50).")
    p.add_argument("--dilate-px", type=int, default=50,
                   help="Dilation kernel size in px (default: 50, 0=disable). "
                        "Kept for backward compat but close is preferred.")
    p.add_argument("--dilate-iters", type=int, default=2,
                   help="Dilation iterations (default: 2).")
    p.add_argument("--close-px", type=int, default=25,
                   help="Morphological close kernel size in px to bridge gaps "
                        "without thickening (default: 25, 0=disable).")
    p.add_argument("--close-iters", type=int, default=2,
                   help="Number of close iterations (default: 2).")
    p.add_argument("--canny-low", type=int, default=30,
                   help="Canny lower hysteresis threshold (default: 30).")
    p.add_argument("--canny-high", type=int, default=100,
                   help="Canny upper hysteresis threshold (default: 100).")
    p.add_argument("--hough-thresh", type=int, default=20,
                   help="HoughLinesP accumulator threshold (default: 20).")
    p.add_argument("--hough-min-len", type=int, default=20,
                   help="HoughLinesP minimum line length in px (default: 20).")
    p.add_argument("--hough-max-gap", type=int, default=30,
                   help="HoughLinesP maximum gap between segments in px (default: 30).")
    p.add_argument("--roi-top-frac", type=float, default=0.55,
                   help="Fraction from top where floor ROI starts (default: 0.55).")
    p.add_argument("--roi-bottom-frac", type=float, default=0.95,
                   help="Fraction from top where floor ROI ends (default: 0.95).")
    p.add_argument("--save", metavar="PATH",
                   help="Save figure to this path instead of displaying.")
    return p.parse_args()


def main():
    args = _parse()

    if args.ros_topic:
        _ros_mode(args)
        return

    if not args.image:
        print("error: provide an image path or --ros-topic", file=sys.stderr)
        sys.exit(1)

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"error: could not read '{args.image}'", file=sys.stderr)
        sys.exit(1)

    stages = run_pipeline(bgr, args)
    _show_stages(stages, args.save)


if __name__ == "__main__":
    main()
