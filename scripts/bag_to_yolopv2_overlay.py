#!/usr/bin/env python3
"""Offline YOLOPv2 overlay generator for a ROS 2 rosbag.

Reads a single camera topic from a rosbag (mcap), runs YOLOPv2 on every
frame and writes two MP4s:

    * ``<out_dir>/raw.mp4``      — just the decoded camera frames
    * ``<out_dir>/overlay.mp4``  — drivable-area (green) + lane-lines (red)

This bypasses ROS entirely — no TF, no node graph, no sync.  Handy for
debugging a bag without spinning up the full launch file.

Usage
-----
    python3 scripts/bag_to_yolopv2_overlay.py \
        --bag rosbags/rosbag2_2026_04_21-23_19_04 \
        --weights models/yolopv2.pt \
        --out yolopv2_bag
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Allow ``from igvc_lane_detection.yolopv2_infer import YolopV2`` without
# installing the ROS package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(
    0, str(REPO_ROOT / 'src' / 'igvc_lane_detection'))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from igvc_lane_detection.yolopv2_infer import YolopV2  # noqa: E402

# ROS 2 deps — only used for bag reading + message deserialization.
from rclpy.serialization import deserialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402
import rosbag2_py  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag', required=True, type=Path,
                   help='Path to the rosbag directory (containing metadata.yaml).')
    p.add_argument('--topic', default='/front_zed_camera_x/zed_node/rgb/color/rect/image',
                   help='Image topic to extract.')
    p.add_argument('--weights', type=Path,
                   default=REPO_ROOT / 'models' / 'yolopv2.pt',
                   help='Path to yolopv2.pt TorchScript weights.')
    p.add_argument('--out', type=Path, default=Path('/tmp/yolopv2_bag'),
                   help='Output directory for raw.mp4 and overlay.mp4.')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-half', action='store_true', help='Disable FP16.')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--fps', type=float, default=15.0,
                   help='Output video framerate.')
    p.add_argument('--max-frames', type=int, default=0,
                   help='Stop after N frames (0 = all).')
    return p.parse_args()


def open_reader(bag_path: Path):
    storage = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='mcap')
    converter = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)
    return reader


def image_msg_to_bgr(msg) -> np.ndarray | None:
    """Decode a sensor_msgs/Image to BGR without cv_bridge."""
    h, w = msg.height, msg.width
    enc = msg.encoding
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ('bgr8', 'rgb8'):
        arr = buf.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)
        if enc == 'rgb8':
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr.copy()
    if enc in ('bgra8', 'rgba8'):
        arr = buf.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4)
        code = cv2.COLOR_RGBA2BGR if enc == 'rgba8' else cv2.COLOR_BGRA2BGR
        return cv2.cvtColor(arr, code)
    if enc == 'mono8':
        arr = buf.reshape(h, msg.step)[:, :w].reshape(h, w)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    print(f'[warn] unsupported encoding {enc!r}', file=sys.stderr)
    return None


def make_overlay(bgr: np.ndarray, da: np.ndarray, ll: np.ndarray) -> np.ndarray:
    ov = bgr.copy()
    colour = np.zeros_like(ov)
    colour[da > 0] = (0, 255, 0)   # green = drivable area
    colour[ll > 0] = (0, 0, 255)   # red   = lane lines
    mask = (da > 0) | (ll > 0)
    ov[mask] = cv2.addWeighted(ov, 0.5, colour, 0.5, 0.0)[mask]
    return ov


def main() -> int:
    args = parse_args()

    if not args.bag.exists():
        print(f'Bag not found: {args.bag}', file=sys.stderr)
        return 2
    if not args.weights.exists():
        print(f'Weights not found: {args.weights}', file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    print(f'Loading YOLOPv2 from {args.weights} on {args.device} '
          f'(half={not args.no_half})…', flush=True)
    model = YolopV2(
        weights_path=str(args.weights),
        device=args.device,
        half=not args.no_half,
        img_size=args.img_size,
    )
    model.load()
    if model.fallback_warning:
        print(f'[warn] {model.fallback_warning}', file=sys.stderr)
    print(f'Model ready on {model.device} (half={model.half}).')

    reader = open_reader(args.bag)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if args.topic not in topic_types:
        print(f'Topic {args.topic!r} not in bag.\nAvailable:')
        for name, type_ in topic_types.items():
            if type_ == 'sensor_msgs/msg/Image':
                print(f'  {name}')
        return 3

    msg_type = get_message(topic_types[args.topic])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))

    ffmpeg_bin = shutil.which('ffmpeg')
    if ffmpeg_bin is None:
        print('ERROR: ffmpeg not found on PATH — install it '
              '(`sudo apt install ffmpeg`) for playable H.264 output.',
              file=sys.stderr)
        return 4

    raw_proc = None
    ov_proc = None
    raw_path = args.out / 'raw.mp4'
    ov_path  = args.out / 'overlay.mp4'

    def spawn_ffmpeg(path: Path, w: int, h: int) -> subprocess.Popen:
        cmd = [
            ffmpeg_bin, '-y',
            '-loglevel', 'error',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}',
            '-r', f'{args.fps}',
            '-i', '-',
            '-an',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'veryfast',
            '-crf', '20',
            '-movflags', '+faststart',
            str(path),
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n = 0
    try:
        while reader.has_next():
            if args.max_frames and n >= args.max_frames:
                break
            _, raw, _ = reader.read_next()
            msg = deserialize_message(raw, msg_type)
            bgr = image_msg_to_bgr(msg)
            if bgr is None:
                continue

            if raw_proc is None:
                h, w = bgr.shape[:2]
                # H.264 needs even dimensions.
                if w % 2 or h % 2:
                    print(f'[warn] padding {w}x{h} to even dims for H.264.')
                raw_proc = spawn_ffmpeg(raw_path, w - (w % 2), h - (h % 2))
                ov_proc  = spawn_ffmpeg(ov_path,  w - (w % 2), h - (h % 2))
                print(f'Writing {w}x{h} H.264 @ {args.fps} fps → {args.out}')
                target_w, target_h = w - (w % 2), h - (h % 2)

            da, ll = model.infer(bgr)
            if da.shape[:2] != bgr.shape[:2]:
                da = cv2.resize(da, (bgr.shape[1], bgr.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
                ll = cv2.resize(ll, (bgr.shape[1], bgr.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

            ov = make_overlay(bgr, da, ll)
            if (target_w, target_h) != (bgr.shape[1], bgr.shape[0]):
                bgr = bgr[:target_h, :target_w]
                ov  = ov[:target_h, :target_w]

            raw_proc.stdin.write(bgr.tobytes())
            ov_proc.stdin.write(ov.tobytes())

            n += 1
            if n % 20 == 0:
                print(f'  processed {n} frames', flush=True)
    finally:
        for proc in (raw_proc, ov_proc):
            if proc is not None:
                proc.stdin.close()
                proc.wait()

    print(f'Done. {n} frames → {raw_path} , {ov_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
