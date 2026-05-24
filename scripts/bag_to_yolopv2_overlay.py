#!/usr/bin/env python3
"""Offline YOLOPv2 overlay generator for a ROS 2 rosbag.

Reads one or more camera topics from a rosbag (mcap), runs YOLOPv2 on
every frame and writes, per topic, two MP4s into ``<out>/<cam_name>/``:

    * ``raw.mp4``      — decoded camera frames
    * ``overlay.mp4``  — drivable-area (green) + lane-lines (red)

This bypasses ROS entirely — no TF, no node graph, no sync.  Handy for
debugging a bag without spinning up the full launch file.

The side ZED cameras are mounted 90 deg off-axis, so the same rotation
scheme used by the ROS segmentation node is supported here via
``--rotations``: CCW degrees per topic (0 / 90 / 180 / 270).  The image
is rotated into road-up orientation before inference and the masks are
rotated back before being overlaid on the original frame.

Usage
-----
    python3 scripts/bag_to_yolopv2_overlay.py \
        --bag rosbags/rosbag2_2026_04_21-23_19_04 \
        --weights models/yolopv2.pt \
        --out yolopv2_bag
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

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


DEFAULT_TOPICS = [
    '/front_zed_camera_x/zed_node/rgb/color/rect/image',
    '/left_zed_camera_x/zed_node/rgb/color/rect/image',
    '/right_zed_camera_x/zed_node/rgb/color/rect/image',
]
# Matches lane_segmentation_config.yaml: front=0, left=90 CCW, right=270 CCW.
DEFAULT_ROTATIONS = [0, 90, 270]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag', required=True, type=Path,
                   help='Path to the rosbag directory (containing metadata.yaml).')
    p.add_argument('--topics', nargs='+', default=DEFAULT_TOPICS,
                   help='Image topics to extract (one MP4 pair per topic).')
    p.add_argument('--rotations', nargs='+', type=int,
                   default=DEFAULT_ROTATIONS,
                   help='CCW degrees per topic — 0/90/180/270. Applied '
                        'before inference; masks are rotated back before '
                        'overlay. Defaults match the ROS node config.')
    p.add_argument('--swap-da-ll', action='store_true',
                   help='Swap drivable-area and lane-line masks (matches '
                        'lane_segmentation swap_da_ll=true).')
    p.add_argument('--weights', type=Path,
                   default=REPO_ROOT / 'models' / 'yolopv2.pt',
                   help='Path to yolopv2.pt TorchScript weights.')
    p.add_argument('--out', type=Path, default=Path('/tmp/yolopv2_bag'),
                   help='Output directory (per-topic subdirs created inside).')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-half', action='store_true', help='Disable FP16.')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--fps', type=float, default=15.0,
                   help='Output video framerate.')
    p.add_argument('--max-frames', type=int, default=0,
                   help='Stop after N frames *per topic* (0 = all).')
    return p.parse_args()


def open_reader(bag_path: Path):
    storage = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='mcap')
    converter = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)
    return reader


def image_msg_to_bgr(msg) -> Optional[np.ndarray]:
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


def rotate_image(img: np.ndarray, deg: int) -> np.ndarray:
    """Rotate image/mask by multiple of 90 deg CCW (no interpolation)."""
    if img is None or img.size == 0:
        return img
    deg = int(deg) % 360
    if deg == 0:
        return img
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f'rotation must be a multiple of 90 deg, got {deg}')


def make_overlay(bgr: np.ndarray, da: np.ndarray, ll: np.ndarray) -> np.ndarray:
    ov = bgr.copy()
    color = np.zeros_like(ov)
    color[da > 0] = (0, 255, 0)   # green = drivable area
    color[ll > 0] = (0, 0, 255)   # red   = lane lines
    mask = (da > 0) | (ll > 0)
    ov[mask] = cv2.addWeighted(ov, 0.5, color, 0.5, 0.0)[mask]
    return ov


def topic_slug(topic: str) -> str:
    """Filesystem-safe short name for a topic (e.g. 'front_zed_camera_x')."""
    parts = [p for p in topic.strip('/').split('/') if p]
    if not parts:
        return 'cam'
    return parts[0]


def spawn_ffmpeg(ffmpeg_bin: str, path: Path,
                 w: int, h: int, fps: float) -> subprocess.Popen:
    cmd = [
        ffmpeg_bin, '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{w}x{h}',
        '-r', f'{fps}',
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


class TopicWriter:
    """Per-topic lazily-initialised raw + overlay MP4 writer."""

    def __init__(self, topic: str, rotation: int,
                 out_dir: Path, ffmpeg_bin: str, fps: float):
        self.topic = topic
        self.rotation = rotation
        self.out_dir = out_dir
        self.ffmpeg_bin = ffmpeg_bin
        self.fps = fps
        self.raw_proc: Optional[subprocess.Popen] = None
        self.ov_proc: Optional[subprocess.Popen] = None
        self.target_w = 0
        self.target_h = 0
        self.n = 0

    def _spawn(self, w: int, h: int) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.target_w = w - (w % 2)
        self.target_h = h - (h % 2)
        raw_path = self.out_dir / 'raw.mp4'
        ov_path  = self.out_dir / 'overlay.mp4'
        self.raw_proc = spawn_ffmpeg(
            self.ffmpeg_bin, raw_path, self.target_w, self.target_h, self.fps)
        self.ov_proc = spawn_ffmpeg(
            self.ffmpeg_bin, ov_path,  self.target_w, self.target_h, self.fps)
        print(f'[{self.topic}] writing {self.target_w}x{self.target_h} '
              f'H.264 @ {self.fps} fps -> {self.out_dir}')

    def write(self, bgr: np.ndarray, overlay: np.ndarray) -> None:
        if self.raw_proc is None:
            self._spawn(bgr.shape[1], bgr.shape[0])
        assert self.raw_proc is not None and self.ov_proc is not None
        if (self.target_w, self.target_h) != (bgr.shape[1], bgr.shape[0]):
            bgr = bgr[:self.target_h, :self.target_w]
            overlay = overlay[:self.target_h, :self.target_w]
        self.raw_proc.stdin.write(bgr.tobytes())
        self.ov_proc.stdin.write(overlay.tobytes())
        self.n += 1

    def close(self) -> None:
        for proc in (self.raw_proc, self.ov_proc):
            if proc is not None and proc.stdin is not None:
                proc.stdin.close()
                proc.wait()


def main() -> int:
    args = parse_args()

    if not args.bag.exists():
        print(f'Bag not found: {args.bag}', file=sys.stderr)
        return 2
    if not args.weights.exists():
        print(f'Weights not found: {args.weights}', file=sys.stderr)
        return 2

    # Align rotations to topics: pad with 0 or truncate.
    rotations: List[int] = list(args.rotations)
    if len(rotations) < len(args.topics):
        rotations += [0] * (len(args.topics) - len(rotations))
    rotations = rotations[:len(args.topics)]
    for r in rotations:
        if r % 90 != 0:
            print(f'ERROR: rotation {r} is not a multiple of 90.',
                  file=sys.stderr)
            return 2

    ffmpeg_bin = shutil.which('ffmpeg')
    if ffmpeg_bin is None:
        print('ERROR: ffmpeg not found on PATH — install it '
              '(`sudo apt install ffmpeg`) for playable H.264 output.',
              file=sys.stderr)
        return 4

    args.out.mkdir(parents=True, exist_ok=True)

    print(f'Loading YOLOPv2 from {args.weights} on {args.device} '
          f'(half={not args.no_half})...', flush=True)
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

    # Validate every requested topic; print available image topics if any
    # are missing so the user can correct their CLI invocation quickly.
    missing = [t for t in args.topics if t not in topic_types]
    if missing:
        print(f'Topics not in bag: {missing}\nAvailable image topics:')
        for name, type_ in topic_types.items():
            if type_ == 'sensor_msgs/msg/Image':
                print(f'  {name}')
        return 3

    msg_classes: Dict[str, type] = {
        t: get_message(topic_types[t]) for t in args.topics
    }

    reader.set_filter(rosbag2_py.StorageFilter(topics=list(args.topics)))

    writers: Dict[str, TopicWriter] = {}
    for topic, rot in zip(args.topics, rotations):
        sub = args.out / topic_slug(topic)
        writers[topic] = TopicWriter(
            topic=topic, rotation=rot,
            out_dir=sub, ffmpeg_bin=ffmpeg_bin, fps=args.fps)
        print(f'  topic={topic}  rotation={rot} deg  out={sub}')

    total = 0
    try:
        while reader.has_next():
            topic, raw, _ = reader.read_next()
            writer = writers.get(topic)
            if writer is None:
                continue
            if args.max_frames and writer.n >= args.max_frames:
                # Skip this topic once its quota is reached; continue so
                # other topics still get processed.
                if all(w.n >= args.max_frames for w in writers.values()):
                    break
                continue

            msg = deserialize_message(raw, msg_classes[topic])
            bgr = image_msg_to_bgr(msg)
            if bgr is None:
                continue

            # Rotate into canonical road-up orientation before inference.
            rot = writer.rotation
            bgr_in = rotate_image(bgr, rot) if rot else bgr
            try:
                da, ll = model.infer(bgr_in)
            except Exception as e:
                print(f'[{topic}] inference error: {e}', file=sys.stderr)
                continue
            if args.swap_da_ll:
                da, ll = ll, da

            # Resize masks to inference-frame size then rotate back so
            # they line up with the original BGR we're overlaying on.
            if da.shape[:2] != bgr_in.shape[:2]:
                da = cv2.resize(da, (bgr_in.shape[1], bgr_in.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
                ll = cv2.resize(ll, (bgr_in.shape[1], bgr_in.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            if rot:
                da = rotate_image(da, -rot)
                ll = rotate_image(ll, -rot)

            ov = make_overlay(bgr, da, ll)
            writer.write(bgr, ov)

            total += 1
            if total % 20 == 0:
                counts = ' '.join(f'{topic_slug(t)}={w.n}'
                                  for t, w in writers.items())
                print(f'  processed {total} frames  [{counts}]', flush=True)
    finally:
        for w in writers.values():
            w.close()

    for topic, w in writers.items():
        print(f'Done. {w.n} frames -> {w.out_dir}/raw.mp4 , '
              f'{w.out_dir}/overlay.mp4')
    return 0


if __name__ == '__main__':
    sys.exit(main())
