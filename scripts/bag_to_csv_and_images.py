#!/usr/bin/env python3
"""Export a ROS 2 rosbag to ROS-free CSVs, PNGs and PLYs.

For every topic in the bag this script produces a file under ``<out>/``
organised by topic. Everything is keyed on the bag-recorded timestamp
(``bag_time_ns``) and, where available, the message header stamp
(``header_stamp_ns``) so the different streams can be joined/synced in
pandas without touching ROS.

Output layout
-------------
    <out>/
        topics.csv                        — summary of every topic
        csv/<topic_slug>.csv              — flattened field dumps
        images/<topic_slug>/<stamp>.png   — rgb / mono / compressed images
        depth/<topic_slug>/<stamp>.png    — 16-bit depth PNGs (mm)
        clouds/<topic_slug>/<stamp>.ply   — binary-little-endian PLY files
        tf/tf.csv, tf/tf_static.csv       — one row per transform

Point clouds, images and tf get both the binary artifacts *and* an index
CSV that points at every file, so you can still treat them as a table.

Usage
-----
    python3 scripts/bag_to_csv_and_images.py \
        --bag rosbags/rosbag2_2026_04_10-02_29_45 \
        --out rosbag_export
"""

from __future__ import annotations

import argparse
import array
import csv
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # PNG fallback below handles this.

# ROS deps — only for deserialising messages. No node graph.
from rclpy.serialization import deserialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402
import rosbag2_py  # noqa: E402


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def slugify(topic: str) -> str:
    s = topic.strip('/').replace('/', '__')
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
    return s or 'root'


def stamp_to_ns(stamp) -> Optional[int]:
    """Convert a builtin_interfaces/Time to int ns, or None."""
    if stamp is None:
        return None
    sec = getattr(stamp, 'sec', None)
    nsec = getattr(stamp, 'nanosec', None)
    if sec is None or nsec is None:
        return None
    return int(sec) * 1_000_000_000 + int(nsec)


def header_stamp_ns(msg) -> Optional[int]:
    hdr = getattr(msg, 'header', None)
    if hdr is None:
        return None
    return stamp_to_ns(getattr(hdr, 'stamp', None))


def header_frame(msg) -> str:
    hdr = getattr(msg, 'header', None)
    if hdr is None:
        return ''
    return getattr(hdr, 'frame_id', '') or ''


_PRIM = (bool, int, float, str, bytes)


def flatten(msg: Any, prefix: str = '',
            out: Optional[Dict[str, Any]] = None,
            max_list_inline: int = 16) -> Dict[str, Any]:
    """Flatten a ROS message into a {dotted.name: scalar-or-json} dict.

    Long numeric arrays are JSON-encoded so the CSV stays one row per msg.
    Binary ``bytes`` fields are replaced by their length to avoid bloat.
    """
    if out is None:
        out = {}

    if msg is None or isinstance(msg, _PRIM):
        if isinstance(msg, bytes):
            out[prefix or 'value'] = f'<{len(msg)} bytes>'
        else:
            out[prefix or 'value'] = msg
        return out

    if isinstance(msg, (list, tuple, array.array, np.ndarray)):
        seq = list(msg)
        # Short, all-primitive lists → JSON for readability.
        if len(seq) <= max_list_inline and all(isinstance(x, _PRIM) or x is None for x in seq):
            out[prefix or 'value'] = json.dumps(seq, default=_json_default)
            return out
        # Numeric arrays of any length → JSON (compact).
        if all(isinstance(x, (int, float, bool)) or x is None for x in seq):
            out[prefix or 'value'] = json.dumps(seq, default=_json_default)
            return out
        # Complex list: index each entry.
        for i, item in enumerate(seq):
            flatten(item, f'{prefix}[{i}]', out, max_list_inline)
        if not seq:
            out[prefix or 'value'] = '[]'
        return out

    slots = getattr(msg, '__slots__', None)
    if slots:
        for slot in slots:
            name = slot.lstrip('_')
            try:
                val = getattr(msg, name)
            except AttributeError:
                try:
                    val = getattr(msg, slot)
                except AttributeError:
                    continue
            key = f'{prefix}.{name}' if prefix else name
            flatten(val, key, out, max_list_inline)
        return out

    # Unknown object — stringify.
    out[prefix or 'value'] = repr(msg)
    return out


def _json_default(o: Any):
    if isinstance(o, (bytes, bytearray)):
        return f'<{len(o)} bytes>'
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

def image_msg_to_array(msg) -> Tuple[Optional[np.ndarray], str]:
    """Decode sensor_msgs/Image into a numpy array. Returns (arr, kind).

    kind is one of: 'rgb', 'depth', 'mono', ''.
    """
    h, w = msg.height, msg.width
    enc = msg.encoding
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if h == 0 or w == 0 or buf.size == 0:
        return None, ''

    try:
        if enc in ('bgr8', 'rgb8'):
            arr = buf.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3).copy()
            if enc == 'rgb8' and cv2 is not None:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr, 'rgb'
        if enc in ('bgra8', 'rgba8'):
            arr = buf.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4).copy()
            if cv2 is not None:
                code = cv2.COLOR_RGBA2BGR if enc == 'rgba8' else cv2.COLOR_BGRA2BGR
                arr = cv2.cvtColor(arr, code)
            return arr, 'rgb'
        if enc == 'mono8':
            arr = buf.reshape(h, msg.step)[:, :w].reshape(h, w).copy()
            return arr, 'mono'
        if enc == 'mono16':
            arr = np.frombuffer(msg.data, dtype='<u2').reshape(h, msg.step // 2)[:, :w].copy()
            return arr, 'mono'
        if enc in ('16UC1',):
            arr = np.frombuffer(msg.data, dtype='<u2').reshape(h, msg.step // 2)[:, :w].copy()
            return arr, 'depth'
        if enc in ('32FC1',):
            arr = np.frombuffer(msg.data, dtype='<f4').reshape(h, msg.step // 4)[:, :w].copy()
            return arr, 'depth'
        if enc == 'yuv422_yuy2' and cv2 is not None:
            arr = buf.reshape(h, w, 2)
            return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUYV), 'rgb'
    except Exception as e:  # noqa: BLE001
        print(f'[warn] image decode failed ({enc}): {e}', file=sys.stderr)
        return None, ''

    print(f'[warn] unsupported image encoding: {enc}', file=sys.stderr)
    return None, ''


def write_png(path: Path, arr: np.ndarray, kind: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'depth':
        if arr.dtype == np.float32:
            # meters → millimeters, clip to uint16 range.
            mm = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
            mm = np.clip(mm, 0, 65535).astype(np.uint16)
        else:
            mm = arr.astype(np.uint16, copy=False)
        if cv2 is None:
            return _png_write_u16(path, mm)
        return bool(cv2.imwrite(str(path), mm))

    if cv2 is None:
        # Fallback: 8-bit greyscale / RGB via stdlib not available; require cv2 for color.
        if arr.ndim == 2:
            return _png_write_u8(path, arr)
        print('[warn] cv2 missing, cannot write color PNGs', file=sys.stderr)
        return False
    return bool(cv2.imwrite(str(path), arr))


def compressed_to_bgr(msg) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    # compressedDepth has a 12-byte header before the PNG payload.
    fmt = (msg.format or '').lower()
    if 'compresseddepth' in fmt or 'png' in fmt and ';' in fmt:
        if len(buf) > 12:
            img = cv2.imdecode(buf[12:], cv2.IMREAD_UNCHANGED)
            return img
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return img


def _png_write_u8(path: Path, arr: np.ndarray) -> bool:
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        print('[warn] install opencv-python or pillow to write PNGs', file=sys.stderr)
        return False
    Image.fromarray(arr).save(str(path))
    return True


def _png_write_u16(path: Path, arr: np.ndarray) -> bool:
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        print('[warn] install opencv-python or pillow to write 16-bit PNGs',
              file=sys.stderr)
        return False
    Image.fromarray(arr, mode='I;16').save(str(path))
    return True


# ---------------------------------------------------------------------------
# PointCloud2 → PLY
# ---------------------------------------------------------------------------

_PC_TYPE = {
    1: ('i1', 1), 2: ('u1', 1), 3: ('i2', 2), 4: ('u2', 2),
    5: ('i4', 4), 6: ('u4', 4), 7: ('f4', 4), 8: ('f8', 8),
}


def pointcloud2_to_xyz(msg) -> Optional[np.ndarray]:
    fields = {f.name: f for f in msg.fields}
    if not {'x', 'y', 'z'} <= fields.keys():
        return None
    dtype_items: List[Tuple[str, str]] = []
    for f in sorted(msg.fields, key=lambda f: f.offset):
        if f.datatype not in _PC_TYPE:
            continue
        code, _ = _PC_TYPE[f.datatype]
        dtype_items.append((f.name, ('<' if not msg.is_bigendian else '>') + code))

    try:
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        stride = msg.point_step
        n = (msg.width * msg.height) if msg.height else (raw.size // stride)
        if n == 0:
            return None
        # Build a structured dtype that covers the full stride.
        names = [n_ for n_, _ in dtype_items]
        offsets = [fields[n_].offset for n_ in names]
        formats = [t for _, t in dtype_items]
        dt = np.dtype({'names': names, 'formats': formats,
                       'offsets': offsets, 'itemsize': stride})
        arr = np.frombuffer(raw[:n * stride], dtype=dt)
        xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=-1).astype(np.float32)
        finite = np.isfinite(xyz).all(axis=-1)
        xyz = xyz[finite]
        rgb = None
        if 'rgb' in arr.dtype.names:
            packed = arr['rgb'][finite].view(np.uint32)
            r = ((packed >> 16) & 0xFF).astype(np.uint8)
            g = ((packed >> 8) & 0xFF).astype(np.uint8)
            b = (packed & 0xFF).astype(np.uint8)
            rgb = np.stack([r, g, b], axis=-1)
        if rgb is None:
            return xyz
        return np.concatenate([xyz, rgb.astype(np.float32)], axis=-1)
    except Exception as e:  # noqa: BLE001
        print(f'[warn] pointcloud decode failed: {e}', file=sys.stderr)
        return None


def write_ply(path: Path, pts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = pts.shape[0]
    has_rgb = pts.shape[1] >= 6
    header = [
        'ply',
        'format binary_little_endian 1.0',
        f'element vertex {n}',
        'property float x',
        'property float y',
        'property float z',
    ]
    if has_rgb:
        header += [
            'property uchar red',
            'property uchar green',
            'property uchar blue',
        ]
    header.append('end_header\n')
    with open(path, 'wb') as f:
        f.write('\n'.join(header).encode('ascii'))
        if has_rgb:
            xyz = pts[:, :3].astype('<f4')
            rgb = pts[:, 3:6].astype(np.uint8)
            rec = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                     ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
            rec['x'], rec['y'], rec['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            rec['r'], rec['g'], rec['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            f.write(rec.tobytes())
        else:
            f.write(pts[:, :3].astype('<f4').tobytes())


# ---------------------------------------------------------------------------
# CSV writer helper
# ---------------------------------------------------------------------------

class LazyCsv:
    """A DictWriter that lazily learns its columns from the first row,
    and grows the header when a new key appears."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = None
        self._rows: List[Dict[str, Any]] = []
        self._fields: List[str] = []
        self._field_set = set()

    def write(self, row: Dict[str, Any]) -> None:
        for k in row.keys():
            if k not in self._field_set:
                self._field_set.add(k)
                self._fields.append(k)
        self._rows.append(row)
        # Flush in chunks to keep memory bounded.
        if len(self._rows) >= 2000:
            self._flush(final=False)

    def _flush(self, final: bool) -> None:
        if self._fp is None:
            self._fp = open(self.path, 'w', newline='')
            self._writer = csv.DictWriter(self._fp, fieldnames=self._fields,
                                          extrasaction='ignore')
            self._writer.writeheader()
            self._written_fields = list(self._fields)
        elif self._fields != self._written_fields:
            # Header grew — rewrite file with the new header.
            self._fp.close()
            old = self.path.read_text()
            first_nl = old.find('\n')
            body = old[first_nl + 1:] if first_nl >= 0 else ''
            self._fp = open(self.path, 'w', newline='')
            self._writer = csv.DictWriter(self._fp, fieldnames=self._fields,
                                          extrasaction='ignore')
            self._writer.writeheader()
            self._fp.write(body)
            self._written_fields = list(self._fields)

        for r in self._rows:
            self._writer.writerow(r)
        self._rows.clear()
        if final:
            self._fp.close()
            self._fp = None

    def close(self) -> None:
        if self._rows or self._fp is None:
            # Ensure file exists even if empty.
            if self._fp is None and self._fields:
                self._fp = open(self.path, 'w', newline='')
                self._writer = csv.DictWriter(self._fp, fieldnames=self._fields,
                                              extrasaction='ignore')
                self._writer.writeheader()
                self._written_fields = list(self._fields)
            self._flush(final=True)
        elif self._fp is not None:
            self._fp.close()
            self._fp = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag', required=True, type=Path,
                   help='Rosbag directory or .mcap file.')
    p.add_argument('--out', required=True, type=Path,
                   help='Output directory.')
    p.add_argument('--topics', nargs='*', default=None,
                   help='Whitelist of topics to export (default: all).')
    p.add_argument('--skip-topics', nargs='*', default=[],
                   help='Topics to skip.')
    p.add_argument('--image-stride', type=int, default=1,
                   help='Save every Nth image/pointcloud frame (default 1).')
    p.add_argument('--max-messages', type=int, default=0,
                   help='Stop after N total messages (0 = all).')
    p.add_argument('--no-clouds', action='store_true',
                   help='Skip PointCloud2 PLY export.')
    p.add_argument('--no-images', action='store_true',
                   help='Skip image PNG export.')
    return p.parse_args()


def open_reader(bag_path: Path) -> rosbag2_py.SequentialReader:
    # If pointed at a directory, look for metadata.yaml; if it's missing,
    # fall back to the first .mcap file inside. This lets us read bags
    # whose metadata was never written.
    uri = str(bag_path)
    if bag_path.is_dir():
        if not (bag_path / 'metadata.yaml').exists():
            mcaps = sorted(bag_path.glob('*.mcap'))
            if not mcaps:
                raise FileNotFoundError(
                    f'No metadata.yaml and no .mcap files in {bag_path}')
            uri = str(mcaps[0])
    storage = rosbag2_py.StorageOptions(uri=uri, storage_id='mcap')
    conv = rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                       output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, conv)
    return reader


def main() -> int:
    args = parse_args()
    if not args.bag.exists():
        print(f'Bag not found: {args.bag}', file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'csv').mkdir(exist_ok=True)

    reader = open_reader(args.bag)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    # Resolve whitelist / skiplist.
    selected = {}
    for name, type_ in topic_types.items():
        if args.topics and name not in args.topics:
            continue
        if name in args.skip_topics:
            continue
        selected[name] = type_

    if not selected:
        print('No topics matched filters.', file=sys.stderr)
        return 3

    reader.set_filter(rosbag2_py.StorageFilter(topics=list(selected.keys())))

    # Preload message classes.
    msg_classes: Dict[str, Any] = {}
    failures: Dict[str, str] = {}
    for name, type_ in selected.items():
        try:
            msg_classes[name] = get_message(type_)
        except Exception as e:  # noqa: BLE001
            failures[name] = f'{type_}: {e}'

    # Per-topic handlers and writers.
    csv_writers: Dict[str, LazyCsv] = {}
    index_writers: Dict[str, LazyCsv] = {}
    counters: Dict[str, int] = {n: 0 for n in selected}

    # Summary of what we did.
    summary_rows: List[Dict[str, Any]] = []

    def get_csv(topic: str) -> LazyCsv:
        w = csv_writers.get(topic)
        if w is None:
            w = LazyCsv(args.out / 'csv' / f'{slugify(topic)}.csv')
            csv_writers[topic] = w
        return w

    def get_index(topic: str, sub: str) -> LazyCsv:
        key = f'{sub}/{topic}'
        w = index_writers.get(key)
        if w is None:
            w = LazyCsv(args.out / sub / f'{slugify(topic)}_index.csv')
            index_writers[key] = w
        return w

    tf_dyn = LazyCsv(args.out / 'tf' / 'tf.csv')
    tf_static = LazyCsv(args.out / 'tf' / 'tf_static.csv')

    def handle_tf(topic: str, msg, bag_ns: int) -> None:
        w = tf_static if topic == '/tf_static' else tf_dyn
        for tr in msg.transforms:
            w.write({
                'bag_time_ns': bag_ns,
                'header_stamp_ns': stamp_to_ns(tr.header.stamp),
                'frame_id': tr.header.frame_id,
                'child_frame_id': tr.child_frame_id,
                'tx': tr.transform.translation.x,
                'ty': tr.transform.translation.y,
                'tz': tr.transform.translation.z,
                'qx': tr.transform.rotation.x,
                'qy': tr.transform.rotation.y,
                'qz': tr.transform.rotation.z,
                'qw': tr.transform.rotation.w,
            })

    def handle_image(topic: str, msg, bag_ns: int) -> None:
        if args.no_images:
            return
        counters[topic] += 1
        if (counters[topic] - 1) % args.image_stride:
            return
        arr, kind = image_msg_to_array(msg)
        if arr is None:
            return
        stamp = header_stamp_ns(msg) or bag_ns
        sub = 'depth' if kind == 'depth' else 'images'
        out_path = args.out / sub / slugify(topic) / f'{stamp}.png'
        if write_png(out_path, arr, kind):
            get_index(topic, sub).write({
                'bag_time_ns': bag_ns,
                'header_stamp_ns': header_stamp_ns(msg),
                'frame_id': header_frame(msg),
                'encoding': msg.encoding,
                'width': msg.width,
                'height': msg.height,
                'file': str(out_path.relative_to(args.out)),
            })

    def handle_compressed(topic: str, msg, bag_ns: int) -> None:
        if args.no_images or cv2 is None:
            return
        counters[topic] += 1
        if (counters[topic] - 1) % args.image_stride:
            return
        img = compressed_to_bgr(msg)
        if img is None:
            return
        stamp = header_stamp_ns(msg) or bag_ns
        fmt = (msg.format or '').lower()
        kind = 'depth' if 'depth' in fmt else 'rgb'
        sub = 'depth' if kind == 'depth' else 'images'
        out_path = args.out / sub / slugify(topic) / f'{stamp}.png'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        get_index(topic, sub).write({
            'bag_time_ns': bag_ns,
            'header_stamp_ns': header_stamp_ns(msg),
            'frame_id': header_frame(msg),
            'format': msg.format,
            'file': str(out_path.relative_to(args.out)),
        })

    def handle_pointcloud(topic: str, msg, bag_ns: int) -> None:
        if args.no_clouds:
            return
        counters[topic] += 1
        if (counters[topic] - 1) % args.image_stride:
            return
        pts = pointcloud2_to_xyz(msg)
        if pts is None or pts.size == 0:
            return
        stamp = header_stamp_ns(msg) or bag_ns
        out_path = args.out / 'clouds' / slugify(topic) / f'{stamp}.ply'
        write_ply(out_path, pts)
        get_index(topic, 'clouds').write({
            'bag_time_ns': bag_ns,
            'header_stamp_ns': header_stamp_ns(msg),
            'frame_id': header_frame(msg),
            'num_points': int(pts.shape[0]),
            'file': str(out_path.relative_to(args.out)),
        })

    def handle_generic(topic: str, msg, bag_ns: int) -> None:
        row: Dict[str, Any] = {
            'bag_time_ns': bag_ns,
            'header_stamp_ns': header_stamp_ns(msg),
            'frame_id': header_frame(msg),
        }
        row.update(flatten(msg))
        get_csv(topic).write(row)

    # Dispatch table keyed on type string.
    def dispatcher(type_name: str) -> Callable:
        if type_name == 'sensor_msgs/msg/Image':
            return handle_image
        if type_name == 'sensor_msgs/msg/CompressedImage':
            return handle_compressed
        if type_name == 'sensor_msgs/msg/PointCloud2':
            return handle_pointcloud
        if type_name == 'tf2_msgs/msg/TFMessage':
            return handle_tf
        return handle_generic

    dispatchers = {n: dispatcher(t) for n, t in selected.items()}

    total = 0
    progress_every = 5000
    print(f'Exporting {len(selected)} topics from {args.bag} → {args.out}')
    while reader.has_next():
        if args.max_messages and total >= args.max_messages:
            break
        topic, raw, t_ns = reader.read_next()
        cls = msg_classes.get(topic)
        if cls is None:
            continue
        try:
            msg = deserialize_message(raw, cls)
        except Exception as e:  # noqa: BLE001
            print(f'[warn] deserialize {topic}: {e}', file=sys.stderr)
            continue
        try:
            dispatchers[topic](topic, msg, int(t_ns))
        except Exception as e:  # noqa: BLE001
            print(f'[warn] handler {topic}: {e}', file=sys.stderr)
        total += 1
        if total % progress_every == 0:
            print(f'  {total} messages…', flush=True)

    for w in list(csv_writers.values()) + list(index_writers.values()) + [tf_dyn, tf_static]:
        w.close()

    # Topic summary.
    sum_path = args.out / 'topics.csv'
    with open(sum_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['topic', 'type', 'messages_seen', 'skipped_reason'])
        for name, type_ in topic_types.items():
            wr.writerow([
                name, type_, counters.get(name, 0),
                failures.get(name, '' if name in selected else 'filtered-out'),
            ])

    print(f'Done. {total} messages processed.')
    print(f'Summary: {sum_path}')
    if failures:
        print('Types that failed to load:', file=sys.stderr)
        for n, e in failures.items():
            print(f'  {n}: {e}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
