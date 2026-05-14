from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32, String


class LaneEvaluatorNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_evaluator_node')

        self.declare_parameter('ground_truth_topic', '/lane_ground_truth')
        self.declare_parameter('predicted_topic', '/lane_map')
        self.declare_parameter('max_stamp_skew_sec', 0.1)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('report_csv_path', '')

        gt_topic = str(self.get_parameter('ground_truth_topic').value)
        pred_topic = str(self.get_parameter('predicted_topic').value)
        self._max_skew = float(self.get_parameter('max_stamp_skew_sec').value)
        self._occ_th = int(self.get_parameter('occupied_threshold').value)
        report_path = str(self.get_parameter('report_csv_path').value)

        self._last_gt: Optional[OccupancyGrid] = None
        self._last_pred: Optional[OccupancyGrid] = None
        self._frames_total = 0
        self._frames_dropped = 0

        self._iou_pub = self.create_publisher(Float32, '/lane_eval/iou', 10)
        self._prec_pub = self.create_publisher(Float32, '/lane_eval/precision', 10)
        self._rec_pub = self.create_publisher(Float32, '/lane_eval/recall', 10)
        self._lat_pub = self.create_publisher(Float32, '/lane_eval/latency_ms', 10)
        self._report_pub = self.create_publisher(String, '/lane_eval/report', 10)

        self.create_subscription(OccupancyGrid, gt_topic, self._on_gt, 10)
        self.create_subscription(OccupancyGrid, pred_topic, self._on_pred, 10)

        self._csv_file = None
        self._csv_writer = None
        if report_path:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = path.open('a', newline='', encoding='utf-8')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                'stamp_sec', 'stamp_nanosec', 'iou', 'precision', 'recall',
                'latency_ms', 'frames_total', 'frames_dropped',
            ])

        self.get_logger().info(
            f'Lane evaluator started: gt={gt_topic}, pred={pred_topic}, max_skew={self._max_skew:.3f}s')

    def destroy_node(self):
        if self._csv_file is not None:
            self._csv_file.close()
        return super().destroy_node()

    def _on_gt(self, msg: OccupancyGrid) -> None:
        self._last_gt = msg
        self._try_eval()

    def _on_pred(self, msg: OccupancyGrid) -> None:
        self._last_pred = msg
        self._try_eval()

    @staticmethod
    def _stamp_delta_sec(a, b) -> float:
        ta = Time.from_msg(a)
        tb = Time.from_msg(b)
        if ta.nanoseconds == 0 or tb.nanoseconds == 0:
            return float('inf')
        return abs((ta - tb).nanoseconds / 1e9)

    def _try_eval(self) -> None:
        if self._last_gt is None or self._last_pred is None:
            return

        skew = self._stamp_delta_sec(
            self._last_gt.header.stamp,
            self._last_pred.header.stamp,
        )
        self._frames_total += 1
        if skew > self._max_skew:
            self._frames_dropped += 1
            self.get_logger().warn(
                f'Dropping eval frame due to stamp skew={skew:.3f}s (> {self._max_skew:.3f}s).',
                throttle_duration_sec=2.0,
            )
            return

        iou, precision, recall = self._compute_metrics(self._last_gt, self._last_pred)
        latency_ms = float(skew * 1000.0)

        self._iou_pub.publish(Float32(data=float(iou)))
        self._prec_pub.publish(Float32(data=float(precision)))
        self._rec_pub.publish(Float32(data=float(recall)))
        self._lat_pub.publish(Float32(data=float(latency_ms)))

        report = {
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'latency_ms': latency_ms,
            'frames_total': self._frames_total,
            'frames_dropped': self._frames_dropped,
            'stamp_skew_sec': skew,
        }
        self._report_pub.publish(String(data=json.dumps(report, sort_keys=True)))

        if self._csv_writer is not None:
            stamp = self._last_pred.header.stamp
            self._csv_writer.writerow([
                int(stamp.sec),
                int(stamp.nanosec),
                float(iou),
                float(precision),
                float(recall),
                latency_ms,
                self._frames_total,
                self._frames_dropped,
            ])
            self._csv_file.flush()

    def _compute_metrics(self, gt: OccupancyGrid, pred: OccupancyGrid) -> Tuple[float, float, float]:
        gt_img = np.array(gt.data, dtype=np.int16).reshape((gt.info.height, gt.info.width))
        pred_img = np.array(pred.data, dtype=np.int16).reshape((pred.info.height, pred.info.width))

        if (
            gt.info.width == pred.info.width
            and gt.info.height == pred.info.height
            and abs(float(gt.info.resolution) - float(pred.info.resolution)) < 1e-9
            and abs(float(gt.info.origin.position.x) - float(pred.info.origin.position.x)) < 1e-6
            and abs(float(gt.info.origin.position.y) - float(pred.info.origin.position.y)) < 1e-6
        ):
            gt_occ = gt_img >= self._occ_th
            pred_occ = pred_img >= self._occ_th
        else:
            gt_occ = self._sample_gt_on_pred(gt, gt_img, pred)
            pred_occ = pred_img >= self._occ_th

        tp = int(np.logical_and(pred_occ, gt_occ).sum())
        fp = int(np.logical_and(pred_occ, np.logical_not(gt_occ)).sum())
        fn = int(np.logical_and(np.logical_not(pred_occ), gt_occ).sum())

        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else 1.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        return iou, precision, recall

    def _sample_gt_on_pred(self, gt: OccupancyGrid, gt_img: np.ndarray, pred: OccupancyGrid) -> np.ndarray:
        h = int(pred.info.height)
        w = int(pred.info.width)
        pred_res = float(pred.info.resolution)
        pred_ox = float(pred.info.origin.position.x)
        pred_oy = float(pred.info.origin.position.y)

        gt_res = float(gt.info.resolution)
        gt_ox = float(gt.info.origin.position.x)
        gt_oy = float(gt.info.origin.position.y)
        gt_w = int(gt.info.width)
        gt_h = int(gt.info.height)

        sampled = np.zeros((h, w), dtype=bool)
        for row in range(h):
            wy = pred_oy + (row + 0.5) * pred_res
            gy = int((wy - gt_oy) / gt_res)
            if gy < 0 or gy >= gt_h:
                continue
            for col in range(w):
                wx = pred_ox + (col + 0.5) * pred_res
                gx = int((wx - gt_ox) / gt_res)
                if gx < 0 or gx >= gt_w:
                    continue
                sampled[row, col] = gt_img[gy, gx] >= self._occ_th
        return sampled


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneEvaluatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
