from __future__ import annotations

import json
import sys
import threading
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QWidget


class UiSignal(QObject):
    gt_grid = pyqtSignal(object)
    pred_grid = pyqtSignal(object)
    diff_grid = pyqtSignal(object)
    metrics = pyqtSignal(str)


def _grid_to_img(grid: OccupancyGrid) -> np.ndarray:
    arr = np.array(grid.data, dtype=np.int16).reshape((grid.info.height, grid.info.width))
    img = np.zeros((grid.info.height, grid.info.width, 3), dtype=np.uint8)
    unknown = arr < 0
    occ = arr >= 50
    free = np.logical_and(arr >= 0, arr < 50)
    img[unknown] = (30, 30, 30)
    img[free] = (180, 180, 180)
    img[occ] = (0, 220, 255)
    return np.flipud(img)


def _overlay_diff(gt: OccupancyGrid, pred: OccupancyGrid) -> np.ndarray:
    g = np.array(gt.data, dtype=np.int16).reshape((gt.info.height, gt.info.width)) >= 50
    p = np.array(pred.data, dtype=np.int16).reshape((pred.info.height, pred.info.width)) >= 50

    if g.shape != p.shape:
        h = min(g.shape[0], p.shape[0])
        w = min(g.shape[1], p.shape[1])
        g = g[:h, :w]
        p = p[:h, :w]

    img = np.zeros((g.shape[0], g.shape[1], 3), dtype=np.uint8)
    tp = np.logical_and(g, p)
    fp = np.logical_and(np.logical_not(g), p)
    fn = np.logical_and(g, np.logical_not(p))
    tn = np.logical_and(np.logical_not(g), np.logical_not(p))

    img[tp] = (0, 220, 0)
    img[fp] = (220, 0, 0)
    img[fn] = (0, 0, 220)
    img[tn] = (40, 40, 40)
    return np.flipud(img)


class LaneCompareNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_compare_ui_node')
        self.declare_parameter('ground_truth_topic', '/lane_ground_truth')
        self.declare_parameter('predicted_topic', '/lane_map')

        gt_topic = str(self.get_parameter('ground_truth_topic').value)
        pred_topic = str(self.get_parameter('predicted_topic').value)

        self._sig = UiSignal()
        self._last_gt: Optional[OccupancyGrid] = None
        self._last_pred: Optional[OccupancyGrid] = None

        self.create_subscription(OccupancyGrid, gt_topic, self._on_gt, 10)
        self.create_subscription(OccupancyGrid, pred_topic, self._on_pred, 10)
        self.create_subscription(String, '/lane_eval/report', self._on_metrics, 10)

    @property
    def signal(self) -> UiSignal:
        return self._sig

    def _on_gt(self, msg: OccupancyGrid) -> None:
        self._last_gt = msg
        self._sig.gt_grid.emit(msg)
        self._emit_diff_if_ready()

    def _on_pred(self, msg: OccupancyGrid) -> None:
        self._last_pred = msg
        self._sig.pred_grid.emit(msg)
        self._emit_diff_if_ready()

    def _emit_diff_if_ready(self) -> None:
        if self._last_gt is None or self._last_pred is None:
            return
        self._sig.diff_grid.emit((self._last_gt, self._last_pred))

    def _on_metrics(self, msg: String) -> None:
        self._sig.metrics.emit(msg.data)


class LaneCompareWindow(QWidget):
    def __init__(self, node: LaneCompareNode) -> None:
        super().__init__()
        self._node = node
        self._thread = threading.Thread(target=self._spin, daemon=True)

        self._gt_img = QLabel('GT grid')
        self._pred_img = QLabel('Predicted grid')
        self._diff_img = QLabel('Diff overlay')
        self._metrics = QLabel('metrics: waiting')

        for w in (self._gt_img, self._pred_img, self._diff_img):
            w.setMinimumSize(420, 420)
            w.setAlignment(Qt.AlignCenter)

        self._metrics.setMinimumHeight(60)

        grid = QGridLayout()
        grid.addWidget(QLabel('Ground Truth'), 0, 0)
        grid.addWidget(QLabel('Predicted'), 0, 1)
        grid.addWidget(QLabel('Diff (green=TP red=FP blue=FN)'), 0, 2)
        grid.addWidget(self._gt_img, 1, 0)
        grid.addWidget(self._pred_img, 1, 1)
        grid.addWidget(self._diff_img, 1, 2)
        grid.addWidget(self._metrics, 2, 0, 1, 3)
        self.setLayout(grid)
        self.setWindowTitle('Lane Comparison UI')

        node.signal.gt_grid.connect(self._update_gt_grid)
        node.signal.pred_grid.connect(self._update_pred_grid)
        node.signal.diff_grid.connect(self._update_diff_grid)
        node.signal.metrics.connect(self._update_metrics)

        self._thread.start()

    def _spin(self) -> None:
        rclpy.spin(self._node)

    def closeEvent(self, event):
        rclpy.shutdown()
        self._thread.join(timeout=2.0)
        event.accept()

    @staticmethod
    def _to_pixmap(img: np.ndarray) -> QPixmap:
        h, w, _ = img.shape
        q = QImage(img.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(q.copy())

    def _set_scaled(self, label: QLabel, pix: QPixmap) -> None:
        label.setPixmap(
            pix.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _update_gt_grid(self, gt: OccupancyGrid) -> None:
        gt_img = _grid_to_img(gt)
        self._set_scaled(self._gt_img, self._to_pixmap(gt_img))

    def _update_pred_grid(self, pred: OccupancyGrid) -> None:
        pred_img = _grid_to_img(pred)
        self._set_scaled(self._pred_img, self._to_pixmap(pred_img))

    def _update_diff_grid(self, grids: tuple[OccupancyGrid, OccupancyGrid]) -> None:
        gt, pred = grids
        gt_img = _grid_to_img(gt)
        pred_img = _grid_to_img(pred)
        diff_img = _overlay_diff(gt, pred)
        self._set_scaled(self._diff_img, self._to_pixmap(diff_img))

    def _update_metrics(self, text: str) -> None:
        try:
            data = json.loads(text)
            nice = (
                f"IoU={data.get('iou', 0.0):.3f}   "
                f"Precision={data.get('precision', 0.0):.3f}   "
                f"Recall={data.get('recall', 0.0):.3f}   "
                f"Latency={data.get('latency_ms', 0.0):.1f} ms   "
                f"Dropped={data.get('frames_dropped', 0)}/{data.get('frames_total', 0)}"
            )
            self._metrics.setText(nice)
        except Exception:
            self._metrics.setText(text)


def main(args=None) -> None:
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    node = LaneCompareNode()
    win = LaneCompareWindow(node)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
