"""Ultra-Fast-Lane-Detection-v2 inference wrapper.

This is a small ROS-free adapter around the upstream PyTorch project:
https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2

The upstream model returns structural lane coordinates rather than dense
segmentation masks.  This wrapper decodes those coordinates and rasterizes
them into the same ``(drivable_mask, lane_mask)`` pair used by the existing
lane segmentation pipeline.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


Lane = Dict[str, object]


def _real_init_weights(module) -> None:
    if torch is None:
        return
    if isinstance(module, list):
        for child in module:
            _real_init_weights(child)
        return
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(0.0, std=0.01)
    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(module.weight, 1)
        torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.Module):
        for child in module.children():
            _real_init_weights(child)


class UltraFastLaneDetectionV2:
    """Load UFLDv2 checkpoints and return dense masks for one BGR image."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        root_path: str,
        config_path: str,
        weights_path: str,
        device: str = 'cuda:0',
        half: bool = True,
        lane_width_px: int = 8,
        drivable_fill: bool = True,
        drivable_lane_dilation_px: int = 45,
        min_points_per_lane: int = 4,
        local_width: int = 1,
        row_lane_indices: Optional[Sequence[int]] = None,
        col_lane_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.root_path = Path(os.path.expanduser(root_path)).resolve()
        self.config_path = Path(os.path.expanduser(config_path))
        self.weights_path = Path(os.path.expanduser(weights_path)).resolve()
        if not self.config_path.is_absolute():
            self.config_path = self.root_path / self.config_path

        self.requested_device = str(device)
        self.half = bool(half)
        self.lane_width_px = max(1, int(lane_width_px))
        self.drivable_fill = bool(drivable_fill)
        self.drivable_lane_dilation_px = max(0, int(drivable_lane_dilation_px))
        self.min_points_per_lane = max(1, int(min_points_per_lane))
        self.local_width = max(0, int(local_width))
        self.row_lane_indices = list(row_lane_indices) if row_lane_indices else None
        self.col_lane_indices = list(col_lane_indices) if col_lane_indices else None

        self.cfg: Optional[SimpleNamespace] = None
        self.model = None
        self._device = None
        self._loaded_fallback_warning: Optional[str] = None
        self.last_lane_count = 0
        self.last_lane_point_counts: List[int] = []

    @property
    def device(self):
        return self._device

    @property
    def fallback_warning(self) -> Optional[str]:
        return self._loaded_fallback_warning

    def load(self) -> None:
        if torch is None:
            raise RuntimeError(
                'PyTorch is not installed. On Jetson, install the JetPack-matched '
                'torch/torchvision wheels before enabling UFLDv2.')
        if not self.root_path.is_dir():
            raise FileNotFoundError(
                f'UFLDv2 root not found: {self.root_path}. Run scripts/setup_ufldv2.sh first.')
        if not self.config_path.is_file():
            raise FileNotFoundError(f'UFLDv2 config not found: {self.config_path}')
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f'UFLDv2 checkpoint not found: {self.weights_path}. '
                'Run scripts/fetch_ufldv2_weights.sh first.')
        if self._looks_like_torchscript_archive(self.weights_path):
            raise RuntimeError(
                f'UFLDv2 expected an upstream .pth checkpoint dict, but '
                f'{self.weights_path} is a TorchScript archive. This usually '
                'means model_weights is still pointing at yolopv2.pt. Set '
                'model_weights to $UFLDV2_WEIGHTS or rerun '
                'scripts/fetch_ufldv2_weights.sh culane_res18.')

        self._prepend_upstream_path()
        self._install_initialize_weights_stub()
        self.cfg = self._load_config(self.config_path)
        self._attach_anchors(self.cfg)

        want_cuda = self.requested_device.startswith('cuda')
        if want_cuda and not torch.cuda.is_available():
            self._device = torch.device('cpu')
            self.half = False
            self._loaded_fallback_warning = (
                f'CUDA requested ({self.requested_device}) but not available; '
                'falling back to CPU.')
        else:
            self._device = torch.device(self.requested_device)
            self._loaded_fallback_warning = None

        model = self._build_model(self.cfg)
        checkpoint = torch.load(str(self.weights_path), map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        if not hasattr(state_dict, 'items'):
            raise RuntimeError(
                f'UFLDv2 checkpoint {self.weights_path} did not contain a '
                'state-dict-like object. Expected a .pth file from '
                'scripts/fetch_ufldv2_weights.sh, not a TorchScript/ONNX/TRT export.')
        compatible_state_dict = {
            key[7:] if key.startswith('module.') else key: value
            for key, value in state_dict.items()
        }
        model.load_state_dict(compatible_state_dict, strict=False)
        model = model.to(self._device)
        if self._device.type == 'cuda' and self.half:
            model = model.half()
        else:
            self.half = False
        model.eval()
        self.model = model

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                3,
                int(self.cfg.train_height),
                int(self.cfg.train_width),
                device=self._device)
            if self.half:
                dummy = dummy.half()
            _ = self.model(dummy)

    def infer(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lanes = self.infer_lanes(bgr)
        return self._lanes_to_masks(lanes, bgr.shape[:2])

    def infer_lanes(self, bgr: np.ndarray) -> List[Lane]:
        if self.model is None or self.cfg is None or self._device is None:
            raise RuntimeError('UltraFastLaneDetectionV2.load() must be called before infer().')

        tensor = self._preprocess(bgr)
        with torch.no_grad():
            pred = self.model(tensor)
        lanes = self._decode_lanes(pred, bgr.shape[1], bgr.shape[0])
        self.last_lane_count = len(lanes)
        self.last_lane_point_counts = [
            len(lane.get('points', [])) for lane in lanes
        ]
        return lanes

    def _prepend_upstream_path(self) -> None:
        root_str = str(self.root_path)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    @staticmethod
    def _looks_like_torchscript_archive(path: Path) -> bool:
        if not zipfile.is_zipfile(path):
            return False
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        except zipfile.BadZipFile:
            return False
        return any('/code/' in name or name.startswith('code/') for name in names)

    def _install_initialize_weights_stub(self) -> None:
        # model_culane.py imports utils.common.initialize_weights.  The real
        # utils.common imports DALI/training-only modules, so provide just the
        # initializer needed for model construction.
        if 'utils.common' in sys.modules:
            return
        utils_pkg = sys.modules.get('utils')
        if utils_pkg is None:
            utils_pkg = types.ModuleType('utils')
            utils_pkg.__path__ = [str(self.root_path / 'utils')]
            sys.modules['utils'] = utils_pkg
        common_stub = types.ModuleType('utils.common')
        common_stub.initialize_weights = _real_init_weights
        sys.modules['utils.common'] = common_stub

    @staticmethod
    def _load_config(config_path: Path) -> SimpleNamespace:
        spec = importlib.util.spec_from_file_location('ufldv2_config', str(config_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Unable to load UFLDv2 config: {config_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = {
            key: value
            for key, value in vars(module).items()
            if not key.startswith('_')
        }
        return SimpleNamespace(**values)

    @staticmethod
    def _attach_anchors(cfg: SimpleNamespace) -> None:
        dataset = str(cfg.dataset)
        if dataset == 'CULane':
            cfg.row_anchor = np.linspace(0.42, 1.0, int(cfg.num_row))
            cfg.col_anchor = np.linspace(0.0, 1.0, int(cfg.num_col))
        elif dataset == 'Tusimple':
            cfg.row_anchor = np.linspace(160, 710, int(cfg.num_row)) / 720.0
            cfg.col_anchor = np.linspace(0.0, 1.0, int(cfg.num_col))
        elif dataset == 'CurveLanes':
            cfg.row_anchor = np.linspace(0.4, 1.0, int(cfg.num_row))
            cfg.col_anchor = np.linspace(0.0, 1.0, int(cfg.num_col))
        else:
            raise RuntimeError(f'Unsupported UFLDv2 dataset in config: {dataset}')

    @staticmethod
    def _lane_indices(num_lanes: int, requested: Optional[Sequence[int]], axis: str) -> List[int]:
        if requested is not None:
            return [idx for idx in requested if 0 <= idx < num_lanes]
        if num_lanes == 4:
            return [1, 2] if axis == 'row' else [0, 3]
        return list(range(num_lanes))

    def _build_model(self, cfg: SimpleNamespace):
        dataset_module = str(cfg.dataset).lower()
        module = importlib.import_module(f'model.model_{dataset_module}')
        parsing_net = getattr(module, 'parsingNet')
        kwargs = dict(
            pretrained=False,
            backbone=cfg.backbone,
            num_grid_row=cfg.num_cell_row,
            num_cls_row=cfg.num_row,
            num_grid_col=cfg.num_cell_col,
            num_cls_col=cfg.num_col,
            num_lane_on_row=cfg.num_lanes,
            num_lane_on_col=cfg.num_lanes,
            use_aux=cfg.use_aux,
            input_height=cfg.train_height,
            input_width=cfg.train_width,
        )
        if dataset_module in ('culane', 'tusimple'):
            kwargs['fc_norm'] = getattr(cfg, 'fc_norm', False)
        return parsing_net(**kwargs)

    def _preprocess(self, bgr: np.ndarray):
        assert self.cfg is not None and self._device is not None
        resized_h = int(round(float(self.cfg.train_height) / float(self.cfg.crop_ratio)))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (int(self.cfg.train_width), resized_h),
            interpolation=cv2.INTER_LINEAR)
        rgb = rgb[-int(self.cfg.train_height):, :, :]
        img = rgb.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0).to(self._device)
        if self.half:
            tensor = tensor.half()
        else:
            tensor = tensor.float()
        return tensor

    def _decode_lanes(self, pred, image_width: int, image_height: int) -> List[Lane]:
        assert self.cfg is not None
        loc_row = pred['loc_row'].detach().cpu()
        loc_col = pred['loc_col'].detach().cpu()
        valid_row = pred['exist_row'].detach().argmax(1).cpu()
        valid_col = pred['exist_col'].detach().argmax(1).cpu()

        max_indices_row = loc_row.argmax(1)
        max_indices_col = loc_col.argmax(1)
        num_grid_row = int(loc_row.shape[1])
        num_grid_col = int(loc_col.shape[1])
        num_cls_row = int(loc_row.shape[2])
        num_cls_col = int(loc_col.shape[2])
        num_lanes = int(loc_row.shape[3])

        lanes: List[Lane] = []
        for lane_idx in self._lane_indices(num_lanes, self.row_lane_indices, 'row'):
            points: List[Tuple[int, int]] = []
            if int(valid_row[0, :, lane_idx].sum()) > num_cls_row / 2:
                for cls_idx in range(valid_row.shape[1]):
                    if int(valid_row[0, cls_idx, lane_idx]) == 0:
                        continue
                    max_idx = int(max_indices_row[0, cls_idx, lane_idx])
                    start = max(0, max_idx - self.local_width)
                    stop = min(num_grid_row - 1, max_idx + self.local_width)
                    all_ind = torch.arange(start, stop + 1, dtype=torch.long)
                    probs = loc_row[0, all_ind, cls_idx, lane_idx].softmax(0)
                    out = float((probs * all_ind.float()).sum()) + 0.5
                    x = int(round(out / float(num_grid_row - 1) * image_width))
                    y = int(round(float(self.cfg.row_anchor[cls_idx]) * image_height))
                    points.append((x, y))
            if len(points) >= self.min_points_per_lane:
                lanes.append({'axis': 'row', 'lane_index': lane_idx, 'points': points})

        for lane_idx in self._lane_indices(num_lanes, self.col_lane_indices, 'col'):
            points = []
            if int(valid_col[0, :, lane_idx].sum()) > num_cls_col / 4:
                for cls_idx in range(valid_col.shape[1]):
                    if int(valid_col[0, cls_idx, lane_idx]) == 0:
                        continue
                    max_idx = int(max_indices_col[0, cls_idx, lane_idx])
                    start = max(0, max_idx - self.local_width)
                    stop = min(num_grid_col - 1, max_idx + self.local_width)
                    all_ind = torch.arange(start, stop + 1, dtype=torch.long)
                    probs = loc_col[0, all_ind, cls_idx, lane_idx].softmax(0)
                    out = float((probs * all_ind.float()).sum()) + 0.5
                    x = int(round(float(self.cfg.col_anchor[cls_idx]) * image_width))
                    y = int(round(out / float(num_grid_col - 1) * image_height))
                    points.append((x, y))
            if len(points) >= self.min_points_per_lane:
                lanes.append({'axis': 'col', 'lane_index': lane_idx, 'points': points})

        return lanes

    def _lanes_to_masks(self, lanes: Sequence[Lane], shape_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        height, width = shape_hw
        lane_mask = np.zeros((height, width), dtype=np.uint8)
        for lane in lanes:
            points = np.asarray(lane['points'], dtype=np.int32)
            if points.shape[0] < 2:
                continue
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            order = np.argsort(points[:, 1])
            points = points[order]
            cv2.polylines(
                lane_mask,
                [points.reshape(-1, 1, 2)],
                isClosed=False,
                color=1,
                thickness=self.lane_width_px,
                lineType=cv2.LINE_AA)

        drivable = np.zeros_like(lane_mask)
        if self.drivable_fill:
            self._fill_drivable_between_inner_lanes(drivable, lanes, width, height)
        if self.drivable_lane_dilation_px > 0 and np.any(lane_mask):
            k = self.drivable_lane_dilation_px | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            lane_band = cv2.dilate(lane_mask, kernel, iterations=1)
            drivable = ((drivable > 0) | (lane_band > 0)).astype(np.uint8)
        return drivable.astype(np.uint8), (lane_mask > 0).astype(np.uint8)

    @staticmethod
    def _fill_drivable_between_inner_lanes(
        drivable: np.ndarray,
        lanes: Sequence[Lane],
        width: int,
        height: int,
    ) -> None:
        row_lanes = []
        for lane in lanes:
            if lane.get('axis') != 'row':
                continue
            points = np.asarray(lane['points'], dtype=np.float32)
            if points.shape[0] >= 2:
                row_lanes.append(points[np.argsort(points[:, 1])])
        if len(row_lanes) < 2:
            return

        center_x = width * 0.5
        pairs = []
        for left_idx in range(len(row_lanes)):
            for right_idx in range(left_idx + 1, len(row_lanes)):
                a = row_lanes[left_idx]
                b = row_lanes[right_idx]
                mean_a = float(np.mean(a[:, 0]))
                mean_b = float(np.mean(b[:, 0]))
                left, right = (a, b) if mean_a <= mean_b else (b, a)
                left_mean = min(mean_a, mean_b)
                right_mean = max(mean_a, mean_b)
                center_penalty = 0.0 if left_mean <= center_x <= right_mean else min(
                    abs(center_x - left_mean), abs(center_x - right_mean))
                width_penalty = abs((right_mean - left_mean) - width * 0.35)
                pairs.append((center_penalty + width_penalty * 0.1, left, right))
        if not pairs:
            return

        _, left, right = min(pairs, key=lambda item: item[0])
        y_min = int(max(np.min(left[:, 1]), np.min(right[:, 1]), 0))
        y_max = int(min(np.max(left[:, 1]), np.max(right[:, 1]), height - 1))
        if y_max <= y_min:
            return
        y_samples = np.linspace(y_min, y_max, max(8, min(80, y_max - y_min + 1)))
        left_x = np.interp(y_samples, left[:, 1], left[:, 0])
        right_x = np.interp(y_samples, right[:, 1], right[:, 0])
        left_pts = np.stack([left_x, y_samples], axis=1)
        right_pts = np.stack([right_x, y_samples], axis=1)
        polygon = np.vstack([left_pts, right_pts[::-1]])
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillPoly(drivable, [polygon.astype(np.int32).reshape(-1, 1, 2)], 1)


UFLDv2 = UltraFastLaneDetectionV2