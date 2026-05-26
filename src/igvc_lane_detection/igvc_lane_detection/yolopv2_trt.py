"""TensorRT runtime wrapper for the YOLOPv2 segmentation model.

Drop-in replacement for :class:`yolopv2_infer.YolopV2` when the weight
file is a TensorRT engine (``.engine``).  Pre/post-processing matches
the TorchScript path so downstream code is unchanged — only the forward
pass is swapped.

Engine build pipeline (run **once** on the deployment host):

    python3 scripts/export_yolopv2_onnx.py \\
        --weights models/yolopv2.pt \\
        --output  models/yolopv2_384.onnx \\
        --img-size 384 --half

    scripts/export_yolopv2_tensorrt.sh \\
        models/yolopv2_384.onnx \\
        models/yolopv2_384.engine \\
        fp16

Then point ``model_weights`` at the resulting ``.engine`` in
``lane_segmentation_config.yaml``.  The node detects the extension and
loads this backend automatically.

Requires ``tensorrt`` + ``cuda-python`` (both bundled with JetPack 6).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import tensorrt as trt  # type: ignore
except ImportError:  # pragma: no cover
    trt = None  # type: ignore

try:
    from cuda import cudart  # type: ignore
except ImportError:  # pragma: no cover
    cudart = None  # type: ignore

from .yolopv2_infer import _letterbox


def _check(err) -> None:
    """Raise on a non-success cudart error tuple."""
    if isinstance(err, tuple):
        code = err[0]
    else:
        code = err
    if int(code) != 0:
        raise RuntimeError(f"cudart error: {code}")


class YolopV2TRT:
    """TensorRT-backed YOLOPv2 with the same ``infer()`` API as ``YolopV2``."""

    def __init__(
        self,
        engine_path: str,
        img_size: int = 384,
        resize_hw: Optional[Tuple[int, int]] = None,
        preprocess: bool = True,
        clahe_clip: float = 2.0,
        clahe_tile: Tuple[int, int] = (8, 8),
        blur_ksize: Tuple[int, int] = (5, 5),
        blur_sigma: float = 0.0,
        lane_threshold: float = 0.5,
    ) -> None:
        if trt is None or cudart is None:
            raise RuntimeError(
                "TensorRT runtime requires the 'tensorrt' and 'cuda-python' "
                "packages.  On Jetson both ship with JetPack.")
        self.engine_path = engine_path
        self.img_size = int(img_size)
        self.resize_hw = tuple(resize_hw) if resize_hw is not None else None
        self.preprocess_enabled = bool(preprocess)
        self.blur_ksize = tuple(blur_ksize)
        self.blur_sigma = float(blur_sigma)
        self.lane_threshold = float(lane_threshold)
        self._clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_tile))
        self._stride = 32

        self._logger: Optional[trt.Logger] = None
        self._engine: Optional[trt.ICudaEngine] = None
        self._context: Optional[trt.IExecutionContext] = None
        self._stream = None
        self._bindings: list = []
        self._input_name: Optional[str] = None
        self._input_dtype = np.float16  # rebound on load
        self._output_names: list[str] = []
        self._output_shapes: list[tuple] = []
        self._output_dtypes: list = []
        self._device_ptrs: dict = {}
        self._host_outputs: dict = {}
        self._loaded_fallback_warning: Optional[str] = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    def load(self) -> None:
        self._logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self._logger)
        with open(self.engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to load engine: {self.engine_path}")
        self._context = self._engine.create_execution_context()

        err, self._stream = cudart.cudaStreamCreate()
        _check(err)

        # Discover bindings.  TRT 10 uses get_tensor_*; older uses
        # binding-index API.  Prefer the modern path.
        n = self._engine.num_io_tensors
        for i in range(n):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            err, dev_ptr = cudart.cudaMalloc(nbytes)
            _check(err)
            self._device_ptrs[name] = dev_ptr
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
                self._input_dtype = dtype
                self._context.set_input_shape(name, shape)
            else:
                self._output_names.append(name)
                self._output_shapes.append(shape)
                self._output_dtypes.append(dtype)
                self._host_outputs[name] = np.empty(shape, dtype=dtype)
            self._context.set_tensor_address(name, int(dev_ptr))

        if self._input_name is None:
            raise RuntimeError("Engine has no input tensor")

        # Warm-up
        zero = np.zeros(
            (1, 3, self.img_size, self.img_size), dtype=self._input_dtype)
        self._forward(zero)

    @property
    def fallback_warning(self) -> Optional[str]:
        return self._loaded_fallback_warning

    @property
    def device(self):
        return "cuda:0 (tensorrt)"

    @property
    def half(self) -> bool:
        return self._input_dtype == np.float16

    # ── Preprocess (mirrors YolopV2._preprocess) ───────────────────────

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = self._clahe.apply(yuv[:, :, 0])
        out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        if self.blur_ksize[0] > 1:
            out = cv2.GaussianBlur(out, self.blur_ksize, self.blur_sigma)
        return out

    # ── Forward ────────────────────────────────────────────────────────

    def _forward(self, chw_batch: np.ndarray) -> dict:
        assert self._context is not None
        chw_batch = np.ascontiguousarray(chw_batch.astype(self._input_dtype))
        in_ptr = self._device_ptrs[self._input_name]
        nbytes = chw_batch.nbytes
        _check(cudart.cudaMemcpyAsync(
            in_ptr, chw_batch.ctypes.data, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream))

        self._context.execute_async_v3(self._stream)

        for name in self._output_names:
            host = self._host_outputs[name]
            _check(cudart.cudaMemcpyAsync(
                host.ctypes.data, self._device_ptrs[name], host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream))

        _check(cudart.cudaStreamSynchronize(self._stream))
        return dict(self._host_outputs)

    # ── Public API ─────────────────────────────────────────────────────

    def infer(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._context is None:
            raise RuntimeError("YolopV2TRT.load() must be called before infer().")

        src_h, src_w = bgr.shape[:2]
        if self.resize_hw is None:
            resized = bgr
        else:
            resized = cv2.resize(
                bgr, (self.resize_hw[1], self.resize_hw[0]),
                interpolation=cv2.INTER_LINEAR)
        if self.preprocess_enabled:
            resized = self._preprocess(resized)

        lb, _ratio, (dw, dh) = _letterbox(
            resized, new_shape=self.img_size, stride=self._stride,
            auto=True, scaleup=True)
        lb_h, lb_w = lb.shape[:2]

        img = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        outputs = self._forward(img)

        # Outputs assumed in (seg, ll) order matching the ONNX export.
        seg = outputs[self._output_names[0]].astype(np.float32)
        ll  = outputs[self._output_names[1]].astype(np.float32)

        # Strip letterbox padding on the seg heads.  The exported model
        # operates on the full letterboxed square, so padding is
        # symmetric.
        pad_top    = int(round(dh - 0.1))
        pad_bottom = int(round(dh + 0.1))
        pad_left   = int(round(dw - 0.1))
        pad_right  = int(round(dw + 0.1))
        seg_h = seg.shape[-2]
        seg_w = seg.shape[-1]
        if seg_h == lb_h and seg_w == lb_w:
            t, b = pad_top, seg_h - pad_bottom
            l, r = pad_left, seg_w - pad_right
            seg = seg[..., t:b, l:r]
            ll  = ll[..., t:b, l:r]

        # Resize to the YOLOPv2 demo intermediate, then to source.
        if seg.ndim == 4:
            da_logits = seg[0]
            ll_logits = ll[0]
        else:
            da_logits = seg
            ll_logits = ll

        if da_logits.shape[0] == 2:
            da = (da_logits[1] > da_logits[0]).astype(np.uint8)
        else:
            da = (da_logits[0] > 0.5).astype(np.uint8)
        if ll_logits.shape[0] == 1:
            lane = (ll_logits[0] > self.lane_threshold).astype(np.uint8)
        else:
            lane = (ll_logits[1] > ll_logits[0]).astype(np.uint8)

        da   = cv2.resize(da,   (src_w, src_h), interpolation=cv2.INTER_NEAREST)
        lane = cv2.resize(lane, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
        return da, lane

    def __del__(self):
        try:
            if cudart is None:
                return
            for ptr in self._device_ptrs.values():
                cudart.cudaFree(ptr)
            if self._stream is not None:
                cudart.cudaStreamDestroy(self._stream)
        except Exception:
            pass
