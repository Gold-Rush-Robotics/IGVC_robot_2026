"""YOLOPv2 TorchScript inference wrapper.

Thin, ROS-free wrapper around the pretrained YOLOPv2 TorchScript weight
(`yolopv2.pt`) published by CAIC-AD under the MIT licence.  Only the two
segmentation heads are exposed — the traffic-object detection head is
discarded to avoid the ``torchvision`` NMS dependency.

The pre/post-processing helpers (``letterbox``, ``driving_area_mask``,
``lane_line_mask``) are ported verbatim (trimmed) from the YOLOPv2
repository so results match the reference demo.  Credit / licence:
https://github.com/CAIC-AD/YOLOPv2 (MIT).

Target hardware: NVIDIA Jetson AGX Orin Dev Kit (Ampere, aarch64).  The
JetPack-matched ``torch`` wheel from
https://developer.download.nvidia.com/compute/redist/jp/ is required
for CUDA support — stock PyPI ``torch`` has no CUDA on aarch64.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

try:  # torch import is deferred until ``load()`` so module import doesn't
      # fail on machines that haven't installed the Jetson wheel yet.
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────
# YOLOPv2-style helpers (ported from CAIC-AD/YOLOPv2 utils/utils.py, MIT)
# ──────────────────────────────────────────────────────────────────────────

def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114),
               auto=True, scale_fill=False, scaleup=True, stride=32):
    """Resize + pad while keeping aspect ratio and a stride multiple.

    Returns ``(padded_img, (r, r), (dw, dh))`` matching YOLOPv2 semantics
    so we can invert the padding afterwards.
    """
    shape = img.shape[:2]  # (H, W)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1]

    dw /= 2.0
    dh /= 2.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (dw, dh)


# ──────────────────────────────────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────────────────────────────────

class YolopV2:
    """End-to-end wrapper: load weight, run inference, return raw masks.

    Parameters
    ----------
    weights_path:
        Filesystem path to ``yolopv2.pt`` (TorchScript).
    device:
        ``cuda:0``, ``cpu``, ``cuda:N``.  Falls back to CPU with a warning
        if CUDA is requested but not available.
    half:
        Run inference in FP16.  Only honored on CUDA.
    img_size:
        Letterboxed side length (stride 32).  640 matches the trained
        model; other values are untested.
    resize_hw:
        Optional intermediate resize ``(H, W)`` applied before
        letterboxing.  ``None`` keeps the native input resolution and
        avoids aspect-ratio distortion on non-4:3 cameras.
    """

    # Crop offsets used by YOLOPv2 to strip the stride-32 padding on the
    # vertical axis of the seg heads (see utils/utils.driving_area_mask).
    _SEG_CROP_TOP = 12
    _SEG_CROP_BOTTOM = 372  # exclusive

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda:0",
        half: bool = True,
        img_size: int = 640,
        resize_hw: Optional[Tuple[int, int]] = None,
        preprocess: bool = True,
        clahe_clip: float = 2.0,
        clahe_tile: Tuple[int, int] = (8, 8),
        blur_ksize: Tuple[int, int] = (5, 5),
        blur_sigma: float = 0.0,
        lane_threshold: float = 0.5,
    ) -> None:
        self.weights_path = weights_path
        self.requested_device = device
        self.half = bool(half)
        self.img_size = int(img_size)
        self.resize_hw = tuple(resize_hw) if resize_hw is not None else None  # (H, W)
        # Probability threshold applied to the lane-line head.  Lower
        # values recover thin / faint lane paint (e.g. IRL 0.5–2 inch
        # markings that activate the head only weakly) at the cost of
        # more salt-and-pepper noise — the caller is expected to clean
        # up with morphology + color filtering.
        self.lane_threshold = float(lane_threshold)

        # Pre-processing: CLAHE histogram equalisation on the luma channel
        # (boosts contrast in shadows / bright sun without color shifts)
        # followed by a Gaussian blur (suppresses high-frequency texture
        # noise that otherwise produces speckle in the lane mask).
        self.preprocess_enabled = bool(preprocess)
        self.blur_ksize = tuple(blur_ksize)
        self.blur_sigma = float(blur_sigma)
        self._clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_tile))

        self._model = None  # torch.jit.ScriptModule once loaded
        self._device = None  # torch.device once loaded
        self._stride = 32

    # ── Lifecycle ──────────────────────────────────────────────────────

    def load(self) -> None:
        """Load TorchScript weight onto the requested device and warm up."""
        if torch is None:
            raise RuntimeError(
                "PyTorch is not installed. On Jetson AGX Orin, install the "
                "JetPack-matched wheel from "
                "https://developer.download.nvidia.com/compute/redist/jp/")

        want_cuda = self.requested_device.startswith("cuda")
        if want_cuda and not torch.cuda.is_available():
            self._device = torch.device("cpu")
            self.half = False
            self._loaded_fallback_warning = (
                f"CUDA requested ({self.requested_device}) but not available;"
                " falling back to CPU.")
        else:
            self._device = torch.device(self.requested_device)
            self._loaded_fallback_warning = None

        model = torch.jit.load(self.weights_path, map_location=self._device)
        model = model.to(self._device)
        if self._device.type == "cuda" and self.half:
            model.half()
        else:
            self.half = False
        model.eval()
        self._model = model

        # Warm-up so the first real frame isn't TorchScript's first-call
        # autotune pass.
        with torch.no_grad():
            dummy = torch.zeros(
                1, 3, self.img_size, self.img_size, device=self._device)
            if self.half:
                dummy = dummy.half()
            _ = self._model(dummy)

    @property
    def device(self):
        return self._device

    @property
    def fallback_warning(self) -> Optional[str]:
        return getattr(self, "_loaded_fallback_warning", None)

    # ── Inference ──────────────────────────────────────────────────────

    def infer(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run YOLOPv2 on a BGR image and return two uint8 masks.

        Both returned masks have the **same shape as ``bgr``** (i.e. the
        caller's original resolution), dtype ``uint8``, values in
        ``{0, 1}``.  ``da_mask`` = drivable-area; ``ll_mask`` = lane
        lines.
        """
        if self._model is None:
            raise RuntimeError("YolopV2.load() must be called before infer().")

        src_h, src_w = bgr.shape[:2]

        # 1) Optionally resize before letterbox.
        if self.resize_hw is None:
            resized = bgr
        else:
            resized = cv2.resize(
                bgr, (self.resize_hw[1], self.resize_hw[0]),
                interpolation=cv2.INTER_LINEAR)

        # 1b) Optional contrast/denoise preprocessor.
        if self.preprocess_enabled:
            resized = self._preprocess(resized)

        # 2) Letterbox to a stride-multiple square.
        lb, ratio, (dw, dh) = _letterbox(
            resized, new_shape=self.img_size, stride=self._stride,
            auto=True, scaleup=True)
        lb_h, lb_w = lb.shape[:2]

        # 3) Tensor prep: BGR→RGB, HWC→CHW, /255, add batch dim, cast.
        img = lb[:, :, ::-1].transpose(2, 0, 1)  # RGB, CHW
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(self._device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        tensor = tensor.unsqueeze(0)

        # 4) Forward pass.  YOLOPv2 returns
        #    ``[pred, anchor_grid], seg, ll`` — we discard the detection
        #    head (traffic objects) to avoid a torchvision NMS dep.
        with torch.no_grad():
            out = self._model(tensor)

        seg, ll = out[1], out[2]

        # 5) Post-process each seg head.  The YOLOPv2 crop + ×2 upsample
        #    yields a mask whose shape matches the pre-letterbox resized
        #    image (``resize_hw``) — the crop removes exactly the top/
        #    bottom letterbox padding rows.
        da_mask = self._postprocess_da(seg)
        ll_mask = self._postprocess_ll(ll)

        # 6) Resize back to the caller's original resolution.
        if (src_h, src_w) != da_mask.shape[:2]:
            da_mask = cv2.resize(
                da_mask, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
            ll_mask = cv2.resize(
                ll_mask, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
        
        return da_mask.astype(np.uint8), ll_mask.astype(np.uint8)

    # ── Pre-processing ────────────────────────────────────────────────

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """CLAHE histogram equalisation on Y (YUV) + Gaussian blur.

        Operates on the luma channel only so color balance is preserved
        and the network still sees a natural-looking RGB image.
        """
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = self._clahe.apply(yuv[:, :, 0])
        out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        if self.blur_ksize and self.blur_ksize[0] > 1 and self.blur_ksize[1] > 1:
            out = cv2.GaussianBlur(out, self.blur_ksize, self.blur_sigma)
        return out

    # ── Post-processing (YOLOPv2-equivalent) ──────────────────────────

    def _postprocess_da(self, seg) -> np.ndarray:
        """Drivable-area mask at the pre-letterbox resolution (uint8)."""
        crop_top = self._SEG_CROP_TOP
        crop_bot = self._SEG_CROP_BOTTOM
        da_predict = seg[:, :, crop_top:crop_bot, :]
        da_up = torch.nn.functional.interpolate(
            da_predict, scale_factor=2, mode="bilinear", align_corners=False)
        da_mask = torch.argmax(da_up, dim=1).to(torch.uint8)
        return da_mask.squeeze(0).detach().cpu().numpy()

    def _postprocess_ll(self, ll) -> np.ndarray:
        """Lane-line mask at the pre-letterbox resolution (uint8).

        Uses ``self.lane_threshold`` instead of a hard 0.5 round so faint
        / thin lane responses can be recovered.
        """
        crop_top = self._SEG_CROP_TOP
        crop_bot = self._SEG_CROP_BOTTOM
        ll_predict = ll[:, :, crop_top:crop_bot, :]
        ll_up = torch.nn.functional.interpolate(
            ll_predict, scale_factor=2, mode="bilinear", align_corners=False)
        ll_mask = (ll_up > self.lane_threshold).to(torch.uint8)
        return ll_mask.squeeze(0).squeeze(0).detach().cpu().numpy()
