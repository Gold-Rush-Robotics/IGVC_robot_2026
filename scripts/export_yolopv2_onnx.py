#!/usr/bin/env python3
"""Export the YOLOPv2 TorchScript weight to ONNX.

This is step 1 of the Jetson optimisation pipeline:

    1. ``export_yolopv2_onnx.py``  — TorchScript .pt → .onnx
    2. ``export_yolopv2_tensorrt.sh`` — .onnx → .engine via ``trtexec``

The resulting ``.engine`` can be loaded by ``YolopV2TRT`` (see
``src/igvc_lane_detection/igvc_lane_detection/yolopv2_trt.py``) and gives
a ~2-4× speedup vs the TorchScript path on Jetson Orin.

Usage
-----
    python3 scripts/export_yolopv2_onnx.py \\
        --weights models/yolopv2.pt \\
        --output  models/yolopv2_384.onnx \\
        --img-size 384

Only the two segmentation heads are exported.  The traffic-object
detection head is discarded (matches the runtime YolopV2 wrapper).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


class _SegOnly(nn.Module):
    """Wrap the YOLOPv2 TorchScript model and return only seg + lane outputs."""

    def __init__(self, jit_model: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.model = jit_model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        # YOLOPv2 forward: [pred, anchor_grid], seg, ll
        seg, ll = out[1], out[2]
        return seg, ll


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="Path to yolopv2.pt (TorchScript)")
    ap.add_argument("--output",  required=True, help="Output .onnx path")
    ap.add_argument("--img-size", type=int, default=384,
                    help="Input height in pixels (stride 32). Default 384.")
    ap.add_argument("--img-width", type=int, default=None,
                    help="Input width in pixels (stride 32). Defaults to --img-size (square).")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--half", action="store_true",
                    help="Export with FP16 weights. Recommended for Jetson.")
    args = ap.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    output  = Path(args.output).expanduser().resolve()
    if not weights.is_file():
        print(f"weights not found: {weights}", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    img_h = args.img_size
    img_w = args.img_width if args.img_width is not None else img_h
    for dim, name in ((img_h, 'img-size'), (img_w, 'img-width')):
        if dim % 32 != 0:
            print(f"{name} must be a multiple of 32 (got {dim})", file=sys.stderr)
            return 1

    device = torch.device(args.device)
    print(f"[export] device={device}  half={args.half}  size={img_h}x{img_w}")

    jit = torch.jit.load(str(weights), map_location=device).eval()
    if args.half and device.type == "cuda":
        jit = jit.half()
    model = _SegOnly(jit).to(device).eval()

    dummy = torch.zeros(1, 3, img_h, img_w, device=device)
    if args.half and device.type == "cuda":
        dummy = dummy.half()

    # Warm up + sanity check the wrapped forward.
    with torch.no_grad():
        seg, ll = model(dummy)
    print(f"[export] seg shape={tuple(seg.shape)}  ll shape={tuple(ll.shape)}")

    # torch.onnx.export's internal ONNXTracedModule cannot re-enter a
    # torch.jit.ScriptModule that is nested inside an nn.Module wrapper.
    # Fix: pre-trace with torch.jit.trace first (which CAN descend into
    # ScriptModules), then pass the resulting ScriptModule to onnx.export.
    # The exporter detects a ScriptModule and converts its TorchScript IR
    # to ONNX directly instead of trying to re-trace it.
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, check_trace=False)

    torch.onnx.export(
        traced,
        dummy,
        str(output),
        input_names=["images"],
        output_names=["seg", "ll"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,  # fixed batch=1, fixed size for trtexec
    )
    print(f"[export] wrote {output}  ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
