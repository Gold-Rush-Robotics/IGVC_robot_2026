#!/usr/bin/env bash
# Compile a YOLOPv2 ONNX model into a TensorRT engine for Jetson Orin.
#
# Usage:
#   scripts/export_yolopv2_tensorrt.sh \
#       models/yolopv2_384.onnx \
#       models/yolopv2_384.engine \
#       [fp16|fp32]
#
# Defaults: fp16. Requires the JetPack-bundled trtexec on $PATH.
set -euo pipefail

ONNX="${1:-models/yolopv2_384.onnx}"
ENGINE="${2:-${ONNX%.onnx}.engine}"
PRECISION="${3:-fp16}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found on PATH. On Jetson it lives at:" >&2
  echo "  /usr/src/tensorrt/bin/trtexec" >&2
  echo "Add that directory to PATH or run trtexec directly." >&2
  exit 1
fi

if [[ ! -f "$ONNX" ]]; then
  echo "ONNX file not found: $ONNX" >&2
  exit 1
fi

mkdir -p "$(dirname "$ENGINE")"

EXTRA=()
case "$PRECISION" in
  fp16) EXTRA+=("--fp16") ;;
  fp32) ;;  # nothing
  *)
    echo "unknown precision: $PRECISION (use fp16|fp32)" >&2
    exit 1 ;;
esac

echo "[trtexec] onnx=$ONNX  engine=$ENGINE  precision=$PRECISION"
trtexec \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --memPoolSize=workspace:2048 \
  --useSpinWait \
  "${EXTRA[@]}"

echo "[trtexec] wrote $ENGINE"
