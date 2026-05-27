#!/usr/bin/env bash
# Download the YOLOPv2 TorchScript weights into $REPO_ROOT/models/yolopv2.pt.
#
# The weights (MIT-licensed) come from the upstream CAIC-AD/YOLOPv2 GitHub
# release and are intentionally kept out of version control (see .gitignore).
# Run this once per machine:
#
#     ./src/igvc_lane_detection/scripts/fetch_yolopv2_weights.sh
#     export YOLOPV2_WEIGHTS=$PWD/models/yolopv2.pt
#
# Jetson AGX Orin users: PyTorch itself must be installed separately from
# the NVIDIA JetPack wheel index — see training/README.md.

set -euo pipefail

URL="https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives at <repo>/src/igvc_lane_detection/scripts/
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEST_DIR="${REPO_ROOT}/models"
DEST_FILE="${DEST_DIR}/yolopv2.pt"

mkdir -p "${DEST_DIR}"

if [[ -f "${DEST_FILE}" ]]; then
  size_bytes=$(stat -c '%s' "${DEST_FILE}" 2>/dev/null || stat -f '%z' "${DEST_FILE}")
  if (( size_bytes > 10000000 )); then
    echo "yolopv2.pt already present at ${DEST_FILE} (${size_bytes} bytes) — skipping."
    echo "Delete the file and rerun to force a redownload."
    exit 0
  fi
  echo "Existing ${DEST_FILE} looks incomplete (${size_bytes} B); redownloading."
  rm -f "${DEST_FILE}"
fi

echo "Downloading YOLOPv2 weights → ${DEST_FILE}"
if command -v curl >/dev/null 2>&1; then
  curl --fail --location --progress-bar -o "${DEST_FILE}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget --show-progress -O "${DEST_FILE}" "${URL}"
else
  echo "ERROR: neither curl nor wget is installed." >&2
  exit 1
fi

echo
echo "Done. Export the path so the launch file picks it up:"
echo "  export YOLOPV2_WEIGHTS=\"${DEST_FILE}\""
