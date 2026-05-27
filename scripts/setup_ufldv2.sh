#!/usr/bin/env bash
# Clone Ultra-Fast-Lane-Detection-v2 and install the Python packages needed
# by the IGVC wrapper.
#
# Usage:
#   scripts/setup_ufldv2.sh [INSTALL_DIR]
#
# INSTALL_DIR defaults to ~/Ultra-Fast-Lane-Detection-v2.
# PyTorch itself must already be installed. On Jetson, install the
# JetPack-matched torch/torchvision wheels from NVIDIA before running this.
set -euo pipefail

INSTALL_DIR="${1:-$HOME/Ultra-Fast-Lane-Detection-v2}"
REPO_URL="https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${CYAN}[ufldv2]${NC} $*"; }
success() { echo -e "${GREEN}[ufldv2]${NC} $*"; }
die()     { echo -e "${RED}[ufldv2] ERROR:${NC} $*" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v git >/dev/null 2>&1 || die "git not found"
python3 -m pip --version >/dev/null 2>&1 || die "python3 -m pip not available"

if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Repo already exists at $INSTALL_DIR; pulling latest"
    git -C "$INSTALL_DIR" pull --ff-only
else
    info "Cloning Ultra-Fast-Lane-Detection-v2 into $INSTALL_DIR"
    git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
fi

info "Checking PyTorch"
python3 - <<'PY' || die "PyTorch is not installed. Install the JetPack-matched wheel on Jetson, then rerun."
import torch
print('torch', torch.__version__)
PY

if ! python3 - <<'PY' >/dev/null 2>&1
import torchvision
PY
then
    if [[ "$(uname -m)" == "aarch64" ]]; then
        die "torchvision is missing. On Jetson, install the torchvision wheel that matches your JetPack/PyTorch build."
    fi
    info "torchvision missing; installing from PyPI for this non-Jetson machine"
    python3 -m pip install torchvision --break-system-packages
fi

info "Installing UFLDv2 runtime dependencies"
python3 -m pip install \
    addict \
    gdown \
    imagesize \
    opencv-python-headless \
    pathspec \
    pillow \
    scikit-learn \
    tensorboard \
    tqdm \
    ujson \
    --break-system-packages

info "Verifying runtime imports"
python3 - <<'PY'
import cv2
import gdown
import torch
import torchvision
print('cv2', cv2.__version__)
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
PY

success "UFLDv2 source is ready at $INSTALL_DIR"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "  scripts/fetch_ufldv2_weights.sh culane_res18"
echo "  export UFLDV2_ROOT=\"$INSTALL_DIR\""
echo "  export UFLDV2_CONFIG=\"configs/culane_res18.py\""
echo "  export UFLDV2_WEIGHTS=\"$REPO_ROOT/models/ufldv2_culane_res18.pth\""
echo ""
echo "Then set lane_segmentation detection_mode to ufldv2 and pass model_weights:=\$UFLDV2_WEIGHTS."