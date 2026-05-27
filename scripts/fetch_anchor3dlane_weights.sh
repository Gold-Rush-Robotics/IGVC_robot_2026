#!/usr/bin/env bash
# Download Anchor3DLane++ model weights from HuggingFace.
#
# Usage:
#   scripts/fetch_anchor3dlane_weights.sh [WEIGHTS_DIR] [ANCHOR3DLANE_DIR]
#
# WEIGHTS_DIR       defaults to ~/anchor3dlane_weights
# ANCHOR3DLANE_DIR  defaults to ~/anchor3dlane
#                   (used only to print the config path)
#
# Downloads (OpenLane-v1.2):
#   openlane_anchor3dlane++_r50.pth     — ResNet-50 camera-only 360x480 F1=59.4 (recommended on Py3.12)
#   openlane_anchor3dlane++_r18_se.pth  — ResNet-18 + SECOND    360x480 F1=59.8 (requires spconv)
#   openlane_anchor3dlane++_r50_se.pth  — ResNet-50 + SECOND    360x480 F1=61.4 (requires spconv)
#
# Requires: huggingface_hub  (pip3 install huggingface_hub)
set -euo pipefail

WEIGHTS_DIR="${1:-$HOME/anchor3dlane_weights}"
A3D_DIR="${2:-$HOME/anchor3dlane}"

BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${CYAN}[weights]${NC} $*"; }
success() { echo -e "${GREEN}[weights]${NC} $*"; }
die()     { echo -e "${RED}[weights] ERROR:${NC} $*" >&2; exit 1; }

HF_REPO="nowherespyfly/anchor3dlane"
# HuggingFace paths (include subfolder)
R50_CAM_HF="Openlane/openlane_anchor3dlane++_r50.pth"    # ResNet-50 camera-only, 360x480, F1=59.4
R18_HF="Openlane/openlane_anchor3dlane++_r18_se.pth"   # ResNet-18 + SECOND, 360x480, F1=59.8
R50_HF="Openlane/openlane_anchor3dlane++_r50_se.pth"   # ResNet-50 + SECOND, 360x480, F1=61.4
# Local filenames (basename only)
R50_CAM_CKPT="openlane_anchor3dlane++_r50.pth"
R18_CKPT="openlane_anchor3dlane++_r18_se.pth"
R50_CKPT="openlane_anchor3dlane++_r50_se.pth"

mkdir -p "$WEIGHTS_DIR"

# ── Check huggingface_hub ─────────────────────────────────────────────────
python3 -c "import huggingface_hub" 2>/dev/null \
    || die "huggingface_hub not installed.\n  pip3 install huggingface_hub"

# ── Download checkpoints ──────────────────────────────────────────────────
# $1 = HuggingFace path (e.g. Openlane/foo.pth)  $2 = local basename
download_weight() {
    local hf_path="$1"
    local basename="$2"
    local dest="$WEIGHTS_DIR/$basename"
    if [[ -f "$dest" ]]; then
        info "$basename already present — skipping"
    else
        info "Downloading $basename from $HF_REPO …"
        python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id='$HF_REPO', filename='$hf_path')
os.makedirs('$WEIGHTS_DIR', exist_ok=True)
shutil.copy(path, '$dest')
print('  saved to $dest')
"
    fi
}

download_weight "$R50_CAM_HF" "$R50_CAM_CKPT"
download_weight "$R18_HF" "$R18_CKPT"
download_weight "$R50_HF" "$R50_CKPT"

# ── Resolve config path ─────────────────────────────────────────────────
# Verify these paths exist after cloning:
R50_CAM_CFG="$A3D_DIR/configs/openlane/anchor3dlane_iter_r50.py"
R18_CFG="$A3D_DIR/configs/openlane/anchor3dlane_mf.py"
R50_CFG="$A3D_DIR/configs/openlane/anchor3dlane_mf.py"  # same fusion config, backbone set inside

# Warn if config not found
if [[ ! -f "$R50_CAM_CFG" ]]; then
    echo -e "${RED}WARNING:${NC} camera-only config not found at $R50_CAM_CFG" >&2
    echo "  Check available configs with:  ls $A3D_DIR/configs/openlane/" >&2
fi
if [[ ! -f "$R18_CFG" ]]; then
    echo -e "${RED}WARNING:${NC} fusion config not found at $R18_CFG" >&2
    echo "  Check available configs with:  ls $A3D_DIR/configs/openlane/" >&2
fi

success "Weights downloaded to $WEIGHTS_DIR"
echo ""
echo -e "${BOLD}Update anchor3dlane_config.yaml with these values:${NC}"
echo ""
echo "  # ResNet-50 camera-only (recommended on Jetson Py3.12: no spconv)"
echo "  anchor3dlane_root: \"$A3D_DIR\""
echo "  config_path:       \"$R50_CAM_CFG\""
echo "  checkpoint_path:   \"$WEIGHTS_DIR/$R50_CAM_CKPT\""
echo ""
echo "  # ResNet-18 + SECOND (requires spconv, F1=59.8)"
echo "  # config_path:     \"$R18_CFG\""
echo "  # checkpoint_path: \"$WEIGHTS_DIR/$R18_CKPT\""
echo ""
echo "  # ResNet-50 + SECOND (requires spconv, F1=61.4)"
echo "  # config_path:     \"$R50_CFG\""
echo "  # checkpoint_path: \"$WEIGHTS_DIR/$R50_CKPT\""
