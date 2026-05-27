#!/usr/bin/env bash
# Install Anchor3DLane++ and its dependencies.
#
# Usage (run from the repo root, or from anywhere):
#   scripts/setup_anchor3dlane.sh [INSTALL_DIR]
#
# INSTALL_DIR defaults to ~/anchor3dlane
#
# After this script succeeds, run:
#   scripts/fetch_anchor3dlane_weights.sh
# and update anchor3dlane_config.yaml with the resulting paths.
#
# Tested on: Jetson AGX Orin — JetPack 6, CUDA 12, Python 3.10
set -euo pipefail

INSTALL_DIR="${1:-$HOME/anchor3dlane}"

# ── Colours ───────────────────────────────────────────────────────────────
BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${CYAN}[anchor3dlane]${NC} $*"; }
success() { echo -e "${GREEN}[anchor3dlane]${NC} $*"; }
die()     { echo -e "${RED}[anchor3dlane] ERROR:${NC} $*" >&2; exit 1; }

# ── Prerequisites ─────────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v pip3    >/dev/null 2>&1 || die "pip3 not found"
command -v git     >/dev/null 2>&1 || die "git not found"
command -v nvcc    >/dev/null 2>&1 || die "nvcc not found — CUDA toolkit required (JetPack 6)"

CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
info "Detected CUDA $CUDA_VER"

TORCH_AVAILABLE=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || true)
if [[ -z "$TORCH_AVAILABLE" ]]; then
    die "PyTorch not found. Install the Jetson-specific wheel first:\n" \
        "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
fi
info "PyTorch $TORCH_AVAILABLE already installed — skipping"

# ── Clone repo ────────────────────────────────────────────────────────────
if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Repo already exists at $INSTALL_DIR — pulling latest"
    git -C "$INSTALL_DIR" pull --ff-only
else
    info "Cloning Anchor3DLane++ into $INSTALL_DIR …"
    git clone --depth 1 -b anchor3dlane++ \
        https://github.com/tusen-ai/Anchor3DLane "$INSTALL_DIR"
fi

# ── Python dependencies ───────────────────────────────────────────────────
info "Installing mmcv-full for CUDA $CUDA_VER …"
# mmcv-full requires a CUDA-compatible wheel; the pre-built index below
# covers CUDA 11.x–12.x builds.  Adjust the torch/cuda version tags if needed.
TORCH_SHORT=$(python3 -c "import torch; v=torch.__version__.split('+')[0]; print(''.join(v.split('.')[:2]))")
pip3 install --quiet \
    "mmcv-full" \
    "mmdet==2.28.2" \
    "mmsegmentation==0.30.0" \
    --index-url https://pypi.org/simple/ \
    || die "mmcv-full install failed.\n  If pre-built wheels are unavailable try:\n  pip3 install mmcv-full --find-links https://download.openmmlab.com/mmcv/dist/cu${CUDA_VER//./}/torch${TORCH_SHORT}/index.html"

info "Installing remaining Python requirements …"
pip3 install --quiet \
    numpy \
    opencv-python-headless \
    scipy \
    tqdm \
    terminaltables \
    yapf \
    huggingface_hub \
    --index-url https://pypi.org/simple/

# ── Build Anchor3DLane++ package ──────────────────────────────────────────
info "Running python setup.py develop in $INSTALL_DIR …"
pushd "$INSTALL_DIR" >/dev/null
python3 setup.py develop --quiet

# Deformable attention CUDA extension
DFORM_DIR="mmseg/models/utils/ops"
if [[ -d "$DFORM_DIR" ]]; then
    info "Patching mmseg/__init__.py version check for mmcv==1.7.2 …"
    # The repo asserts  mmcv_version < mmcv_max_version  with strict <.
    # mmcv 1.7.2 (the latest 1.x release) fails because max is 1.7.2.
    # Relax to <= so 1.7.2 is accepted.
    MMSEG_INIT="mmseg/__init__.py"
    if [[ -f "$MMSEG_INIT" ]]; then
        sed -i 's/mmcv_version < mmcv_max_version/mmcv_version <= mmcv_max_version/g' "$MMSEG_INIT"
        info "Patched $MMSEG_INIT"
    fi

    info "Patching ms_deform_attn_cuda.cu for PyTorch >= 2.0 API …"
    # Newer PyTorch removed at::DeprecatedTypeProperties; AT_DISPATCH_FLOATING_TYPES
    # must use tensor.scalar_type() instead of the deprecated tensor.type().
    CU_FILE="$DFORM_DIR/src/cuda/ms_deform_attn_cuda.cu"
    if [[ -f "$CU_FILE" ]]; then
        # Replace all occurrences of AT_DISPATCH_FLOATING_TYPES(<tensor>.type(),
        # with AT_DISPATCH_FLOATING_TYPES(<tensor>.scalar_type(),
        sed -i -E 's/AT_DISPATCH_FLOATING_TYPES\(([a-zA-Z_][a-zA-Z0-9_]*)\.type\(\)/AT_DISPATCH_FLOATING_TYPES(\1.scalar_type()/g' "$CU_FILE"
        info "Patch applied to $CU_FILE"
    else
        info "  $CU_FILE not found — skipping patch"
    fi

    info "Building deformable attention CUDA extension …"
    pushd "$DFORM_DIR" >/dev/null
    bash make.sh
    popd >/dev/null
else
    info "Deformable attention dir not found at $DFORM_DIR — skipping extension build"
fi

# ── NumPy >= 1.24 compatibility patches ──────────────────────────────────
# np.float/int/complex/bool/object/str were removed in NumPy 1.24.
# Patch every Python file under mmseg/ that uses these aliases.
info "Patching repo for NumPy >= 1.24 removed aliases …"
grep -rl 'np\.float\b\|np\.int\b\|np\.complex\b\|np\.bool\b\|np\.object\b\|np\.str\b' mmseg/ 2>/dev/null \
    | xargs --no-run-if-empty sed -i -E \
        's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.complex\b/complex/g; s/np\.bool\b/bool/g; s/np\.object\b/object/g; s/np\.str\b/str/g'

popd >/dev/null

# ── Verify import ─────────────────────────────────────────────────────────
info "Verifying import …"
python3 -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
import mmcv
from mmseg.models import build_segmentor
print('mmcv', mmcv.__version__, '— OK')
" || die "Import check failed. Check the output above for missing dependencies."

success "Anchor3DLane++ installed to $INSTALL_DIR"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "  1. Download model weights:"
echo "       scripts/fetch_anchor3dlane_weights.sh"
echo "  2. Update config:"
echo "       src/igvc_test_bringup/config/anchor3dlane_config.yaml"
echo "     Set anchor3dlane_root: \"$INSTALL_DIR\""
echo "     Set config_path, checkpoint_path (printed by fetch script)"
