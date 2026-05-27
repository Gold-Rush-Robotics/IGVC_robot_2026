#!/usr/bin/env bash
# Download one of the upstream Ultra-Fast-Lane-Detection-v2 checkpoints.
#
# Usage:
#   scripts/fetch_ufldv2_weights.sh [variant] [DEST_DIR]
#
# Variants:
#   culane_res18     default, configs/culane_res18.py
#   culane_res34
#   tusimple_res18
#   tusimple_res34
#   curvelanes_res18
#   curvelanes_res34
#
# Checkpoints are downloaded from the Google Drive links published in the
# upstream README and stored under $REPO_ROOT/models by default.
set -euo pipefail

VARIANT="${1:-culane_res18}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${2:-${REPO_ROOT}/models}"

BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${CYAN}[ufldv2-weights]${NC} $*"; }
success() { echo -e "${GREEN}[ufldv2-weights]${NC} $*"; }
die()     { echo -e "${RED}[ufldv2-weights] ERROR:${NC} $*" >&2; exit 1; }

case "$VARIANT" in
    culane_res18)
        FILE_ID="1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W"
        OUT_NAME="ufldv2_culane_res18.pth"
        CONFIG_PATH="configs/culane_res18.py"
        ;;
    culane_res34)
        FILE_ID="1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj"
        OUT_NAME="ufldv2_culane_res34.pth"
        CONFIG_PATH="configs/culane_res34.py"
        ;;
    tusimple_res18)
        FILE_ID="1Clnj9-dLz81S3wXiYtlkc4HVusCb978t"
        OUT_NAME="ufldv2_tusimple_res18.pth"
        CONFIG_PATH="configs/tusimple_res18.py"
        ;;
    tusimple_res34)
        FILE_ID="1pkz8homK433z39uStGK3ZWkDXrnBAMmX"
        OUT_NAME="ufldv2_tusimple_res34.pth"
        CONFIG_PATH="configs/tusimple_res34.py"
        ;;
    curvelanes_res18)
        FILE_ID="1VfbUvorKKMG4tUePNbLYPp63axgd-8BX"
        OUT_NAME="ufldv2_curvelanes_res18.pth"
        CONFIG_PATH="configs/curvelanes_res18.py"
        ;;
    curvelanes_res34)
        FILE_ID="1O1kPSr85Icl2JbwV3RBlxWZYhLEHo8EN"
        OUT_NAME="ufldv2_curvelanes_res34.pth"
        CONFIG_PATH="configs/curvelanes_res34.py"
        ;;
    *)
        die "Unknown variant '$VARIANT'. Use one of: culane_res18, culane_res34, tusimple_res18, tusimple_res34, curvelanes_res18, curvelanes_res34"
        ;;
esac

mkdir -p "$DEST_DIR"
DEST_FILE="$DEST_DIR/$OUT_NAME"

looks_like_torchscript() {
    local path="$1"
    python3 - "$path" <<'PY'
import sys
import zipfile
path = sys.argv[1]
if not zipfile.is_zipfile(path):
    raise SystemExit(1)
with zipfile.ZipFile(path) as archive:
    for name in archive.namelist():
        if '/code/' in name or name.startswith('code/'):
            raise SystemExit(0)
raise SystemExit(1)
PY
}

if [[ -f "$DEST_FILE" ]]; then
    size_bytes=$(stat -c '%s' "$DEST_FILE" 2>/dev/null || stat -f '%z' "$DEST_FILE")
    if looks_like_torchscript "$DEST_FILE"; then
        info "Existing $DEST_FILE is a TorchScript archive, not a UFLDv2 checkpoint; redownloading"
        rm -f "$DEST_FILE"
    elif (( size_bytes > 10000000 )); then
        info "$OUT_NAME already present at $DEST_FILE (${size_bytes} bytes); skipping"
        echo "Delete the file and rerun to force a redownload."
        exit 0
    else
        info "Existing $DEST_FILE looks incomplete (${size_bytes} B); redownloading"
        rm -f "$DEST_FILE"
    fi
fi

python3 - <<'PY' >/dev/null 2>&1 || python3 -m pip install gdown
import gdown
PY

TMP_FILE="${DEST_FILE}.tmp"
rm -f "$TMP_FILE"
info "Downloading $VARIANT checkpoint to $DEST_FILE"
python3 - <<PY
import gdown
import os
url = 'https://drive.google.com/uc?id=${FILE_ID}'
output = '${TMP_FILE}'
result = gdown.download(url, output, quiet=False)
if result is None or not os.path.exists(output):
    raise SystemExit('gdown failed to download ${VARIANT}')
PY
mv "$TMP_FILE" "$DEST_FILE"
if looks_like_torchscript "$DEST_FILE"; then
    rm -f "$DEST_FILE"
    die "Downloaded file is a TorchScript archive, not a UFLDv2 checkpoint. Check the Google Drive file id for $VARIANT."
fi

success "Downloaded $DEST_FILE"
echo ""
echo -e "${BOLD}Use these values with lane_segmentation:${NC}"
echo "  export UFLDV2_CONFIG=\"$CONFIG_PATH\""
echo "  export UFLDV2_WEIGHTS=\"$DEST_FILE\""
echo ""
echo "  detection_mode: \"ufldv2\""
echo "  ufldv2_config:  \"$CONFIG_PATH\""
echo "  model_weights:  \"$DEST_FILE\""