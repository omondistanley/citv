#!/usr/bin/env bash
# =============================================================================
# CITV — one-shot setup script
#
# Run from project root after cloning:
#   git clone https://github.com/<you>/citv.git && cd citv
#   bash setup.sh
#
# What this does (in order):
#   0. Disk-space check
#   1. Clone SAM2 repo (if not already present)
#   2. Download SAM2 checkpoint
#   3. Install PyTorch with the correct CUDA version
#   4. Install root Python dependencies (requirements.txt + extras)
#   5. Build SAM2 _C CUDA extension
#   6. Pre-download all HuggingFace models (Depth Anything V2 ×2, Florence-2,
#      GroundingDINO, CLIP) so the first pipeline run is instant.
#
# Requirements:
#   - Python 3.9–3.11
#   - CUDA-capable GPU (≥8 GB VRAM recommended)
#   - nvcc installed and on PATH (sudo apt install nvidia-cuda-toolkit)
#   - ~20 GB free disk space
#   - HF_TOKEN (optional) — Hugging Face access token if a weight download is gated
# =============================================================================
set -euo pipefail

CITV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CITV_ROOT"

PIP_INSTALL="pip install --no-cache-dir"

echo "=== CITV setup (project root: $CITV_ROOT) ==="
echo ""

# -----------------------------------------------------------------------------
# Python version guard (project supports 3.9–3.11)
# -----------------------------------------------------------------------------
PY_VER=$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
case "$PY_VER" in
  3.9|3.10|3.11|3.12) ;;
  *)
    echo "ERROR: Python $PY_VER detected. CITV supports Python 3.9–3.11."
    echo "Create a Python 3.11 venv and rerun setup.sh."
    exit 1
    ;;
esac

# -----------------------------------------------------------------------------
# 0. Disk-space check
# -----------------------------------------------------------------------------
FREE_KB=$(df -k . | awk 'NR==2 {print $4}')
if [ -n "$FREE_KB" ] && [ "$FREE_KB" -lt 20000000 ]; then
  echo "WARNING: Only ${FREE_KB} KB free. Full setup needs ~20 GB."
  echo "  Run: ./free_disk_space.sh  or resize the disk before continuing."
  echo "Continue anyway? [y/N]"
  read -r _reply
  [ "$_reply" = "y" ] || [ "$_reply" = "Y" ] || exit 1
fi

# -----------------------------------------------------------------------------
# 1. Clone third-party repo
# -----------------------------------------------------------------------------
echo "--- 1. Third-party repo ---"

if [ -d "sam2/.git" ]; then
  echo "  SAM2 already present; skipping clone."
else
  echo "  Cloning SAM2..."
  git clone https://github.com/facebookresearch/sam2.git
fi

# -----------------------------------------------------------------------------
# 2. SAM2 checkpoint
# -----------------------------------------------------------------------------
echo ""
echo "--- 2. SAM2 checkpoint ---"
SAM2_CKPT="sam2/checkpoints/sam2.1_hiera_large.pt"
if [ -f "$SAM2_CKPT" ]; then
  echo "  Checkpoint already present: $SAM2_CKPT"
else
  mkdir -p sam2/checkpoints
  echo "  Downloading sam2.1_hiera_large.pt (~900 MB)..."
  wget -q --show-progress -P sam2/checkpoints \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
  echo "  Downloaded: $SAM2_CKPT"
fi

# -----------------------------------------------------------------------------
# 3. PyTorch — install with matching CUDA version
# -----------------------------------------------------------------------------
echo ""
echo "--- 3. PyTorch ---"

# Detect system CUDA from nvcc
NVCC_VER=""
if command -v nvcc &>/dev/null; then
  NVCC_VER=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | head -1 | awk '{print $2}')
  [ -n "$NVCC_VER" ] && echo "  System CUDA: $NVCC_VER"
fi

if python -c "import torch; import torchvision" 2>/dev/null; then
  TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or 'cpu')" 2>/dev/null)
  echo "  PyTorch already installed (CUDA $TORCH_CUDA); skipping."
else
  if [[ "$NVCC_VER" == 11.* ]]; then
    echo "  Installing PyTorch for CUDA 11.8 (matches system nvcc $NVCC_VER)..."
    $PIP_INSTALL torch torchvision --index-url https://download.pytorch.org/whl/cu118
  elif [[ "$NVCC_VER" == 12.* ]]; then
    echo "  Installing PyTorch for CUDA 12.1 (matches system nvcc $NVCC_VER)..."
    $PIP_INSTALL torch torchvision --index-url https://download.pytorch.org/whl/cu121
  else
    echo "  nvcc not found or unknown version; installing PyTorch (CPU/default)."
    echo "  For GPU support, install manually: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    $PIP_INSTALL torch torchvision
  fi
fi

# -----------------------------------------------------------------------------
# 4. Root Python dependencies
# -----------------------------------------------------------------------------
echo ""
echo "--- 4. Root Python dependencies (pinned for Florence-2 compatibility) ---"
# Install pinned HF stack first to avoid Florence-2 incompatibilities.
$PIP_INSTALL \
  "transformers==4.46.3" \
  "tokenizers==0.20.1" \
  "accelerate==0.34.2" \
  "huggingface_hub==0.26.5" \
  "safetensors>=0.4.5"

# Install project dependencies.
$PIP_INSTALL -r requirements.txt

# Re-pin in case transitive dependencies changed versions.
$PIP_INSTALL \
  "transformers==4.46.3" \
  "tokenizers==0.20.1" \
  "accelerate==0.34.2" \
  "huggingface_hub==0.26.5"

# Ensure torch is recent enough for secure HF weight loading behavior.
python3 - <<'PY'
import sys
try:
    import torch
    base = torch.__version__.split('+')[0]
    parts = base.split('.')
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    if (major, minor) < (2, 6):
        print(f"ERROR: torch {torch.__version__} is too old. Need >= 2.6.")
        sys.exit(1)
    print(f"  torch OK: {torch.__version__}")
except Exception as e:
    print(f"ERROR: torch version check failed: {e}")
    sys.exit(1)
PY

# -----------------------------------------------------------------------------
# 5. SAM2 _C CUDA extension
# -----------------------------------------------------------------------------
# sam2/_C.so enables connected-component post-processing on masks.
# Without it SAM2 still works but with slightly lower mask quality.
echo ""
echo "--- 5. SAM2 _C CUDA extension ---"
if python -c "import sam2._C" 2>/dev/null; then
  echo "  sam2._C already built; skipping."
else
  echo "  Building SAM2 C extension (pip install --no-build-isolation -e ./sam2)..."
  $PIP_INSTALL --no-build-isolation -e "./sam2"
  # Verify
  if python -c "import sam2._C" 2>/dev/null; then
    echo "  sam2._C built successfully."
  else
    echo "  WARNING: sam2._C did not compile (nvcc error?). SAM2 will still run without it."
  fi
fi

# -----------------------------------------------------------------------------
# 6. RAM++ install + checkpoint
# -----------------------------------------------------------------------------
echo ""
echo "--- 6. RAM++ (Recognize Anything) ---"

RAM_REPO="$CITV_ROOT/third_party/recognize-anything"
RAMPP_CKPT="$CITV_ROOT/checkpoints/ram_plus_swin_large_14m.pth"

# Fast path: repo + checkpoint already present.
if [ -d "$RAM_REPO/.git" ] && [ -f "$RAMPP_CKPT" ]; then
  echo "  Found RAM++ repo and checkpoint."
  echo "  Installing recognize-anything (editable) from $RAM_REPO ..."
  $PIP_INSTALL -e "$RAM_REPO"
else
  # Ensure repo exists.
  if [ ! -d "$RAM_REPO/.git" ]; then
    echo "  Cloning Recognize Anything into $RAM_REPO ..."
    git clone https://github.com/xinyu1205/recognize-anything.git "$RAM_REPO"
  else
    echo "  recognize-anything repo already present; skipping clone."
  fi

  # Ensure editable install.
  echo "  Installing recognize-anything (editable) from $RAM_REPO ..."
  $PIP_INSTALL -e "$RAM_REPO"

  # Ensure checkpoint exists.
  if [ -f "$RAMPP_CKPT" ]; then
    echo "  RAM++ checkpoint already present: $RAMPP_CKPT"
  else
    mkdir -p "$CITV_ROOT/checkpoints"
    echo "  Downloading RAM++ checkpoint (Hugging Face Hub)..."
    echo "  Uses HF_TOKEN if set (required for some gated repos)."
    CITV_ROOT="$CITV_ROOT" python3 - <<'PY'
import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError as e:
    print(f"ERROR: huggingface_hub required: {e}")
    sys.exit(1)

root = os.environ["CITV_ROOT"]
dest_dir = os.path.join(root, "checkpoints")
os.makedirs(dest_dir, exist_ok=True)

path = hf_hub_download(
    repo_id="xinyu1205/recognize-anything",
    filename="ram_plus_swin_large_14m.pth",
    local_dir=dest_dir,
)
print(f"  RAM++ checkpoint at: {path}")
PY
  fi
fi

# Smoke test.
python3 - <<'PY'
import sys
try:
    import ram
    print("  RAM++ import OK")
except Exception as e:
    print(f"ERROR: RAM++ import failed: {e}")
    sys.exit(1)
PY

# -----------------------------------------------------------------------------
# 7. Pre-download HuggingFace models
# -----------------------------------------------------------------------------
# Uses Hugging Face CLI (hf or huggingface-cli) with a python fallback.
# Each call is idempotent — already-cached files are skipped automatically.
# Models land in ~/.cache/huggingface/hub/ (the default HF cache).
echo ""
echo "--- 7. Pre-downloading HuggingFace models ---"
echo "  This fetches ~6 GB total; already-cached files are skipped."
echo ""

FREE_KB_MODELS=$(df -k . | awk 'NR==2 {print $4}')
if [ -n "$FREE_KB_MODELS" ] && [ "$FREE_KB_MODELS" -lt 6000000 ]; then
  echo "ERROR: Only ${FREE_KB_MODELS} KB free before model download."
  echo "Need at least ~6 GB free for model predownload."
  exit 1
fi

FAILED_DOWNLOADS=0

hf_download() {
  local repo_id="$1"
  local friendly="$2"
  echo "  Downloading $friendly ($repo_id)..."
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo_id" --quiet
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo_id" --quiet
  else
    python3 -m huggingface_hub download "$repo_id" --quiet
  fi

  if [ $? -eq 0 ]; then
    echo "    ✓ $friendly"
  else
    echo "    ✗ $friendly — failed (check disk/network/HF_TOKEN)."
    FAILED_DOWNLOADS=1
  fi
}

# Depth Anything V2 — indoor metric model (~1.3 GB)
hf_download "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf" \
            "Depth-Anything-V2-Metric-Indoor-Large-hf"

# Depth Anything V2 — outdoor metric model (~1.3 GB)
hf_download "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf" \
            "Depth-Anything-V2-Metric-Outdoor-Large-hf"

# CLIP — used to classify indoor/outdoor before loading depth model (~340 MB)
hf_download "openai/clip-vit-base-patch32" \
            "CLIP ViT-B/32"

# Grounding DINO — open-vocab object detector for GroundedSAM2 (~341 MB)
hf_download "IDEA-Research/grounding-dino-base" \
            "GroundingDINO-base"

# Florence-2 — object labelling + relation prediction (~900 MB)
hf_download "microsoft/Florence-2-large" \
            "Florence-2-large"

if [ "$FAILED_DOWNLOADS" -ne 0 ]; then
  echo ""
  echo "ERROR: One or more HuggingFace model downloads failed."
  echo "Fix the reported issue(s) and rerun setup.sh."
  exit 1
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "=== CITV setup complete!              ==="
echo "=========================================="
echo ""
echo "All models are pre-downloaded. Run the pipeline:"
echo "  python -m scene_understanding --input_dir images --output_dir output_scene"
echo ""
echo "Optional — camera calibration for accurate intrinsics:"
echo "  python tools/calibrate_camera.py --images path/to/checkerboard/ --out calibration.json"
echo "  Then set camera_calibration_file = 'calibration.json' in config.py"
