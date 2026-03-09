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
# =============================================================================
set -e

CITV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CITV_ROOT"

PIP_INSTALL="pip install --no-cache-dir"

echo "=== CITV setup (project root: $CITV_ROOT) ==="
echo ""

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
echo "--- 4. Root Python dependencies (requirements.txt) ---"
$PIP_INSTALL -r requirements.txt

# Extra runtime deps not in requirements.txt
echo "  Installing extra runtime deps (scipy, matplotlib, safetensors, huggingface_hub)..."
$PIP_INSTALL \
  "scipy" \
  "matplotlib" \
  "safetensors" \
  "huggingface_hub>=0.22.0"

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
  echo "  Building SAM2 C extension (pip install -e ./sam2)..."
  $PIP_INSTALL -e "./sam2"
  # Verify
  if python -c "import sam2._C" 2>/dev/null; then
    echo "  sam2._C built successfully."
  else
    echo "  WARNING: sam2._C did not compile (nvcc error?). SAM2 will still run without it."
  fi
fi

# -----------------------------------------------------------------------------
# 6. Pre-download HuggingFace models
# -----------------------------------------------------------------------------
# Uses `huggingface-cli download` (ships with huggingface_hub, installed above).
# Each call is idempotent — already-cached files are skipped automatically.
# Models land in ~/.cache/huggingface/hub/ (the default HF cache).
echo ""
echo "--- 6. Pre-downloading HuggingFace models ---"
echo "  This fetches ~6 GB total; already-cached files are skipped."
echo ""

hf_download() {
  local repo_id="$1"
  local friendly="$2"
  echo "  Downloading $friendly ($repo_id)..."
  huggingface-cli download "$repo_id" --quiet && \
    echo "    ✓ $friendly" || \
    echo "    ✗ $friendly — failed (check network / HF_TOKEN if private)"
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

# RAM++ checkpoint is not downloaded automatically here.
# Set rampp_checkpoint_path in config.py after installing/cloning Recognize Anything.

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "=== CITV setup complete!              ==="
echo "=========================================="
echo ""
echo "All models are pre-downloaded. Run the pipeline:"
echo "  python scene_understanding.py --input_dir images --output_dir output_scene"
echo ""
echo "Optional — camera calibration for accurate intrinsics:"
echo "  python tools/calibrate_camera.py --images path/to/checkerboard/ --out calibration.json"
echo "  Then set camera_calibration_file = 'calibration.json' in config.py"
