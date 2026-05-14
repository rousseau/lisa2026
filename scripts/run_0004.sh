#!/bin/bash

# RUN_0004 – Task 1b BasicUNet Self-supervised Denoising Launcher
# Usage: bash scripts/run_0004.sh [--smoke-test]

set -e

SMOKE_TEST=0
for arg in "$@"; do
  [[ "$arg" == "--smoke-test" ]] && SMOKE_TEST=1
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "RUN_0004 – Task 1b BasicUNet Denoising"
echo "Date: $(date)"
[[ $SMOKE_TEST -eq 1 ]] && echo "MODE: SMOKE TEST"
echo "=========================================="

echo ""
echo "[1/5] Loading Python environment..."

USE_CURRENT_ENV=0
if [[ -n "${LISA_SKIP_CONDA:-}" || -n "${SLURM_JOB_ID:-}" ]]; then
  USE_CURRENT_ENV=1
fi

if [[ "$USE_CURRENT_ENV" -eq 0 ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
  elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]]; then
    source "$HOME/mambaforge/etc/profile.d/conda.sh"
  fi
fi

if [[ "$USE_CURRENT_ENV" -eq 1 ]]; then
  echo "  ! using current environment (SLURM/module or LISA_SKIP_CONDA set)"
elif command -v conda >/dev/null 2>&1; then
  conda activate lisa2026 || echo "  ! conda env lisa2026 unavailable, using current environment"
else
  echo "  ! conda not available, using current environment"
fi

PYTHON_BIN="$(command -v python3 || command -v python)"

echo "✓ Environment loaded"
"$PYTHON_BIN" --version
pytorch_version=$("$PYTHON_BIN" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
monai_version=$("$PYTHON_BIN" -c "import monai; print(monai.__version__)" 2>/dev/null || echo "unknown")
lpips_version=$("$PYTHON_BIN" -c "import lpips; print(lpips.__version__)" 2>/dev/null || echo "unknown")
echo "  PyTorch: $pytorch_version"
echo "  MONAI:   $monai_version"
echo "  LPIPS:   $lpips_version"
echo "  Python:  $PYTHON_BIN"

DATA_ROOT="${LISA_DATA_ROOT:-/home/rousseau/Data/LISA2026}"
echo "  Data root: $DATA_ROOT"

echo ""
echo "[2/5] Creating output directories..."
mkdir -p outputs/checkpoints/RUN_0004
mkdir -p outputs/logs/RUN_0004
mkdir -p results/runs/RUN_0004/plots
mkdir -p results/splits
echo "✓ Directories created"

echo ""
echo "[3/5] Ensuring patient-level split exists..."
# Task 1b reuses the task1a patient split (task1a_fixed.pkl) if available.
# The Task1bDataset handles the index-to-subject mapping automatically.
# Alternatively, task2_fixed.pkl (subject-level) can be pointed to in the config.
if [[ -f results/splits/task1a_fixed.pkl ]]; then
  echo "  ✓ task1a_fixed.pkl already exists – will be used for Task 1b split"
elif [[ -f results/splits/task2_fixed.pkl ]]; then
  echo "  ✓ task2_fixed.pkl found – update config split_pkl to use it for subject-level split"
else
  echo "  ! No split file found. Task1bDataset will create a deterministic 80/20 split on-the-fly."
  echo "    To create a proper split, run:"
  echo "      python prepare_split_task2.py --data-root $DATA_ROOT --output results/splits/task1b_fixed.pkl"
fi

echo ""
echo "[4/5] Starting Task 1b training..."
START_TIME=$(date +%s)

SMOKE_FLAG=""
[[ $SMOKE_TEST -eq 1 ]] && SMOKE_FLAG="--smoke_test"

"$PYTHON_BIN" train_task1b.py \
  --config configs/run_0004_task1b_unet.yaml \
  $SMOKE_FLAG || { echo "Error during training"; exit 1; }

END_TIME=$(date +%s)
echo "✓ Training complete ($(( END_TIME - START_TIME ))s)"

echo ""
echo "[5/5] Running Task 1b evaluation..."

"$PYTHON_BIN" evaluate_task1b.py \
  --config configs/run_0004_task1b_unet.yaml \
  $SMOKE_FLAG || { echo "Error during evaluation"; exit 1; }

echo ""
echo "=========================================="
echo "RUN_0004 COMPLETE"
echo "Results: results/runs/RUN_0004/metrics.json"
echo "=========================================="
