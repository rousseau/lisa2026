#!/bin/bash

# RUN_0003 - Task2 DynUNet Segmentation Launcher
# Usage: bash scripts/run_0003.sh [--smoke-test]

set -e

SMOKE_TEST=0
for arg in "$@"; do
  [[ "$arg" == "--smoke-test" ]] && SMOKE_TEST=1
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "RUN_0003 - Task2 DynUNet Launcher"
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
echo "  PyTorch: $pytorch_version"
echo "  MONAI: $monai_version"
echo "  Python: $PYTHON_BIN"

DATA_ROOT="${LISA_DATA_ROOT:-/home/rousseau/Data/LISA2026}"
echo "  Data root: $DATA_ROOT"

echo ""
echo "[2/5] Creating output directories..."
mkdir -p outputs/checkpoints/RUN_0003
mkdir -p outputs/logs/RUN_0003
mkdir -p results/runs/RUN_0003/plots
mkdir -p results/splits
echo "✓ Directories created"

echo ""
echo "[3/5] Ensuring Task2 patient-level split exists..."
if [[ -f results/splits/task2_fixed.pkl ]]; then
  echo "  ✓ Split already exists"
else
  "$PYTHON_BIN" prepare_split_task2.py \
    --data-root "$DATA_ROOT" \
    --seed 42 \
    --val-fraction 0.2 \
    --output results/splits/task2_fixed.pkl \
    --manifest-output results/splits/task2_manifest.csv
fi

echo ""
echo "[4/5] Starting Task2 DynUNet training..."
START_TIME=$(date +%s)

SMOKE_FLAG=""
[[ $SMOKE_TEST -eq 1 ]] && SMOKE_FLAG="--smoke_test"

"$PYTHON_BIN" train_task2_dynunet.py \
  --config configs/run_0003_task2_dynunet.yaml \
  $SMOKE_FLAG || { echo "Error during training"; exit 1; }

END_TIME=$(date +%s)
echo "✓ Training complete ($(( END_TIME - START_TIME ))s)"

echo ""
echo "[5/5] Running Task2 evaluation..."

"$PYTHON_BIN" evaluate_task2_dynunet.py \
  --config configs/run_0003_task2_dynunet.yaml \
  $SMOKE_FLAG || { echo "Error during evaluation"; exit 1; }

echo ""
echo "=========================================="
echo "RUN_0003 COMPLETE"
echo "Results: results/runs/RUN_0003/metrics.json"
echo "=========================================="
