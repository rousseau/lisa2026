#!/bin/bash

# RUN_0002 – UPF-inspired Multi-label Task 1a Launcher
# Single DenseNet264 + 7 heads, EMD + Focal loss
# Usage: bash scripts/run_0002.sh [--smoke-test]

set -e

SMOKE_TEST=0
for arg in "$@"; do
  [[ "$arg" == "--smoke-test" ]] && SMOKE_TEST=1
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "RUN_0002 – Multi-label EMD+Focal Launcher"
echo "Date: $(date)"
[[ $SMOKE_TEST -eq 1 ]] && echo "MODE: SMOKE TEST"
echo "=========================================="

# Load environment
echo ""
echo "[1/5] Loading Python environment..."

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]]; then
  source "$HOME/mambaforge/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate lisa2026 || echo "  ! conda env lisa2026 unavailable, using current environment"
else
  echo "  ! conda not available, using current environment"
fi

echo "✓ Environment loaded"
python --version
pytorch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "  PyTorch: $pytorch_version"
echo "  CUDA: $cuda_version"

DATA_ROOT="${LISA_DATA_ROOT:-/home/rousseau/Data/LISA2026}"
CSV_PATH="${LISA_CSV_PATH:-$DATA_ROOT/LISA_Task1a_2026.csv}"
echo "  Data root: $DATA_ROOT"
echo "  CSV path:  $CSV_PATH"

# Create directories
echo ""
echo "[2/5] Creating output directories..."
mkdir -p outputs/checkpoints/RUN_0002
mkdir -p outputs/logs/RUN_0002/tensorboard
mkdir -p results/runs/RUN_0002/plots
mkdir -p results/splits
echo "✓ Directories created"

# Generate fixed split (reuse from RUN_0001 if already present)
echo ""
echo "[3/5] Ensuring patient-level split exists..."
if [[ -f results/splits/task1a_fixed.pkl ]]; then
  echo "  ✓ Split already exists – reusing for fair comparison with RUN_0001"
else
  python prepare_split.py \
    --csv "$CSV_PATH" \
    --seed 42 \
    --output results/splits/task1a_fixed.pkl
fi

# Training
echo ""
echo "[4/5] Starting multi-label training (7 tasks simultaneously)..."
START_TIME=$(date +%s)

SMOKE_FLAG=""
[[ $SMOKE_TEST -eq 1 ]] && SMOKE_FLAG="--smoke_test"

python train_task1a_multilabel.py \
  --config configs/run_0002_upf.yaml \
  $SMOKE_FLAG || { echo "Error during training"; exit 1; }

END_TIME=$(date +%s)
echo "✓ Training complete ($(( END_TIME - START_TIME ))s)"

# Evaluation
echo ""
echo "[5/5] Running evaluation and computing metrics..."

python evaluate_task1a_multilabel.py \
  --config configs/run_0002_upf.yaml \
  $SMOKE_FLAG || { echo "Error during evaluation"; exit 1; }

python compute_metrics.py \
  --predictions results/runs/RUN_0002/predictions_val.csv \
  --ground-truth "$CSV_PATH" \
  --output results/runs/RUN_0002/metrics.json \
  --run-id 0002 \
  --run-date "$(date +%Y-%m-%d)" || { echo "Error computing metrics"; exit 1; }

echo ""
echo "=========================================="
echo "RUN_0002 COMPLETE"
echo "Results: results/runs/RUN_0002/metrics.json"
echo "=========================================="
