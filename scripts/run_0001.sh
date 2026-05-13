#!/bin/bash

# RUN_0001 – Baseline Task 1a Launcher
# Supervised Contrastive Learning + Ordinal Classification Baseline
# Usage: bash scripts/run_0001.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "RUN_0001 – Baseline Task 1a Launcher"
echo "Date: $(date)"
echo "=========================================="

# Load environment
echo "[1/6] Loading conda environment..."
source ~/.bashrc
conda activate lisa2026 || { echo "Error: conda env lisa2026 not found"; exit 1; }

echo "✓ Environment loaded"
python --version
pytorch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "  PyTorch: $pytorch_version"
echo "  CUDA: $cuda_version"

# Create directories
echo ""
echo "[2/6] Creating output directories..."
mkdir -p outputs/checkpoints/RUN_0001
mkdir -p outputs/logs/RUN_0001
mkdir -p results/runs/RUN_0001/plots
mkdir -p results/splits
echo "✓ Directories created"

# Generate fixed split
echo ""
echo "[3/6] Generating fixed patient-level split..."
python prepare_split.py \
  --csv /home/rousseau/Data/LISA2026/LISA_Task1a_2026.csv \
  --seed 42 \
  --output results/splits/task1a_fixed.pkl

# Training phase
echo ""
echo "[4/6] Starting training for all 7 tasks..."
echo "Tasks: Noise, Zipper, Positioning, Banding, Motion, Contrast, Distortion"
echo ""

TASKS=("Noise" "Zipper" "Positioning" "Banding" "Motion" "Contrast" "Distortion")
START_TIME=$(date +%s)

for TASK in "${TASKS[@]}"; do
  echo "---------- Training $TASK ----------"
  python train_task1a.py \
    --config configs/run_0001_baseline.yaml \
    --task $TASK || { echo "Error training $TASK"; exit 1; }
  echo "✓ $TASK training complete"
  echo ""
done

END_TIME=$(date +%s)
TRAINING_DURATION=$((END_TIME - START_TIME))

# Evaluation phase
echo ""
echo "[5/6] Running evaluation..."
python evaluate_task1a.py --config configs/run_0001_baseline.yaml || { echo "Error during evaluation"; exit 1; }

echo ""
echo "[6/6] Computing metrics..."
python compute_metrics.py \
  --predictions results/runs/RUN_0001/predictions_val.csv \
  --ground-truth /home/rousseau/Data/LISA2026/LISA_Task1a_2026.csv \
  --output results/runs/RUN_0001/metrics.json || { echo "Error computing metrics"; exit 1; }

# Consolidation
echo ""
echo "=========================================="
echo "RUN_0001 COMPLETED"
echo "=========================================="
echo ""
echo "Artifacts:"
echo "  ✓ Checkpoints: outputs/checkpoints/RUN_0001/"
echo "  ✓ Logs: outputs/logs/RUN_0001/"
echo "  ✓ Predictions: results/runs/RUN_0001/predictions_val.csv"
echo "  ✓ Metrics: results/runs/RUN_0001/metrics.json"
echo "  ✓ Documentation: results/runs/RUN_0001/AGENTS.md"
echo ""
echo "Training duration: $TRAINING_DURATION seconds ($(($TRAINING_DURATION / 60)) min)"
echo ""
echo "Next: Review results/runs/RUN_0001/metrics.json"
echo ""
echo "$(date)"
