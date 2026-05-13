#!/bin/bash

# Full training pipeline for RUN_0001
# Execute all 7 tasks sequentially with logging

set -e

cd /home/rousseau/Exp/lisa2026
source /home/rousseau/miniforge3/bin/activate lisa2026

echo "=========================================="
echo "RUN_0001 – Full Training Pipeline"
echo "Date: $(date)"
echo "=========================================="

TASKS=("Noise" "Zipper" "Positioning" "Banding" "Motion" "Contrast" "Distortion")
START_TIME=$(date +%s)

for i in "${!TASKS[@]}"; do
  TASK="${TASKS[$i]}"
  TASK_NUM=$((i + 1))
  
  echo ""
  echo "========== Task $TASK_NUM/7: $TASK =========="
  echo "Start time: $(date)"
  
  python train_task1a.py \
    --config configs/run_0001_baseline.yaml \
    --task "$TASK" \
    > "outputs/logs/RUN_0001/task1a_${TASK}.log" 2>&1
  
  if [ $? -eq 0 ]; then
    echo "✓ $TASK completed successfully"
    tail -5 "outputs/logs/RUN_0001/task1a_${TASK}.log"
  else
    echo "✗ Error training $TASK - check logs"
    tail -20 "outputs/logs/RUN_0001/task1a_${TASK}.log"
    exit 1
  fi
done

END_TIME=$(date +%s)
TRAINING_DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "✓ All training tasks completed!"
echo "=========================================="
echo "Total duration: $TRAINING_DURATION seconds ($(($TRAINING_DURATION / 60)) min)"
echo ""

# Evaluation
echo "Starting evaluation..."
python evaluate_task1a.py --config configs/run_0001_baseline.yaml

echo ""
echo "Computing metrics..."
python compute_metrics.py \
  --predictions results/runs/RUN_0001/predictions_val.csv \
  --ground-truth /home/rousseau/Data/LISA2026/LISA_Task1a_2026.csv \
  --output results/runs/RUN_0001/metrics.json

echo ""
echo "=========================================="
echo "✓ FULL PIPELINE COMPLETED"
echo "=========================================="
echo "Date: $(date)"
