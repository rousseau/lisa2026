# RUN_0001 - Execution Notes

## Overview
Execution log for RUN_0001 baseline (Task 1a only).

Run status: completed

## Phase 1: Data Preparation

- Split generation: completed
- Command: python prepare_split.py --csv /home/rousseau/Data/LISA2026/LISA_Task1a_2026.csv --seed 42 --output results/splits/task1a_fixed.pkl
- Output:
  - Train samples: 436
  - Val samples: 96
  - Subjects: 244
- Split file: results/splits/task1a_fixed.pkl

## Phase 2: Training

Training executed sequentially for 7 independent tasks with early stopping.

- Noise:
  - Best validation F1: 0.6867
  - Early stopping epoch: 15
- Zipper:
  - Best validation F1: 0.5879
  - Early stopping epoch: 22
- Positioning:
  - Best validation F1: 0.7787
  - Early stopping epoch: 23
- Banding:
  - Best validation F1: 0.5520
  - Early stopping epoch: 19
- Motion:
  - Best validation F1: 0.6025
  - Early stopping epoch: 19
- Contrast:
  - Best validation F1: 0.6640
  - Early stopping epoch: 30
- Distortion:
  - Best validation F1: 0.6015
  - Early stopping epoch: 28

Training summary:
- All tasks completed without runtime failures
- Total training duration: 12,398 seconds (206 minutes)

## Phase 3: Evaluation

- Inference completed for all 7 tasks on validation split
- Predictions file: results/runs/RUN_0001/predictions_val.csv
- Metrics file: results/runs/RUN_0001/metrics.json

Global metrics:
- Aggregate: 0.6887
- Accuracy: 0.8437
- F1_macro: 0.6390
- F2_macro: 0.6188
- Precision_macro: 0.7317
- Recall_macro: 0.6103

## Phase 4: Consolidation

- AGENTS file updated with final metrics and decision
- RUNS_INDEX updated (status=completed)
- Baseline decision: accepted

## Issues and Resolutions

- Issue: conda activate failed in non-initialized shell
- Resolution: switched to direct activation command source /home/rousseau/miniforge3/bin/activate lisa2026 for execution context
- Impact: no impact on model outputs

## Decision Record

Baseline acceptance criteria:
- Global aggregate score >= 0.60: met (0.6887)
- All 7 tasks completed: met
- Early stopping behavior normal: met
- Reproducibility setup captured: met

Decision: baseline accepted

## Last Updated

- Date: 2026-05-12
- Final status: completed
