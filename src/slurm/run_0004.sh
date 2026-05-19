#!/bin/bash
# RUN_0004 – Task 1b BasicUNet Denoising Launcher
# Usage: bash src/slurm/run_0004.sh [--smoke-test]
# Must be run from the project root: cd /path/to/lisa2026 && bash src/slurm/run_0004.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

SMOKE_TEST=0
for arg in "$@"; do [[ "$arg" == "--smoke-test" ]] && SMOKE_TEST=1; done

echo "=========================================="
echo "RUN_0004 – Task 1b BasicUNet Launcher"
echo "Date: $(date)"
[[ $SMOKE_TEST -eq 1 ]] && echo "MODE: SMOKE TEST"
echo "=========================================="

# ── Environment ────────────────────────────────────────────────────────────
if [[ -z "${SLURM_JOB_ID:-}" && -z "${LISA_SKIP_CONDA:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate lisa2026 || echo "  ! conda env lisa2026 unavailable"
  fi
fi
PYTHON_BIN="$(command -v python3 || command -v python)"
echo "  Python : $PYTHON_BIN  ($("$PYTHON_BIN" --version 2>&1))"
DATA_ROOT="${LISA_DATA_ROOT:-/home/rousseau/Data/LISA2026}"

# ── Directories ────────────────────────────────────────────────────────────
mkdir -p outputs/checkpoints/RUN_0004 outputs/logs/RUN_0004 results/runs/RUN_0004/plots results/splits

# ── Split (reuses task2 subject-level split if available) ──────────────────
if [[ ! -f results/splits/task1a_fixed.pkl && ! -f results/splits/task2_fixed.pkl ]]; then
  echo "  ! No split file found. Task1bDataset will create a deterministic 80/20 split."
fi

# ── Training ───────────────────────────────────────────────────────────────
SMOKE_FLAG=""; [[ $SMOKE_TEST -eq 1 ]] && SMOKE_FLAG="--smoke_test"
echo "[train] BasicUNet self-supervised denoising..."
"$PYTHON_BIN" -m src.train_task1b \
  --config configs/run_0004_task1b_unet.yaml $SMOKE_FLAG

# ── Evaluation ─────────────────────────────────────────────────────────────
echo "[eval] Running Task 1b evaluation (FID/PSNR/LPIPS)..."
"$PYTHON_BIN" -m src.evaluate_task1b \
  --config configs/run_0004_task1b_unet.yaml $SMOKE_FLAG

echo "=========================================="
echo "RUN_0004 COMPLETE → results/runs/RUN_0004/metrics.json"
echo "=========================================="
