#!/bin/bash
# RUN_0003 – Task 2 DynUNet Segmentation Launcher
# Usage: bash src/slurm/run_0003.sh [--smoke-test]
# Must be run from the project root: cd /path/to/lisa2026 && bash src/slurm/run_0003.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

SMOKE_TEST=0
for arg in "$@"; do [[ "$arg" == "--smoke-test" ]] && SMOKE_TEST=1; done

echo "=========================================="
echo "RUN_0003 – Task 2 DynUNet Launcher"
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
mkdir -p outputs/checkpoints/RUN_0003 outputs/logs/RUN_0003 results/runs/RUN_0003/plots results/splits

# ── Split ──────────────────────────────────────────────────────────────────
if [[ ! -f results/splits/task2_fixed.pkl ]]; then
  echo "[split] Generating fixed subject-level split..."
  "$PYTHON_BIN" -m src.prepare_split_task2 \
    --data-root "$DATA_ROOT" \
    --seed 42 \
    --val-fraction 0.2 \
    --output results/splits/task2_fixed.pkl \
    --manifest-output results/splits/task2_manifest.csv
fi

# ── Training ───────────────────────────────────────────────────────────────
SMOKE_FLAG=""; [[ $SMOKE_TEST -eq 1 ]] && SMOKE_FLAG="--smoke_test"
echo "[train] DynUNet 12-class segmentation..."
"$PYTHON_BIN" -m src.train_task2_dynunet \
  --config configs/run_0003_task2_dynunet.yaml $SMOKE_FLAG

# ── Evaluation ─────────────────────────────────────────────────────────────
echo "[eval] Running segmentation evaluation..."
"$PYTHON_BIN" -m src.evaluate_task2_dynunet \
  --config configs/run_0003_task2_dynunet.yaml $SMOKE_FLAG

echo "=========================================="
echo "RUN_0003 COMPLETE → results/runs/RUN_0003/metrics.json"
echo "=========================================="
