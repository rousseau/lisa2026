#!/bin/bash
# RUN_0002 – Task 1a Multi-label EMD+Focal Launcher
# Usage: bash src/slurm/run_0002.sh [--smoke-test]
# Must be run from the project root: cd /path/to/lisa2026 && bash src/slurm/run_0002.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

SMOKE_TEST=0
for arg in "$@"; do [[ "$arg" == "--smoke-test" ]] && SMOKE_TEST=1; done

echo "=========================================="
echo "RUN_0002 – Task 1a Multi-label Launcher"
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
mkdir -p outputs/checkpoints/RUN_0002 outputs/logs/RUN_0002 results/runs/RUN_0002/plots results/splits

# ── Split (reuse from RUN_0001 for fair comparison) ────────────────────────
if [[ ! -f results/splits/task1a_fixed.pkl ]]; then
  echo "[split] Generating fixed patient-level split..."
  "$PYTHON_BIN" -m src.prepare_split \
    --csv "$DATA_ROOT/LISA_Task1a_2026.csv" \
    --seed 42 \
    --output results/splits/task1a_fixed.pkl
fi

# ── Training ───────────────────────────────────────────────────────────────
SMOKE_FLAG=""; [[ $SMOKE_TEST -eq 1 ]] && SMOKE_FLAG="--smoke_test"
echo "[train] Multi-label model (7 heads)..."
"$PYTHON_BIN" -m src.train_task1a_multilabel \
  --config configs/run_0002_upf.yaml $SMOKE_FLAG

# ── Evaluation ─────────────────────────────────────────────────────────────
echo "[eval] Generating predictions..."
"$PYTHON_BIN" -m src.evaluate_task1a_multilabel \
  --config configs/run_0002_upf.yaml $SMOKE_FLAG
echo "[metrics] Computing metrics..."
"$PYTHON_BIN" -m src.compute_metrics \
  --predictions results/runs/RUN_0002/predictions_val.csv \
  --ground-truth "$DATA_ROOT/LISA_Task1a_2026.csv" \
  --output results/runs/RUN_0002/metrics.json \
  --run-id 0002

echo "=========================================="
echo "RUN_0002 COMPLETE → results/runs/RUN_0002/metrics.json"
echo "=========================================="
