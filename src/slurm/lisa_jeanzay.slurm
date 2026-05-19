#!/bin/bash
# =============================================================================
# Script SLURM générique — LISA 2026 sur Jean Zay (H100 SXM5 80 GB)
#
# Usage (depuis la racine du projet) :
#   sbatch src/slurm/lisa_jeanzay.slurm --run 0001
#   sbatch src/slurm/lisa_jeanzay.slurm --run 0003
#   sbatch src/slurm/lisa_jeanzay.slurm --run 0004          ← multi-task
#   sbatch src/slurm/lisa_jeanzay.slurm --run 0004 --smoke-test
#
# Pour RUN_0004, préférer le script dédié :
#   sbatch src/slurm/run_0004_multitask.slurm
# =============================================================================

#SBATCH --job-name=lisa2026
#SBATCH -A crp@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=100:00:00
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --signal=USR1@120

set -euo pipefail

# ── Répertoire projet ─────────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

# ── Parsing des arguments ─────────────────────────────────────────────────────
RUN_ID="${LISA_RUN:-0004}"
SMOKE_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)        RUN_ID="$2"; shift 2 ;;
        --run=*)      RUN_ID="${1#*=}"; shift ;;
        --smoke-test) SMOKE_FLAG="--smoke-test"; shift ;;
        --help|-h)
            echo "Usage: sbatch src/slurm/lisa_jeanzay.slurm --run <ID> [--smoke-test]"
            echo "  <ID> : 0001 | 0002 | 0003 | 0004"
            exit 0 ;;
        *) echo "[WARN] argument inconnu ignoré : $1"; shift ;;
    esac
done

# Normalise RUN_0003 → 0003
RUN_ID="${RUN_ID#RUN_}"

case "$RUN_ID" in
    0001|0002|0003|0004) ;;
    *) echo "[ERREUR] Run inconnu : $RUN_ID. Valides : 0001 0002 0003 0004"; exit 1 ;;
esac

# ── Chemins données ───────────────────────────────────────────────────────────
DATA_ROOT="${LISA_DATA_ROOT:-${WORK}/LISA/data}"
CSV_PATH="${LISA_CSV_PATH:-${DATA_ROOT}/LISA_Task1a_2026.csv}"
export LISA_DATA_ROOT="$DATA_ROOT"
export LISA_CSV_PATH="$CSV_PATH"
export LISA_SKIP_CONDA=1

# ── Bannière ──────────────────────────────────────────────────────────────────
echo "======================================================================="
echo "  LISA 2026 — RUN_${RUN_ID}"
echo "  Job     : ${SLURM_JOB_ID:-local}"
echo "  Nœud    : ${SLURM_JOB_NODELIST:-local}"
echo "  Projet  : $PROJECT_ROOT"
echo "  Données : $DATA_ROOT"
echo "  Smoke   : ${SMOKE_FLAG:-non}"
echo "  Début   : $(date)"
echo "======================================================================="

# ── Modules Jean Zay ──────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

# ── Dépendances Python ($WORK/.local, persistantes entre jobs) ────────────────
export PYTHONUSERBASE="${WORK}/.local"
export PATH="${WORK}/.local/bin:${PATH}"
export TORCH_HOME="${WORK}/.cache/torch"

_pip_if_missing() {
    python3 -c "import $1" 2>/dev/null \
        || { echo "  [dep] $2 ..."; pip install --user --quiet "$2"; }
}
_pip_if_missing nibabel   nibabel
_pip_if_missing scipy     scipy
_pip_if_missing yaml      pyyaml
_pip_if_missing pandas    pandas
_pip_if_missing sklearn   scikit-learn
_pip_if_missing monai     "monai[nibabel]"

echo "  Python  : $(python3 --version)"
echo "  PyTorch : $(python3 -c 'import torch; print(torch.__version__)')"
echo "  GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ── Vérification des données ──────────────────────────────────────────────────
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "[ERREUR] Données introuvables : $DATA_ROOT"
    echo "         rsync -avP ~/Data/LISA2026/ ${WORK}/LISA/data/"
    exit 1
fi

# ── Splits (versionnés dans git, régénérés si absents) ────────────────────────
if [[ ! -f "results/splits/task1a_fixed.pkl" ]]; then
    echo "[INFO] Génération task1a_fixed.pkl ..."
    python3 -m src.prepare_split \
        --csv "$CSV_PATH" --seed 42 \
        --output results/splits/task1a_fixed.pkl
fi
if [[ "$RUN_ID" =~ ^(0003|0004)$ ]] && [[ ! -f "results/splits/task2_fixed.pkl" ]]; then
    echo "[INFO] Génération task2_fixed.pkl ..."
    python3 -m src.prepare_split_task2 \
        --data-root "$DATA_ROOT" --seed 42 --val-fraction 0.2 \
        --output results/splits/task2_fixed.pkl \
        --manifest-output results/splits/task2_manifest.csv
fi

# ── Répertoires de sortie ─────────────────────────────────────────────────────
mkdir -p "outputs/checkpoints/RUN_${RUN_ID}" \
         "outputs/logs/RUN_${RUN_ID}" \
         "results/runs/RUN_${RUN_ID}/plots" \
         results/splits

# ── Auto-requeue ──────────────────────────────────────────────────────────────
_requeue() {
    echo "[$(date)] Timeout — requeue ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap '_requeue' USR1

# ── Entraînement ──────────────────────────────────────────────────────────────
echo ""
echo "── Entraînement RUN_${RUN_ID} ──────────────────────────────────────────"
T0=$(date +%s)
python3 train.py --run "$RUN_ID" $SMOKE_FLAG
echo "  Durée : $(( ($(date +%s) - T0) / 60 )) min"

# ── Évaluation ────────────────────────────────────────────────────────────────
echo ""
echo "── Évaluation RUN_${RUN_ID} ────────────────────────────────────────────"
T1=$(date +%s)
python3 evaluate.py --run "$RUN_ID" $SMOKE_FLAG
echo "  Durée : $(( ($(date +%s) - T1) / 60 )) min"

# ── Bilan ─────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================="
echo "  TERMINÉ — RUN_${RUN_ID}"
echo "  Durée totale : $(( ($(date +%s) - T0) / 60 )) min"
echo "  Métriques    : results/runs/RUN_${RUN_ID}/metrics.json"
echo "  Fin          : $(date)"
echo "======================================================================="
