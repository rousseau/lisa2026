#!/bin/bash
# =============================================================================
# RUN_0004 — Shared Encoder Multi-task (Tasks 1a, 1b, 2)
# Jean Zay — partition gpu_p6 (H100 SXM5 80 GB)
#
# Durée estimée : ~10-15h (90 epochs warmup+joint, 436 steps/epoch, 3 tâches)
#
# Lancement depuis la racine du projet ($WORK/LISA/lisa2026) :
#   sbatch src/slurm/run_0004_multitask.slurm
#
# Smoke test (2 steps/epoch) :
#   sbatch src/slurm/run_0004_multitask.slurm --smoke-test
#
# Reprise automatique en cas de timeout (signal USR1 → requeue).
# =============================================================================

#SBATCH --job-name=lisa2026_run0004
#SBATCH -A crp@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --output=outputs/logs/RUN_0004/%x_%j.out
#SBATCH --error=outputs/logs/RUN_0004/%x_%j.err
#SBATCH --signal=USR1@120

set -euo pipefail

# ── Répertoire projet ─────────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

# ── Arguments optionnels (ex: --smoke-test) ───────────────────────────────────
SMOKE_FLAG=""
for arg in "$@"; do
    [[ "$arg" == "--smoke-test" ]] && SMOKE_FLAG="--smoke-test"
done

# ── Chemins données ───────────────────────────────────────────────────────────
DATA_ROOT="${LISA_DATA_ROOT:-${WORK}/LISA/data}"
CSV_PATH="${LISA_CSV_PATH:-${DATA_ROOT}/LISA_Task1a_2026.csv}"
export LISA_DATA_ROOT="$DATA_ROOT"
export LISA_CSV_PATH="$CSV_PATH"
export LISA_SKIP_CONDA=1

# ── Bannière ──────────────────────────────────────────────────────────────────
echo "======================================================================="
echo "  LISA 2026 — RUN_0004 Multi-task"
echo "  Job         : ${SLURM_JOB_ID:-local}"
echo "  Nœud        : ${SLURM_JOB_NODELIST:-local}"
echo "  Projet      : $PROJECT_ROOT"
echo "  Données     : $DATA_ROOT"
echo "  Smoke test  : ${SMOKE_FLAG:-non}"
echo "  Début       : $(date)"
echo "======================================================================="

# ── Modules Jean Zay ──────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

# ── Dépendances Python (installées dans $WORK/.local, persistantes) ───────────
export PYTHONUSERBASE="${WORK}/.local"
export PATH="${WORK}/.local/bin:${PATH}"
export TORCH_HOME="${WORK}/.cache/torch"

_pip_if_missing() {
    python3 -c "import $1" 2>/dev/null \
        || { echo "  [dep] installation de $2 ..."; pip install --user --quiet "$2"; }
}
_pip_if_missing nibabel    nibabel
_pip_if_missing scipy      scipy
_pip_if_missing yaml       pyyaml
_pip_if_missing pandas     pandas
_pip_if_missing sklearn    scikit-learn
_pip_if_missing monai      "monai[nibabel]"

echo "  Python  : $(python3 --version)"
echo "  PyTorch : $(python3 -c 'import torch; print(torch.__version__)')"
echo "  MONAI   : $(python3 -c 'import monai; print(monai.__version__)')"
echo "  GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ── Vérifications préalables ──────────────────────────────────────────────────
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "[ERREUR] Données introuvables : $DATA_ROOT"
    echo "         Lancer depuis la machine locale :"
    echo "           rsync -avP ~/Data/LISA2026/ ${WORK}/LISA/data/"
    exit 1
fi
N=$(find "$DATA_ROOT" -name "*_ciso.nii.gz" | wc -l)
echo "  Volumes _ciso.nii.gz : $N"
[[ $N -lt 10 ]] && { echo "[ERREUR] Trop peu de volumes. Vérifier le transfert."; exit 1; }

if [[ ! -f "results/splits/task1a_fixed.pkl" || ! -f "results/splits/task2_fixed.pkl" ]]; then
    echo "[INFO] Splits manquants — génération..."
    python3 -m src.prepare_split \
        --csv "$CSV_PATH" --seed 42 \
        --output results/splits/task1a_fixed.pkl
    python3 -m src.prepare_split_task2 \
        --data-root "$DATA_ROOT" --seed 42 --val-fraction 0.2 \
        --output results/splits/task2_fixed.pkl \
        --manifest-output results/splits/task2_manifest.csv
fi

# ── Création des répertoires de sortie ────────────────────────────────────────
mkdir -p outputs/checkpoints/RUN_0004 \
         outputs/logs/RUN_0004 \
         results/runs/RUN_0004/plots \
         results/splits

# ── Auto-requeue en cas de timeout ────────────────────────────────────────────
_requeue() {
    echo "[$(date)] Timeout imminent (USR1) — requeue de ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap '_requeue' USR1

# ── Entraînement ──────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " Phase 1/2 — Entraînement"
echo "════════════════════════════════════════"
T_TRAIN_START=$(date +%s)

python3 train.py --run 0004 $SMOKE_FLAG

T_TRAIN_END=$(date +%s)
echo "  Durée entraînement : $(( (T_TRAIN_END - T_TRAIN_START) / 60 )) min"

# ── Évaluation ────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " Phase 2/2 — Évaluation (3 tâches)"
echo "════════════════════════════════════════"
T_EVAL_START=$(date +%s)

python3 evaluate.py --run 0004 $SMOKE_FLAG

T_EVAL_END=$(date +%s)
echo "  Durée évaluation : $(( (T_EVAL_END - T_EVAL_START) / 60 )) min"

# ── Résumé final ─────────────────────────────────────────────────────────────
echo ""
echo "======================================================================="
echo "  RUN_0004 TERMINÉ"
echo "  Fin        : $(date)"
echo "  Durée tot. : $(( (T_EVAL_END - T_TRAIN_START) / 60 )) min"
echo "  Métriques  : results/runs/RUN_0004/metrics.json"
echo "  Checkpoint : outputs/checkpoints/RUN_0004/multitask_best.pt"
echo "======================================================================="

# Afficher le résumé des métriques si disponible
if [[ -f "results/runs/RUN_0004/metrics.json" ]]; then
    echo ""
    echo "── Résumé métriques ──────────────────────────────────────────────────"
    python3 -c "
import json, sys
m = json.load(open('results/runs/RUN_0004/metrics.json'))
t1a = m.get('task1a', {}).get('global', {})
t1b = m.get('task1b', {})
t2  = m.get('task2', {}).get('global', {})
print(f'  Task 1a — aggregate : {t1a.get(\"aggregate\", \"n/a\"):.4f}')
print(f'  Task 1b — PSNR      : {t1b.get(\"psnr\", \"n/a\"):.2f} dB')
print(f'  Task 2  — mean DSC  : {t2.get(\"mean_dsc\", \"n/a\"):.4f}')
"
fi
