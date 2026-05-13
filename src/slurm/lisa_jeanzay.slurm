#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script SLURM générique — LISA 2026 sur Jean Zay
#
# Usage:
#   sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0001
#   sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0002 --smoke-test
#
# Le script lance le wrapper correspondant dans scripts/run_<RUN>.sh.
# Si --run n'est pas fourni, RUN_0002 est utilisé par défaut.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=lisa2026
#SBATCH -A crp@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1

#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=100:00:00

#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err

#SBATCH --signal=USR1@120

set -euo pipefail

# En contexte SLURM, le script est exécuté depuis une copie temporaire
# (souvent /var/spool). Le vrai dossier de soumission est SLURM_SUBMIT_DIR.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$PROJECT_ROOT"

RUN_NAME="${LISA_RUN:-RUN_0002}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            RUN_NAME="$2"
            shift 2
            ;;
        --run=*)
            RUN_NAME="${1#*=}"
            shift
            ;;
        --help|-h)
            cat <<'EOF'
Usage: sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0002 [--smoke-test]

Options:
  --run RUN_NAME   Run launcher to execute (default: RUN_0002 or $LISA_RUN)
  --smoke-test     Forwarded to the selected launcher
EOF
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$RUN_NAME" =~ ^[0-9]+$ ]]; then
    RUN_NAME="RUN_${RUN_NAME}"
fi

case "$RUN_NAME" in
    RUN_0001|RUN_0002)
        RUN_SCRIPT="scripts/run_${RUN_NAME#RUN_}.sh"
        ;;
    *)
        echo "[ERREUR] Run inconnu: $RUN_NAME"
        echo "         Runs disponibles: RUN_0001, RUN_0002"
        exit 1
        ;;
esac

DATA_DIR="${WORK:-$HOME}/LISA/data"
export LISA_DATA_ROOT="$DATA_DIR"
export LISA_CSV_PATH="$DATA_DIR/LISA_Task1a_2026.csv"

echo "======================================================================="
echo " Job ID    : ${SLURM_JOB_ID:-unknown}"
echo " Nœud      : ${SLURM_JOB_NODELIST:-unknown}"
echo " Code      : $PROJECT_ROOT"
echo " Run       : $RUN_NAME"
echo " Launcher  : $RUN_SCRIPT"
echo " Args      : ${EXTRA_ARGS[*]:-<none>}"
echo "======================================================================="

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

export PYTHONUSERBASE="${WORK:-$HOME}/.local"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export TORCH_HOME="${WORK:-$HOME}/.cache/torch"

_install_if_missing() {
    python3 -c "import $1" 2>/dev/null || pip install --user --quiet "$2"
}

_install_if_missing nibabel nibabel
_install_if_missing scipy scipy
_install_if_missing yaml pyyaml
_install_if_missing pandas pandas

_requeue_handler() {
    echo "[$(date)] Timeout imminent — requeue du job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap '_requeue_handler' USR1

if [[ ! -d "$DATA_DIR" ]]; then
    echo "[ERREUR] Données introuvables : $DATA_DIR"
    echo "         Lancez d'abord le transfert rsync vers Jean Zay"
    exit 1
fi

N_NIFTI=$(find "$DATA_DIR" -name "*.nii.gz" | wc -l)
echo "Données NIfTI  : OK ($N_NIFTI volumes dans $DATA_DIR)"
echo "GPU            : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/outputs/checkpoints/$RUN_NAME"
mkdir -p "$PROJECT_ROOT/outputs/logs/$RUN_NAME"
mkdir -p "$PROJECT_ROOT/results/runs/$RUN_NAME/plots"
mkdir -p "$PROJECT_ROOT/results/splits"

cd "$PROJECT_ROOT"

echo ""
echo "Lancement : bash $RUN_SCRIPT ${EXTRA_ARGS[*]:-}"
echo ""

bash "$RUN_SCRIPT" "${EXTRA_ARGS[@]}"

echo "Entraînement terminé (job ${SLURM_JOB_ID:-unknown})."