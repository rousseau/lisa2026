#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script SLURM — Entraînement joint LISA 2026 sur Jean Zay (H100 SXM5 80GB)
#
# Partition gpu_p6 : H100 SXM5 80GB, 4 GPUs/nœud
# QOS : qos_gpu_h100-t4 (100h max)
#
# Durée estimée sur H100 :
#   100 epochs, 622 items, batch=2, target_size=96 : ~4-8h
#
# Lancement :
#   sbatch src/slurm/train_jeanzay.slurm
#   sbatch src/slurm/train_jeanzay.slurm --config configs/train_large.yaml
#
# Prérequis (une seule fois via setup_jeanzay.sh) :
#   $WORK/LISA/lisa2026/     ← ce repo (git clone)
#   $WORK/LISA/data/         ← données NIfTI (rsync depuis machine locale)
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
#SBATCH --time=20:00:00

#SBATCH --output=logs/lisa2026_%x_%j.out
#SBATCH --error=logs/lisa2026_%x_%j.err

# Signal 120s avant la fin pour auto-requeue
#SBATCH --signal=USR1@120

set -e

# ─── Arguments optionnels passés à sbatch (ex: --config configs/train_large.yaml)
EXTRA_ARGS="${@}"

CODE_DIR="$WORK/LISA/lisa2026"
DATA_DIR="$WORK/LISA/data"
mkdir -p "$CODE_DIR/logs"

echo "======================================================================="
echo " Job ID    : $SLURM_JOB_ID"
echo " Nœud      : $SLURM_JOB_NODELIST"
echo " Code      : $CODE_DIR"
echo " Données   : $DATA_DIR"
echo " Args      : $EXTRA_ARGS"
echo "======================================================================="

# ─── Modules Jean Zay ────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

# ─── Dépendances utilisateur ($WORK/.local, installées une seule fois) ───────
export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"
export TORCH_HOME="$WORK/.cache/torch"

_install_if_missing() {
    python3 -c "import $1" 2>/dev/null || pip install --user --quiet "$2"
}
_install_if_missing nibabel   nibabel
_install_if_missing scipy     scipy
_install_if_missing yaml      pyyaml
_install_if_missing pandas    pandas

# ─── Gestion auto-requeue ─────────────────────────────────────────────────────
_requeue_handler() {
    echo "[$(date)] Timeout imminent — requeue du job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap '_requeue_handler' USR1

# ─── Vérification des données ────────────────────────────────────────────────
if [ ! -d "$DATA_DIR" ]; then
    echo "[ERREUR] Données introuvables : $DATA_DIR"
    echo "         Transférez les données depuis la machine locale (voir setup_jeanzay.sh)"
    exit 1
fi
N_NIFTI=$(find "$DATA_DIR" -name "*.nii.gz" | wc -l)
echo "Données NIfTI  : OK ($N_NIFTI volumes dans $DATA_DIR)"
echo "GPU            : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ─── Reprise depuis le dernier checkpoint ────────────────────────────────────
CKPT_DIR="$CODE_DIR/outputs/checkpoints"
RESUME_ARG=""
if [ -d "$CKPT_DIR" ]; then
    LAST_CKPT=$(ls "$CKPT_DIR"/epoch_*.pt 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_CKPT" ]; then
        echo "Reprise depuis  : $LAST_CKPT"
        RESUME_ARG="--resume $LAST_CKPT"
    fi
fi

# ─── Lancement ───────────────────────────────────────────────────────────────
cd "$CODE_DIR"

# Config par défaut ; peut être surchargée via EXTRA_ARGS
CONFIG="configs/train_default.yaml"

echo ""
echo "Lancement : python src/train.py --config $CONFIG --data-root $DATA_DIR $RESUME_ARG $EXTRA_ARGS"
echo ""

python3 src/train.py \
    --config "$CONFIG" \
    --data-root "$DATA_DIR" \
    --num-workers 8 \
    $RESUME_ARG \
    $EXTRA_ARGS

echo "Entraînement terminé (job $SLURM_JOB_ID)."
