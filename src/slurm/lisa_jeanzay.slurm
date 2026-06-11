#!/bin/bash
# =============================================================================
# Script SLURM générique — LISA 2026 sur Jean Zay (H100 SXM5 80 GB)
#
# Usage recommandé (depuis la racine du projet) :
#   bash src/slurm/submit.sh --run <ID> [--smoke-test] [--time HH:MM:SS]
#
# Lancement direct (logs flat, pas de RUN_ dans le nom) :
#   sbatch src/slurm/lisa_jeanzay.slurm --run 0001
#   sbatch src/slurm/lisa_jeanzay.slurm --run 0004
#
# Le run ID est validé par train.py / evaluate.py ; aucune whitelist ici.
# =============================================================================

#SBATCH --job-name=lisa2026
#SBATCH -A crp@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00
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
            exit 0 ;;
        *) echo "[WARN] argument inconnu ignoré : $1"; shift ;;
    esac
done

# Normalise RUN_0003 → 0003
RUN_ID="${RUN_ID#RUN_}"

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
_pip_if_missing nibabel     nibabel
_pip_if_missing scipy       scipy
_pip_if_missing yaml        pyyaml
_pip_if_missing pandas      pandas
_pip_if_missing sklearn     scikit-learn
_pip_if_missing monai       "monai[nibabel]"
_pip_if_missing lpips       lpips
_pip_if_missing torchmetrics torchmetrics

echo "  Python  : $(python3 --version)"
echo "  PyTorch : $(python3 -c 'import torch; print(torch.__version__)')"
echo "  GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  N GPUs  : $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"

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

# ── nnU-Net environment variables (for RUN_0003a) ────────────────────────────
if [[ "$RUN_ID" = "0003a" ]]; then
    export nnUNet_raw="${WORK}/nnUNet/raw"
    export nnUNet_preprocessed="${WORK}/nnUNet/preprocessed"
    export nnUNet_results="${WORK}/nnUNet/results"
    mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
    echo "  nnUNet paths configured."
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

# Si une config dédiée Jean Zay existe pour ce run, on appelle le module
# d'entraînement directement plutôt que train.py (qui résout la config lui-même).
# Cela permet d'utiliser batch_size / num_workers adaptés aux 4xH100.
JZ_CONFIG="configs/run_${RUN_ID}_jeanzay.yaml"
if [[ -f "$JZ_CONFIG" ]]; then
    echo "  Config Jean Zay : $JZ_CONFIG"
    # Special handling for run IDs that need custom commands
    case "$RUN_ID" in
        0003a)
            echo "  Running nnU-Net v2 pipeline ..."
            python3 -m src.prepare_nnunet_dataset \
                --data-root "$DATA_ROOT" \
                --split-pkl results/splits/task2_fixed.pkl \
                --output-root nnUNet_raw/Dataset501_LISA
            echo "  nnUNetv2_plan_and_preprocess ..."
            nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -np 8 -nps 8
            echo "  nnUNetv2_train ..."
            nnUNetv2_train 501 3d_fullres 0
            echo "  Evaluation ..."
            python3 -m src.evaluate_task2_nnunet \
                --pred-dir "${nnUNet_results}/Dataset501_LISA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation" \
                --gt-dir "nnUNet_raw/Dataset501_LISA/labelsTr" \
                --config "$JZ_CONFIG"
            ;;
        0003b)
            python3 -m src.train_task2_medsam2 --config "$JZ_CONFIG" $SMOKE_FLAG
            python3 -m src.evaluate_task2_medsam2 --config "$JZ_CONFIG" $SMOKE_FLAG
            ;;
        0003c)
            python3 -m src.train_task2_hybrid --config "$JZ_CONFIG" $SMOKE_FLAG
            python3 -m src.evaluate_task2_hybrid --config "$JZ_CONFIG" $SMOKE_FLAG
            ;;
        *)
            # Generic handling for other runs
            TASK=$(python3 -c "import yaml; print(yaml.safe_load(open('$JZ_CONFIG'))['run']['task'])")
            case "$TASK" in
                1a)        MODULE="src.train_task1a" ;;
                1b)        MODULE="src.train_task1b" ;;
                2)         MODULE="src.train_task2"  ;;
                multitask) MODULE="src.train_multitask" ;;
                *)         echo "[ERREUR] Tache inconnue : $TASK"; exit 1 ;;
            esac
            python3 -m "$MODULE" --config "$JZ_CONFIG" $SMOKE_FLAG
            ;;
    esac
else
    python3 train.py --run "$RUN_ID" $SMOKE_FLAG
fi

echo "  Durée : $(( ($(date +%s) - T0) / 60 )) min"

# Évaluation déjà intégrée dans l'entraînement pour 0003a/0003b/0003c
if [[ "$RUN_ID" =~ ^0003[abc]$ ]]; then
    echo ""
    echo "--> Évaluation déjà effectuée pendant l'entraînement"
else
    echo ""
    echo "── Évaluation RUN_${RUN_ID} ────────────────────────────────────────────"
    T1=$(date +%s)

    JZ_CONFIG="configs/run_${RUN_ID}_jeanzay.yaml"
    if [[ -f "$JZ_CONFIG" ]]; then
        TASK=$(python3 -c "import yaml; print(yaml.safe_load(open('$JZ_CONFIG'))['run']['task'])")
    case "$TASK" in
        1a)        EVAL_MODULE="src.evaluate_task1a" ;;
        1b)        EVAL_MODULE="src.evaluate_task1b" ;;
        2)         EVAL_MODULE="src.evaluate_task2"  ;;
        multitask) EVAL_MODULE="src.evaluate_multitask" ;;
        *)         echo "[ERREUR] Tâche inconnue : $TASK"; exit 1 ;;
    esac
    python3 -m "$EVAL_MODULE" --config "$JZ_CONFIG" $SMOKE_FLAG
else
    python3 evaluate.py --run "$RUN_ID" $SMOKE_FLAG
fi

echo "  Durée : $(( ($(date +%s) - T1) / 60 )) min"

# ── Bilan ─────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================="
echo "  TERMINÉ — RUN_${RUN_ID}"
echo "  Durée totale : $(( ($(date +%s) - T0) / 60 )) min"
echo "  Métriques    : results/runs/RUN_${RUN_ID}/metrics.json"
echo "  Checkpoints  : outputs/checkpoints/RUN_${RUN_ID}/"
echo "  Logs Slurm   : outputs/logs/RUN_${RUN_ID}/"
echo "  Fin          : $(date)"
echo "======================================================================="
echo ""
echo "── Récupération (depuis la machine locale) ────────────────────────────"
echo "  scp -rp -o ProxyCommand=\"ssh froussea@ssh.telecom-bretagne.eu nc %h %p\" \\"
echo "    ulq73oz@jean-zay.idris.fr:$PROJECT_ROOT/outputs/checkpoints/RUN_${RUN_ID}/ \\"
echo "    outputs/checkpoints/RUN_${RUN_ID}/"
echo ""
echo "  scp -rp -o ProxyCommand=\"ssh froussea@ssh.telecom-bretagne.eu nc %h %p\" \\"
echo "    ulq73oz@jean-zay.idris.fr:$PROJECT_ROOT/outputs/logs/RUN_${RUN_ID}/ \\"
echo "    outputs/logs/RUN_${RUN_ID}/"
echo ""
echo "  scp -rp -o ProxyCommand=\"ssh froussea@ssh.telecom-bretagne.eu nc %h %p\" \\"
echo "    ulq73oz@jean-zay.idris.fr:$PROJECT_ROOT/results/runs/RUN_${RUN_ID}/ \\"
echo "    results/runs/RUN_${RUN_ID}/"
echo "======================================================================="
