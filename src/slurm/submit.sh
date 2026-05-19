#!/bin/bash
# =============================================================================
# submit.sh — Point d'entrée unique pour soumettre un job LISA 2026 sur Jean Zay
#
# Les directives #SBATCH sont statiques et ne peuvent pas contenir de variables
# shell. Ce script injecte --job-name, --output et --error dynamiquement via
# la ligne de commande sbatch, de sorte que les logs contiennent le run ID.
#
# Usage (depuis la racine du projet) :
#   bash src/slurm/submit.sh --run 0004
#   bash src/slurm/submit.sh --run 0003
#   bash src/slurm/submit.sh --run 0004 --smoke-test
#   bash src/slurm/submit.sh --run 0001 --time 48:00:00
#
# Les logs seront nommés :
#   outputs/logs/run0004_<JOBID>.out
#   outputs/logs/run0004_<JOBID>.err
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Parsing des arguments ─────────────────────────────────────────────────────
RUN_ID=""
SMOKE_FLAG=""
TIME_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)        RUN_ID="${2#RUN_}"; shift 2 ;;
        --run=*)      RUN_ID="${1#*=}"; RUN_ID="${RUN_ID#RUN_}"; shift ;;
        --smoke-test) SMOKE_FLAG="--smoke-test"; shift ;;
        --time)       TIME_OVERRIDE="$2"; shift 2 ;;
        --time=*)     TIME_OVERRIDE="${1#*=}"; shift ;;
        --help|-h)
            sed -n '2,20p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "[WARN] argument inconnu ignoré : $1"; shift ;;
    esac
done

if [[ -z "$RUN_ID" ]]; then
    echo "[ERREUR] --run <ID> est obligatoire (ex: --run 0004)"
    echo "Usage : bash src/slurm/submit.sh --run <ID> [--smoke-test] [--time HH:MM:SS]"
    exit 1
fi

case "$RUN_ID" in
    0001|0002|0003|0004) ;;
    *) echo "[ERREUR] Run inconnu : $RUN_ID. Valides : 0001 0002 0003 0004"; exit 1 ;;
esac

# ── Sélection du script Slurm ─────────────────────────────────────────────────
if [[ "$RUN_ID" == "0004" ]]; then
    SLURM_SCRIPT="src/slurm/run_0004_multitask.slurm"
else
    SLURM_SCRIPT="src/slurm/lisa_jeanzay.slurm"
fi

# ── Noms de logs avec run ID ──────────────────────────────────────────────────
JOB_NAME="lisa2026_run${RUN_ID}"
LOG_OUT="outputs/logs/${JOB_NAME}_%j.out"
LOG_ERR="outputs/logs/${JOB_NAME}_%j.err"

mkdir -p outputs/logs

# ── Construction de la commande sbatch ───────────────────────────────────────
SBATCH_ARGS=(
    --job-name="$JOB_NAME"
    --output="$LOG_OUT"
    --error="$LOG_ERR"
)
[[ -n "$TIME_OVERRIDE" ]] && SBATCH_ARGS+=(--time="$TIME_OVERRIDE")

# ── Affichage du plan ─────────────────────────────────────────────────────────
echo "======================================================================="
echo "  LISA 2026 — Soumission RUN_${RUN_ID}"
echo "  Script  : $SLURM_SCRIPT"
echo "  Job     : $JOB_NAME"
echo "  Logs    : outputs/logs/${JOB_NAME}_<JOBID>.{out,err}"
[[ -n "$SMOKE_FLAG" ]] && echo "  Mode    : SMOKE TEST"
[[ -n "$TIME_OVERRIDE" ]] && echo "  Durée   : $TIME_OVERRIDE (override)"
echo "======================================================================="

# ── Soumission ────────────────────────────────────────────────────────────────
JOB_OUTPUT=$(sbatch "${SBATCH_ARGS[@]}" "$SLURM_SCRIPT" --run "$RUN_ID" $SMOKE_FLAG)
echo "$JOB_OUTPUT"

JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '(?<=Submitted batch job )\d+')
if [[ -n "$JOB_ID" ]]; then
    echo ""
    echo "  Suivi   : squeue -j $JOB_ID"
    echo "  Logs    : tail -f outputs/logs/${JOB_NAME}_${JOB_ID}.out"
fi
