#!/bin/bash
# =============================================================================
# jz_sync.sh — Transferts local ↔ Jean Zay — LISA 2026
#
# Usage (depuis la racine du projet) :
#   bash src/slurm/jz_sync.sh push_data              # données NIfTI → Jean Zay
#   bash src/slurm/jz_sync.sh push_splits            # splits → Jean Zay
#   bash src/slurm/jz_sync.sh pull_run --run 0002    # résultats RUN_0002 ← Jean Zay
#   bash src/slurm/jz_sync.sh pull_all               # tous les runs ← Jean Zay
# =============================================================================

set -e

JZ_USER="ulq73oz@jean-zay.idris.fr"
PROXY="ssh froussea@ssh.telecom-bretagne.eu nc %h %p"
SCP="scp -rp -o ProxyCommand='${PROXY}'"

BASE_SRC=$(git rev-parse --show-toplevel)
BASE_DST="/gpfswork/rech/crp/ulq73oz/LISA/lisa2026"
BASE_DST_DATA="/gpfswork/rech/crp/ulq73oz/LISA/data"

# ---------- helpers ----------
up()   { eval "$SCP" "$1" "$JZ_USER:$2"; }
down() { eval "$SCP" "$JZ_USER:$1" "$2"; }

pull_run_id() {
    local RUN_ID="$1"
    local TAG="RUN_${RUN_ID}"
    echo "[download] ${TAG} ← Jean Zay"
    mkdir -p "${BASE_SRC}/outputs/checkpoints/${TAG}"
    mkdir -p "${BASE_SRC}/outputs/logs/${TAG}"
    mkdir -p "${BASE_SRC}/results/runs/${TAG}"
    down "${BASE_DST}/outputs/checkpoints/${TAG}/" "${BASE_SRC}/outputs/checkpoints/${TAG}/"
    down "${BASE_DST}/outputs/logs/${TAG}/"        "${BASE_SRC}/outputs/logs/${TAG}/"
    down "${BASE_DST}/results/runs/${TAG}/"        "${BASE_SRC}/results/runs/${TAG}/"
    echo "  [OK] ${TAG}"
}

# ---------- parsing --run ----------
RUN_ID=""
CMD="${1:-help}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)   RUN_ID="${2#RUN_}"; shift 2 ;;
        --run=*) RUN_ID="${1#*=}"; RUN_ID="${RUN_ID#RUN_}"; shift ;;
        *) echo "[WARN] argument inconnu ignoré : $1"; shift ;;
    esac
done

# ---------- commandes ----------
case "$CMD" in

  push_data)
    echo "[upload] Données NIfTI → Jean Zay"
    up ~/Data/LISA2026/ "$BASE_DST_DATA/"
    echo "  [OK] ~/Data/LISA2026/ → ${BASE_DST_DATA}/"
    ;;

  push_splits)
    echo "[upload] Splits → Jean Zay"
    up "${BASE_SRC}/results/splits/" "${BASE_DST}/results/splits/"
    echo "  [OK] results/splits/ → ${BASE_DST}/results/splits/"
    ;;

  pull_run)
    [[ -z "$RUN_ID" ]] && { echo "[ERREUR] --run <ID> obligatoire"; exit 1; }
    pull_run_id "$RUN_ID"
    ;;

  pull_all)
    for ID in 0001 0002 0003 0004; do
        pull_run_id "$ID"
    done
    ;;

  help|*)
    echo "Usage: bash src/slurm/jz_sync.sh <commande> [--run <ID>]"
    echo ""
    echo "Upload (local → Jean Zay) :"
    echo "  push_data              Données NIfTI ~/Data/LISA2026/"
    echo "  push_splits            Fichiers de split results/splits/"
    echo ""
    echo "Download (Jean Zay → local) :"
    echo "  pull_run --run <ID>    Checkpoints + logs + résultats d'un run"
    echo "  pull_all               Tous les runs (0001 0002 0003 0004)"
    ;;

esac
