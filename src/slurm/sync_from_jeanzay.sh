#!/usr/bin/env bash
# =============================================================================
# sync_from_jeanzay.sh — Synchronisation des résultats depuis Jean Zay
#
# Rapatrie pour un run donné (ou tous les runs) :
#   - outputs/checkpoints/RUN_XXXX/   (checkpoints .pt)
#   - outputs/logs/RUN_XXXX/          (logs Slurm .out/.err + train.log)
#   - results/runs/RUN_XXXX/          (métriques, notes, plots)
#
# Principe de connexion (ControlMaster) :
#   Le script ouvre une session SSH maître au démarrage.
#   Les 2 mots de passe (proxy + Jean Zay) sont demandés une seule fois.
#   Tous les scp suivants réutilisent ce tunnel sans redemander de mot de passe.
#   Le tunnel est fermé proprement à la fin.
#
# Usage :
#   bash src/slurm/sync_from_jeanzay.sh pull_run --run 0002
#   bash src/slurm/sync_from_jeanzay.sh pull_all
#   bash src/slurm/sync_from_jeanzay.sh push_splits
#   bash src/slurm/sync_from_jeanzay.sh push_data
#   bash src/slurm/sync_from_jeanzay.sh status
#
# Configuration requise :
#   cp src/slurm/sync_from_jeanzay.sh.env.example .sync_env
#   # Puis éditer .sync_env avec vos identifiants (fichier ignoré par git)
# =============================================================================
set -euo pipefail

# ── Résolution des chemins ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Chargement de .sync_env ───────────────────────────────────────────────────
ENV_FILE="$PROJECT_ROOT/.sync_env"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

_check_var() {
    local var_name="$1"
    if [[ -z "${!var_name:-}" ]]; then
        echo "[ERREUR] Variable '$var_name' non définie."
        echo ""
        echo "Créer le fichier .sync_env à la racine du projet :"
        echo "  cp src/slurm/sync_from_jeanzay.sh.env.example .sync_env"
        echo "  # Puis éditer .sync_env avec vos identifiants"
        exit 1
    fi
}

_check_var JEANZAY_USER
_check_var JEANZAY_HOST
_check_var PROXY_USER
_check_var PROXY_HOST
_check_var REMOTE_BASE
_check_var LOCAL_DATA

# ── Parsing des arguments ─────────────────────────────────────────────────────
CMD="${1:-help}"
shift || true
RUN_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)   RUN_ID="${2#RUN_}"; shift 2 ;;
        --run=*) RUN_ID="${1#*=}"; RUN_ID="${RUN_ID#RUN_}"; shift ;;
        *) echo "[WARN] argument inconnu ignoré : $1"; shift ;;
    esac
done

# Afficher l'aide sans ouvrir le tunnel
if [[ "$CMD" == "help" ]]; then
    echo "Usage: bash src/slurm/sync_from_jeanzay.sh <commande> [--run <ID>]"
    echo ""
    echo "Download (Jean Zay → local) :"
    echo "  pull_run --run <ID>   Checkpoints + logs + résultats d'un run"
    echo "  pull_all              Tous les runs (0001 0002 0003 0004)"
    echo ""
    echo "Upload (local → Jean Zay) :"
    echo "  push_splits           results/splits/ → Jean Zay"
    echo "  push_data             \$LOCAL_DATA/ → Jean Zay"
    echo ""
    echo "Divers :"
    echo "  status                Jobs SLURM en cours (squeue)"
    exit 0
fi

# ── Ouverture du tunnel SSH maître (ControlMaster) ────────────────────────────
SOCK="/tmp/lisa2026_sync_$$.sock"

_close_tunnel() {
    ssh -S "$SOCK" -O exit "${JEANZAY_USER}@${JEANZAY_HOST}" 2>/dev/null || true
    rm -f "$SOCK"
}
trap _close_tunnel EXIT

PROXY_CMD="ssh -i ~/.ssh/id_mrixfields -o StrictHostKeyChecking=no ${PROXY_USER}@${PROXY_HOST} nc %h %p"

echo "======================================================================="
echo "  LISA 2026 — sync depuis Jean Zay"
echo "  Jean Zay : ${JEANZAY_USER}@${JEANZAY_HOST}"
echo "  Proxy    : ${PROXY_USER}@${PROXY_HOST}"
echo "  Remote   : ${REMOTE_BASE}"
echo "======================================================================="
echo ""
echo "Ouverture du tunnel SSH (mots de passe demandés une seule fois) ..."
echo ""

ssh -fNM \
    -S "$SOCK" \
    -o "ProxyCommand=${PROXY_CMD}" \
    -o "StrictHostKeyChecking=no" \
    -o "ConnectTimeout=30" \
    -o "ServerAliveInterval=60" \
    "${JEANZAY_USER}@${JEANZAY_HOST}"

echo "Tunnel ouvert."
echo ""

# Options communes pour scp et ssh via le tunnel
SSH_CTRL="-S $SOCK -o ControlMaster=no -o BatchMode=yes"
SCP_CTRL="-o ControlPath=${SOCK} -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=30"

# ── Helpers ───────────────────────────────────────────────────────────────────
_scp_down() {
    local remote="$1"
    local local_dest="$2"
    # shellcheck disable=SC2086
    scp -rp $SCP_CTRL "${JEANZAY_USER}@${JEANZAY_HOST}:${remote}" "${local_dest}" 2>/dev/null
}

_scp_up() {
    local local_src="$1"
    local remote="$2"
    # shellcheck disable=SC2086
    scp -rp $SCP_CTRL "${local_src}" "${JEANZAY_USER}@${JEANZAY_HOST}:${remote}" 2>/dev/null
}

_pull_run_id() {
    local ID="$1"
    local TAG="RUN_${ID}"
    echo "── ${TAG} ← Jean Zay"

    mkdir -p "${PROJECT_ROOT}/outputs/checkpoints/${TAG}"
    mkdir -p "${PROJECT_ROOT}/outputs/logs/${TAG}"
    mkdir -p "${PROJECT_ROOT}/results/runs/${TAG}"

    echo -n "   checkpoints/ ... "
    if _scp_down "${REMOTE_BASE}/outputs/checkpoints/${TAG}/" \
                 "${PROJECT_ROOT}/outputs/checkpoints/${TAG}/"; then
        N=$(find "${PROJECT_ROOT}/outputs/checkpoints/${TAG}" -name "*.pt" | wc -l)
        echo "OK  ($N fichiers .pt)"
    else
        echo "SKIP (absent ou erreur)"
    fi

    echo -n "   logs/         ... "
    if _scp_down "${REMOTE_BASE}/outputs/logs/${TAG}/" \
                 "${PROJECT_ROOT}/outputs/logs/${TAG}/"; then
        N=$(find "${PROJECT_ROOT}/outputs/logs/${TAG}" -type f | wc -l)
        echo "OK  ($N fichiers)"
    else
        echo "SKIP (absent ou erreur)"
    fi

    echo -n "   results/      ... "
    if _scp_down "${REMOTE_BASE}/results/runs/${TAG}/" \
                 "${PROJECT_ROOT}/results/runs/${TAG}/"; then
        echo "OK"
    else
        echo "SKIP (absent ou erreur)"
    fi

    echo ""
}

# ── Commandes ─────────────────────────────────────────────────────────────────
case "$CMD" in

  pull_run)
    [[ -z "$RUN_ID" ]] && { echo "[ERREUR] --run <ID> obligatoire"; exit 1; }
    _pull_run_id "$RUN_ID"
    ;;

  pull_all)
    for ID in 0001 0002 0003 0004; do
        _pull_run_id "$ID"
    done
    ;;

  push_splits)
    echo "── Splits → Jean Zay"
    # shellcheck disable=SC2086
    ssh $SSH_CTRL "${JEANZAY_USER}@${JEANZAY_HOST}" \
        "mkdir -p ${REMOTE_BASE}/results/splits"
    _scp_up "${PROJECT_ROOT}/results/splits/" "${REMOTE_BASE}/results/splits/"
    echo "  [OK] results/splits/ → ${REMOTE_BASE}/results/splits/"
    ;;

  push_data)
    echo "── Données NIfTI → Jean Zay"
    REMOTE_DATA="${REMOTE_BASE%/lisa2026}/data"
    # shellcheck disable=SC2086
    ssh $SSH_CTRL "${JEANZAY_USER}@${JEANZAY_HOST}" "mkdir -p ${REMOTE_DATA}"
    _scp_up "${LOCAL_DATA}/" "${REMOTE_DATA}/"
    echo "  [OK] ${LOCAL_DATA}/ → ${REMOTE_DATA}/"
    ;;

  status)
    echo "── Jobs SLURM en cours"
    # shellcheck disable=SC2086
    ssh $SSH_CTRL "${JEANZAY_USER}@${JEANZAY_HOST}" \
        "squeue -u ${JEANZAY_USER} --format='%.10i %.20j %.8T %.12M %.5D %R' 2>/dev/null \
         || echo '  (aucun job)'"
    ;;

  *)
    echo "[ERREUR] Commande inconnue : $CMD"
    echo "Lancer 'bash src/slurm/sync_from_jeanzay.sh help' pour l'aide."
    exit 1
    ;;

esac
