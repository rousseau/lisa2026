#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Setup Jean Zay — première installation (nœud de login)
#
# À exécuter UNE SEULE FOIS depuis le nœud de login Jean Zay :
#   bash src/slurm/setup_jeanzay.sh
#
# Ce script :
#   1. Crée la structure de répertoires sous $WORK/LISA/
#   2. Installe les dépendances Python dans $WORK/.local
#   3. Affiche les commandes rsync pour transférer les données NIfTI
# ─────────────────────────────────────────────────────────────────────────────

set -e

LISA_ROOT="$WORK/LISA"
PROJECT_DIR="$LISA_ROOT/lisa2026"
DATA_DIR="$LISA_ROOT/data"

echo "======================================================================="
echo " Setup Jean Zay — LISA 2026"
echo " LISA_ROOT : $LISA_ROOT"
echo "======================================================================="

# ─── 1. Répertoires ───────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR"
mkdir -p "$PROJECT_DIR/outputs/checkpoints"
mkdir -p "$PROJECT_DIR/results/stats"
mkdir -p "$PROJECT_DIR/logs"
echo "[1/3] Répertoires créés"

# ─── 2. Installer les dépendances Python ─────────────────────────────────────
echo ""
echo "[2/3] Installation des dépendances Python..."

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"

pip install --user --quiet nibabel scipy pyyaml pandas
echo "  → nibabel, scipy, pyyaml, pandas installés dans \$WORK/.local"

echo "[2/3] Dépendances installées"

# ─── 3. Cloner / mettre à jour le repo lisa2026 ──────────────────────────────
echo ""
echo "[3/3] Repo lisa2026..."
if [ ! -d "$PROJECT_DIR/.git" ]; then
    echo "      Clonage du repo :"
    echo "      git clone https://github.com/rousseau/lisa2026.git $PROJECT_DIR"
    echo "      (Lancez cette commande manuellement si ce script n'est pas déjà dans le repo)"
else
    echo "      Mise à jour du repo..."
    cd "$PROJECT_DIR" && git pull
fi

# ─── 4. Transfert des données ─────────────────────────────────────────────────
echo ""
echo "======================================================================="
echo " Transfert des données (à lancer depuis la machine locale) :"
echo ""
echo "  # Toutes les données NIfTI LISA 2026 :"
echo "  rsync -avP ~/Data/LISA2026/ \\"
echo "    <login>@jean-zay.idris.fr:$DATA_DIR/"
echo ""
echo "  # Optionnel — transférer des checkpoints existants :"
echo "  rsync -avP ~/Exp/lisa2026/outputs/checkpoints/ \\"
echo "    <login>@jean-zay.idris.fr:$PROJECT_DIR/outputs/checkpoints/"
echo ""
echo "======================================================================="
echo " Setup terminé. Lancer l'entraînement depuis $PROJECT_DIR :"
echo ""
echo "   sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0002"
echo ""
echo "   # Choisir un autre run :"
echo "   sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0001"
echo ""
echo "   # Smoke test local sur Jean Zay :"
echo "   sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0002 --smoke-test"
echo "======================================================================="
