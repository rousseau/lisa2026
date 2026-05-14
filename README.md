# LISA 2026

Challenge MICCAI 2026 — Low-field Infant and child brain Segmentation and Analysis  
https://www.synapse.org/Synapse:syn72118611/wiki/

## Structure

```
lisa2026/
├── src/           # code Python (modèles, pipelines, utilitaires)
├── configs/       # fichiers de configuration des expériences (YAML)
├── results/       # résultats légers : métriques CSV, courbes PNG
├── outputs/       # résultats lourds : images, prédictions (gitignore)
└── paper/         # sources LaTeX de l'article
```

## Données

- Challenge : `~/Data/LISA2026/`
- Scanner : Hyperfine SWOOP (0.064 T), pédiatrique
- 3 orientations par sujet : axiale (`_LF_axi`), coronale (`_LF_cor`), sagittale (`_LF_sag`)

## Tâches

| Task | Description | Meilleur run |
|------|-------------|---------------|
| Task 1a | Classification multi-label d'artefacts (7 classes : Noise, Zipper, Positioning, Banding, Motion, Contrast, Distortion) | RUN_0002 |
| Task 1b | Amélioration qualité (suppression bruit / mouvement) — à lancer | RUN_0004 (en cours) |
| Task 2  | Segmentation multi-structures (hippocampes, noyaux caudés, putamens, globus pallidus, thalami, corps calleux, ventricules latéraux) | **RUN_0003_EXP_C** (DSC=0.631) |

## Dépendances

```bash
conda activate lisa   # à créer
pip install nibabel pandas numpy matplotlib scikit-learn
```

## Runs actifs

| Run ID | Task | Statut | Description |
|--------|------|--------|-------------|
| RUN_0001 | 1a | ✅ baseline | Classification per-artefact (7 modèles indépendants) |
| RUN_0002 | 1a | ✅ actif | Multi-label ordinal avec EMD + Focal loss |
| RUN_0003 | 2 | ✅ baseline | DynUNet 7-class, patch 96³ |
| RUN_0003_COLLAPSED | 2 | ✅ actif | DynUNet 6-class (hippocampes fusionnés) |
| **RUN_0003_EXP_C** | **2** | **✅ winner** | **DynUNet + augmentation symétrique corrigée, DSC=0.631** |
| RUN_0004 | 1b | 🔜 à lancer | U-Net débruitage / suppression artefacts |

Runs archivés (non maintenus) : RUN_0003_EXP_A (training failed), RUN_0003_EXP_B (checkpoint skip), RUN_0003_EXP_C_TTA (TTA cassé), RUN_0003_EXP_SYM (régression).

## Lancement sur Jean Zay

Le point d'entrée générique est [src/slurm/lisa_jeanzay.slurm](src/slurm/lisa_jeanzay.slurm).

Exemples :

```bash
sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0001
sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0002
sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0003_EXP_C --smoke-test
sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0004
```

Les wrappers shell (`scripts/run_XXXX.sh`) couvrent les runs 0001–0004. Les runs `0003_COLLAPSED` et `0003_EXP_C` sont routés directement via `train.py` / `evaluate.py`.
