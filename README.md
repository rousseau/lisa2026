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

| Task | Description |
|------|-------------|
| Task 1a | Classification multi-label d'artefacts (7 classes : Noise, Zipper, Positioning, Banding, Motion, Contrast, Distortion) |
| Task 1b | Amélioration qualité (suppression bruit / mouvement pour que Task 1a classe sans artefact) |
| Task 2  | Segmentation multi-structures (hippocampes, noyaux caudés, putamens, globus pallidus, thalami, corps calleux, ventricules latéraux) |

## Dépendances

```bash
conda activate lisa   # à créer
pip install nibabel pandas numpy matplotlib scikit-learn
```

## Lancement sur Jean Zay

Le point d'entrée générique est [src/slurm/lisa_jeanzay.slurm](src/slurm/lisa_jeanzay.slurm).

Exemples :

```bash
sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0001
sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0002
sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0002 --smoke-test
```

Les wrappers correspondants sont [scripts/run_0001.sh](scripts/run_0001.sh) et [scripts/run_0002.sh](scripts/run_0002.sh). Le run 0002 reprend l'approche multi-label ordinal avec EMD + Focal.
