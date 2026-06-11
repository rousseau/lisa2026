# LISA 2026

Challenge MICCAI 2026 — Low-field Infant and child brain Segmentation and Analysis  
https://www.synapse.org/Synapse:syn72118611/wiki/

---

## Structure

```
lisa2026/
├── src/                       # code Python (modèles, datasets, pipelines)
│   ├── models/__init__.py     # Task1aMultiLabelModel, DynUNetMultiHeadModel, ...
│   ├── datasets/__init__.py   # Task1aMultiTask128Dataset, CleanImageDataset, ...
│   ├── train_multitask.py     # Multi-task trainer (warmup + joint + calibrate)
│   ├── evaluate_multitask.py  # Multi-task evaluator (Tasks 1a/1b/2)
│   ├── train_*.py             # task-specific trainers
│   ├── evaluate_*.py          # task-specific evaluators
│   └── slurm/                 # SLURM helpers (submit.sh, *.slurm)
├── configs/                   # YAML configuration files per run
├── results/                   # Scientific tracking: metrics.json, AGENTS.md
│   ├── RUNS_INDEX.md          # Global run ranking table
│   └── runs/RUN_XXXX/         # Per-run documentation
├── outputs/                   # Checkpoints, logs (gitignored)
├── paper/                     # LaTeX report sources
└── results/splits/            # Fixed train/val splits (pickles)
```

## Données

- Challenge : `/home/rousseau/Data/LISA2026/` (ou `$LISA_DATA_ROOT`)
- Scanner : Hyperfine SWOOP (0.064 T), pédiatrique
- 3 orientations par sujet : axiale (`_LF_axi`), coronale (`_LF_cor`), sagittale (`_LF_sag`)

## Tâches

| Task | Description | Meilleur run (promoted) |
|------|-------------|--------|---|
| **Task 1a** | Classification multi-label d'artefacts (7 classes : Noise, Zipper, Positioning, Banding, Motion, Contrast, Distortion) | **RUN_0001** — DenseNet264 × 7 independent heads, aggregate **0.6887** |
| **Task 1b** | Amélioration qualité (suppression bruit / mouvement) — à lancer | — (aucun run promu) |
| **Task 2** | Segmentation multi-structures (12 classes : hippocampes, noyaux caudés, putamens, globus pallidus, thalami, corps calleux, ventricules latéraux, etc.) | **RUN_0003** — DynUNet 12-class [32..320], DSC **0.4647**, HD95 12.30 |

## Environnement

```bash
conda activate lisa2026
# Python 3.12/3.13 · PyTorch 2.5.0+cu124 · MONAI 1.5.2
```

## Runs actifs (index)

Voir [`results/RUNS_INDEX.md`](results/RUNS_INDEX.md) pour le tableau complet.

### Task 1a

| Run ID | Date | Model | Aggregate | Decision |
|--------|------|-------|------|--------|
| [RUN_0001](results/runs/RUN_0001/AGENTS.md) | 2026-05-12 | DenseNet264 × 7 | **0.6887** | ✅ Promoted |
| [RUN_0002](results/runs/RUN_0002/AGENTS.md) | 2026-05-13 | DenseNet264 × 1 + 7 heads | 0.6695 | ❌ Rejeté |

### Task 1b

| Run ID | Date | Model | PSNR | Decision |
|--------|------|-------|------|------|
| [RUN_0004](results/runs/RUN_0004/AGENTS.md) | 2026-05-19 v2 | SharedEncoder 3T heads | 3.48 dB | 🔜 Jean Zay |

### Task 2

| Run ID | Date | Model | DSC | HD95 | Decision |
|--------|------|-------|-----|------|------|
| [RUN_0003](results/runs/RUN_0003/AGENTS.md) | 2026-05-14 | DynUNet 12-class | **0.4647** | 12.30 | ✅ Promoted |

### Multi-task (Tasks 1a + 1b + 2)

| Run ID | Date | Architecture | T1a Aggr | T2 DSC | Decision |
|--------|------|------|---|--|------|
| [RUN_0004](results/runs/RUN_0004/AGENTS.md) v2 | 2026-05-19 | DynUNet backbone + 3 heads | 0.7143 | 0.5532 | 🔜 Jean Zay |

**Note** : RUN_0004 v1 (job 878620, 2026-05-19) a été rejeté (DSC=0.0083).
Des corrections majeures ont été appliquées (v2) — voir ci-dessous.

## RUN_0004 — Architecture multi-tâche

### Principe

Un **encodeur DynUNet partagé** [32→64→128→256→320] avec 3 têtes :

```
Volume IRM 3D (1×128³)
        │
        ▼
┌───────────────┐
│  DynUNet      │  ← Backbone (102/102 keys = RUN_0003 compatible)
│  Shared Encoder│
└───┬─────┬─────┘
    ▼     ▼     ▼
 ┌─┴──┐ ┌─┴──┐ ┌─┴──┐
 │1a  │ │1b  │ │ 2  │  ← Têtes independentes
 │Class│ │Recon│ │Seg │
 └────┘ └────┘ └────┘
```

### Correctifs v2 (contre les 3 causes racines de v1)

| Cause (v1) | Correction (v2) |
|---|---|
| 0/102 keys warm-start | `DynUNetMultiHeadModel` → 46/84 keys (~55%) |
| Loss 1a domine (ratio 2.7:1) | `_calibrate_losses()` — normalisation par magnitude initiale |
| Warmup 10 epochs insuffisant | 30 epochs max + early exit si DSC ≥ 0.15 |

### Pipeline d'entraînement

1. **Warmup** (Task 2 seul) — jusqu'à 30 epochs ou DSC ≥ 0.15
2. **Calibration** — mesure des pertes initiales pour équilibrage des gradients
3. **Joint** (3 têtes) — pertes normalisées, early stopping sur DSC (patience=15)

### Commandes

```bash
# Entrainement
python train.py --run 0004

# Evaluation
python evaluate.py --run 0004

# Smoke test local
python train.py --run 0004 --smoke-test
python evaluate.py --run 0004 --smoke-test

# Soumission Jean Zay (H100)
bash src/slurm/submit.sh --run 0004
```

## Lancement Jean Zay

### Méthode recommandée

Le script `src/slurm/submit.sh` injecte automatiquement le run-ID dans les paramètres SLURM :

```bash
bash src/slurm/submit.sh --run 0004
bash src/slurm/submit.sh --run 0004 --smoke-test
```

Logs : `outputs/logs/lisa2026_<JOBID>.out`

### Méthode directe (SLURM script dédié)

Chaque run peut avoir un SLURM script dédié dans `src/slurm/` :

```bash
sbatch src/slurm/run_0004_multitask.slurm
```

### Runs Task 1a / Task 2 directs

```bash
python train.py --run 0001
python train.py --run 0003
python evaluate.py --run 0001
python evaluate.py --run 0003
```

## Governance

Conforme à [`AGENTS.md`](lisa2026/AGENTS.md) : chaque run a un dossier dédié avec AGENTS.md,
implementation_plan.md, metrics.json, config_snapshot.yaml, notes.md. Les comparaisons se font
sur la table [`results/RUNS_INDEX.md`](results/RUNS_INDEX.md).
