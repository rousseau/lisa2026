# AGENTS.md

## Project Overview

This repository is an experimental work for the LISA 2026 challenge (https://www.synapse.org/Synapse:syn72118611/wiki/637239). The purpose is to implement in PyTorch techniques to solve three tasks, in a fully reproducible research spirit, using the metrics and ranking proposed by the challenge. 

In Task 1a, develop an automatic method to assess image artifacts/quality of 0.064T magnetic resonance images.

In Task 1b, develop an automatic method to improve image quality via artifact removal or reduction for 0.064T magnetic resonance images.

In Task 2, develop an automatic multi-structure segmentation method for 0.064T magnetic resonance images.

A dedicated Python environment using Conda is used: lisa2026.

## Experiment governance

- A run may include multiple technical changes, including architecture shifts.
- Each run must have a dedicated AGENTS file that records exactly what was changed and why.
- Runs are tracked as scientific units, not only training executions.
- A run must define a parent run used as baseline for comparison.
- If two runs are not strictly comparable (major pipeline drift), this must be explicitly stated.

## Evaluation metrics

Task 1a: Accuracy, F1 score, F2 score, Precision, Recall. Ranking using an average of all metrics (https://www.synapse.org/Synapse:syn72118611/wiki/637243)

Task 1b: FID (Fréchet Inception Distance), PSNR (Peak Signal-to-Noise Ratio), LPIPS (Learned Perceptual Image Patch Similarity). Ranking using normalized mean across their respective metrics (https://www.synapse.org/Synapse:syn72118611/wiki/639002)

Task 2: DSC (Dice Similarity Coefficient), HD95, HD, RVE, ASSD. The mean across their respective metrics is used to obtain rankings (https://www.synapse.org/Synapse:syn72118611/wiki/637246).

## Learning procedures

- Fixed split patient-level for an entire cycle of experiments.
- Same normalization, same crop/resize strategy, same post-processing.
- Same evaluation script version across compared runs.
- For each compared run, report environment details (Python, PyTorch, CUDA, GPU).

## Evaluation

For each run (called also experiment), report all metrics and update the common ranking table.

## Run documentation policy

- Raw execution artifacts are stored under outputs.
- Scientific run documentation is stored under results.
- Each run has a dedicated folder and AGENTS file.
- A global index tracks all runs and their status.

Required files per run:

- results/runs/RUN_XXXX/AGENTS.md
- results/runs/RUN_XXXX/implementation_plan.md
- results/runs/RUN_XXXX/metrics.json
- results/runs/RUN_XXXX/config_snapshot.yaml
- results/runs/RUN_XXXX/notes.md

Optional files per run:

- results/runs/RUN_XXXX/plots/
- results/runs/RUN_XXXX/predictions/

Global tracking file:

- results/RUNS_INDEX.md

Minimum content required in each run AGENTS file:

- Run ID and date.
- Tasks covered (1a, 1b, 2).
- Parent run.
- Change scope (incremental or architectural).
- Full list of changes.
- Assumptions and hypothesis.
- Training and evaluation configuration.
- Implementation plan.
- Results summary.
- Comparability statement.
- Decision (promoted, rejected, or to retest).

## Commands

All entry points are unified and accept `--run <ID>` (with or without `RUN_` prefix).

```bash
# Discovery
python train.py --list-runs
python evaluate.py --list-runs

# Training
python train.py --run 0001
python train.py --run 0004 --smoke-test

# Evaluation
python evaluate.py --run 0001
python evaluate.py --run 0003 --smoke-test

# Remote (Jean Zay)
bash src/slurm/submit.sh --run 0001
bash src/slurm/sync_from_jeanzay.sh pull_run --run 0001
bash src/slurm/sync_from_jeanzay.sh pull_all
bash src/slurm/sync_from_jeanzay.sh status

# Visualisation
python visualize.py --run RUN_0001
python visualize.py --run RUN_0003 --qualitative
python visualize.py --compare RUN_0001 RUN_0003 RUN_0004
python visualize.py --auto-compare
```

## Artifacts management and cleanup policy

- `outputs/` contains raw execution artifacts (checkpoints, logs) and is excluded from version control.
- `results/` contains scientific documentation and **must** remain tracked, except heavy `*.csv` or raw predictions.
- Periodic cleanup should remove:
  - Orphan SLURM logs (`lisa2026_*.err`, `lisa2026_*.out`) that do not correspond to a validated run.
  - Redundant training logs (intermediate or failed runs superseded by a corrected version).
  - Empty placeholder directories.
  - `__pycache__` directories to prevent import ghosting after module renames or deletions.
  Run `find src/ results/ configs/ -type d -name "__pycache__" -exec rm -rf {} +` periodically.
- Checkpoint retention: keep only the best or final checkpoint used for evaluation. Full-model snapshots (e.g. `cyclegan_full_best.pt`) may be archived externally if redundant with task-specific checkpoints.

```
.
├── configs/                     # Configuration files for runs
├── doc/                         # LISA proceedings (2024, 2025)
├── outputs/                     # Raw execution artifacts
│   ├── checkpoints/
│   └── logs/
├── paper/                       # LaTeX report
├── results/                     # Scientific tracking and comparison
│   ├── RUNS_INDEX.md
│   ├── runs/
│   │   └── RUN_0001/
│   │       ├── AGENTS.md
│   │       ├── implementation_plan.md
│   │       ├── notes.md
│   │       ├── config_snapshot.yaml
│   │       ├── metrics.json
│   │       └── plots/
│   ├── plots/
│   └── stats/
├── src/                         # Source code
│   └──  slurm/                  # Slurm script for remote server
```