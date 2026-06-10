# LISA 2026 — Sync & Evaluation Status

| Run     | Remote Trained | Local Synced | Local Evaluated | Metrics OK | Notes |
|---------|----------------|--------------|-----------------|------------|-------|
| RUN_0001| ✓              | ✓            | ✓               | ✓          | Baseline Task 1a |
| RUN_0002| ✓              | ✓            | ✓               | ✓          | CycleGAN Task 1b |
| RUN_0003| ✓              | ✓            | ✓               | ✓          | DynUNet Task 2 |
| RUN_0003a| ?             | ?            | ?               | ?          | nnU-Net Task 2 |
| RUN_0003b| ?             | ?            | ?               | ?          | MedSAM2 Task 2 |
| RUN_0003c| ?             | ?            | ?               | ?          | Hybrid Task 2 |
| RUN_0004| ✓              | ✓            | ✓               | ✓          | Multi-task |

_(Ce fichier est mis à jour automatiquement par `src/slurm/sync_from_jeanzay.sh` après chaque `pull_run`.)_

## Mise à jour manuelle

Après une évaluation locale réussie, marquer le run comme évalué :
```bash
bash src/slurm/sync_from_jeanzay.sh status_run --run 0003a
```
