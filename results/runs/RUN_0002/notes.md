# RUN_0002 — Notes

## Status

✅ **Terminé** sur Jean Zay (4x H100 80 Go)

## Training configuration (exécuté sur Jean Zay)

| Parameter | Value |
|-----------|-------|
| GPU | 4× H100 80 Go |
| Epochs | 100 |
| Training time | 14 minutes |
| Batch size | 1 (DataParallel sur 4 GPUs) |
| Spatial size | 96³ |
| lr | 2e-4 |

## Results

- **Best validation cycle loss**: 2.0578 @ epoch 94
- **Final validation cycle loss**: 2.0593
- **Discriminator loss stable**: D ≈ 0.0015 (pas de collapse)
- **Epoch 50**: LR decay débuté (lineaire jusqu'epoch 100)

## Domain statistics (exécuté)

- Domain A (artefacted, Noise≥1 OR Motion≥1): 27 sujets
- Domain B (clean, Noise=0 AND Motion=0): 27 sujets
- Validation: 5 sujets domaine A

## Potentiels problèmes identifiés

1. **SSIM négatif** (−0.0018) : Indique des différences importantes entre
   input (domaine A) et output G_AB. Attendu sans GT appariée — proxy
   métrique limité.

2. **FID/PSNR/LPIPS non calculables localement**: Réquiert le jeu de test
   du challenge (non disponible localement).

## Outputs

- Checkpoints: `outputs/checkpoints/RUN_0002/` (G_AB_best.pt, cyclegan_full_best.pt)
- Metrics: `results/runs/RUN_0002/metrics.json`
- Plots: `results/runs/RUN_0002/plots/`
  - training_curves.png
  - metrics_summary.png
  - evolution_30epochs.png

## Commands

```bash
# Training (Jean Zay)
sbatch src/slurm/sync_from_jeanzay.sh pull_run --run 0002

# Evaluation locale
python evaluate.py --run 0002
```
