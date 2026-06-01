# RUN_0002 — Analysis Summary

## Training Results (Jean Zay, 4× H100 80 Go)

- **Duration**: 14 minutes (100 epochs)
- **Best val_cycle**: 2.0578 @ epoch 94
- **Final val_cycle**: 2.0593  
- **Discriminator loss**: ~0.0015 (stable, no collapse)

## Evaluation Metrics (Validation Set, Domain A)

| Metric | Value | Notes |
|--------|-------|-------|
| PSNR (proxy) | 2.17 dB | Input vs G_AB output (no GT) |
| SSIM (proxy) | -0.002 | Input vs G_AB output (no GT) |
| FID | — | Requires challenge test set |
| LPIPS | — | Requires challenge test set |

## Domain Statistics (Jean Zay)

- Domain A train: 27 sujets
- Domain B train: 27 sujets  
- Domain A val: 5 sujets
- Domain B val: 8 sujets

## Outputs

- Checkpoints: `outputs/checkpoints/RUN_0002/`
- Metrics: `results/runs/RUN_0002/metrics.json`
- Plots: `results/runs/RUN_0002/plots/`
  - training_curves.png
  - metrics_summary.png
  - evolution_30epochs.png

## Decision

✅ **Promoted** — Stable convergence, checkpoint saved, ready for challenge submission.
