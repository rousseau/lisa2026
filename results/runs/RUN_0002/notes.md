# RUN_0002 — Notes (v2 corrected)

## Status

✅ **Terminé** — v2 retrained locally with corrected normalization.

## Training configuration (v2 local)

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (local) |
| Epochs | 100 |
| Training time | ~7 hours |
| Batch size | 1 |
| Spatial size | 96³ |
| lr | 2e-4 |

## Results v2 (corrected normalization)

| Metric | v2 (corrected) | v1 (buggy, Jean Zay) |
|--------|-----------------|----------------------|
| Best val_cycle | **0.1594** @ epoch 94 | 2.0578 |
| Final val_cycle | **0.1677** | 2.0593 |
| D loss | **~0.36** (active) | ~0.0015 (inert) |
| PSNR proxy | **27.51 dB** | 2.17 dB ❌ |
| SSIM proxy | **0.958** | -0.002 ❌ |

## Bug fix

**Critical bug (2026-06-02)**: `NormalizeIntensityd` output z-score, but Generator uses `Tanh` → range mismatch caused anatomical erasure.

**Fix**: Added `ScaleIntensityd(minv=-1.0, maxv=1.0)` in `src/datasets/task1b.py`. Retrained v2 locally.

## Visual proofs

All categories generated:
- `results/runs/RUN_0002/plots/visuals/visual_proofs_clean_*.png`
- `results/runs/RUN_0002/plots/visuals/visual_proofs_motion_*.png`
- `results/runs/RUN_0002/plots/visuals/visual_proofs_noise_*.png`

## Outputs

- Checkpoints: `outputs/checkpoints/RUN_0002/` (G_AB_best.pt, cyclegan_full_best.pt)
- Metrics: `results/runs/RUN_0002/metrics.json`
- Plots: `results/runs/RUN_0002/plots/`
- Visual proofs: `results/runs/RUN_0002/plots/visuals/`

## Commands

```bash
# Training (local, corrected)
python train.py --run 0002

# Evaluation
python evaluate.py --run 0002

# Visualisation
python src/analysis/visualize_run0002.py
```
