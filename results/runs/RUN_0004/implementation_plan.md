# RUN_0004 – Implementation Plan

## Goal

Establish a first Task 1b (image quality enhancement) baseline using a 3D BasicUNet trained in a fully self-supervised fashion. No paired clean/degraded data is assumed.

---

## Architecture

```
Task1bUNetModel
  └─ monai.networks.nets.BasicUNet
       spatial_dims : 3
       in_channels  : 1
       out_channels : 1
       features     : (16, 32, 64, 128, 256, 16)
       act          : PRELU
       norm         : INSTANCE
       dropout      : 0.0
       upsample     : deconv
```

Parameter count is intentionally kept modest (≈2M) to allow batch_size=2 at 96³ on a single GPU with AMP.

---

## Data pipeline

1. **Scan** `data_root` for `*_ciso.nii.gz` files (regex `LISA_\d+`).
2. **Split** by patient using `task1a_fixed.pkl` (≈80/20 train/val). The `_subjects_from_split` helper maps both index-level (Task1a) and subject-level (Task2) pickle formats.
3. **Preprocessing** (MONAI transforms, same at train and val):
   - `LoadImaged` (NiBabel reader)
   - `EnsureChannelFirstd`
   - `NormalizeIntensityd` (channel_wise, all voxels)
   - `CenterSpatialCropd` to 96³
   - `SpatialPadd` to 96³ (symmetric)
   - `EnsureTyped`

No geometric augmentations at this stage (added in future runs if needed).

---

## Training loop

```
for epoch in range(num_epochs):
    for clean_batch in train_loader:
        degraded = add_synthetic_noise(clean, std ~ U(0.05, 0.20))
        pred     = model(degraded)
        loss     = L1(pred, clean) + SSIM(pred, clean)
        backward()
```

- Validation: fixed noise std = 0.125 (mean of range) for reproducibility.
- Best checkpoint saved on minimum `val_loss`.
- Training history written to `results/runs/RUN_0004/training_history.json`.

---

## Evaluation protocol

For each validation volume:
1. Add Gaussian noise (std=0.125) → degraded input.
2. Run model forward pass.
3. Normalise pred and clean to [0,1] using the clean volume's global min/max.
4. Compute **PSNR** (3D volumetric).
5. Extract 10 axial slices from the central 50% of the volume.
6. For each slice, compute **LPIPS** (VGG, inputs scaled to [−1,1]).
7. Collect all slice pairs → compute **FID** via InceptionV3 pool features (2048-dim) + Fréchet distance.

---

## Key implementation files

| File | Role |
|---|---|
| `src/models/__init__.py` | `Task1bUNetModel` class |
| `src/datasets/__init__.py` | `Task1bDataset`, `get_task1b_dataloaders` |
| `train_task1b.py` | `Task1bTrainer`, `CombinedL1SSIMLoss`, `add_synthetic_noise` |
| `evaluate_task1b.py` | `compute_psnr`, `compute_fid`, `extract_inception_features` |
| `configs/run_0004_task1b_unet.yaml` | Full hyperparameter config |
| `scripts/run_0004.sh` | End-to-end launcher |

---

## Expected timeline

| Step | Estimated duration (single A100) |
|---|---|
| Smoke test | < 5 min |
| Full training (80 epochs, ~N volumes) | 2–4 h |
| Evaluation | < 15 min |

---

## Future improvements (not in this run)

- Rician noise simulation (in addition to Gaussian).
- Perceptual loss (VGG features) during training.
- Test-time augmentation (TTA) by flipping.
- Real paired data fine-tuning if provided by the challenge.
- Larger crop (128³) with batch_size=1 for higher-resolution features.
