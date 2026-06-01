# RUN_0002 — Implementation Plan

## Task

Task 1b: unpaired artifact removal for 0.064T low-field MRI.
Target artifacts: Gaussian noise and motion artifacts.

## Architecture choice: 3D CycleGAN

### Rationale

- No paired GT available → unpaired adversarial training required.
- CycleGAN is the standard baseline for unpaired image-to-image translation.
- 3D architecture to preserve volumetric coherence of MRI data.
- LSGAN loss for training stability (vs. original GAN BCE).

### Generator (G_AB and G_BA)

```
Input [1, 96, 96, 96]
  → ReplicationPad3d(3) + Conv3d(1→32, k=7) + InstanceNorm + ReLU      # enc0
  → Conv3d(32→64, k=3, s=2) + InstanceNorm + ReLU                      # enc1
  → Conv3d(64→128, k=3, s=2) + InstanceNorm + ReLU                     # enc2
  → Conv3d(128→256, k=3, s=2) + InstanceNorm + ReLU                    # enc3
  → 6 × ResBlock3D(256)                                                  # bottleneck
  → ConvTranspose3d(512→128, k=3, s=2) + skip(enc3) + InstanceNorm + ReLU  # dec3
  → ConvTranspose3d(256→64, k=3, s=2) + skip(enc2) + InstanceNorm + ReLU   # dec2
  → ConvTranspose3d(128→32, k=3, s=2) + skip(enc1) + InstanceNorm + ReLU   # dec1
  → ReplicationPad3d(3) + Conv3d(64→1, k=7) + skip(enc0) + Tanh             # output
Output [1, 96, 96, 96]
```

### Discriminator (D_A and D_B)

```
PatchGAN 3D — 4 strided Conv3d layers (64→128→256→512→1)
Output: patch-level real/fake predictions (no sigmoid, LSGAN)
```

## Domain partition

From Task 1a CSV (LISA_Task1a_2026.csv):

- **Domain A**: rows where `Noise >= 1` OR `Motion >= 1`
- **Domain B**: rows where `Noise == 0` AND `Motion == 0`

Subject-level split from `task2_fixed.pkl` (train_subjects / val_subjects).
Volumes: `{data_root}/{subject}/{subject}_ciso.nii.gz`

## Training schedule

- Epochs 1–50: constant LR = 2e-4
- Epochs 51–100: linear LR decay from 2e-4 to 0
- Checkpoint: best by minimum validation cycle-consistency loss
- Saved artefacts: `G_AB_best.pt`, `cyclegan_full_best.pt`

## Evaluation

Since no GT is available:
1. Apply G_AB to all domain A validation images → fake_B
2. Apply G_BA to fake_B → rec_A (cycle reconstruction)
3. Compute PSNR / SSIM between real_A and rec_A (cycle self-consistency)
4. For challenge submission: submit generated images to official test set
   → FID / PSNR / LPIPS computed by challenge organisers

## Files created

| File | Role |
|------|------|
| `src/models/task1b.py` | Generator3D, Discriminator3D |
| `src/datasets/task1b.py` | Task1bCycleGANDataset |
| `src/training/task1b.py` | CycleGANTrainer |
| `src/train_task1b.py` | Training entry point |
| `src/evaluate_task1b.py` | Evaluation entry point |
| `configs/run_0002_cyclegan_task1b.yaml` | Config |
