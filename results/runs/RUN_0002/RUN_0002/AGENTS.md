# RUN_0002 — AGENTS.md

## Run metadata

- **Run ID**: RUN_0002
- **Date**: 2026-05-29
- **Tasks covered**: 1b (artifact removal / image enhancement)
- **Parent run**: None (first Task 1b baseline)
- **Change scope**: New task — architectural baseline

---

## Objective

Establish a Task 1b baseline for unpaired artifact removal on 0.064T low-field
MRI images. No paired ground truth is available in the challenge dataset.

---

## Full list of changes

- Created `src/models/task1b.py` — `Generator3D` (3D U-Net with residual
  bottleneck) + `Discriminator3D` (3D PatchGAN, LSGAN).
- Created `src/datasets/task1b.py` — `Task1bCycleGANDataset` partitioning
  volumes into domain A (artefacted: Noise≥1 OR Motion≥1) and domain B
  (clean: Noise=0 AND Motion=0) using Task 1a CSV labels.
- Created `src/training/task1b.py` — `CycleGANTrainer` with adversarial
  (LSGAN), cycle-consistency (L1 λ=10), and identity (L1 λ=5) losses.
- Created `src/train_task1b.py` and `src/evaluate_task1b.py` entry points.
- Created `configs/run_0002_cyclegan_task1b.yaml`.
- Updated `train.py` and `evaluate.py` registries.
- Removed obsolete RUN_0002 (Task 1a multi-label, rejected) and all associated
  code (`Task1aMultiLabelModel`, `Task1aMultiLabelTrainer`,
  `get_multilabel_dataloaders`, `src/train_task1a_multilabel.py`,
  `src/evaluate_task1a_multilabel.py`).
- Removed ghost RUN_0005 entries (BasicUNet self-supervised, never formalised).

---

## Assumptions and hypotheses

1. Domains A and B can be defined from Task 1a ordinal labels (threshold ≥ 1
   for Noise and Motion). This is an approximation — severity-0 images are
   not truly artifact-free, only low-artifact.
2. CycleGAN can learn a meaningful unpaired translation between the two domains
   given sufficient data diversity.
3. Patches of 96³ voxels are sufficient to capture relevant artifact patterns
   while fitting in GPU memory with batch_size=1.
4. 100 training epochs (50 constant LR + 50 linear decay) is a standard
   CycleGAN schedule sufficient for convergence on 3D medical data.
5. LSGAN (MSE adversarial loss) is more stable than the original GAN (BCE)
   for 3D volumes.

---

## Training configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Generator3D (U-Net, 4 down/up, 6 ResBlocks, filters 32/64/128/256) |
| Discriminator | PatchGAN3D (4 layers, 64 base filters) |
| Patch size | 96³ |
| Batch size | 1 |
| Epochs | 100 (50 constant + 50 linear decay) |
| Optimizer | Adam (lr=2e-4, β=(0.5, 0.999)) for G and D separately |
| lambda_cycle | 10.0 |
| lambda_identity | 5.0 |
| Adversarial loss | LSGAN (MSE) |
| Image buffer | 50 (discriminator stabilisation) |
| Domain A threshold | Noise ≥ 1 OR Motion ≥ 1 |
| Domain B threshold | Noise = 0 AND Motion = 0 |
| Split | Task 2 subject-level split (task2_fixed.pkl) |

---

## Evaluation configuration

Metrics computed on validation domain A (artefacted images):
- PSNR and SSIM between input and G_AB output (proxy, no GT).
- Official metrics (FID, PSNR, LPIPS) require challenge test set.

---

## Implementation plan

See `implementation_plan.md`.

---

## Results summary

| Metric | Value |
|--------|-------|
| FID | — (pending) |
| PSNR | — (pending) |
| LPIPS | — (pending) |

---

## Comparability statement

This run is the first Task 1b baseline. No prior run to compare with.
RUN_0004 includes a reconstruction head (L1+SSIM autoencoder) but it is
not directly comparable: different architecture, different objective
(denoising vs. unpaired translation), and RUN_0004 multi-task training
is unstable.

---

## Decision

⏳ Pending — training not yet executed.
