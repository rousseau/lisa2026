# RUN_0004 – AGENTS

- **Run ID**: RUN_0004
- **Date**: 2026-05-15 (planned)
- **Tasks covered**: Task 1b (image quality enhancement / denoising)
- **Parent run**: None (first Task 1b run; independent track from RUN_0001–0003)
- **Change scope**: Architectural baseline

---

## Full list of changes

- Added `Task1bUNetModel` to `src/models/__init__.py`: 3D MONAI `BasicUNet` with PRELU activations, instance normalisation, and transposed-convolution upsampling (features: 16–32–64–128–256).
- Added `Task1bDataset`, `build_task1b_records`, `_subjects_from_split`, and `get_task1b_dataloaders` to `src/datasets/__init__.py`.  The dataset supports both subject-level (Task2 format) and index-level (Task1a format) split pickles.
- Created `train_task1b.py`: self-supervised denoising trainer with L1+SSIM loss, AdamW, cosine-annealing LR, AMP, and early stopping.
- Created `evaluate_task1b.py`: evaluator computing 3D PSNR, per-slice LPIPS (VGG), and FID (InceptionV3 pool features on axial slices).
- Created `configs/run_0004_task1b_unet.yaml`: full run configuration.
- Created `scripts/run_0004.sh`: launcher script with environment setup, directory creation, split verification, training, and evaluation steps.

---

## Assumptions and hypothesis

- No paired (degraded, clean) acquisitions are assumed to be available; the model is trained in a fully self-supervised regime.
- Adding synthetic Gaussian noise (std ∈ [0.05, 0.20]) during training teaches the network to suppress real 0.064T noise patterns by exploiting spatial structure in the volumes.
- A 3D U-Net with global context (encoder depth of 4) is sufficient to capture the dominant noise/artifact patterns of low-field MRI at 96³ resolution.
- `_ciso.nii.gz` isotropic volumes are used as both input and reconstruction target; the same preprocessing pipeline as Task 2 is applied (NormalizeIntensity + CenterSpatialCrop + SpatialPad at 96³).
- Using a compact feature map (16→256) limits GPU memory usage and allows batch size of 2 on a single 16 GB GPU.

---

## Training and evaluation configuration

| Parameter | Value |
|---|---|
| Config file | `configs/run_0004_task1b_unet.yaml` |
| Train script | `train_task1b.py` |
| Eval script | `evaluate_task1b.py` |
| Split file | `results/splits/task1a_fixed.pkl` |
| Model | 3D BasicUNet (MONAI), features [16,32,64,128,256,16] |
| Loss | L1 (w=1.0) + SSIM (w=1.0) |
| Optimizer | AdamW, lr=1e-4, wd=1e-5 |
| Scheduler | CosineAnnealingLR, T_max=80, eta_min=1e-6 |
| Spatial size | 96³ |
| Batch size | 2 |
| Max epochs | 80 |
| Early stopping | patience=15 (monitored on val reconstruction loss) |
| Noise std range | [0.05, 0.20] uniform |
| Val noise std | 0.125 (fixed mid-range) |
| AMP | True |
| Seed | 42 |

---

## Implementation plan

See `implementation_plan.md`.

---

## Results summary

| Metric | Value |
|---|---|
| PSNR | pending |
| LPIPS | pending |
| FID | pending |

Status: **not_run** – training and evaluation have not been executed yet.

---

## Comparability statement

RUN_0004 is not directly comparable to RUN_0001–RUN_0003: it targets Task 1b (image enhancement) which uses an entirely different objective, data pipeline, and metric family (PSNR/LPIPS/FID vs classification accuracy or DSC).  Within the Task 1b track, all future runs should use the same split file, spatial size, and evaluation script version to remain comparable with RUN_0004.

---

## Decision

- Status: **planned**
- Promotion: pending execution and metric review
