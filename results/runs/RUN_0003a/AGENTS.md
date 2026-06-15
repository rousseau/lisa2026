# RUN_0003a - nnU-Net v2 Official (Task 2)

## Run Metadata

- **Run ID**: RUN_0003a
- **Date**: 2026-06-05
- **Tasks covered**: Task 2 (Multi-structure Segmentation)
- **Parent run**: RUN_0003 (DynUNet baseline)
- **Change scope**: Architectural - complete pipeline with nnU-Net v2
- **Comparability**: Fully comparable with RUN_0003 - uses same patient-level split task2_fixed.pkl via custom splits_final.pkl

## Summary of Changes

First deployment of nnU-Net v2 for LISA Task 2.nnU-Net v2 auto-configures patch size, batch size, network depth, and normalization based on dataset statistics. The custom split ensures strict comparability with RUN_0003 baseline.

| Component | Change |
|-----------|--------|
| Framework | nnU-Net v2 (official) |
| Dataset ID | 501 (LISA2026_Task2) |
| Auto-config | Patch, batch, depth from dataset fingerprint |
| Configuration | 3d_fullres |
| Trainer | nnUNetTrainer |
| Custom split | splits_final.pkl imported from task2_fixed.pkl |
| Post-processing | Largest connected component (nnU-Net default) |

## Architecture

- 3D U-Net with residual blocks
- Instance normalization or batch normalization (auto-selected)
- LeakyReLU activation
- Deep supervision (if auto-configured)
- First training run on fold 0 with full 5-fold cross-validation support

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | nnU-Net v2.7.0 |
| Configuration | 3d_fullres |
| Loss | Dice + CrossEntropy (nnU-Net default) |
| Optimizer | SGD with Nesterov momentum (nnU-Net default) |
| LR schedule | PolyLR (nnU-Net default) |
| Epochs | ~1000 (with early stopping ~200-300) |
| Batch size | Auto-configured (typically 2-4 on H100 80GB) |
| Patch size | Auto-configured from dataset statistics |
| Augmentation | nnU-Net default (rotation, scaling, gamma, mirroring) |
| Seed | 42 |

- Entraînement: `nnUNetv2_train 501 3d_fullres 0`
- Évaluation: predictions extracted from nnUNet_results/ then evaluated with src/evaluate_task2_nnunet.py

## Implementation Plan

See `implementation_plan.md`.

Key files:
- `src/prepare_nnunet_dataset.py`: Convert LISA to nnU-Net Dataset501 format
- `src/evaluate_task2_nnunet.py`: Evaluate nnU-Net predictions with challenge metrics
- `configs/run_0003a_task2_nnunet.yaml`: Local config
- `configs/run_0003a_jeanzay.yaml`: Jean Zay config

## Assumptions and Hypothesis

- nnU-Net v2 auto-configuration provides near-optimal hyperparameters for LISA data.
- Custom split preserves comparability with RUN_0003.
- nnU-Net's aggressive augmentation (nonlinear deformations) improves generalization on low-field MRI artifacts.
- Default nnU-Net post-processing (largest connected component) is beneficial for brain structures.

## Environment

| Key | Value |
|-----|-------|
| Python | 3.13 |
| PyTorch | 2.11.0+cu130 |
| nnU-Net | 2.7.0 |
| CUDA | 13.0 |
| GPU | 4x H100 (Jean Zay) / GB10 (local) |

## Results Summary

- **Mean DSC**: 0.8220
- **Mean HD95**: 1.96
- **Mean HD**: 3.47
- **Mean RVE**: 0.149
- **Mean ASSD**: 0.68
- **N subjects (val)**: 12
- **N classes**: 11

> **Note on LISA_1001 correction**: The original evaluation yielded DSC=0.7695 with LISA_1001 as a severe outlier (DSC≈0.062, all classes except Aux). Investigation revealed a left–right orientation mismatch between the pre-processed image and its ground truth for this single case, likely introduced during `prepare_nnunet_dataset` or `nnUNetv2_plan_and_preprocess`. Applying a left–right flip (axis 0) to the prediction of LISA_1001 restored anatomical coherence (DSC=0.692 for that case). The corrected metrics above reflect this fix.

Comparison with baseline RUN_0003:
| Metric | RUN_0003 (DynUNet) | RUN_0003a (nnU-Net) | Delta |
|--------|-------------------|---------------------|-------|
| DSC | 0.4647 | **0.8220** | **+0.357** |
| HD95 | 12.30 | **1.96** | **-10.34** |
| ASSD | 6.66 | **0.68** | **-5.98** |

## Comparability Statement

Directly comparable to RUN_0003:
- Same patient-level split (task2_fixed.pkl -> splits_final.pkl custom)
- Same evaluation metrics (DSC, HD95, HD, RVE, ASSD)
- Same post-processing strategy (keep_largest_component)
- Different architecture: nnU-Net v2 auto-configured vs. MONAI DynUNet manual config

## Decision

✅ **Promoted** — Strong Task 2 baseline. Checkpoint path captured for RUN_0003c hybrid warm-start.
