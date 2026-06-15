# RUN_0005 — Multi-task nnU-Net-like Architecture (v2 revised 2026-06-15)

## Run Metadata
- **Run ID**: 0005
- **Date**: 2026-06-12 (v1), 2026-06-15 (v2 revision)
- **Status**: active (training in progress, v2 relaunch on Jean Zay)
- **Parent Run**: RUN_0004
- **Tasks Covered**: 1a, 1b, 2
- **Change Scope**: architectural / hyperparameters

## Parent Baseline
- **RUN_0004** (multi-task with 5-stage DynUNet):
  - Task 2 DSC: 0.6205
  - Task 1a aggregate: 0.3956
  - Task 1b PSNR: 21.40 dB
- **RUN_0003a** (nnU-Net v2 Task 2 only):
  - Task 2 DSC: 0.8220 (after LISA_1001 correction)
  - Key lesson: SGD + PolyLR, deep supervision, 1000 epochs, 6 stages identical to RUN_0005

## Hypothesis & Changes (v2 Revision)

### Motivation
RUN_0005 v1 failed due to:
1. **AdamW + 100 epochs insufficient** for 6-stage convergence from scratch (DSC plateau at ~0.29).
2. **No deep supervision** (gradient vanishing on deep decoder).
3. **No class-weighted T1a** (severe class imbalance).

Learning from nnU-Net (RUN_0003a), we apply its proven training recipe while keeping the multi-task heads.

### Architectural Changes
| Aspect | RUN_0004 | RUN_0005 v1 | RUN_0005 v2 |
|--------|----------|-------------|---------------|
| Stages | 5 (32,64,128,256,320) | 6 (32..320,320) | 6 (32..320,320) |
| Deep supervision | No | No | **Yes (5 levels)** |
| Optimiser | AdamW | AdamW | **SGD + Nesterov** |
| LR schedule | CosineAnnealingLR | CosineAnnealingLR | **PolyLR (power=0.9)** |
| Initial LR | 1e-4 | 1e-4 | **1e-2** |
| Epochs | 100 | 100 | **1000** |
| Patience | 15 | 20 | **200** |
| Class-weighted T1a | No | No | **Yes** |

### Data & Augmentation
- Patch size: 128³
- Samples/volume: 2
- 33% foreground oversampling
- Same spatial & intensity augmentation as 0004

### Training Strategy
- Warm-start: **None** (5→6 stage mismatch with RUN_0004).
- Warm-up: 50 epochs Task 2 only.
- Head warm-up: disabled.
- Calibration: automatic loss-scale calibration at joint phase start.

## Implementation Plan (v2)
- [x] Generalise `DynUNetMultiHeadModel` to N-stage (already done v1)
- [x] Enable `deep_supervision=True` in MONAI `DynUNet`
- [x] Implement `PolyLR` scheduler matching nnU-Net recipe
- [x] Implement SGDOptimizer + momentum 0.99 + Nesterov
- [x] Implement `_seg_loss_with_ds` handling MONAI 6-D deep-supervision tensor
- [x] Add `forward_task2_main` for inference (extracts level 0 from 6-D tensor)
- [x] Create `configs/run_0005_jeanzay.yaml` for Jean Zay single-GPU
- [x] Adapt SLURM script: 1×H100, 24h max
- [x] Smoke test passed locally
- [ ] Full training on Jean Zay H100
- [ ] Evaluation on validation set
- [ ] Compare with RUN_0004

## Training & Evaluation Configuration
- **Config local**: `configs/run_0005_multitask.yaml`
- **Config Jean Zay**: `configs/run_0005_jeanzay.yaml`
- **Batch size**: 1
- **Patch size**: 128×128×128
- **Epochs**: 1000
- **Warm-up**: 50 epochs
- **Early stopping patience**: 200
- **Checkpoint**: `outputs/checkpoints/RUN_0005/multitask_best.pt`
- **Hardware**: 1× H100 80GB (Jean Zay, no DDP)

## Results Summary
- v1: Task 2 DSC=0.2946 (failed — under-trained)
- v2: Pending Jean Zay execution

## Comparability Statement
- **Architecture change** (5→6 stages) makes weight-level comparison impossible, but metrics-level comparison valid since all other hyperparameters, data splits, and augmentations are identical.
- **No warm-start** means convergence speed differs; final metrics are the primary comparison.
- v2 uses nnU-Net training recipe learned from RUN_0003a, making the comparison RUN_0005 v2 vs RUN_0003a particularly informative for multi-task impact on segmentation.

## Decision
- **Pending v2 evaluation** — awaiting Jean Zay training completion.
