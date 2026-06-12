# RUN_0005 — Multi-task nnU-Net-like Architecture

## Run Metadata
- **Run ID**: 0005
- **Date**: 2026-06-12
- **Status**: active (training in progress)
- **Parent Run**: RUN_0004
- **Tasks Covered**: 1a, 1b, 2
- **Change Scope**: architectural

## Parent Baseline
- **RUN_0004** (multi-task with 5-stage DynUNet):
  - Task 2 DSC: 0.6205
  - Task 1a aggregate: 0.3956
  - Task 1b PSNR: 21.40 dB
- **Key insight**: 5 stages sufficient, but auto-config from nnU-Net v2 suggests 6 stages with [32..320,320] filters.

## Hypothesis & Changes

### Motivation
nnU-Net v2 auto-configuration produced strong results on Task 2 (0.60+ pseudo-Dice by epoch 477). We hypothesise that adopting its architectural recipe (6 stages, deeper bottleneck, higher final channel count) within our multi-task framework will improve all three tasks — especially segmentation — without requiring the full nnU-Net training pipeline.

### Architectural Changes
| Aspect | RUN_0004 | RUN_0005 |
|--------|----------|----------|
| Stages | 5 (32,64,128,256,320) | 6 (32,64,128,256,320,320) |
| Strides | [(1,1,1),(2,2,2),(2,2,2),(2,2,2),(1,2,2)] | [(1,1,1),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)] |
| Optimiser | AdamW | AdamW |
| LR curve | 1e-4 → 5e-5 joint | 1e-4 → 5e-5 joint |
| Weight decay | 1e-5 | 3e-5 (nnU-Net default) |
| Epochs | 100 | 100 |

### Data & Augmentation
- Patch size: 128³
- Samples/volume: 2 (nnU-Net style; was 1 in 0004)
- 33% foreground oversampling (`oversample_foreground: 0.333`)
- Same spatial & intensity augmentation pipeline as 0004

### Training Strategy
- Warm-start: **None** (RUN_0004 warm-start incompatible due to stage mismatch).
- Warm-up: 50 epochs Task 2 only (same as 0004).
- Head warm-up: disabled (lesson from RUN_0004).
- Calibration: automatic loss-scale calibration at joint phase start.

### What We Keep from RUN_0004
- Classification head architecture (GAP → MLP)
- Reconstruction decoder (UpBlock3d stack)
- Focal-Dice hybrid loss for segmentation
- Multi-task loss weighting (λ=0.5/1.0/1.0)
- Evaluation bug fixes (`task_name` routing, `torch.no_grad()`)

## Assumptions
1. Deeper encoder (6 stages) improves feature representation for all tasks.
2. `(2,2,2)` stride at level 5 (instead of nnU-Net's `(1,2,2)`) is safe in DynUNet decoder without risk of shape mismatch under time pressure.
3. 3e-5 weight decay (nnU-Net default) generalises better than 1e-5.
4. 2 samples/volume + foreground oversampling improves foreground class learning.

## Risks & Mitigations
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| 6 stages OOM on GB10 | Medium | Batch size 1, no deep supervision |
| Shape mismatch in decoder | Low | Conservative all-(2,2,2) strides |
| Slower convergence (no warm-start) | High | 50-epoch warm-up, patient early stopping (20) |
| Augmentation too aggressive | Medium | Same as 0004 (proven stable) |

## Implementation Plan
- [x] Generalise `DynUNetMultiHeadModel` to N-stage (ModuleList recon_ups, dynamic cls_mlp)
- [x] Create `configs/run_0005_multitask.yaml`
- [x] Add `oversample_foreground` to transforms / loaders
- [x] Smoke test — passed
- [ ] Full training (local GB10 or Jean Zay 4×H100)
- [ ] Evaluation on validation set
- [ ] Compare with RUN_0004

## Training & Evaluation Configuration
- **Config**: `configs/run_0005_multitask.yaml`
- **Batch size**: 1
- **Patch size**: 128×128×128
- **Epochs**: 100
- **Warm-up**: 50 epochs
- **Early stopping patience**: 20
- **Checkpoint**: `outputs/checkpoints/RUN_0005/multitask_best.pt`

## Results Summary
- (training in progress)

## Comparability Statement
- **Architecture change** (5→6 stages) makes weight-level comparison impossible, but metrics-level comparison valid since all other hyperparameters, data splits, and augmentations are identical.
- **No warm-start** means convergence speed differs; final metrics are the primary comparison.

## Decision
- **Pending** — awaiting training completion.
