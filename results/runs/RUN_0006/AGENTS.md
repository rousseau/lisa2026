# RUN_0006 — Multi-task avec Warm-start RUN_0004 (option B)

## Run Metadata
- **Run ID**: 0006
- **Date**: 2026-06-16
- **Status**: en préparation (smoke test OK)
- **Parent Run**: RUN_0005
- **Warm-start Parent**: RUN_0004
- **Tasks Covered**: 1a, 1b, 2
- **Change Scope**: hyperparameters / warm-start / post-processing

## Parent Baseline
- **RUN_0005 v2** (rejected):
  - Task 2 DSC: 0.3523 (failed due to no warm-start + lambda_1a=0.5 + no post-processing)
  - Task 1a aggregate: 0.4887
  - Task 1b PSNR: 25.90 dB
  - Root cause: Task 2 encoder trained from scratch with conflicting gradients, Task 1a dominated joint loss

- **RUN_0004** (warm-start source):
  - Task 2 DSC: 0.6205 (with warm-start from RUN_0003)
  - Architecture: DynUNetMultiHead 5 stages [32..320]
  - Checkpoint: `outputs/checkpoints/RUN_0004/multitask_best.pt`

## Hypothesis & Changes

### Hypothesis
RUN_0005 failed because:
1. **No warm-start** on Task 2 encoder → encoder learned from scratch with 3 conflicting task gradients
2. **lambda_1a=0.5** dominated joint loss (Task 1a loss ~4.5, scaled contribution ~0.4 → 74% of gradient by epoch 750)
3. **No post-processing** (keep_largest_component=false) → predictions contain spurious connected components

RUN_0004 succeeded with DSC=0.62 because its encoder was **warm-started from RUN_0003** (Task 2 only), so it was already locked into anatomical features before the joint phase.

### Changes (RUN_0006)
| Aspect | RUN_0005 v2 | **RUN_0006** |
|--------|-------------|----------------|
| Warm-start | None | **RUN_0004 encoder (5 stages → 6)** |
| Head initialisation | Inherited from previous training | **Reset all 3 heads to random** (option B) |
| lambda_1a | 0.5 | **0.1** |
| lambda_1b | 1.0 | 1.0 |
| lambda_2 | 1.0 | 1.0 |
| Post-processing | false | **true (`keep_largest_component`)** |
| Optimiser | SGD + Nesterov | SGD + Nesterov |
| LR schedule | PolyLR | PolyLR |
| Epochs | 1000 | 1000 |
| Patience | 200 | 200 |
| Deep supervision | Yes (5 levels) | Yes (5 levels) |

### Data & Augmentation
- Identical to RUN_0005 v2
- Patch size: 128³, batch size: 1

### Training Strategy
- Warm-start: **Encoder only** (model.* keys) from RUN_0004 checkpoint.
  - Expected ~30-40 keys loaded (5 stages compatible out of 6).
  - 6th downsample and 5th upsample are new, random-initialised.
- All 3 task heads **reset to random** after warm-start (`model.reset_heads()`).
- Warm-up: 50 epochs Task 2 only (same as RUN_0005).
- Joint: all 3 tasks with recalibrated losses.

## Implementation Plan
- [x] Create `configs/run_0006_multitask.yaml`
- [x] Create `configs/run_0006_jeanzay.yaml`
- [x] Implement `reset_heads()` in `DynUNetMultiHeadModel`
- [x] Implement `_load_pretrained()` with `pretrained_load_mode: "encoder_only"`
- [x] Smoke test passed locally (31/101 keys loaded, heads reset, lambda_1a=0.1)
- [ ] Full training on Jean Zay H100
- [ ] Evaluation on validation set
- [ ] Compare with RUN_0005

## Training & Evaluation Configuration
- **Config local**: `configs/run_0006_multitask.yaml`
- **Config Jean Zay**: `configs/run_0006_jeanzay.yaml`
- **Batch size**: 1
- **Patch size**: 128×128×128
- **Epochs**: 1000
- **Warm-up**: 50 epochs
- **Early stopping patience**: 200
- **Checkpoint**: `outputs/checkpoints/RUN_0006/multitask_best.pt`
- **Hardware**: 1× H100 80GB (Jean Zay)

## Expected Outcomes
| Metric | RUN_0005 v2 | **RUN_0006 target** | RUN_0004 |
|--------|-------------|---------------------|----------|
| Task 2 DSC | 0.3523 | **≥ 0.55** | 0.6205 |
| Task 1a aggregate | 0.4887 | ~0.40-0.50 | 0.3956 |
| Task 1b PSNR | 25.90 dB | ~23-26 dB | 21.40 dB |

## Comparability Statement
- **Not directly comparable to RUN_0005** in interface because `lambda_1a` changed (0.5→0.1) and warm-start introduced. Metrics-level comparison still informative for Task 2 improvement.
- **Comparable to RUN_0004** for Task 2 DSC since encoder warm-started from the same architecture baseline (RUN_0003 → RUN_0004 encoder).

## Decision
- [ ] Promoted
- [ ] Rejected
- [ ] To retest
- [x] **In progress** — awaiting Jean Zay training

---

## Smoke Test (local, NVIDIA GB10)

```
[INFO] Partial warm-start from outputs/checkpoints/RUN_0004/multitask_best.pt: 31/101 keys.
[INFO] All task heads reset to random initialisation.
Device: cuda | AMP: True
Phases — warmup=1ep  head_warmup=0ep  joint=1ep | λ=(1a=0.1, 1b=1.0, 2=1.0) grad_clip=12.0
[Warmup] 001/001 | loss=2.9926 | val_dice_2=0.0005
[Joint] 001/001 (g=002) | total=1.9364 1a=5.5631 1b=0.5645 2=2.8791 | val_dice_2=0.0005
```

- 31 encoder keys loaded (input_block, downsamples[0-3], bottleneck).
- 6th stage blocks (downsample[4], upsample[4]) new, random-init.
- All heads reset to random.
- No NaN, no crash. OK to submit.
