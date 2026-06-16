# RUN_0005 — Multi-task nnU-Net-like Architecture (v2 revised 2026-06-15, evaluated 2026-06-16)

## Run Metadata
- **Run ID**: 0005
- **Date**: 2026-06-12 (v1), 2026-06-15 (v2 training), 2026-06-16 (v2 evaluation)
- **Status**: completed
- **Parent Run**: RUN_0004
- **Tasks Covered**: 1a, 1b, 2
- **Change Scope**: architectural / hyperparameters
- **Execution**: Job 424170, Jean Zay jzxh052, NVIDIA H100 80GB
- **Training duration**: 973 min (~16 h)

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
- [x] Full training on Jean Zay H100 (job 424170, 973 min)
- [x] Evaluation on validation set (2026-06-16)
- [x] Compare with RUN_0004

## Training Execution (Jean Zay H100, job 424170)

### Training Dynamics
- **Total epochs**: 764 (warmup 30 + joint 734)
- **Early stopping**: patience=200, triggered at epoch 784
- **Best `val_dice_2`**: **0.3523** at epoch **584** (joint phase)
- **Aucun NaN** throughout training (numerical stability OK)
- **Severe overfitting**: `train_loss_2` keeps dropping to ~0.058 while `val_dice_2` collapses after epoch 584 and never recovers above 0.29
- **Missing warm-start**: no pretrained encoder from RUN_0004 (5→6 stage mismatch)

### Training Phase Breakdown

| Phase | Epochs | `val_dice_2` Evolution |
|-------|--------|------------------------|
| Warmup | 1-30 | 0.0000 → **0.1309** (early exit at 30, threshold 0.10 reached) |
| Joint (early) | 51-100 | 0.09 → **0.29** (rapid rise, comparable to RUN_0004 early phase) |
| Joint (mid) | 250-400 | 0.17 → **0.33** (local peaks, strong oscillations) |
| Joint (peak) | **584** | **0.3523** (global maximum, checkpoint saved) |
| Joint (decline) | 585-784 | Drop to ~0.22, then stagnation 0.18-0.23 |
| Last epoch | **784** | **0.2194** (early stopped after 200 epochs without improvement) |

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

## Results Summary (v2 evaluated 2026-06-16)

### Final Metrics (checkpoint epoch 584, val_dice_2=0.3523)

| Task | Metric | RUN_0005 v1 | **RUN_0005 v2** | RUN_0004 (parent) | Δ v2 vs v1 | Δ v2 vs 0004 |
|------|--------|-------------|-----------------|-------------------|------------|--------------|
| **Task 1a** | Aggregate | 0.3956 | **0.4887** ✅ | 0.3956 | **+23.5%** | **+23.5%** |
| | Accuracy | 0.7798 | **0.7664** | 0.7798 | -1.7% | -1.7% |
| | Recall macro | 0.3333 (constant) | **0.4225** | 0.3333 (partially constant) | **+26.8%** | **+26.8%** |
| **Task 1b** | PSNR | 21.61 dB | **25.90 dB** ✅ | 21.40 dB | **+19.9%** | **+21.0%** |
| | LPIPS | 0.3479 | **0.0693** ✅ | 0.3992 | **-80.1%** | **-82.6%** |
| | FID | 68.70 | **27.32** ✅ | 52.01 | **-60.2%** | **-47.5%** |
| | L1 | 0.0590 | **0.1299** | 0.5570 | +120% | **-76.7%** |
| **Task 2** | Mean DSC | 0.2946 | **0.3523** ✅ | **0.6205** | **+19.6%** | **-43.2%** ❌ |
| | Mean HD95 | 28.79 | **14.13** ✅ | 12.31 | **-51.0%** | +14.8% |
| | Mean ASSD | 14.08 | **4.70** ✅ | 2.78 | **-66.6%** | +69.1% |
| | N subjects | 12 | 12 | 12 | — | — |

### Per-class Task 2 DSC (v2)

| Class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|-------|---|---|---|---|---|---|---|---|---|----|----|
| DSC | ~0 | ~0 | 0.515 | **0.685** | 0.151 | 0.266 | 0.399 | 0.365 | 0.438 | **0.638** | 0.419 |

**Observations**:
- Classes 1 and 2 completely failed (DSC ≈ 0, HD95 = NaN for class 2)
- Classes 4 and 10 performed best (>0.60 DSC)
- Overall Task 2 remains far below RUN_0004 (0.6205) and RUN_0003a (0.8220)

### Task 1a Per-artifact Aggregate (v2)

| Artifact | Noise | Zipper | Positioning | Banding | Motion | Contrast | Distortion |
|----------|-------|--------|-------------|---------|--------|----------|------------|
| Aggregate | **0.648** | 0.554 | 0.430 | 0.457 | 0.354 | **0.453** | **0.525** |

- Noise and Distortion highest; Motion and Positioning lowest.
- Strong improvement vs v1 on Noise (+0.227) and Zipper (+0.186) thanks class-weighted CE.

## Comparability Statement
- **Architecture change** (5→6 stages) makes weight-level comparison impossible, but metrics-level comparison valid since all other hyperparameters, data splits, and augmentations are identical.
- **No warm-start** means convergence speed differs; final metrics are the primary comparison.
- v2 uses nnU-Net training recipe learned from RUN_0003a, making the comparison RUN_0005 v2 vs RUN_0003a particularly informative for multi-task impact on segmentation.
- **Task 2 still underperforms** despite nnU-Net recipe: the multi-task setting (shared encoder + 3 heads) appears to degrade segmentation vs mono-task nnU-Net. This confirms H5 (gradient conflict between tasks).

## Decision
- [ ] Promoted
- [x] **Rejected**
- [ ] To retest

**Decision : REJECTED**

### Why rejected
1. **Task 2 DSC = 0.3523** is **-43.2%** below parent RUN_0004 (0.6205) and **-57.1%** below mono-task nnU-Net RUN_0003a (0.8220). The multi-task shared encoder hypothesis (H1) is **not validated** for segmentation.
2. **Severe overfitting** after epoch 584: validation DSC never recovers despite continued training loss decrease. This suggests the model memorizes training data but fails to generalize.
3. **Classes 1 and 2 completely failed** in Task 2 (DSC ≈ 0), whereas RUN_0004 achieved non-zero DSC on all classes.

### What worked
- **Task 1a improved significantly**: aggregate **0.4887** (+23.5% vs RUN_0004) thanks to class-weighted CE and longer training.
- **Task 1b improved dramatically**: PSNR **25.90 dB** (+4.5 dB vs RUN_0004), LPIPS **0.0693** (-82.6%), FID **27.32** (-47.5%). The reconstruction head benefits from the nnU-Net recipe.

### Lessons for future runs
- The multi-task setup **hurts Task 2** badly. Consider:
  1. **Task-specific branches with limited sharing** (early layers shared, late layers task-specific)
  2. **Gradient surgery** (PCGrad, GradNorm) to reduce task conflict
  3. **Sequential training**: pre-train Task 2 encoder separately, then freeze and add Task 1a/1b heads
  4. **Reduce Task 1a/1b loss weights** (`lambda_1a` currently 0.5, try 0.1 or 0.05)
- The lack of **warm-start** for the 6-stage encoder is a major handicap. Future 6-stage multi-task runs should warm-start from a converged 6-stage Task 2 checkpoint (e.g., RUN_0003a).
