# RUN_0003 — DynUNet 12-class Baseline (Task 2)

## Run Metadata

- **Run ID**: RUN_0003
- **Date**: 2026-05-14
- **Tasks covered**: Task 2 (Multi-structure Segmentation)
- **Parent run**: None (independent track — Task 2 is a new pipeline distinct from RUN_0001/RUN_0002)
- **Change scope**: Architectural — full Task 2 pipeline from scratch
- **Comparability**: Not comparable to RUN_0001 or RUN_0002 (different task, different metric family).
  RUN_0003 is the Task 2 reference baseline for all subsequent EXP_* runs.

## Summary of Changes

First dedicated Task 2 segmentation pipeline. Builds the complete stack: data loading
with MONAI transforms and patient-level fixed split, MONAI DynUNet with 12-class output,
Dice+Focal loss to handle class imbalance, sliding-window inference with TTA (flip axes
[2,3,4]) and largest-connected-component post-processing. Evaluation with challenge
metrics: DSC, HD95, HD, RVE, ASSD per class and globally.

## Full List of Changes vs. Parent

| Component           | Change                                                             |
|---------------------|--------------------------------------------------------------------|
| Task                | New: Task 2 multi-structure segmentation                           |
| Data pipeline       | New: MONAI transforms, patch 128³, sliding-window eval            |
| Split               | New: task2_fixed.pkl (patient-level, frozen)                       |
| Model               | New: MONAI DynUNet, out_channels=12, filters=[48,96,192,384,512]  |
| Loss                | New: DiceFocalLoss (λ_dice=1.0, λ_focal=1.0, γ=2.0)              |
| Optimizer           | New: AdamW, LR=2e-4, weight_decay=1e-5                            |
| Scheduler           | New: CosineAnnealingLR, T_max=120, eta_min=1e-6                   |
| Early stopping      | New: patience=25                                                   |
| Inference           | New: TTA flip axes [[2],[3],[4]], keep_largest=true, overlap=0.5   |
| Evaluation          | New: DSC, HD95, HD, RVE, ASSD per class + global mean             |
| Mixed precision     | New: AMP enabled                                                   |

## Assumptions and Hypothesis

- Task 2 labels in `*_LF_seg.nii.gz` are correctly aligned with `*_ciso.nii.gz`.
- 12 label classes: bg(0), L-Hippo(1), R-Hippo(2), L-Caudate(3), R-Caudate(4),
  L-Lentiform(5), R-Lentiform(6), L-Ventricle(7), R-Ventricle(8), L-ExV(9),
  R-ExV(10), Aux(11).
- DynUNet + DiceFocal is a competitive default for medical image segmentation.
- Focal loss (γ=2.0) helps with small/rare structures (hippocampus).
- TTA on flip axes improves inference robustness on symmetric anatomy.
- Patient-level split prevents leakage and enables reproducible comparison across runs.

## Training Configuration

See `config_snapshot.yaml` for full details.

| Parameter       | Value                         |
|-----------------|-------------------------------|
| Model           | MONAI DynUNet                 |
| out_channels    | 12                            |
| filters         | [48, 96, 192, 384, 512]       |
| norm_name       | instance                      |
| Loss            | dice_focal (λ_dice=1.0, λ_focal=1.0, γ=2.0) |
| LR              | 2e-4                          |
| Weight decay    | 1e-5                          |
| Epochs          | 120                           |
| Patience        | 25                            |
| Patch size      | [128, 128, 128]               |
| TTA axes        | [[2], [3], [4]]               |
| Keep largest    | true                          |
| Mixed precision | true                          |
| Seed            | 42                            |

- Train script: `train_task2_dynunet.py --config configs/run_0003_task2_dynunet.yaml`
- Eval script:  `evaluate_task2_dynunet.py --config configs/run_0003_task2_dynunet.yaml`
- Split file:   `results/splits/task2_fixed.pkl`

## Implementation Plan

See `implementation_plan.md`.

## Results Summary

Metrics computed on validation split (12 subjects, 11 non-background classes).

| Metric      | Value   |
|-------------|---------|
| mean_dsc    | 0.5003  |
| mean_hd95   | 23.17   |
| mean_hd     | 29.21   |
| mean_rve    | 0.3549  |
| mean_assd   | 6.14    |
| n_subjects  | 12      |
| n_classes_eval | 11   |

### Per-class DSC

| Class | Structure     | DSC   |
|-------|---------------|-------|
| 1     | L Hippocampus | 0.052 |
| 2     | R Hippocampus | 0.304 |
| 3     | L Caudate     | 0.517 |
| 4     | R Caudate     | 0.443 |
| 5     | L Lentiform   | 0.576 |
| 6     | R Lentiform   | 0.454 |
| 7     | L Ventricle   | 0.581 |
| 8     | R Ventricle   | 0.573 |
| 9     | L ExV         | 0.632 |
| 10    | R ExV         | 0.733 |
| 11    | Aux           | 0.639 |

**Key observations:**
- Hippocampus (L/R) is systematically the worst class (DSC < 0.31), motivating
  the collapsed-label strategy in RUN_0003_COLLAPSED.
- Large structures (ExV, Aux) perform significantly better (DSC > 0.63).

## Comparability Statement

RUN_0003 is not comparable to any Task 1a/1b run. Within Task 2, it is the **reference
baseline** for all EXP_* variants. All subsequent runs (EXP_A through EXP_SYM) share the
same fixed split, image/label suffixes, and evaluation script version.

## Decision

✅ **Valid baseline** — Metrics confirmed on validation split (12 subjects, 11 classes,
corrected evaluation script). Retained as the Task 2 reference baseline for the RUN_0003
experimental cycle.

**Promoted run from this cycle : `RUN_0003_EXP_C`** (mean DSC=0.6309, mean_HD95=9.58),
which trains from scratch with filters=[32,64,128,256,320] + DiceCE + LR=1e-4 and
achieves +0.131 DSC over this baseline. See `results/runs/RUN_0003_EXP_C/AGENTS.md`
and `results/runs/RUN_0003_EXP_C/analysis.md`.
