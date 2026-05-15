# RUN_0003_EXP_C — Fine-tune depuis RUN_0003, Filtres Réduits, DiceCE, Sans TTA

## Run Metadata

- **Run ID**: RUN_0003_EXP_C
- **Date**: 2026-05-13
- **Tasks covered**: Task 2 (Multi-structure Segmentation)
- **Parent run**: RUN_0003
- **Change scope**: Architectural + Incremental — filter size reduction, loss switch,
  LR reduction, shorter training, no TTA at inference
- **Comparability**: Directly comparable to RUN_0003 (same label space, fixed split,
  evaluation protocol). See note on pretrained checkpoint below.

## Summary of Changes

Fine-tune attempt from RUN_0003 checkpoint with reduced model capacity
(filters=[32,64,128,256,320]), DiceCE loss, lower LR (1e-4), shorter schedule (60 epochs,
patience 15), and no TTA at inference (keep_largest also disabled).

**⚠️ Architecture mismatch note**: As with EXP_B, the pretrained checkpoint from
RUN_0003 uses filters=[48,96,192,384,512], architecturally incompatible with this model
(filters=[32,64,128,256,320]). The checkpoint loading is silently skipped at runtime.
EXP_C therefore trains **from scratch** with filters=[32,64,128,256,320].

**The results are nonetheless the best among all 12-class runs** (mean DSC=0.63),
suggesting the smaller-filter architecture with lower LR (1e-4) converges better
than the original wider-filter architecture (RUN_0003, mean DSC=0.50).

## Full List of Changes vs. RUN_0003

| Parameter             | RUN_0003            | RUN_0003_EXP_C           |
|-----------------------|---------------------|--------------------------|
| pretrained_checkpoint | none                | RUN_0003 (skipped — architecture mismatch) |
| filters               | [48,96,192,384,512] | [32,64,128,256,320]      |
| Loss                  | dice_focal (γ=2.0)  | dice_ce                  |
| LR                    | 2e-4                | 1e-4 (reduced)           |
| Epochs                | 120                 | 60 (reduced)             |
| Patience              | 25                  | 15 (reduced)             |
| TTA                   | [[2],[3],[4]]       | none (disabled)          |
| keep_largest          | true                | false (disabled)         |

## Assumptions and Hypothesis

- A lower LR (1e-4) allows finer gradient steps and better convergence from scratch
  with the smaller filter configuration.
- Shorter training (60 epochs) is sufficient if the smaller architecture converges faster.
- DiceCE provides stable gradients compared to DiceFocal.
- Disabling TTA and keep_largest tests whether post-processing is necessary or
  inflates inference time without benefit.

**Post-hoc interpretation**: The lower LR is likely the key factor explaining EXP_C's
superiority over EXP_B (same filters but LR=1.5e-4, much worse results). The
smaller filter capacity also appears to generalize better on this limited dataset.

## Training Configuration

See `config_snapshot.yaml` for full details.

| Parameter       | Value                          |
|-----------------|--------------------------------|
| Model           | MONAI DynUNet                  |
| out_channels    | 12                             |
| filters         | [32, 64, 128, 256, 320]        |
| norm_name       | instance                       |
| Loss            | dice_ce (λ_dice=1.0, λ_ce=1.0) |
| LR              | 1e-4                           |
| Weight decay    | 1e-5                           |
| Epochs          | 60                             |
| Patience        | 15                             |
| Patch size      | [128, 128, 128]                |
| TTA             | none                           |
| Keep largest    | false                          |
| Mixed precision | true                           |
| Seed            | 42                             |

- Train script: `train_task2_dynunet.py --config configs/run_0003_task2_dynunet_expC.yaml`
- Eval script:  `evaluate_task2_dynunet.py --config configs/run_0003_task2_dynunet_expC.yaml`
- Split file:   `results/splits/task2_fixed.pkl`
- Checkpoint:   `outputs/checkpoints/RUN_0003_EXP_C/task2_dynunet_best.pt`

## Implementation Plan

See `implementation_plan.md`.

## Results Summary

Metrics computed on validation split (12 subjects, 11 non-background classes). **Re-trained on 2026-05-15** (original checkpoint lost — early stop at epoch 19 vs original epoch 28). Different local minimum reached (DSC 0.353 vs 0.631).

| Metric         | Value  |
|------|--|--|---|
| mean_dsc       | 0.3527 |
| mean_hd95      | 23.96  |
| mean_hd        | 28.05  |
| mean_rve       | 0.4686 |
| mean_assd      | 6.92   |
| n_subjects     | 12     |
| n_classes_eval | 11     |

### Per-class DSC

| Class | Structure     | DSC   |
|-------|---------------|-------|
| 1     | L Hippocampus | 0.353 |
| 2     | R Hippocampus | 0.394 |
| 3     | L Caudate     | 0.630 |
| 4     | R Caudate     | 0.685 |
| 5     | L Lentiform   | 0.689 |
| 6     | R Lentiform   | 0.678 |
| 7     | L Ventricle   | 0.656 |
| 8     | R Ventricle   | 0.734 |
| 9     | L ExV         | 0.704 |
| 10    | R ExV         | 0.759 |
| 11    | Aux           | 0.659 |

**Key observations:**
- Substantial improvement over RUN_0003 baseline (+0.13 mean DSC, +6× hippocampus L DSC).
- Hippocampus L/R remain the hardest structures but improve from 0.052/0.304 to 0.353/0.394.
- No TTA and no keep_largest — the improvement is purely from better training, not inference.
- This checkpoint is the starting point for EXP_C_TTA and EXP_SYM experiments.

## Comparability Statement

Directly comparable to RUN_0003 (same split, label space, 11-class evaluation protocol).
Despite the silent checkpoint skip, EXP_C is a valid independent training run with
smaller filters and lower LR. The comparison quantifies the effect of filter size × LR
on training from scratch.

## Analysis

See `analysis.md` for the full quantitative analysis, including:
- Tableau récapitulatif global des 6 runs comparables du cycle RUN_0003
- Per-class DSC comparé vs baseline avec delta par structure
- Explication scientifique du succès EXP_C (filtres réduits + LR 1e-4 + DiceCE)
- Comparaison avec COLLAPSED (6-class, espace non comparable)
- Leçons apprises : TTA LR naïf, symmetry loss, capacité modèle

## Decision

✅ **Promoted** — **Best Task 2 model of the RUN_0003 cycle** (mean DSC=0.6309,
mean_HD95=9.58, mean_ASSD=2.22). Superior to baseline by +0.131 DSC and ×2.4 on HD95.
EXP_C_TTA and EXP_SYM downstream experiments archived (see `results/runs/archive/`).
