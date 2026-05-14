# RUN_0003_COLLAPSED — DynUNet 7-class (Labels L/R Fusionnés)

## Run Metadata

- **Run ID**: RUN_0003_COLLAPSED
- **Date**: 2026-05-14
- **Tasks covered**: Task 2 (Multi-structure Segmentation)
- **Parent run**: RUN_0003 (same experimental cycle, alternative label strategy)
- **Change scope**: Architectural — label space reduction from 12 to 7 classes
- **Comparability**: ⚠️ Not directly comparable to RUN_0003. The evaluation is performed
  on 6 collapsed classes (bilateral structures) vs 11 L/R-separate classes. The aggregate
  metrics numerically appear better but solve a simpler problem.

## Summary of Changes

Alternative label strategy motivated by the extremely poor hippocampus performance
in RUN_0003 (L-Hippo DSC=0.052, R-Hippo DSC=0.304). Symmetric L/R structures are
fused into a single bilateral class (12 → 7 classes) using `COLLAPSED_MAP` from
`src/collapsed_labels.py`. The model is trained end-to-end with the collapsed labels.
Loss is switched from DiceFocal to DiceCE. No TTA at inference. Strategy inspired by
the nnU-Net ResEnc approach described in the LISA 2025 proceedings.

## Full List of Changes vs. RUN_0003

| Component         | RUN_0003                      | RUN_0003_COLLAPSED            |
|-------------------|-------------------------------|-------------------------------|
| out_channels      | 12                            | 7                             |
| collapse_labels   | false                         | true                          |
| Label mapping     | original L/R separate         | L+R fused (COLLAPSED_MAP)     |
| Loss              | dice_focal (γ=2.0)            | dice_ce                       |
| TTA               | axes [[2],[3],[4]]            | none (tta_flip_axes=[])        |
| keep_largest      | true                          | false                         |
| Eval classes      | 11 (classes 1–11)             | 6 (collapsed classes 1–6)     |
| Epochs / Patience | 120 / 25                      | 120 / 25 (unchanged)          |
| LR                | 2e-4                          | 2e-4 (unchanged)              |
| filters           | [48, 96, 192, 384, 512]       | [48, 96, 192, 384, 512] (same)|

## Assumptions and Hypothesis

- Collapsing L/R pairs reduces class imbalance and simplifies the learning objective,
  particularly for the hippocampus which is the smallest structure.
- DiceCE is sufficient for 7 balanced classes without the focal weighting.
- The collapsed model may achieve higher bilateral DSC even though L/R lateralization
  information is lost.
- If the collapsed model outperforms the 12-class model on merged-structure DSC, the
  strategy may be combined with a post-processing lateralization step.

## Training Configuration

See `config_snapshot.yaml` for full details.

| Parameter       | Value                          |
|-----------------|--------------------------------|
| Model           | MONAI DynUNet                  |
| out_channels    | 7                              |
| filters         | [48, 96, 192, 384, 512]        |
| norm_name       | instance                       |
| Loss            | dice_ce (λ_dice=1.0, λ_ce=1.0) |
| LR              | 2e-4                           |
| Weight decay    | 1e-5                           |
| Epochs          | 120                            |
| Patience        | 25                             |
| Patch size      | [128, 128, 128]                |
| TTA             | none                           |
| Keep largest    | false                          |
| Mixed precision | true                           |
| Seed            | 42                             |

- Train script: `train_task2_dynunet.py --config configs/run_0003_task2_dynunet_collapsed.yaml`
- Eval script:  `evaluate_task2_dynunet.py --config configs/run_0003_task2_dynunet_collapsed.yaml`
- Split file:   `results/splits/task2_fixed.pkl`

## Implementation Plan

See `implementation_plan.md`.

## Results Summary

Metrics computed on validation split (12 subjects, 6 collapsed non-background classes).

| Metric         | Value  |
|----------------|--------|
| mean_dsc       | 0.6837 |
| mean_hd95      | 10.29  |
| mean_hd        | 17.01  |
| mean_rve       | 0.2717 |
| mean_assd      | 2.67   |
| n_subjects     | 12     |
| n_classes_eval | 6      |

### Per-class DSC (collapsed classes)

| Class | Structure          | DSC   |
|-------|--------------------|-------|
| 1     | Hippocampus (L+R)  | 0.406 |
| 2     | Caudate (L+R)      | 0.801 |
| 3     | Lentiform (L+R)    | 0.735 |
| 4     | Ventricle (L+R)    | 0.720 |
| 5     | ExV (L+R)          | 0.750 |
| 6     | Aux                | 0.691 |

**Key observations:**
- The mean DSC improvement (0.68 vs 0.50 in RUN_0003) is partly attributable to the
  simpler 6-class evaluation problem.
- The hippocampus remains the hardest structure (0.406) even after L/R fusion, but is
  substantially better than the separate L/R performance (0.052 / 0.304).
- Caudate and Lentiform show strong performance (DSC > 0.73) when evaluated bilaterally.

## Comparability Statement

This run is **not directly comparable** to RUN_0003 in aggregate metrics because:
1. The output space differs (7 vs 12 classes).
2. The evaluation is performed on 6 vs 11 non-background classes.
3. Mean DSC over 6 collapsed classes is not the same metric as mean DSC over 11 separate classes.

A fair comparison would require either (a) mapping collapsed predictions back to L/R before
evaluation, or (b) evaluating RUN_0003 predictions with the same collapsing protocol.

## Decision

⏳ **To retest** — Checkpoint at `outputs/checkpoints/RUN_0003_COLLAPSED/task2_dynunet_best.pt`.
The collapsed strategy shows high absolute DSC (0.68) but the comparison with RUN_0003 requires
metric reconciliation. Pending full evaluation on Jean Zay with consistent protocols.
