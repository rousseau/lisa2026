# RUN_0003_COLLAPSED — Implementation Plan

## Objective

Train an alternative 7-class DynUNet where symmetric L/R structures are fused into
bilateral classes, to address the poor hippocampus performance observed in RUN_0003.

## Steps

1. Verify `src/collapsed_labels.py` COLLAPSED_MAP covers all 12 original classes.
2. Enable `collapse_labels: true` in the data pipeline to remap labels on-the-fly.
3. Configure DynUNet with `out_channels=7` and same architecture as RUN_0003.
4. Train with DiceCE loss (simpler than DiceFocal for 7 classes).
5. Evaluate with `evaluate_task2_dynunet.py` using `collapse_labels: true`
   (evaluates on 6 collapsed classes, no reverse mapping).
6. Compare per-class DSC at the bilateral structure level vs RUN_0003.
7. Store metrics and checkpoint in `results/runs/RUN_0003_COLLAPSED/`.

## Validation Checklist

- [ ] Label collapse verified in dataset __getitem__ (COLLAPSED_MAP applied correctly)
- [ ] Model output shape: [B, 7, D, H, W]
- [ ] Evaluation runs on 6 non-background classes (collapsed)
- [ ] Hippocampus bilateral DSC > 0.40 (target)
- [ ] Checkpoint saved to `outputs/checkpoints/RUN_0003_COLLAPSED/task2_dynunet_best.pt`

## Notes on Comparability

The collapsed evaluation (6 classes) is NOT directly comparable to RUN_0003 (11 classes).
For a fair comparison, post-processing to re-split bilateral predictions into L/R would
be needed, or RUN_0003 predictions must be evaluated with the same collapsing.
