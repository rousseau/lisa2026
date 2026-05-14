# RUN_0003_COLLAPSED — Notes

## Motivation

RUN_0003 showed extremely poor hippocampus performance (L-Hippo DSC=0.052, R-Hippo
DSC=0.304). The hypothesis was that fusing L/R labels into a single bilateral class
would simplify the learning problem and improve overall segmentation quality.

## Results Interpretation

- Mean DSC=0.68 vs 0.50 for RUN_0003, but over 6 vs 11 classes (not directly comparable).
- Bilateral hippocampus DSC=0.41 is substantially better than L+R separate (0.052+0.304).
- The collapsed model is simpler to optimize but loses lateralization information.

## Comparability Issue

The aggregate metrics (mean DSC) are computed over different class sets:
- RUN_0003: 11 classes (1–11, L and R separate)
- RUN_0003_COLLAPSED: 6 collapsed classes (bilateral structures)

A fair comparison requires metric reconciliation (collapse-then-evaluate vs evaluate-then-compare).
See `implementation_plan.md` for details.

## Labels Reference

| Collapsed class | Original classes | Structure          |
|-----------------|------------------|--------------------|
| 0               | 0                | Background         |
| 1               | 1, 2             | Hippocampus (L+R)  |
| 2               | 3, 4             | Caudate (L+R)      |
| 3               | 5, 6             | Lentiform (L+R)    |
| 4               | 7, 8             | Ventricle (L+R)    |
| 5               | 9, 10            | ExV (L+R)          |
| 6               | 11               | Aux                |

## Checkpoint

`outputs/checkpoints/RUN_0003_COLLAPSED/task2_dynunet_best.pt`
