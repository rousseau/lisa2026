# RUN_0003 - Implementation Plan

## Objective

Build an independent Task2 baseline using MONAI DynUNet for multi-structure segmentation.

## Steps

1. Build Task2 image/mask manifest from LISA filenames.
2. Create fixed patient-level split and freeze it.
3. Train DynUNet with Dice+CE loss and augmentation.
4. Evaluate on validation split with challenge metrics.
5. Store reproducible artifacts in `results/runs/RUN_0003`.
6. Launch and monitor on Jean Zay through generic SLURM script.

## Validation checklist

- Split and manifest generated.
- Local smoke test passes (train + eval).
- Metrics JSON and per-class table generated.
- SLURM run dispatch `RUN_0003` works.
