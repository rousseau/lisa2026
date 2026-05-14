# RUN_0003_EXP_C — Notes

## Best 12-class Run

EXP_C achieves the best 12-class segmentation in this experimental cycle (mean DSC=0.631,
vs RUN_0003 baseline 0.500). This is despite:
- No TTA at inference
- No keep_largest post-processing
- A silently-skipped pretrained checkpoint (architecture mismatch)

## Architecture Mismatch (Pretrained Checkpoint)

Like EXP_B, the pretrained_checkpoint points to RUN_0003 (filters=[48,96,192,384,512])
but the EXP_C model uses filters=[32,64,128,256,320]. The checkpoint is silently skipped.
EXP_C trains from scratch.

## Why EXP_C > RUN_0003 Despite Fresh Training?

Two factors likely contribute:
1. **Smaller filter capacity** [32,64,128,256,320] generalizes better on the limited
   LISA 2026 dataset (fewer parameters to overfit).
2. **Lower LR** (1e-4 vs 2e-4) allows finer gradient steps and more stable convergence.

## Why EXP_C > EXP_B?

Both use the same filters [32,64,128,256,320] and DiceCE loss. The key differences:
- EXP_B: LR=1.5e-4, epochs=90 → poor results (DSC=0.32)
- EXP_C: LR=1e-4, epochs=60 → best results (DSC=0.63)

The lower LR (1e-4 vs 1.5e-4) is almost certainly the dominant factor.

## Downstream Usage

This checkpoint is used as:
- Starting point for EXP_C_TTA (evaluation with TTA — but evaluation failed due to bug)
- Starting point for EXP_SYM (fine-tune with symmetry loss)

## Checkpoint

`outputs/checkpoints/RUN_0003_EXP_C/task2_dynunet_best.pt`

## Label Class Reference (12-class)

| Class | Structure     | DSC (EXP_C) |
|-------|---------------|-------------|
| 0     | Background    | —           |
| 1     | L Hippocampus | 0.353       |
| 2     | R Hippocampus | 0.394       |
| 3     | L Caudate     | 0.630       |
| 4     | R Caudate     | 0.685       |
| 5     | L Lentiform   | 0.689       |
| 6     | R Lentiform   | 0.678       |
| 7     | L Ventricle   | 0.656       |
| 8     | R Ventricle   | 0.734       |
| 9     | L ExV         | 0.704       |
| 10    | R ExV         | 0.759       |
| 11    | Aux           | 0.659       |
