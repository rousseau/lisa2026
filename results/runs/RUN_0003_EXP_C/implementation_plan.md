# RUN_0003_EXP_C — Implementation Plan

## Objective

Test whether a smaller DynUNet (filters=[32,64,128,256,320]) with DiceCE loss and
lower LR (1e-4) can outperform the RUN_0003 baseline when trained from scratch with
a shorter schedule (60 epochs, patience 15). No TTA or keep_largest at inference.

## Steps

1. Configure DynUNet with filters=[32,64,128,256,320] and pretrained checkpoint
   from RUN_0003 (note: checkpoint will be silently skipped due to filter mismatch).
2. Train with DiceCE, LR=1e-4, epochs=60, patience=15.
3. Evaluate WITHOUT TTA, WITHOUT keep_largest to measure raw model performance.
4. If results are good, use this checkpoint for:
   - EXP_C_TTA: re-evaluate the same checkpoint with TTA enabled
   - EXP_SYM: fine-tune with symmetry consistency loss

## Validation Checklist

- [x] Checkpoint saved to `outputs/checkpoints/RUN_0003_EXP_C/task2_dynunet_best.pt`
- [x] Evaluation metrics saved to `results/runs/RUN_0003_EXP_C/metrics.json`
- [x] Predictions CSV saved to `results/runs/RUN_0003_EXP_C/predictions_val_task2.csv`
- [x] Mean DSC > 0.50 (achieved: 0.63) ✅

## Known Issue

The `pretrained_checkpoint` in the config points to RUN_0003 (wider filters), causing
a silent skip. To properly test fine-tuning, the config should either (a) use matching
filter sizes, or (b) implement partial weight loading. This will be addressed in future
runs if fine-tuning is the target strategy.
