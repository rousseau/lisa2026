# RUN_0002 — Notes

## Status

Pending training. Code complete, config ready.

## Domain statistics (estimated from prior analysis)

- Domain A (artefacted, Noise≥1 OR Motion≥1): ~60–70% of volumes
- Domain B (clean, Noise=0 AND Motion=0): ~30–40% of volumes

Exact counts to be confirmed when running with the actual CSV.

## Potential issues

1. **Class imbalance between domains**: Domain A may be significantly larger
   than domain B. The dataset handles this by cycling the smaller domain
   (modulo indexing). May need to verify balance at runtime.

2. **Memory**: 3D volumes at 96³ with batch=1 should fit on GB10 (24 GB).
   If OOM, reduce to 80³ or use gradient checkpointing in the generator.

3. **Training instability**: CycleGAN on 3D data is known to be less stable
   than 2D due to the larger parameter count. Monitor discriminator loss
   (should not collapse to 0 or diverge). Image buffer (size 50) helps.

4. **Task 2 split vs Task 1a split**: Using task2_fixed.pkl for the split.
   If a subject appears in the Task 1a CSV but not in task2_fixed.pkl
   (because they have no segmentation label), they will be excluded.
   This is acceptable for a baseline.

## Commands

```bash
# Training
python train.py --run 0002

# Smoke test (2 epochs, 2 batches)
python train.py --run 0002 --smoke-test

# Evaluation
python evaluate.py --run 0002
```
