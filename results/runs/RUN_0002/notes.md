# RUN_0002 — Notes

## Execution Notes

- Trained on Jean Zay (H100), single GPU.
- Single training job for all 7 tasks simultaneously.
- Checkpoint: `outputs/checkpoints/RUN_0002/multilabel_best.pt`

## Known Issues

- `results/` was gitignored at the time of this run. Metrics were computed but not versioned via git.
- **Status**: Metrics recovered from on-disk artifacts after fixing `.gitignore`. Aggregate = 0.6695.

## Observations

- EMD loss naturally handles ordinal severity (none/moderate/severe) without treating distance-2 errors the same as distance-1 errors.
- The `focal_alpha=[0.25, 0.5, 1.0]` weighting is key for Banding which has ~96% class-0 samples.
- Single model is 7× more efficient at inference than RUN_0001.
- Despite architectural improvements, aggregate score (0.6695) is slightly lower than RUN_0001 (0.6887).
  - Most affected tasks: Zipper (-0.089), Motion (-0.040).
  - Distortion improved slightly (+0.021).
  - The multi-head shared backbone may require more epochs or task-specific LR tuning.

## Next Steps

- Compare RUN_0002 vs RUN_0001 in detail (per-task breakdown available in metrics.json).
- Task 2 addressed independently in RUN_0003.
- Consider RUN_0002b: longer training + per-head LR or task-specific loss weighting.
