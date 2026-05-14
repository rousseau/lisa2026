# RUN_0002 — Task 1a Multi-label EMD+Focal

## Run Metadata

- **Run ID**: RUN_0002
- **Date**: 2026-05-13
- **Tasks covered**: Task 1a (Image Quality Assessment)
- **Parent run**: RUN_0001
- **Change scope**: Incremental (loss function + model architecture)
- **Comparability**: Directly comparable to RUN_0001 — same fixed split, same preprocessing, same evaluation script.

---

## Summary of Changes

Replaces 7 independent models with a single shared DenseNet264 backbone and 7 independent classification heads. Changes the loss from CrossEntropy to EMD (ordinal Earth Mover's Distance) + Focal Loss.

Inspired by the UPF best single-model result (Table 1, LISA 2025 proceedings, score ≈ 0.836).

---

## Full List of Changes vs. RUN_0001

| Component | RUN_0001 | RUN_0002 |
|-----------|----------|----------|
| Models | 7 independent DenseNet264 | 1 shared DenseNet264 + 7 heads |
| Loss | CrossEntropy | EMD (w=1.0) + Focal (w=1.0, γ=2.0) |
| Optimizer | Adam, lr=1e-4 | Adam, lr=1e-4, wd=1e-5 |
| LR scheduler | None | CosineAnnealing (T=70, η_min=1e-6) |
| Batch size | 8 | 4 |
| Inference | 7 separate forward passes | 1 forward pass → 7 outputs |

---

## Assumptions and Hypothesis

- Shared features between artifact types should improve generalization (low-field MRI artifacts share underlying physics).
- Ordinal EMD loss penalizes predictions proportionally to distance from true severity (0 < 1 < 2), unlike CE which treats all errors equally.
- Focal loss addresses severe class imbalance (e.g. Banding: ~96% class 0).
- Per-class focal weights α=[0.25, 0.5, 1.0] (none/moderate/severe) follow UPF ablation study.

---

## Training Configuration

See `config_snapshot.yaml` for full configuration.

- Config file: `configs/run_0002_upf.yaml`
- Script: `train_task1a_multilabel.py`
- Epochs: 70 (early stopping patience=10)
- Batch size: 4
- Split: same as RUN_0001 (`results/splits/task1a_fixed.pkl`)
- Device: CUDA
- Mixed precision: disabled
- Seed: 42

---

## Implementation Plan

See `implementation_plan.md`.

---

## Results Summary

Metrics recovered from on-disk artifacts (results/ was gitignored at run time, files recovered after fixing .gitignore).

Re-run: `python evaluate.py --run 0002` to re-verify.

| Metric | RUN_0001 | RUN_0002 | Delta |
|--------|----------|----------|-------|
| Accuracy | 0.8437 | 0.8259 | -0.0178 |
| F1 macro | 0.6390 | 0.6246 | -0.0144 |
| F2 macro | 0.6188 | 0.6160 | -0.0028 |
| Precision macro | 0.7317 | 0.6669 | -0.0648 |
| Recall macro | 0.6103 | 0.6139 | +0.0036 |
| **Aggregate** | **0.6887** | **0.6695** | **-0.0192** |

### Per-task Aggregate Scores

| Task | RUN_0001 | RUN_0002 |
|------|----------|----------|
| Noise | 0.7394 | 0.7334 |
| Zipper | 0.6638 | 0.5745 |
| Positioning | 0.8094 | 0.8023 |
| Banding | 0.6416 | 0.6378 |
| Motion | 0.6394 | 0.5990 |
| Contrast | 0.6902 | 0.6808 |
| Distortion | 0.6373 | 0.6585 |

---

## Comparability Statement

✅ Directly comparable to RUN_0001. Same fixed split (task1a_fixed.pkl, seed=42), same spatial preprocessing (CenterCrop 150³ + pad), same evaluation script (evaluate_task1a_multilabel.py + compute_metrics.py).

---

## Decision

✅ **Promoted** — multi-label + EMD+Focal is the architectural baseline for Task 1a multi-head training.

**Note**: aggregate score (0.6695) is slightly lower than RUN_0001 (0.6887). The multi-head architecture may need further tuning (learning rate, loss weights, longer training). Promoted as an architecture reference, not a score improvement.
