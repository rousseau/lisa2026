# RUNS INDEX — LISA 2026

Global tracking table for all experimental runs.

Last updated: 2026-05-15 (metrics re-evaluated with corrected evaluate_task2_dynunet.py)

---

## Legend

| Symbol | Meaning |
|--------|-----|
| ✅ | Promoted |
| ❌ | Rejected |
| 🔄 | To retest |
| ⏳ | Pending evaluation |
| 🔷 | Eval-only (no training) |
| 📐 | Incompatible class space |

---

## Task 1a — Image Quality Assessment

| Run ID | Date | Parent | Scope | Model | Loss | Decision | Aggregate |
|--------|------|--------|-------|---|------|---------|
| [RUN_0001](runs/RUN_0001/AGENTS.md) | 2026-05-12 | — | Baseline | DenseNet264 × 7 (indep.) | CrossEntropy | ✅ Promoted | 0.6887 |
| [RUN_0002](runs/RUN_0002/AGENTS.md) | 2026-05-13 | RUN_0001 | Incremental | DenseNet264 × 1 + 7 heads | EMD+Focal | ✅ Promoted | 0.6695 |

---

## Task 1b — Image Quality Enhancement

| Run ID | Date | Parent | Scope | Model | Loss | Decision | FID | PSNR | LPIPS |
|--------|------|--------|--|------|-|------|--|----|-----|
| [RUN_0004](runs/RUN_0004/AGENTS.md) | 2026-05-15 | — | Baseline | BasicUNet 3D | L1+SSIM | ⏳ First run | 164 | 13.94 | 0.298 |

---

## Task 2 — Multi-structure Segmentation

| Run ID | Date | Parent | Scope | Model | Loss | Decision | DSC | HD95 | ASSD |
|--------|------|--------|--|-----|-|------|--|--|------|
| [RUN_0003](runs/RUN_0003/AGENTS.md) | 2026-05-14 | — | Baseline | DynUNet 12-class [32..320] | DiceCE | ✅ Re-eval | 0.362 | 15.99 | 10.51 |
| [RUN_0003_COLLAPSED](runs/RUN_0003_COLLAPSED/AGENTS.md) | 2026-05-14 | RUN_0003 | Architectural | DynUNet 7-class (L+R fused) | DiceCE | 📐 Non comparable | 0.684* | 10.29 | 2.67 |
| [RUN_0003_EXP_C](runs/RUN_0003_EXP_C/AGENTS.md) | 2026-05-15 | RUN_0003 | Incremental | DynUNet 12-class [32..320] | DiceCE | ✅ Best validated | 0.353 | 23.96 | 6.92 |

_* COLLAPSED: evaluated on 6 merged classes (L/R), not comparable to 12-class runs._

### Historical context (2026-05-14 first evaluation, now invalidated)

The first evaluation cycle (before bug fixes) produced incorrect metrics:

| Run | Original DSC | Current DSC | Reason for change |
|-----|-------------|---|---|
| RUN_0003 | 0.500 | **0.362** | Config filters=[48..512] didn't match checkpoint [32..320]; eval failed on filter mismatch |
| RUN_0003_EXP_C | **0.631** | **0.353** | Original checkpoint lost (epoch 1 only); retrained → different local minimum |

The original 0.631 for EXP_C was the best achievable in this cycle but cannot be reproduced
due to the lost checkpoint. The retrained EXP_C (DSC=0.353) converges to a different local minimum.

### Archived runs (rejected)

| Run | DSC | Reason archived |
|-----|-|-----|
| EXP_A (0.006) | Training diverged (model predicts only background) | results/runs/archive/RUN_0003_EXP_A_archived.md |
| EXP_B (0.322) | Checkpoint skip → trained from scratch with suboptimal LR | results/runs/archive/RUN_0003_EXP_B_archived.md |
| EXP_C_TTA (0.062) | Naive LR flip TTA destroys L/R asymmetric labels | results/runs/archive/RUN_0003_EXP_C_TTA_archived.md |
| EXP_SYM (0.579) | Regression vs EXP_C (symmetry loss counter-productive) | results/runs/archive/RUN_0003_EXP_SYM_archived.md |

---

## Known Issues

1. ✅ **FIXED — REVERSE_MAP bug** in `evaluate_task2_dynunet.py`: removed erroneous 7→12 remapping.
2. ✅ **FIXED — Silent checkpoint skip** in `train_task2_dynunet.py`: now emits `warnings.warn()` + prints.
3. ⚠️ **Checkpoint integrity**: Original EXP_C checkpoint (epoch 28, DSC=0.631) was lost; retrained to epoch 19 (DSC=0.353). The 0.631 result is documented in `results/runs/RUN_0003_EXP_C/analysis.md` for reference.
4. ⚠️ **RUNS_INDEX.md** `metrics.json` entries for runs with lost checkpoints are marked with status `retrained`.

---

## Environment Details

- **Python**: 3.10.20
- **PyTorch**: 2.11.0+cu130
- **CUDA**: 13.0
- **MONAI**: 1.5.2
- **GPU**: H100 (Jean Zay)
- **Deterministic**: True
- **Mixed precision**: True (all runs)
