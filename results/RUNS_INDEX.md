# RUNS INDEX — LISA 2026

Global tracking table for all experimental runs.

Last updated: 2026-05-15

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Promoted |
| ❌ | Rejected |
| 🔄 | To retest |
| ⏳ | Pending evaluation |

---

## Task 1a — Image Quality Assessment

| Run ID | Date | Parent | Model | Loss | Aggregate | Decision |
|--------|------|--------|-------|------|-----------|---------|
| [RUN_0001](runs/RUN_0001/AGENTS.md) | 2026-05-12 | — | DenseNet264 × 7 (independent) | CrossEntropy | **0.6887** | ✅ Promoted |
| [RUN_0002](runs/RUN_0002/AGENTS.md) | 2026-05-13 | RUN_0001 | DenseNet264 × 1 + 7 heads | EMD + Focal | 0.6695 | ❌ Rejected |

---

## Task 1b — Image Quality Enhancement

| Run ID | Date | Parent | Model | Loss | FID | PSNR | LPIPS | Decision |
|--------|------|--------|-------|------|-----|------|-------|---------|
| [RUN_0004](runs/RUN_0004/AGENTS.md) | 2026-05-15 | — | BasicUNet 3D | L1 + SSIM | 164 | 13.94 | 0.298 | ⏳ Baseline |

---

## Task 2 — Multi-structure Segmentation

| Run ID | Date | Parent | Model | Loss | DSC | HD95 | ASSD | Decision |
|--------|------|--------|-------|------|-----|------|------|---------|
| [RUN_0003](runs/RUN_0003/AGENTS.md) | 2026-05-14 | — | DynUNet 12-class [32..320] | DiceCE | **0.362** | 15.99 | 10.51 | ✅ Promoted |

---

## Environment

| Key | Value |
|-----|-------|
| Python | 3.13 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| MONAI | 1.5.2 |
| GPU | NVIDIA GB10 (local) / H100 (Jean Zay) |
| Seed | 42 (all runs) |
| Mixed precision | True (Task 2) / False (Task 1a) |

---

## Known Issues

1. ✅ **FIXED** — `REVERSE_MAP` bug in `src/evaluate_task2_dynunet.py` (erroneous 7→12 remapping removed).
2. ✅ **FIXED** — Silent checkpoint skip in `src/train_task2_dynunet.py` (now warns explicitly).
