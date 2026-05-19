# RUNS INDEX — LISA 2026

Global tracking table for all experimental runs.

Last updated: 2026-05-19

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
| [RUN_0004](runs/RUN_0004/AGENTS.md) | 2026-05-19 | — | SharedEncoder 3T heads | L1+SSIM+DiceCE | — | 3.48 | — | ❌ Rejeté |

---

## Task 2 — Multi-structure Segmentation

| Run ID | Date | Parent | Model | Loss | DSC | HD95 | ASSD | Decision |
|--------|------|--------|-------|------|-----|------|------|---------|
| [RUN_0003](runs/RUN_0003/AGENTS.md) | 2026-05-14 | — | DynUNet 12-class [32..320] | DiceCE | **0.362** | 15.99 | 10.51 | ✅ Promoted |

---

## Multi-task (Tasks 1a + 1b + 2 — Shared Encoder)

| Run ID | Date | Parent | Architecture | T1a Aggr | T1b PSNR | T2 DSC | Decision |
|--------|------|--------|-------------|----------|----------|--------|----------|
| [RUN_0004](runs/RUN_0004/AGENTS.md) | 2026-05-19 | RUN_0001/RUN_0003 | SharedEncoder [32..320] + 3 heads | 0.1293 | 3.48 dB | 0.0083 | ❌ Rejeté |

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
3. ⚠️ **RUN_0004** — Architecture `SharedEncoderMultiTaskModel` incompatible avec poids DynUNet RUN_0003 (0/102 keys matched). Loss 1a domine phase joint (ratio 2.7:1), collapse segmentation. Early stopping déclenché epoch 25/90.
