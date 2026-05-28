# RUNS INDEX — LISA 2026

Global tracking table for all experimental runs.

Last updated: 2026-05-27

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
| [RUN_0004](runs/RUN_0004/AGENTS.md) | 2026-05-20 | — | DynUNetMultiHead [32..320] + 3 heads | L1+SSIM+DiceCE | — | 3.38 | — | 🔄 À retester |
| RUN_0005 | ⏳ pending | — | BasicUNet 3D (self-supervised, Gaussian noise) | L1+SSIM | — | — | — | ⏳ Pending |

---

## Task 2 — Multi-structure Segmentation

| Run ID | Date | Parent | Model | Loss | DSC | HD95 | ASSD | Decision |
|--------|------|--------|-------|------|-----|------|------|---------|
| [RUN_0003](runs/RUN_0003/AGENTS.md) | 2026-05-14 | — | DynUNet 12-class [32..320] | DiceCE | **0.362** | 15.99 | 10.51 | ✅ Promoted |

> **Note — informal RUN_0003 experimental sub-cycle:** During development of RUN_0003, several
> experimental variants (EXP_A through EXP_SYM) were explored informally. EXP_C reportedly achieved
> DSC=0.6309 (filters [32,64,128,256,320], DiceCE, LR=1e-4). These sub-runs were **not formally
> tracked**: their result directories (`RUN_0003_EXP_B`, `RUN_0003_EXP_C`, `RUN_0003_EXP_SYM`) do
> not exist in the repository and their configs/checkpoints are lost. They are referenced only in
> `results/runs/RUN_0003/AGENTS.md`.

---

## Multi-task (Tasks 1a + 1b + 2 — Shared Encoder)

| Run ID | Date | Parent | Architecture | T1a Aggr | T1b PSNR | T2 DSC | Decision |
|--------|------|--------|-------------|----------|----------|--------|----------|
| [RUN_0004](runs/RUN_0004/AGENTS.md) | 2026-05-20 | RUN_0001/RUN_0003 | DynUNetMultiHead [32..320] + 3 heads | 0.1944 | 3.38 dB | **0.1649** | 🔄 À retester (v4 code non sync, v5 prête) |

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
3. ⚠️ **RUN_0004 v3** — Head warmup NaN dès epoch 46 (LR scheduler partagé T_max=130, LR≈7.56e-5 trop élevé pour têtes aléatoires). Bug Python `max(nan, 1e-6)` → calibration NaN → phase joint 100% inerte. Warmup convergent (DSC=0.1104 ✅). Corrections F1–F3 identifiées pour v4.
4. ⚠️ **RUN_0004 v4** (job 1076254) — Code v4 non synchronisé sur Jean Zay → run exécuté avec code v3. Head warmup encore NaN (LR≈1e-4, plus rapide qu'en v3 : hw epoch 2 vs hw epoch 3). Warm-start RUN_0003 présent : DSC warmup = **0.1649** (best so far). Corrections v5 appliquées localement (head warmup supprimé + CE pondérée Task 1a) et validées par smoke test.
