# RUNS INDEX — LISA 2026

Global tracking table for all experimental runs.

Last updated: 2026-06-01 (RUN_0002 analysis completed)

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
| [RUN_0001](runs/RUN_0001/AGENTS.md) | 2026-05-12 / 2026-05-29 | — | DenseNet264 × 7 (independent) | CrossEntropy | **0.6887** | ✅ Promoted |

## SOTA Reference — Task 1a (LISA 2025)

| Team / System              | Aggregate | F1-macro (mean) | Split          | Notes |
|---------------------------|-----------|-----------------|----------------|-------|
| BRIQA best (1st place)    | 0.799     | 0.682 (weighted)| test (official)| DenseNet + rotating batches + TorchIO aug |
| UPF (ordinal + aug)       | 0.840–0.850 | —             | val (internal) | EMD loss + Bayesian network + TorchIO aug |
| 5th place (MaxViT 2D)     | 0.777     | 0.691           | test (official)| 2D view-conditional, slice aggregation |
| BRIQA baseline (CE only)  | 0.745     | ~0.560          | test (official)| No class balancing, no rotation |
| **RUN_0001 (ours)**       | **0.6887**| **0.6390**      | val (local)    | DenseNet264 3D, CE, no aug simulation |

Gap to 1st place: ~−0.11 aggregate. Main gap drivers: no artifact simulation, no class-weighted loss.

---

## Task 1b — Artifact Removal / Image Enhancement

| Run ID | Date | Parent | Model | Method | FID | PSNR | LPIPS | Decision |
|--------|------|--------|-------|--------|-----|------|-------|---------|
| [RUN_0002](runs/RUN_0002/AGENTS.md) | 2026-06-03 (v2 local, corrected) | — | 3D CycleGAN (G: U-Net [32..256] + 6 ResBlocks; D: PatchGAN) | Unpaired (LSGAN + cycle L1 + identity L1) | — | **27.51 dB** (proxy) | — | ✅ Promoted |

> **Note**: No paired ground truth available. Domains A (Noise≥1 or Motion≥1) and B (Noise=0 and
> Motion=0) derived from Task 1a CSV. Official metrics (FID/PSNR/LPIPS) require challenge test set.
> 
> **RUN_0002 Results (Jean Zay, 14 min, 100 epochs)**: Best val_cycle=2.0578 @ epoch 94, D loss≈0.0015 (stable).

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
| [RUN_0004](runs/RUN_0004/AGENTS.md) | 2026-05-20 | RUN_0001/RUN_0003 | DynUNetMultiHead [32..320] + 3 heads | 0.1944 | 3.38 dB | **0.1649** | 🔄 À retester (v5 prête) |

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
| Mixed precision | True (Task 2) / False (Task 1a, Task 1b) |

---

## Known Issues

1. ✅ **FIXED** — `REVERSE_MAP` bug in `src/evaluate_task2_dynunet.py` (erroneous 7→12 remapping removed).
2. ✅ **FIXED** — Silent checkpoint skip in `src/train_task2_dynunet.py` (now warns explicitly).
3. ⚠️ **RUN_0004 v3** — Head warmup NaN dès epoch 46 (LR scheduler partagé T_max=130, LR≈7.56e-5 trop élevé pour têtes aléatoires). Bug Python `max(nan, 1e-6)` → calibration NaN → phase joint 100% inerte. Warmup convergent (DSC=0.1104 ✅). Corrections F1–F3 identifiées pour v4.
4. ⚠️ **RUN_0004 v4** (job 1076254) — Code v4 non synchronisé sur Jean Zay → run exécuté avec code v3. Head warmup encore NaN (LR≈1e-4, plus rapide qu'en v3 : hw epoch 2 vs hw epoch 3). Warm-start RUN_0003 présent : DSC warmup = **0.1649** (best so far). Corrections v5 appliquées localement (head warmup supprimé + CE pondérée Task 1a) et validées par smoke test.
