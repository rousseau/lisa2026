# RUN_0001 - Execution Notes

## Overview

Execution log for RUN_0001 baseline (Task 1a only).

Run status: completed (local GB10 rerun — 2026-05-29)

---

## Phase 1: Data Preparation

- Split generation: completed
- Command: python prepare_split.py --csv /home/rousseau/Data/LISA2026/LISA_Task1a_2026.csv --seed 42 --output results/splits/task1a_fixed.pkl
- Output:
  - Train samples: 436
  - Val samples: 96
  - Subjects: 244
- Split file: results/splits/task1a_fixed.pkl

---

## Phase 2: Training

Training executed sequentially for 7 independent DenseNet264 3D models with early stopping (patience=10, monitor=val_f1_macro).

### Per-task training summary (local GB10 rerun, 2026-05-29)

| Artifact    | Best val_f1 | Early stop epoch |
|-------------|-------------|-----------------|
| Noise       | 0.6729      | 20              |
| Zipper      | 0.6472      | 27              |
| Positioning | 0.7600      | 19              |
| Banding     | 0.5520      | 21              |
| Motion      | 0.6423      | 27              |
| Contrast    | 0.6500      | 15              |
| Distortion  | 0.5825      | 24              |

> Note: slight differences vs. the 2026-05-12 run (same code, same split, different random GPU state).
> The evaluate.py run used the newly saved checkpoints and produced the metrics.json below.

---

## Phase 3: Evaluation (metrics.json)

Global metrics (macro-averaged over 7 tasks, validation split):

| Metric       | RUN_0001 |
|--------------|----------|
| Accuracy     | 0.8437   |
| F1_macro     | 0.6390   |
| F2_macro     | 0.6188   |
| Precision    | 0.7317   |
| Recall       | 0.6103   |
| **Aggregate**| **0.6887** |

Per-task F1_macro:

| Artifact    | F1_macro | Precision | Recall | Accuracy | Aggregate |
|-------------|----------|-----------|--------|----------|-----------|
| Noise       | 0.6867   | 0.8068    | 0.6348 | 0.9167   | 0.7394    |
| Zipper      | 0.5879   | 0.8256    | 0.5524 | 0.7917   | 0.6638    |
| Positioning | 0.7787   | 0.7939    | 0.7660 | 0.9375   | 0.8094    |
| Banding     | 0.5520   | 0.6596    | 0.5000 | 0.9792   | 0.6416    |
| Motion      | 0.6025   | 0.6524    | 0.6095 | 0.7292   | 0.6394    |
| Contrast    | 0.6640   | 0.7336    | 0.6374 | 0.7708   | 0.6902    |
| Distortion  | 0.6015   | 0.6501    | 0.5718 | 0.7813   | 0.6373    |

---

## Quantitative Analysis — RUN_0001 vs LISA 2025 State of the Art

### LISA 2025 Task 1a — Reference Results

Source: LISA 2025 Proceedings (LNCS 16411). All figures are on the official test set (held-out).

#### BRIQA (1st place, Tsinghua/MBZUAI — F1-macro aggregate ~0.799)

Best configuration: DenseNet + rotating batches + cross-entropy. Key per-artifact F1 (macro):

| Artifact    | BRIQA best | BRIQA baseline (CE, no rotation) |
|-------------|-----------|----------------------------------|
| Noise       | 0.725     | 0.295 → +0.430 gain              |
| Zipper      | 0.731     | 0.633                            |
| Positioning | 0.732     | 0.635                            |
| Banding     | 0.605     | 0.593                            |
| Motion      | 0.625     | 0.603                            |
| Contrast    | 0.698     | 0.720 (−0.022, regression)       |
| Distortion  | 0.657     | 0.440                            |
| **Mean**    | **0.682** | 0.560                            |

Note: BRIQA uses weighted F1 (not macro). Their best overall score (5 metrics averaged) = **0.799**.

#### UPF Team (3rd overall, aggregate = 0.84 on online validation)

- Ordinal loss (EMD) + Bayesian network + aggressive augmentation (TorchIO artifact simulation)
- Reported composite accuracy: **0.84** (online validation); 0.85 with ordinal loss on internal val
- Architecture: DenseNet or ResNet variants (per-artifact model selection)

#### 5th-place team (MaxViT 2D view-conditional, weighted F1 = 0.771 on test set)

Per-artifact F1-macro on test set:

| Artifact    | F1-macro |
|-------------|----------|
| Noise       | 0.797    |
| Zipper      | 0.709    |
| Positioning | 0.432    |
| Banding     | 0.596    |
| Motion      | 0.594    |
| Contrast    | 0.443    |
| Distortion  | 0.754    |
| **Average** | **0.691** |

---

### Head-to-Head Comparison: RUN_0001 vs SOTA

> Important caveat: RUN_0001 metrics are on a **local validation split** (96 samples, patient-level
> fixed split, val=20% of 532 total). LISA 2025 SOTA figures are on the **official held-out test set**.
> Direct numerical comparison is indicative only — the ranking on the official test may differ.

| Artifact    | RUN_0001 (val) | BRIQA best (test) | 5th place (test) | Gap vs BRIQA |
|-------------|---------------|-------------------|------------------|-------------|
| Noise       | **0.6867**    | 0.725             | 0.797            | −0.038      |
| Zipper      | **0.5879**    | 0.731             | 0.709            | −0.143      |
| Positioning | **0.7787**    | 0.732             | 0.432            | **+0.047**  |
| Banding     | **0.5520**    | 0.605             | 0.596            | −0.053      |
| Motion      | **0.6025**    | 0.625             | 0.594            | −0.023      |
| Contrast    | **0.6640**    | 0.698             | 0.443            | −0.034      |
| Distortion  | **0.6015**    | 0.657             | 0.754            | −0.056      |
| **Mean F1** | **0.6390**    | **0.682**         | **0.691**        | **−0.043**  |

Global aggregate comparison:

| System                     | Aggregate score | Split          |
|----------------------------|-----------------|----------------|
| **RUN_0001 (ours)**        | **0.6887**      | val (local)    |
| BRIQA baseline (CE, no rot)| 0.745           | test (official)|
| BRIQA best                 | 0.799           | test (official)|
| UPF ordinal + aug          | 0.840–0.850     | val (internal) |
| 5th place (MaxViT 2D)      | 0.777           | test (official)|

---

### Strengths and Weaknesses of RUN_0001

#### Strengths

1. **Positioning**: F1=0.779, above BRIQA best (0.732) and far above 5th place (0.432).
   This suggests that 3D volumetric modeling has an advantage for spatial/positioning artifacts.

2. **Noise and Contrast**: close to BRIQA best (gap < 0.04), competitive despite no class reweighting
   or artifact simulation.

3. **Architecture**: DenseNet264 3D converges stably across all 7 tasks (no crash, no NaN).

4. **Training speed**: ~1.4s/batch on GB10, ~90s/epoch. Total ~3.5h for 7 tasks (early stopping
   triggered at 15–27 epochs).

#### Weaknesses

1. **Zipper**: largest gap (−0.143 vs BRIQA). Likely explanation: zipper artifacts are high-frequency
   patterns strongly localized in k-space — 2D slice-based models may capture them better than 3D.

2. **Banding**: F1=0.552, lowest artifact. Very severe class imbalance (501/15/13 per 0/1/2 class).
   No explicit class reweighting in RUN_0001 → dominated by class-0. The accuracy is 0.979 (trivial
   all-zeros predictor) while F1 is poor.

3. **Distortion**: F1=0.601, below 5th place (0.754). Distortion is a global spatial pattern —
   the 5th-place 2D model at higher resolution may have better resolution sensitivity.

4. **No artifact simulation**: BRIQA and UPF both use TorchIO to synthesize artifacts and increase
   minority class representation. RUN_0001 has no such augmentation — the main driver of the gap.

5. **No ordinal loss**: EMD/ordinal loss is used by UPF and BRIQA best configs. RUN_0001 uses
   standard cross-entropy which does not penalize confusion across adjacent severity levels.

6. **3D at 150³ vs 2D slice-level**: 3D approach has advantage for spatial/global artifacts
   (Positioning, Contrast) but is limited by GPU memory and effective resolution. 2D slice-level
   approaches (5th place, BRIQA) process at higher in-plane resolution.

---

### Gap Analysis and Next Run Priorities

Estimated gain potential from individual improvements (based on LISA 2025 ablations):

| Improvement                        | Estimated Δ aggregate | Priority |
|------------------------------------|----------------------|----------|
| Artifact simulation (TorchIO)      | +0.05 to +0.12       | HIGH     |
| Ordinal loss (EMD)                 | +0.01 to +0.02       | MEDIUM   |
| Class-weighted CE (per artifact)   | +0.01 to +0.03       | HIGH     |
| Rotating batches (BRIQA scheme)    | +0.01 to +0.03       | MEDIUM   |
| View-conditioning / scan plane aux | +0.03 to +0.05       | MEDIUM   |
| Architecture diversity per task    | +0.02 to +0.04       | LOW      |

**Conclusion**: RUN_0001 is **~0.11 below BRIQA best** on aggregate. The dominant gap drivers are:
(1) no artifact simulation for class rebalancing, and (2) no class-weighted loss for banding/zipper.
These are the two changes most likely to close the gap in the next run.

---

## Phase 4: Consolidation

- AGENTS file updated with final metrics and decision
- RUNS_INDEX updated (status=completed)
- Baseline decision: accepted

## Issues and Resolutions

- Issue: conda activate failed in non-initialized shell
- Resolution: switched to direct activation command source /home/rousseau/miniforge3/bin/activate lisa2026 for execution context
- Impact: no impact on model outputs

- Issue: nohup / conda run caused silent process termination
- Resolution: launched in tmux session (train0001) with direct python binary path
- Impact: no impact on model outputs

## Decision Record

Baseline acceptance criteria:
- Global aggregate score >= 0.60: met (0.6887)
- All 7 tasks completed: met
- Early stopping behavior normal: met
- Reproducibility setup captured: met

Decision: baseline accepted — to be superseded by class-reweighting + artifact simulation run

## Last Updated

- Date: 2026-05-29
- Final status: completed (2nd local execution, GB10 NVIDIA)
