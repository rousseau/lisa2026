# Runs rapatriés en local - statut au 2026-06-09

## Synthèse par run

| Run | Date(s) expérience | Source date | Statut exécution | Évaluations disponibles | Évaluations à faire |
|---|---|---|---|---|---|
| RUN_0001 (Task 1a) | 2026-05-12 (run initial), 2026-05-29 (rerun local) | AGENTS + notes | Fait | metrics.json présent (Accuracy/F1/F2/Precision/Recall + aggregate) | Aucune critique (éventuel rerun benchmark officiel si besoin challenge) |
| RUN_0002 (Task 1b) | 2026-05-29 (Jean Zay v1), 2026-06-03 (v2 corrigé local), analyse 2026-06-01/2026-06-02 | AGENTS + notes + ANALYSIS | Fait (v2 promue) | metrics.json présent (PSNR/SSIM proxy), visual proofs présents | FID/LPIPS officiels (requiert test set challenge) |
| RUN_0003 (Task 2 DynUNet) | 2026-05-14 | AGENTS | Fait | metrics.json présent (DSC/HD95/HD/RVE/ASSD) | Vérifier cohérence finale des chiffres entre AGENTS et metrics.json (historique multiple versions) |
| RUN_0003a (Task 2 nnU-Net v2) | 2026-06-05 | AGENTS | Non exécuté ou non rapatrié (aucun artefact de résultat) | Aucune métrique trouvée | Lancer entraînement + inférence val + évaluation challenge metrics |
| RUN_0003b (Task 2 MedSAM2 + DynUNet) | 2026-06-05 | AGENTS | Non exécuté ou non rapatrié (aucun artefact de résultat) | Aucune métrique trouvée | Lancer entraînement + inférence val + évaluation challenge metrics |
| RUN_0003c (Task 2 hybride) | 2026-06-05 | AGENTS | Non exécuté ou non rapatrié (pas de metrics.json/training_history) | Aucune métrique consolidée | Exécuter après 0003a/0003b puis évaluer (DSC/HD95/HD/RVE/ASSD) |
| RUN_0004 (multi-task) | 2026-05-19 à 2026-05-20 (v1-v4 Jean Zay) | AGENTS + notes | Exécuté, non promu (à retester v5) | metrics.json présent + post-mortem détaillé | Exécuter v5 corrigée puis réévaluation complète Task1a/1b/2 |

## Détail des évaluations - fait vs à faire

### RUN_0001
- Fait:
  - Évaluation Task 1a complète sur split fixe (global + per-task).
- À faire:
  - Optionnel: benchmark officiel challenge (si test set accessible).

### RUN_0002
- Fait:
  - Évaluation proxy locale (PSNR/SSIM input vs output), visual proofs par catégorie.
  - Validation de la correction de normalisation (v2 promue).
- À faire:
  - Évaluation officielle challenge: FID + LPIPS (et PSNR officiel si protocole test set).

### RUN_0003
- Fait:
  - Évaluation Task 2 (DSC/HD95/HD/RVE/ASSD) exportée en metrics.json.
- À faire:
  - Harmoniser la version de référence (AGENTS vs metrics.json) avant comparaison finale inter-runs.

### RUN_0003a / RUN_0003b / RUN_0003c
- Fait:
  - Plans et protocole définis.
- À faire:
  - Exécution sur Jean Zay (ou local), puis import des artefacts:
    - checkpoints
    - training history
    - predictions
    - metrics.json
  - Comparatif direct vs RUN_0003 baseline.

### RUN_0004
- Fait:
  - Plusieurs exécutions Jean Zay (v1-v4) + analyses d'échec (NaN/head warmup) documentées.
  - Évaluations partielles disponibles.
- À faire:
  - Exécuter v5 (fixes appliqués localement) et refaire l'évaluation complète multi-tâche.

## Vérifications de présence des artefacts (local)

- RUN_0001: AGENTS.md, notes.md, metrics.json, predictions_val.csv, training_history.json.
- RUN_0002: AGENTS.md, notes.md, ANALYSIS.md, metrics.json, plots/.
- RUN_0003: AGENTS.md, notes.md, metrics.json, predictions_val_task2.csv, training_history.json.
- RUN_0003a: AGENTS.md, implementation_plan.md uniquement.
- RUN_0003b: AGENTS.md, implementation_plan.md uniquement.
- RUN_0003c: AGENTS.md, implementation_plan.md (+ plots/), sans metrics consolidées.
- RUN_0004: AGENTS.md, notes.md, metrics.json, training_history.json.
