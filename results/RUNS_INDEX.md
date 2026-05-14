# RUNS_INDEX

Global run registry for LISA2026 experiments.

| Run ID | Date | Tasks | Parent Run | Scope | Comparable | Status | Primary Score | Secondary Notes | Run AGENTS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RUN_0001 | 2026-05-12 | Task 1a | none | baseline | yes | completed | acc=0.6887 | Mono-task DenseNet264x7 baseline, fixed split, CE loss | results/runs/RUN_0001/AGENTS.md |
| RUN_0002 | — | Task 1a | RUN_0001 | incremental | yes | — | — | See results/runs/RUN_0002/ | results/runs/RUN_0002/AGENTS.md |
| RUN_0003 | 2026-05-14 | Task 2 | none | architectural | no (new task) | ✅ baseline | mean_dsc=0.500 | DynUNet 12-class baseline, DiceFocal, TTA+LC, 12 subjects val | results/runs/RUN_0003/AGENTS.md |
| RUN_0003_COLLAPSED | 2026-05-14 | Task 2 | RUN_0003 | architectural | ⚠️ not comparable (6 vs 11 eval classes) | to_compare | mean_dsc=0.684 (6 classes) | DynUNet 7-class, L/R fused, DiceCE, no TTA — metric space differs | results/runs/RUN_0003_COLLAPSED/AGENTS.md |
| **RUN_0003_EXP_C** | 2026-05-13 | Task 2 | RUN_0003 | architectural | yes | **✅ promoted** | **mean_dsc=0.631** | Best 12-class run — filters=[32,64,128,256,320], DiceCE, LR=1e-4, no TTA | results/runs/RUN_0003_EXP_C/AGENTS.md |

## Status legend

- planned
- running
- completed
- promoted
- rejected
- to_compare
- archived

## Task 2 — Multi-structure Segmentation

| Run ID | Date | Parent | Scope | Model | Loss | Decision | DSC | HD95 | ASSD |
|--------|------|--------|-------|-------|------|----------|-----|------|------|
| RUN_0003 | 2026-05-14 | — | Baseline | DynUNet 12-class filters=[48,96,192,384,512] | DiceFocal | ✅ Baseline | 0.500 | 23.17 | 6.14 |
| RUN_0003_COLLAPSED | 2026-05-14 | RUN_0003 | Architectural | DynUNet 7-class (L+R merged) | DiceCE | 🔄 À comparer | 0.684† | 10.29 | 2.67 |
| **RUN_0003_EXP_C** | 2026-05-13 | RUN_0003 | Incremental | DynUNet 12-class filters=[32,64,128,256,320] | DiceCE | **✅ Promoted** | **0.631** | **9.58** | **2.22** |

_† : COLLAPSED évalué sur 6 classes fusionnées (espace non comparable aux runs 12-class)_

### Runs archivés (cycle RUN_0003)

| Run | Raison | DSC | Archive |
|-----|--------|-----|---------|
| RUN_0003_EXP_A | Training divergé (modèle prédit fond uniquement) | 0.006 | results/runs/archive/RUN_0003_EXP_A_archived.md |
| RUN_0003_EXP_B | Checkpoint skip silencieux → from scratch, sous-performant | 0.322 | results/runs/archive/RUN_0003_EXP_B_archived.md |
| RUN_0003_EXP_C_TTA | TTA LR naïf détruit labels L/R asymétriques | 0.062 | results/runs/archive/RUN_0003_EXP_C_TTA_archived.md |
| RUN_0003_EXP_SYM | Régression vs EXP_C (symmetry loss nuisible) | 0.579 | results/runs/archive/RUN_0003_EXP_SYM_archived.md |

## Known Issues

1. ✅ **FIXED — REVERSE_MAP bug** (`evaluate_task2_dynunet.py`, fixed 2026-05-14) :
   branche non-collapsed appliquait incorrectement `REVERSE_MAP`, scramblant les classes 7–11
   vers le fond (0). Correction : suppression du remapping dans le chemin non-collapsed.
   Toutes les métriques 12-class du tableau ci-dessus sont issues de la version corrigée.

2. ✅ **FIXED — Silent checkpoint skip** (`train_task2_dynunet.py`, fixed post-EXP_B) :
   ajout d'un `warnings.warn()` et de messages explicites lors d'un skip de checkpoint pour
   incompatibilité architecturale. Le `training_history.json` logue désormais
   `pretrained_checkpoint_loaded: true/false` à chaque run.
