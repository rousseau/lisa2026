# RUN_0006 — Multi-task nnU-Net v2 PlainConvUNet (refactor depuis DynUNet)

## Run Metadata
- **Run ID**: 0006
- **Date**: 2026-06-18
- **Status**: **Promoted — en entraînement**
- **Parent Run**: RUN_0005 (rejected)
- **Warm-start Source**: RUN_0003a (nnU-Net v2, DSC=0.8220)
- **Tasks Covered**: 1a, 1b, 2
- **Change Scope**: **architectural** (remplacement du backbone complet)

## Parent Baseline
- **RUN_0005** (rejected):
  - Task 2 DSC: 0.3523 (failed: no warm-start + lambda_1a=0.5 + no post-processing)
  - Architecture: DynUNetMultiHead 5 stages [32..320] from scratch
  
- **RUN_0004** (warm-start reference):
  - Task 2 DSC: 0.6205 (with 5-stage warm-start from RUN_0003)
  - Architecture: DynUNetMultiHead 5 stages [32..320]

- **RUN_0003a** (warm-start source):
  - Task 2 DSC: 0.8220 (nnU-Net v2, Task 2 only)
  - Architecture: PlainConvUNet 6 stages [32..320,320]
  - Checkpoint: `nnUNet_results/.../fold_0/checkpoint_best.pth`

## Root Cause of Previous Failure

**RUN_0006 original (DynUNet option B) a échoué au smoke test** :
- `val_dice_2 = 0.0272` (désastre)
- Cause: DynUNet monolithic architecture — impossible de charger le stage 6 depuis un modèle 5-stage.
- Seulement 31/101 clés chargées ; tout le decoder restait random.
- **Décision**: abandonner DynUNet, passer au backbone nnU-Net v2 (PlainConvUNet).

## Hypothesis & Changes

### Hypothesis
La limite fondamentale de DynUNet multi-tâche est DSC≈0.62 (RUN_0004), car :
1. DynUNet max 5 stages avec maxpooling (pas de strides custom)
2. Warm-start cross-architecture impossible (stages monolithiques)
3. Le backbone nnU-Net v2 atteint DSC=0.82 en mono-tâche → même backbone en multi-tâche devrait garder ≥0.70

### Changes (RUN_0006 → PLAN A)
| Aspect | RUN_0005/DynUNet | **RUN_0006 PlainConv** |
|--------|------------------|------------------------|
| Backbone | DynUNet (5 stages) | **PlainConvUNet (6 stages)** |
| Architecture | maxpooling monolithic | **strided conv, modular encoder/decoder** |
| Warm-start source | Aucun → RUN_0004 (5 stg) | **RUN_0003a checkpoint_best.pth** |
| Keys loaded | 31/101 (échec) | **292/318 (succès)** |
| Patch size | 128³ | **112×160×128** (nnU-Net optimal) |
| Batch size | 1 | **2** |
| Features | [32,64,128,256,320] | **[32,64,128,256,320,320]** |
| Strides | maxpool 2×2×2 | **[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,2,2]]** |
| Optimiser | SGD + Nesterov | **idem (nnU-Net default)** |
| LR | 1e-2 poly | **idem** |
| lambda_1a | 0.5 → 0.1 | **0.1** |
| lambda_1b | 1.0 | **1.0** |
| lambda_2 | 1.0 | **1.0** |
| Post-processing | false → true | **true (LCC)** |
| Deep supervision | 5 levels [0.5,0.25,0.125,0.0625,0.03125] | **idem** |

## Implementation Changes

### Code modifications
1. **Créé** `src/models/plainconv_multihead.py`:
   - `PlainConvMultiHeadModel` avec `PlainConvEncoder` + `UNetDecoder`
   - 3 têtes: Task 2 (decoder nnU-Net), Task 1a (GAP+MLP), Task 1b (reconstruction ConvTranspose)
   - `load_pretrained_nnunet()` — charge uniquement `network_weights` du checkpoint nnU-Net v2
   - Gestion des strides asymétriques via `interpolate()` dans Task 1b

2. **Modifié** `src/models/__init__.py`:
   - Export `PlainConvMultiHeadModel`

3. **Modifié** `src/training/multitask.py`:
   - `_build_model()` supporte `model.type=plainconv/nnunet`
   - `_load_pretrained()` utilise `load_pretrained_nnunet()` pour PlainConv

4. **Modifié** `src/runners/evaluate_multitask.py`:
   - Instancie `PlainConvMultiHeadModel` selon config

5. **Modifié** `src/cli/registry.py`:
   - Ajout mapping `("1a+1b+2", "plainconv")` → modules multitask

6. **Créé** `configs/run_0006_plainconv.yaml`:
   - Params nnU-Net v2 exacts (patch [112,160,128], batch=2, strides)
   - Warm-start vers `checkpoint_best.pth` RUN_0003a

## Training & Evaluation Configuration
- **Config local**: `configs/run_0006_plainconv.yaml`
- **Config Jean Zay**: `configs/run_0006_jeanzay.yaml` (à créer)
- **Patch size**: 112×160×128
- **Batch size**: 2
- **Epochs**: 1000
- **Warm-up**: 50 epochs
- **Early stopping patience**: 200
- **Checkpoint**: `outputs/checkpoints/RUN_0006/multitask_best.pt`
- **Hardware**: 1× H100 80GB (Jean Zay) ou local GB10

## Warm-start Details

```
[INFO] Loaded 292/318 keys from nnU-Net checkpoint_best.pth
```
- **292 clés** : encoder stages 0-5 (100%), decoder stages 0-4 (100%), skip connections
- **26 clés restantes random** :
  - Task 1a head: `cls_gap`, `cls_mlp` (2 keys)
  - Task 1b decoder: `recon_ups.*`, `recon_out` (~24 keys)

## Smoke Test (local, NVIDIA GB10, 2026-06-18)

```
[INFO] PlainConvMultiHead warm-started: 292 keys loaded.
Device: cuda | AMP: True
Phases — warmup=1ep  head_warmup=0ep  joint=1ep | λ=(1a=0.1, 1b=1.0, 2=1.0) grad_clip=12.0

[Warmup] 001/001 | loss=0.8877 | val_dice_2=0.2334
  -> Warmup early exit (dice=0.2334 >= 0.10)
[Joint] 001/001 (g=002) | total=1.9993 1a=5.1838 1b=0.9246 2=0.8208 | val_dice_2=0.2932
  -> New best (val_dice_2=0.2932)
```

| Métrique | Smoke Test | Commentaire |
|----------|------------|-------------|
| val_dice_2 warmup | **0.2334** | Encodeur pré-entraîné → convergence rapide |
| val_dice_2 joint | **0.2932** | Progression en 1 epoch |
| Task 1a loss | 5.18 | Calibration normale (pas de NaN) |
| Task 1b loss | 0.92 | Normal |
| Temps/epoch | ~7 min | Largement acceptable |

**vs RUN_0006 original (DynUNet)** : 0.0272 → **0.2932** (+10×) — architecture validée.

## Expected Outcomes
| Metric | RUN_0005 | RUN_0004 | RUN_0003a | **RUN_0006 target** |
|--------|----------|----------|-----------|---------------------|
| Task 2 DSC | 0.3523 | 0.6205 | **0.8220** | **≥ 0.70** |
| Task 1a aggregate | 0.4887 | 0.3956 | N/A | ~0.40-0.50 |
| Task 1b PSNR | 25.90 | 21.40 | N/A | ~20-26 dB |

## Comparability Statement
- **Non comparable à RUN_0005** en architecture (changement total du backbone).
- **Comparable à RUN_0004** pour le concept multi-tâche, mais backbone différent.
- **Référence RUN_0003a** pour Task 2 seul — RUN_0006 utilise le même backbone + têtes supplémentaires.

## Decision
- [x] **Promoted** — smoke test validé, prêt pour full training
- [ ] Rejected
- [ ] To retest

## Next Steps
1. [ ] Full training sur Jean Zay H100
2. [ ] Évaluation validation complète
3. [ ] Comparaison avec RUN_0003a (Task 2) et RUN_0004 (multi-tâche)
4. [ ] Si DSC ≥ 0.70 → documenter comme baseline multi-tâche
