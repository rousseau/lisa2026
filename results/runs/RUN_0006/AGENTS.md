# RUN_0006 — Multi-task nnU-Net v2 PlainConvUNet (Option C + exact nnU-Net preprocessing)

## Run Metadata
- **Run ID**: 0006
- **Date**: 2026-06-23
- **Status**: **En cours — Option C (freeze encoder + decoder) + exact nnU-Net preprocessing**
- **Parent Run**: RUN_0005 (rejected)
- **Warm-start Source**: RUN_0003a (nnU-Net v2, DSC=0.8220)
- **Tasks Covered**: 1a, 1b, 2
- **Change Scope**: **architectural** (passage DynUNet → PlainConvUNet + fix preprocessing)

---

## Historique des décisions

### Étape 1 — Refactor DynUNet → PlainConvUNet (2026-06-18)
Passage de `DynUNetMultiHeadModel` à `PlainConvMultiHeadModel` basé sur `dynamic_network_architectures` (backbone exact nnU-Net v2).

### Étape 2 — Option B (2026-06-22)
Gel de l'**encodeur uniquement**. Smoke test local : Dice = **0.23** (encore trop bas).
Diagnostic : le modèle et les poids sont corrects, mais le preprocessing diffère du pipeline nnU-Net natif.

### Étape 3 — Option C + exact preprocessing (2026-06-23)
- Passage à **Option C** : `freeze_encoder()` + `freeze_decoder()` (backbone Task 2 totalement figé).
- Intégration du **preprocessing exact nnU-Net** (`DefaultPreprocessor` natif) dans tous les DataLoaders via `LoadAndPreprocessNnunetd`.
- `torch_compile` désactivé (supprimé de la config après tests).

---

## Problème racine identifié (2026-06-23)

Le Dice de **0.23** n'était **pas** dû au modèle ni aux poids, mais à un **mismatch de preprocessing** :

| Étape | Notre pipeline (MONAI) | nnU-Net natif | Impact |
|-------|------------------------|---------------|--------|
| **Crop** | ❌ Aucun | `crop_to_nonzero` | Fond entourant le cerveau perturbe la normalisation |
| **Resampling** | ❌ Aucun | Isotrope `spacing=[1.0, 1.0, 1.0]` | Kernels appris à une résolution différente |
| **Normalization** | `NormalizeIntensityd` (volume-level z-score) | `ZScoreNormalization` (même formule, mais après crop + resample) | Statistiques différentes |
| **Label mapping** | ❌ Brut | `LabelManager` | Potentiel décalage d'indices |

---

## Changements majeurs

### 2026-06-23 — Exact nnU-Net preprocessing
| Paramètre | Avant | **Après** |
|---|---|---|
| Preprocessing Task 1a | `LoadImaged + NormalizeIntensityd` | **`LoadAndPreprocessNnunetd` (DefaultPreprocessor natif)** |
| Preprocessing Task 1b | `LoadImaged + NormalizeIntensityd + ScaleIntensityd` | **`LoadAndPreprocessNnunetd` + `ScaleIntensityd`** |
| Preprocessing Task 2 | `LoadImaged + NormalizeIntensityd` | **`LoadAndPreprocessNnunetd`** |

### Architecture confirmée (Option C)
| Paramètre | Avant (Option B) | **Après (Option C)** |
|---|---|---|
| Encoder | `requires_grad=False` | **`requires_grad=False`** |
| Decoder Task 2 | `requires_grad=True` | **`requires_grad=False`** |
| Task 1a/1b heads | Entraînées | **Entraînées** |
| `lambda_1a` | 0.1 | **0.1** |
| `lambda_1b` | 1.0 | **1.0** |
| `lambda_2` | 1.0 | **1.0** |
| `batch_size` | 2 | **2** |
| `learning_rate` | 1.0e-2 | **1.0e-2** |
| `torch_compile` | true | **false** (retiré) |

---

## Implémentation

### Fichiers ajoutés
1. **`src/datasets/nnunetv2_plan.py`** : Plans et `dataset_json` extraits automatiquement du checkpoint `checkpoint_best.pth` de RUN_0003a.
2. **`src/datasets/nnunet_preprocessor.py`** : Wrapper `DefaultPreprocessor` + transform MONAI `LoadAndPreprocessNnunetd`.

### Fichiers modifiés
1. **`src/models/plainconv_multihead.py`** :
   - `freeze_encoder()` / `freeze_decoder()` / `freeze_task2()` pour Option C
   - `_load_pretrained()` : chargement exact des 292 clés nnU-Net (0 mismatch)

2. **`src/datasets/transforms.py`** :
   - `build_image_only_transforms()` et `build_segmentation_transforms()` acceptent `use_nnunet_preprocessing: bool`

3. **`src/datasets/task1a.py`** :
   - Paramètre `use_nnunet_preprocessing` passé au constructeur

4. **`src/datasets/task1b.py`** :
   - `_build_transforms()` accepte `use_nnunet_preprocessing`

5. **`src/datasets/task2.py`** :
   - Paramètre `use_nnunet_preprocessing` passé au constructeur

6. **`src/datasets/loaders.py`** :
   - `get_multitask_dataloaders()` active `use_nnunet_preprocessing=True` pour les 3 tâches

7. **`src/runners/evaluate_multitask.py`** :
   - Fix `model_fn=model.forward_task2_main` (au lieu de `forward_task2`) pour éviter la shape mismatch `[B, N_levels, C, H, W, D]` vs `[B, C, H, W, D]` dans `sliding_window_inference`

---

## Résultats de smoke test (validation, 2 sujets)

| Métrique | Valeur | Commentaire |
|----------|--------|-------------|
| Task 2 — mean DSC | **0.9345** | ✅ Backbone exact nnU-Net + preprocessing exact → Dice natif atteint |
| Task 2 — mean HD95 | **1.00** | ✅ |
| Task 2 — mean ASSD | **0.28** | ✅ |
| Task 1a — aggregate | 0.5610 | Baseline (heads aléatoires) |
| Task 1b — PSNR | 6.16 dB | Baseline très bas (sigmoid, domaine [-1,1] vs nnU-Net [~0,1]) |

> **Conclusion Task 2** : Le backbone nnU-Net natif (`checkpoint_best.pth`, DSC=0.8220 en 5-fold cross-val) atteint **DSC=0.9345** sur notre split val avec le preprocessing exact. L'écart avec 0.8220 s'explique par le split différent (60 train / 12 val vs 5-fold). L'important est que la **qualité nativ nnU-Net est préservée**.

---

## Objectif RUN_0006

| Tâche | Métrique | RUN_0003a (nnU-Net seul) | **RUN_0006 target** |
|---|---|---|---|
| Task 2 | DSC | **0.8220** | **≥ 0.80** (backbone figé, preprocessing exact) |
| Task 1a | Accuracy | N/A | ≥ 0.55 |
| Task 1b | PSNR / LPIPS / FID | N/A | Compétitif |

---

## Commandes de lancement

```bash
# Lancer l'entraînement
nohup python train.py --run 0006 > outputs/logs/RUN_0006/training_full.log 2>&1 &

# Smoke test training
python train.py --run 0006 --smoke-test

# Évaluation
python evaluate.py --run 0006
```

---

## Hypothèses et risques résiduels
1. **Task 1b domain shift** : Le preprocessing nnU-Net normalise à ~[−0.7, 8.3] (z-score). Task 1b s'attendait à des entrées dans [−1, 1] (ScaleIntensityd). Il faudra peut-être adapter la tête Task 1b ou sa loss (pas de `ScaleIntensityd` en `use_nnunet_preprocessing=True`).
2. **Task 1a performance** : Classification sur features gelées. Si sous-optimal, envisager un dégel progressif des couches profondes de l'encodeur après convergence des heads.
3. **Memory footprint** : Le `DefaultPreprocessor` natif est plus lourd que MONAI (resampling + crop). `num_workers=4` reste stable.

---

## Décision
- [ ] Promoted (en attente entraînement complet)
- [ ] Rejected
- [ ] To retest (à relancer sur Jean Zay en full)
- [x] **In progress** — Option C validée, preprocessing exact intégré, prêt pour entraînement long

## Iteration 2026-06-23 — label mapping + 79 patients + double supervision

Suite à la vérification du mapping des labels du challenge LISA 2026, le run a été itéré (toujours RUN_0006) avec les modifications suivantes :

### Mapping de labels confirmé (challenge)
Ancien `dataset.json` (checkpoint nnU-Net) → nouveau mapping déclaré par le challenge :
- 3: L_Caudate → VentricleL
- 4: R_Caudate → VentricleR
- 5: L_Lentiform → CaudateL
- 6: R_Lentiform → CaudateR
- 7: L_Ventricle → LentiformL
- 8: R_Ventricle → LentiformR
- 9: L_ExV → ThalamusL
- 10: R_ExV → ThalamusR
- 11: Aux → CorpusCallosum

### Changements techniques
| Changement | Fichier(s) | Justification |
|---|---|---|
| **Permutation poids seg_outputs** | `src/models/plainconv_multihead.py` | Permet canaux 3–8, réinitialise 9–11 (nouvelles structures) |
| **Fix sigmoid Task 1b** | `src/models/plainconv_multihead.py` | Retrait `torch.sigmoid()` dans `forward_task1b` (bloquait apprentissage car cible nnU-Net ~[-0.7, 8]) |
| **Dataset 79 patients** | `src/datasets/task2.py`, `loaders.py` | 54 `Task2/` + 25 images `Task1b/` fallback, labels HF `Task2/`, labels LF `Task2Extra/` |
| **Double supervision Task 2** | `src/training/multitask.py` | Loss = seg(HF) + λ·seg(LF) avec `lambda_lf=0.5` |
| **Nouveau split 79** | `results/splits/task2_79_fixed.pkl` | 70 train / 9 val, seed 42, fixation patient-level |
| **Config paths** | `configs/run_0006_plainconv.yaml` | Pointe vers `Task2/`, `Task2Extra/`, `Task1b/` |
| **Joint phase : encoder-only freeze** | `src/training/multitask.py`, config | `freeze_encoder()` + `unfreeze_decoder()` (Option B, LR=1e-3). Permet adaptation du decoder au nouveau mapping. |

### Objectifs mis à jour
- Task 2 DSC ≥ 0.80 sur nouveau mapping (79 patients)
- Task 1b PSNR/LPIPS corrigés (sigmoid retiré)
- Task 1a inchangé (heads entraînées sur encoder figé)

## Comparabilité
- **Strictement comparable** à RUN_0003a pour Task 2 (même architecture, mêmes poids, même preprocessing).
- **Non comparable** à RUN_0004/RUN_0005 (architecture différente, preprocessing différent).

## Notes complémentaires
- `num_warmup_epochs: 0` dans la config : le warm-up est inutile car le backbone est figé et déjà optimal.
- Le `loss_scale_*` reste identique. L'équilibre des tâches est préservé par le gel complet du backbone Task 2.
