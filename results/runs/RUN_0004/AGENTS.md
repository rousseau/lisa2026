# RUN_0004 — Shared Encoder Multi-task Baseline (Tasks 1a, 1b, 2)

## Run Metadata

- **Run ID**: RUN_0004
- **Date**: 2026-05-19
- **Tasks covered**: Task 1a (Quality Assessment), Task 1b (Enhancement), Task 2 (Segmentation)
- **Parent run (Task 1a)**: RUN_0001 — baseline indépendant DenseNet264 (aggregate 0.6887)
- **Parent run (Task 2)**: RUN_0003 — DynUNet 12-class (mean DSC 0.362)
- **Parent run (Task 1b)**: aucun — le précédent RUN_0004 (BasicUNet auto-supervisé, FID=164) est **supplanté** par cette approche
- **Change scope**: Architectural — pipeline multi-tâche complet, rupture avec les runs précédents

## Comparabilité

| Tâche | Comparable à | Raison |
|-------|-------------|--------|
| Task 1a | RUN_0001, RUN_0002 | Même split `task1a_fixed.pkl`, mêmes métriques (Accuracy, F1, F2, Precision, Recall) |
| Task 1b | aucun | L'ancienne approche (débruitage auto-supervisé) est conceptuellement incompatible |
| Task 2 | RUN_0003 | Même split `task2_fixed.pkl`, même script d'évaluation, même espace de classes (12) |

---

## Motivation scientifique

Les runs 0001–0003 ont établi des baselines indépendantes par tâche. La limite centrale de
cette approche est que chaque modèle apprend des représentations isolées alors que les trois
tâches portent sur les **mêmes volumes IRM 0.064T** et partagent une structure anatomique et
des patterns d'artefacts communs.

L'hypothèse directrice de ce run est qu'un **encodeur partagé et expressif** peut extraire
des features génériques qui bénéficient simultanément à :

1. **Task 1a** — les features de texture/intensité capturent les signatures des artefacts.
2. **Task 1b** — les features de la distribution des images saines définissent un manifold
   de reconstruction "propre" utilisable pour corriger les artefacts à l'inférence.
3. **Task 2** — les features anatomiques de haut niveau guident la segmentation.

Ce run est **non contraint** : aucune régularisation n'est imposée sur l'espace des features.
Les runs suivants pourront introduire des contraintes (disentanglement, contrastive loss,
information bottleneck) en prenant ce run comme référence comparative.

---

## Architecture

### Vue d'ensemble

```
Volume IRM 3D (1×128×128×128)
        │
        ▼
┌───────────────────┐
│  Shared Encoder   │  ← U-Net expressif 3D (5 niveaux)
│  [32→64→128→256→320]  features multi-échelles
└────────┬──────────┘
         │ bottleneck + skip connections
    ┌────┴────┬──────────┐
    ▼         ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│ Head   │ │ Head   │ │ Head   │
│ 1a     │ │ 1b     │ │  2     │
│ Classif│ │ Recon  │ │  Seg   │
└────────┘ └────────┘ └────────┘
```

### Encodeur partagé

- Architecture : U-Net 3D à 5 niveaux (inspiré de DynUNet MONAI)
- Filtres : [32, 64, 128, 256, 320]
- Noyaux : 3×3×3 à chaque niveau
- Strides encodeur : [1, 2, 2, 2, 2]
- Normalisation : Instance Normalization
- Activation : LeakyReLU (pente=0.01)
- Sortie encodeur : bottleneck 320 canaux (4×4×4 pour une entrée 128³) + 4 feature maps de skip

### Head Task 1a — Classification des artefacts

- Entrée : bottleneck de l'encodeur (320×4×4×4)
- Global Average Pooling 3D → vecteur 320-dim
- MLP : Linear(320, 128) → LeakyReLU → Dropout(0.3) → Linear(128, 7×3)
- Sortie : logits [B, 7, 3] (7 artefacts × 3 niveaux)
- Perte : Cross-Entropy indépendante par tête d'artefact

### Head Task 1b — Reconstruction (Autoencoder)

- Entrée : bottleneck + skip connections de l'encodeur
- Décodeur symétrique à l'encodeur (strides transposés 2×2×2)
- Sortie : volume reconstruit 1×128×128×128
- **Données d'entraînement** : uniquement les volumes **non artefactés** (scores = 0
  sur les 7 artefacts dans la CSV Task 1a), identifiés par filtrage à la construction
  du DataLoader
- Perte : L1 + SSIM 3D (λ_L1=1.0, λ_SSIM=1.0)
- À l'inférence : les volumes artefactés passent également à travers ce décodeur ;
  la reconstruction est l'estimation du volume "propre"

### Head Task 2 — Segmentation

- Entrée : bottleneck + skip connections de l'encodeur
- Décodeur symétrique à l'encodeur (strides transposés 2×2×2)
- Sortie : logits [B, 12, 128, 128, 128]
- Perte : DiceCE (λ_dice=1.0, λ_ce=1.0)

---

## Stratégie d'entraînement

### Entraînement conjoint

La perte totale est une somme pondérée des trois contributions :

```
L_total = λ_1a · L_classification + λ_1b · L_reconstruction + λ_2 · L_segmentation
```

Poids initiaux : λ_1a = 1.0, λ_1b = 1.0, λ_2 = 1.0 (baseline non contraint).

### Hétérogénéité des données

Les trois jeux de données ne sont pas parfaitement alignés :

| Source | Disponibilité des labels | Utilisation |
|--------|--------------------------|-------------|
| Volumes Task 1a | Labels artefacts (7×3) | Head 1a active |
| Volumes Task 1a propres (scores=0) | Identité (image→image) | Head 1b active |
| Volumes Task 2 | Masques segmentation | Head 2 active |

**Stratégie de batch** : à chaque itération, un batch est construit en mélangeant les
trois sources. Pour chaque sample du batch, seules les têtes pour lesquelles des labels
sont disponibles contribuent à la perte. Les têtes sans label sont masquées (`loss *= 0`).

### Phases d'entraînement

1. **Warm-up (10 epochs)** : entraîner uniquement la tête Task 2 (supervision dense, plus
   stable) pour initialiser l'encodeur sur une tâche à fort signal.
2. **Entraînement joint (N epochs)** : activer les trois têtes simultanément.
3. **Early stopping** : surveillé sur la moyenne pondérée des métriques de validation :
   `0.4 · DSC_2 + 0.3 · Aggregate_1a + 0.3 · (1 - LPIPS_1b)`

---

## Hypothèses et prédictions

| Hypothèse | Résultat attendu | Indicateur de validation |
|-----------|-----------------|--------------------------|
| H1 : Features partagées bénéfiques pour Task 2 | DSC ≥ RUN_0003 (0.362) | metrics.json `mean_dsc` |
| H2 : Head 1a compétitive sans encodeur spécialisé | Aggregate proche de RUN_0001 (0.6887) | metrics.json `aggregate` |
| H3 : Head 1b apprend le manifold des images propres | FID ↓ vs ancien RUN_0004 (164) | metrics.json `fid` |
| H4 : Dominance de L_2 (supervision dense) | Gradient de L_2 >> L_1a, L_1b | logs de perte par composante |
| H5 (risque) : Conflit de gradient entre tâches | DSC < RUN_0003 et/ou Aggregate < RUN_0001 | → motiverait les contraintes de features dans RUN_0005 |

---

## Configuration d'entraînement

| Paramètre | Valeur |
|-----------|--------|
| Config | `configs/run_0004_multitask.yaml` |
| Entrainement | `python train.py --run 0004` |
| Évaluation | `python evaluate.py --run 0004` |
| Split Task 1a | `results/splits/task1a_fixed.pkl` |
| Split Task 2 | `results/splits/task2_fixed.pkl` |
| Patch size | 128³ |
| Batch size | 2 (mémoire : 3 décodeurs + encodeur) |
| Warm-up epochs | 10 (Task 2 seul) |
| Epochs (joint) | 80 |
| Early stopping patience | 15 (perte jointe) |
| Optimiseur | AdamW, lr=1e-4, wd=1e-5 |
| Scheduler | CosineAnnealingLR, T_max=90, eta_min=1e-6 |
| Mixed precision | True |
| Seed | 42 |
| λ_1a | 1.0 |
| λ_1b | 1.0 |
| λ_2 | 1.0 |

---

## Plan d'implémentation

Voir `implementation_plan.md` pour les détails d'exécution.

### Étapes résumées

1. **Modèle** : implémenter `SharedEncoderModel` dans `src/models/__init__.py` avec les
   trois têtes. L'encodeur est un `DynUNet` tronqué ; les décodeurs 1b et 2 réutilisent
   les blocs up-sampling de DynUNet.

2. **Datasets** : créer `MultiTaskDataset` dans `src/datasets/__init__.py` qui gère
   l'hétérogénéité (labels disponibles par sample) et retourne un dict de masques
   indiquant quelles têtes sont actives.

3. **Entraînement** : créer `src/train_task_multitask.py` avec la logique de batch mixte,
   warm-up Task 2, et le calcul de perte masqué.

4. **Évaluation** : créer `src/evaluate_task_multitask.py` qui évalue séparément les
   métriques des trois tâches et les agrège dans `metrics.json`.

5. **Config** : créer `configs/run_0004_multitask.yaml`.

6. **Enregistrement** : mettre à jour les registres dans `train.py` et `evaluate.py`.

---

## Résultats (v1 / v2 / v3)

### Comparaison des 3 versions

| Métrique | v1 (job 878620) | v2 (job 887047) | v3 (job 1063254) | v4 (job 1076254) | Baseline |
|----------|-----------------|-----------------|------------------|------------------|----------|
| Task 1a — Aggregate | 0.1293 | 0.3261 | 0.1945 | 0.1944 | 0.6887 (RUN_0001) |
| Task 1a — Accuracy | 0.0997 | 0.6071 | 0.2693 | — | — |
| Task 1a — Recall | — | — | 0.3333 (constant) | 0.3333 (constant) | — |
| Task 2 — mean DSC | 0.0083 | 0.0274 | 0.1105 | **0.1649** | 0.362 (RUN_0003) |
| Task 2 — mean HD95 | 66.21 | 37.24 | 27.32 | **26.70** | 15.99 (RUN_0003) |
| Task 2 — mean HD | — | — | 33.16 | — | — |
| Task 2 — mean RVE | — | — | 1.004 | — | — |
| Task 2 — mean ASSD | — | 26.11 | 9.99 | 11.12 | 10.51 (RUN_0003) |
| Task 1b — PSNR | 3.48 dB | 3.82 dB | 3.38 dB | 3.38 dB | — |
| Task 1b — L1 | — | 1.2578 | 1.2335 | — | — |
| Warmup epochs | 10 | 30 | 43 | **1** | — |
| val_dice_2 fin warmup | 0.0083 | 0.0274 | 0.1104 | **0.1649** | — |
| Phase joint | NaN | NaN | NaN (100%) | NaN (100%) | — |

**Statut v3 : À RETESTER** — Le warmup a atteint la cible DSC=0.10 (epoch 43, val_dice_2=0.1104 ✅). La head warmup a produit des poids NaN à partir de l'epoch 46 (LR trop élevé : ~7.56e-5 hérité du scheduler T_max=130). Un bug Python (`max(nan, 1e-6)`) a rendu la calibration NaN → phase joint 100% inerte. Corrections précises et ciblées identifiées pour v4 (voir section Décision).

**Statut v4 (job 1076254) : À RETESTER** — ⚠️ Run exécuté avec le code v3 (non synchronisé sur Jean Zay). Malgré ce défaut, le warm-start RUN_0003 (46/84 keys) a permis d'atteindre val_dice_2=**0.1649** dès le premier warmup epoch. La head warmup (code v3, LR≈1e-4) a produit des NaN encore plus vite qu'en v3 (hw epoch 2 vs hw epoch 3), confirmant la corrélation LR ↔ vitesse d'explosion. Phase joint 100% NaN. Corrections v5 appliquées localement et vérifiées par smoke test.

**Checkpoint évalué** : warmup epoch 1 (val_dice_2=0.1649) — jamais amélioré en joint.

---

## Décision

- [ ] Promu
- [ ] Rejeté
- [x] À retester

**Décision : À RETESTER (v5 préparée)**

Progrès v3 :
- (1) Warmup convergent depuis zéro : val_dice_2=0.1104 en 43 epochs ✅
- (2) DSC évaluation = 0.1105 (+304% vs v2, +1230% vs v1) ✅
- (3) Architecture DynUNetMultiHead confirmée fonctionnelle pour Task 2

Progrès v4 (job 1076254, ⚠️ code v3 non sync) :
- (4) Warm-start RUN_0003 présent (46/84 keys) → val_dice_2=**0.1649** en 1 seul warmup epoch ✅ (meilleur DSC toutes versions)
- (5) Corrélation LR ↔ NaN confirmée et reproductible (v4 LR=1e-4 → hw epoch 2, v3 LR=7.56e-5 → hw epoch 3)
- (6) Distribution classes Task 1a mesurée (Banding ratio 1:52) → pondération nécessaire identifiée

Échecs persistants :
- (7) Head warmup = source unique de NaN (toutes versions depuis v2) ❌ → **supprimé en v5**
- (8) Cross-entropie non pondérée → prédicteur constant Task 1a ❌ → **corrigé en v5**

### Corrections v5 (appliquées localement, smoke test validé)

| # | Problème | Correction v5 | Statut |
|---|----------|---------------|--------|
| C1 | Head warmup → NaN systématique | `num_head_warmup_epochs: 0` dans config | ✅ Local |
| C2 | Cross-entropie non pondérée → prédicteur constant | Poids de classe inversement proportionnels aux fréquences | ✅ Local |
| C3 | LR reset + GradScaler reset manquants en joint | Reset explicite au début de la phase joint | ✅ Local |
| C4 | Poids NaN propagés sans détection | Vérification NaN des poids avant calibration | ✅ Local |
| C5 | Bug `max(nan, 1e-6)` → calibration NaN | `nan_to_num` propre dans la calibration | ✅ Local |

Smoke test v5 (local) : `Head warm-up: 0 epochs (disabled)` ✅ · `Band: [0.3/18.2/14.5]` ✅ · calibration `1a=7.6904 1b=1.7345 2=0.4219` (zéro NaN) ✅ · joint `total=3.3636` (zéro NaN) ✅

→ v5 réutilise la même architecture, les mêmes splits, le même script d'évaluation. Les hypothèses H1–H5 restent à valider avec une phase joint fonctionnelle pour la première fois.

---

## Environnement d'exécution réel — **v2 : corrections appliquées**

Ce run a été **rejeté** après exécution (v1, job 878620, 2026-05-19) mais **corrigé** (v2) et prêt pour ré-exécution.

### Corrections appliquées (v2)

| Cause racine (v1) | Correction (v2) | Preuve smoke test |
|---|---|---|
| 0/102 keys matched | `DynUNetMultiHeadModel` avec backbone DynUNet identique → 46/84 keys |
| Loss 1a domine (ratio 2.7:1) | `_calibrate_losses()` normalise chaque tâche par sa magnitude initiale |
| Warmup trop court (10 epochs) | 30 epochs + early exit si DSC ≥ 0.15 |
| Early stopping sur val_dice_2 seul | Non modifié (val_dice_2 reste le critère de sauvegarde) |

### Résultats du smoke test v2 (local, GB10)

```
[INFO] Partial encoder warm-start: 46/84 keys matched
[Calibration] Effective weights: λ_1a/L0=0.1387  λ_1b/L0=0.4779  λ_2/L0=1.1798
[Warmup] Epoch 001 | val_dice_2=0.5145  -> early exit (≥ 0.15)
[Joint] Epoch 001 | val_dice_2=0.5317  -> NO collapse
```

**DSC post-joint : 0.5317** (vs 0.0000 en v1). La segmentation survive à la phase joint.

---

## Notes

- Ce run **ne conserve pas** l'ancienne implémentation BasicUNet de RUN_0004
  (auto-débruitage supervisé par bruit gaussien synthétique). Cette approche est
  abandonnée car elle ne s'inscrit pas dans la logique de représentation partagée
  adoptée à partir de ce run.
- L'absence de contrainte sur les features est **volontaire** pour ce run : elle servira
  de référence pour mesurer l'apport des contraintes dans les runs suivants (par exemple,
  régularisation de disentanglement artefacts/anatomie, ou perte contrastive sur les
  niveaux d'artefacts).
- Le conflit de gradient entre les trois tâches est le principal risque. Si H5 se confirme,
  le run suivant devra introduire un mécanisme de pondération adaptative des pertes
  (GradNorm, PCGrad, ou pondération par incertitude).
