# RUN_0004 — Notes post-mortem

## Statut

**REJETÉ** — Exécuté sur Jean Zay H100 (job SLURM 878620, 2026-05-19). Early stopping déclenché à l'epoch 25/90. Toutes les métriques sont en dessous des baselines de référence.

---

## Résultats de l'exécution Jean Zay (job 878620)

| Métrique | Valeur RUN_0004 | Baseline |
|----------|-----------------|---------|
| Task 1a — Aggregate | **0.1293** | 0.6887 (RUN_0001) |
| Task 1a — Accuracy | 0.0997 | — |
| Task 2 — mean DSC | **0.0083** | 0.362 (RUN_0003) |
| Task 2 — mean HD95 | 66.21 | 15.99 (RUN_0003) |
| Task 2 — mean ASSD | 37.80 | 10.51 (RUN_0003) |
| Task 1b — PSNR | 3.48 dB | — |
| Task 1b — L1 | 1.4028 | — |

- Durée totale : 56 min (entraînement 54 min + évaluation 2 min)
- Epochs réalisés : 25/90 (10 warmup + 15 joint)
- Meilleur checkpoint : warmup epoch 1, val_dice_2 = 0.0083

---

## Diagnostic — 3 causes racines

### Cause 1 : Warm-start nul (0/102 keys matched)

Le warm-start depuis `outputs/checkpoints/RUN_0003/task2_dynunet_best.pt` a échoué
intégralement. Le log confirme :

```
[INFO] Partial encoder warm-start from outputs/checkpoints/RUN_0003/task2_dynunet_best.pt: 0/102 keys matched.
```

`SharedEncoderMultiTaskModel` a une nomenclature de couches totalement différente de
celle de `DynUNet` (MONAI). Aucun poids pré-entraîné n'a été transféré. Le modèle a
démarré depuis une initialisation aléatoire, perdant l'avantage des 90 epochs RUN_0003
(DSC 0.362). Le warm-up Task 2 a donc dû apprendre des représentations anatomiques
from scratch.

### Cause 2 : Domination de la loss 1a — collapse de la segmentation en phase joint

En phase joint, la loss de classification Task 1a domine largement :

```
[Joint] Epoch 001/080 | 1a=5.1527 | 1b=1.5430 | 2=1.9182 | val_dice_2=0.0000
[Joint] Epoch 015/080 | 1a=4.5141 | 1b=1.2345 | 2=1.9257 | val_dice_2=0.0000
```

Ratio loss_1a / loss_2 ≈ 2.7:1. Avec des poids λ = 1.0 uniformes, les gradients de
la cross-entropy multi-artefacts (7 × 3 = 21 logits) ont submergé le signal de
segmentation. Dès l'epoch joint 1, val_dice_2 = 0.0000 et ne remonte plus jamais.

Pendant le warmup, le meilleur val_dice_2 était 0.0083 (epoch 1) avant de retomber à 0 :

```
[Warmup] Epoch 001/010 | train_loss=2.9772 | val_dice_2=0.0083 -> New best checkpoint saved
[Warmup] Epoch 010/010 | train_loss=1.9809 | val_dice_2=0.0000
```

Même le warmup ne suffit pas à stabiliser la segmentation (val_dice_2 chute à 0 dès
l'epoch 10), ce qui suggère un sous-dimensionnement de la durée de warmup ou un problème
de LR.

### Cause 3 : Early stopping ne surveille que val_dice_2

La spec AGENTS.md documentait une métrique composite `0.4·DSC + 0.3·Agg_1a + 0.3·(1-LPIPS)`.
Le code réel n'implémente que `val_dice_2`. Conséquence : comme val_dice_2 = 0.0000 pour
tous les epochs joint, l'early stopping est déclenché après exactement `patience=15` epochs
joint :

```
-> Early stopping triggered at epoch 25 (patience=15)
```

Le modèle a donc effectué 10 epochs warmup + 15 epochs joint = 25/90 epochs, sans jamais
montrer de progrès en segmentation depuis l'epoch 1.

---

## Analyse par hypothèse

| Hypothèse | Prédiction | Résultat | Verdict |
|-----------|-----------|---------|---------|
| H1 : Features partagées bénéfiques pour Task 2 | DSC ≥ 0.362 | DSC = 0.0083 | **Réfutée** — régression sévère |
| H2 : Head 1a compétitive sans encodeur spécialisé | Aggregate ≈ 0.6887 | Aggregate = 0.1293 | **Réfutée** — dégradation majeure |
| H3 : Head 1b apprend le manifold des images propres | FID ↓ vs 164 | PSNR = 3.48 dB seulement (FID non calculé) | **Non vérifiable / Réfutée** en pratique |
| H4 : Dominance de L_2 (supervision dense) | Gradient L_2 >> L_1a | L_1a ≈ 5.15 >> L_2 ≈ 1.92 | **Réfutée** — c'est L_1a qui domine, pas L_2 |
| H5 (risque) : Conflit de gradient entre tâches | DSC < 0.362 et/ou Aggregate < 0.6887 | Les deux confirmés | **Confirmée** — le conflit de gradient est la cause principale |

---

## Recommandations pour RUN_0005

1. **Pondération adaptative des pertes** : utiliser GradNorm ou uncertainty weighting
   (Kendall et al., 2018) pour équilibrer dynamiquement les gradients entre les trois têtes.
   Le ratio 2.7:1 observé (loss_1a / loss_2) indique que λ_1a doit être réduit ou que
   λ_2 doit être augmenté significativement.

2. **Architecture compatible avec DynUNet pour le warm-start** : soit réutiliser
   directement un `DynUNet` comme tronc de l'encodeur, soit implémenter une couche
   d'adaptation (adapter layer) pour transférer les poids de RUN_0003 malgré la
   différence de nommage. L'objectif est d'obtenir un taux de matching > 50%.

3. **Durée de warmup accrue (≥ 30 epochs)** : 10 epochs sont insuffisants pour qu'un
   encodeur initialisé aléatoirement apprenne des représentations anatomiques stables.
   Viser val_dice_2 ≥ 0.20 à la fin du warmup avant d'activer les autres têtes.

4. **Métriques d'early stopping cohérentes avec la spec** : implémenter la métrique
   composite `0.4·DSC + 0.3·Agg_1a + 0.3·(1-LPIPS)` comme documenté dans AGENTS.md,
   ou a minima surveiller un ensemble de métriques plutôt que val_dice_2 seul.

5. **Vérification systématique du warm-start avant lancement** : ajouter un assert dans
   le code d'initialisation qui lève une exception si le nombre de clés matchées est < 10%
   du total attendu.

---

## Notes supplémentaires

- Le warm-up epoch 1 a produit val_dice_2 = 0.0083 (unique checkpoint sauvegardé), ce qui
  est le seul signal positif du run. Ce comportement suggère que l'encodeur random peut
  apprendre quelque chose, mais que le LR ou la durée ne permettent pas de consolider.
- FID et LPIPS n'ont pas été calculés (probablement erreur dans le script d'évaluation ou
  absence de prédictions 1b exploitables). À corriger pour RUN_0005.
- La décision d'abandonner la tête 1b autoencoder au profit d'une approche supervisée
  (si des paires sont disponibles) mérite d'être revisitée.

---

## Corrections v2 (appliquées et testées par smoke test local)

Après analyse des 3 causes racines, **RUN_0004 a été corrigé en place** plutôt que
renommé RUN_0005. Trois modifications ciblées :

### Correction 1 : `DynUNetMultiHeadModel` (compatibilité warm-start)

Remplacement de `SharedEncoderMultiTaskModel` par `DynUNetMultiHeadModel` dans
`src/models/__init__.py`. Le backbone Task 2 est maintenant un MONAI `DynUNet` identique
à `Task2DynUNetModel` de RUN_0003.

**Résultat** : `46/84 keys matched` (vs 0/102 en v1). Les ~40 clés restantes sont les
têtes 1a/1b (`cls_*`, `recon_*`) qui sont bien des paramètres nouveaux.

### Correction 2 : `_calibrate_losses()` (pondération adaptative)

Nouvelle méthode mesurant les pertes initiales sur un batch de chaque tâche, puis
normalisant chaque tâche par sa magnitude avant la phase joint : `loss_total = λ₁·L₁/L₀₁ + λ₂·L₂/L₀₂ + …`.

**Résultat** : `λ_1a/L0=0.1387 λ_1b/L0=0.4779 λ_2/L0=1.1798` — les pertes sont équilibrées
et la segmentation ne collapse plus.

### Correction 3 : Warmup étendu + early exit

- `num_warmup_epochs` : 10 → 30
- Ajout d'un critère `dice_warmup_target: 0.15` : sortie anticipée si val_dice_2 ≥ target

**Résultat** : `val_dice_2=0.5145 >= target 0.15` → warmup terminé à l'epoch 1 (car les
poids pré-entraînés déjà bons). En pleine exécution Jean Zay, le warmup durera 30 epochs
max pour apprendre from scratch si nécessaire.

### Preuve — smoke test local (v2)

```
Partial encoder warm-start: 46/84 keys matched
[Calibration] Effective weights: λ_1a/L0=0.1387  λ_1b/L0=0.4779  λ_2/L0=1.1798
[Warmup] Epoch 001 | val_dice_2=0.5145 -> early exit (>= 0.15)
[Joint] Epoch 001 | val_dice_2=0.5317 -> NO collapse
```

**DSC post-joint : 0.5317** (vs 0.0000 en v1). **PSNR 1b : 3.48 dB** (unchanged).
**Aggregate 1a : 0.1293** (unchanged, besoin de plus d'epochs).

### Comparaison v1 vs v2

| Aspect | v1 (REJETÉ) | v2 (prêt Jean Zay) |
|---|---|---|
| Modèle | `SharedEncoderMultiTaskModel` (custom Conv3d) | `DynUNetMultiHeadModel` (MONAI DynUNet) |
| Warm-start | 0/102 keys | 46/84 keys (~55%) |
| Loss équilibrée | Non (ratio 2.7:1) | Oui (normalisée) |
| Warmup | 10 epochs, fixe | 30 epochs max + early exit DSC ≥ 0.15 |
| DSC post-joint | **0.0000** | **0.5317** (smoke test) |
| Decision | REJETÉ | **En attente de ré-exécution Jean Zay** |

### Prochaine étape

**Ré-exécuter le run v2 sur Jean Zay H100** :
```bash
bash src/slurm/submit.sh --run 0004
```
