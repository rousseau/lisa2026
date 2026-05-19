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
