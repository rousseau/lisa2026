# RUN_0004 — Notes post-mortem (Jean Zay v2 : job 887047)

## Statut

**REJETÉ (v2)** — Exécuté sur Jean Zay H100 (job SLURM **887047**, 2026-05-19). La phase joint a produit des pertes **NaN** dès le premier epoch. L'évaluation a été faite sur le meilleur checkpoint du warmup (epoch 30, val_dice_2=0.0274).

---

## Résultats Jean Zay v2 (job 887047)

| Aspect | Résultat |
|------|--|
| Warm-start | **46/84 keys** — **réussi** ✅ |
| Loss calibration | 1a=7.63  1b=2.22  2=1.37 ✅ |
| Effective weights | λ₁/L₀ = 0.131 / 0.450 / 0.731 ✅ |
| Warmup 30 epochs | val_dice_2 : 0.0040 → 0.0274 ⚠️ (cible 0.15 jamais atteinte) |
| Phase joint | **NaN dès epoch 1** ❌ (tous les epochs joint = NaN) |
| Early stopping | epoch 45 (patience 15 de NaN) |
| DSC post-éval | 0.0274 (sur checkpoint warmup) |
| Aggregate post-éval | 0.3261 |
| PSNR post-éval | 3.82 dB |
| Durée | 66 min (65 min training + 1 min eval) |

---

## Résultats Jean Zay v1 (job 878620) — Ancien

| Métrique | Valeur v1 | Baseline |
|------|--|---|
| Task 1a Aggregate | 0.1293 | 0.6887 (RUN_0001) |
| Task 1a Accuracy | 0.0997 | — |
| Task 2 DSC | 0.0083 | 0.362 (RUN_0003) |
| Task 2 HD95 | 66.21 | 15.99 (RUN_0003) |
| Task 1b PSNR | 3.48 dB | — |

- Durée : 56 min
- Epochs : 25/90 (10 warmup + 15 joint)
- Meilleur checkpoint : warmup epoch 1 (val_dice_2=0.0083)

---

## Analyse détaillée v2

### Warmup (30 epochs, Task 2 seul)

| Epoch | train_loss | val_dice_2 |
|------|--|----|
| 1 | 3.3681 | 0.0040 |
| 8-25 | ~1.6-2.2 | **0.0000** (plateau) |
| 26 | 1.4703 | 0.0008 (reprise) |
| 28 | 1.4258 | 0.0097 |
| 29 | 1.4043 | 0.0231 |
| 30 | 1.3836 | **0.0274** |

**Analyse** : val_dice_2 est resté à 0 pendant 18 epochs (8→25). L'encodeur n'apprend que lentement et n'atteint jamais la cible de 0.15. La reprise tardive (epochs 26→30) suggère que l'encodeur doit apprendre des représentations anatomiques de zéro.

### Calibration

```
[Calibration] Initial losses — 1a=7.6333  1b=2.2211  2=1.3690
[Calibration] Effective weights after normalisation — 
  λ_1a/L0=0.1310  λ_1b/L0=0.4502  λ_2/L0=0.7305
```

**Interprétation** : Les pertes initiales sont équilibrées (λ/L0 ≈ 1.0 pour la tâche la plus facile). λ_2/L0=0.7305 est le plus proche de 1.0 → la tâche 2 est la plus facile à optimiser.

### Phase joint — Échec NaN

```
[Joint] Epoch 001/080 (global 031) | total=nan | 1a=nan | 1b=nan | 2=nan | val_dice_2=0.0000
[Joint] Epoch 015/080 (global 045) | total=nan | val_dice_2=0.0000
  -> Early stopping triggered at epoch 45 (patience=15)
```

**Toutes les 15 epochs joint ont produit NaN**. Le modèle est devenu instable dès le premier forward.

### Causes du NaN

1. **Tête 1a aléatoire** : `cls_mlp` est initialisé aléatoirement (pas de warm-start). Le gradient de la cross-entropy sur 21 logits (7×3) est instable quand l'encodeur shared n'est pas encore stabilisé.

2. **AMP + gradient explosion** : Le GradScaler de PyTorch AMP peut overflow si un gradient dépasse 65504 (fp16 max). Avec batch_size=1 et une tête random, le gradient peut exploser dans l'encodeur partagé.

3. **Pas de gradient clipping** : Le code n'utilise pas `torch.nn.utils.clip_grad_norm_`, laissant les gradients non bornés.

---

## Analyse par hypothèse

| Hypothèse | Résultat | Verdict |
|------|--|---|--|
| H1 : Features partagées bénéfiques pour Task 2 | DSC=0.0274 (vs 0.362) | **Réfutée** |
| H2 : Head 1a compétitive sans encodeur spécialisé | Aggr=0.3261 (vs 0.6887) | **Partiellement** (0.33 vs 0.69) |
| H3 : Head 1b apprend le manifold | PSNR=3.82 dB | **Inconnu** |
| H4 : Dominance de L_2 (supervision dense) | L_2/L_0=0.731 (le + proche de 1.0) | **Réfutée** (L_1a/L_0=0.131 est le + petit) |
| H5 (risque) : Conflit de gradient | **NaN** au lieu de gradient conflict | **Confirmée** (pire que prévu) |

---

## Comparaison v1 vs v2

| Aspect | v1 (job 878620) | v2 (job 887047) | Delta |
|------|---|---|---|
| Modèle | SharedEncoderMultiTaskModel | DynUNetMultiHeadModel | ✅ |
| Warm-start | 0/102 keys ❌ | **46/84 keys** ✅ | +46 keys |
| Loss calibration | Non | **Oui** ✅ | OK |
| Warmup | 10 epochs | **30 epochs** ✅ | +20 |
| Phase joint | val_dice_2=0.0000 | **NaN** | Pire |
| DSC post-éval | 0.0083 | **0.0274** | +234% |
| Aggregate post-éval | 0.1293 | **0.3261** | +152% |
| PSNR post-éval | 3.48 dB | **3.82 dB** | +9.8% |
| Decision | REJETÉ | **REJETÉ** | — |

---

## Recommandations pour RUN_0005 (v3)

1. **Gradient clipping obligatoire** : `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
2. **Têtes 1a/1b initialisées séparément** : pas aléatoire. Copier des poids pré-entraînés ou utiliser un warm-up spécifique par tête.
3. **Warmup ≥ 50 epochs** ou critère DSC ≥ 0.10 minimum avant d'activer les autres têtes.
4. **LR réduit pour la phase joint** : par exemple 1e-5 au lieu de 1e-4.
5. **Monitoring des gradients** : ajouter `torch.autograd.profiler` pour détecter les gradients NaN/Infinite avant qu'ils n'explosent.

---

## Notes supplémentaires

- La tête 1a (classification) est la plus problématique : 21 logits avec initialisation aléatoire.
- Le warmup de 30 epochs n'a été **pas assez** pour atteindre 0.15 DSC.
- Le NaN est une **régression** par rapport au collapse (v1 val_dice_2=0.0000) : le NaN empêche même la sauvegarde du modèle.
- Le meilleur checkpoint reste celui du warmup epoch 30 (DSC=0.0274). C'est un résultat minimal mais non nul.
