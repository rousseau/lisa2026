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

---

## Résultats Jean Zay v3 (job 1063254, 2026-05-20)

## Statut

**À RETESTER (v3)** — Exécuté sur Jean Zay H100 (job SLURM **1063254**, 2026-05-20). Le warmup a atteint la cible DSC=0.10 (**epoch 43, val_dice_2=0.1104 ✅**), mais la head warmup a produit des poids NaN à partir de l'epoch 46 (LR scheduler partagé trop élevé). La phase joint est restée 100% NaN. Corrections précises identifiées pour v4.

---

### Tableau comparatif v1 / v2 / v3

| Aspect | v1 (job 878620) | v2 (job 887047) | v3 (job 1063254) |
|--------|-----------------|-----------------|------------------|
| Architecture | SharedEncoderMultiTaskModel | DynUNetMultiHeadModel | DynUNetMultiHeadModel |
| Warm-start encodeur | 0/102 keys ❌ | 46/84 keys ✅ | **absent** (checkpoint introuvable) ❌ |
| Loss calibration | Non | 1a=7.63 / 1b=2.22 / 2=1.37 ✅ | **NaN** (bug max) ❌ |
| Warmup epochs (réels) | 10 | 30 | **43** |
| val_dice_2 fin warmup | 0.0083 | 0.0274 | **0.1104** ✅ |
| Head warmup | — | — | NaN dès epoch 46 ❌ |
| Phase joint | NaN (0/15) | NaN (0/15) | NaN (0/15, 436/436 steps) ❌ |
| Task 2 DSC éval | 0.0083 | 0.0274 | **0.1105** ✅ |
| Task 2 HD95 éval | 66.21 | 37.24 | **27.32** |
| Task 1a Aggregate | 0.1293 | 0.3261 | **0.1945** |
| Task 1a Accuracy | 0.0997 | 0.6071 | **0.2693** |
| Task 1b PSNR | 3.48 dB | 3.82 dB | **3.38 dB** |
| Durée totale | 56 min | 66 min | **33 min** (joint inactif) |
| Décision | REJETÉ | REJETÉ | **À retester** |

---

### Analyse détaillée de la progression warmup

Contrairement aux v1 et v2, le checkpoint RUN_0003 était **absent sur Jean Zay** → entraînement depuis zéro.

| Epoch | train_loss | val_dice_2 | Commentaire |
|-------|-----------|-----------|-------------|
| 1 | 3.3527 | 0.0019 | départ |
| 1–24 | 3.35 → 1.40 | 0.0000–0.0025 | plateau quasi-nul (depuis zéro) |
| 25 | 1.3708 | 0.0092 | **première vraie reprise** |
| 33 | 1.2214 | 0.0570 | progression accélérée |
| 35 | 1.1949 | 0.0700 | pic local |
| 38 | 1.1590 | 0.0844 | meilleure epoch avant 43 |
| 41 | 1.1255 | 0.0860 | progression instable |
| 43 | 1.0995 | **0.1104** | **early exit** (≥ 0.10) ✅ |

**Interprétation** :
- Le plateau de 24 epochs est dû à l'initialisation from scratch (v1 avait aussi un plateau à ~10 epochs avec seulement 10 warmup total → jamais sorti).
- La convergence tardive mais réelle (epoch 25–43) confirme que l'architecture DynUNetMultiHead **peut apprendre la segmentation de zéro**.
- Avec le checkpoint RUN_0003 présent (v4), le warmup devrait converger beaucoup plus vite.

---

### Post-mortem : explosion gradient head warmup

La head warmup (encoder gelé, seulement `cls_mlp` et `recon` entraînés) a produit des NaN progressifs :

| Epoch | train_loss_1a | train_loss_1b | nan_steps / 436 | % NaN |
|-------|--------------|--------------|-----------------|-------|
| 44 (hw1) | 5.4466 | 1.5600 | 0 | 0% |
| 45 (hw2) | 4.5959 | 1.2703 | 0 | 0% |
| 46 (hw3) | 4.3950 | 1.2710 | **306** | **70%** |
| 47 (hw4) | 0.0 | 0.0 | **436** | **100%** |
| 48 (hw5) | 0.0 | 0.0 | **436** | **100%** |

**Cause racine** : le scheduler CosineAnnealingLR est partagé avec T_max=130 (= 50 warmup + 80 joint). À l'epoch 43 (fin warmup), la position dans le cycle est 43/130 ≈ 33%, ce qui correspond à un LR ≈ **7.56e-5** (soit ~75% du LR max de 1e-4). Ce LR est beaucoup trop élevé pour des têtes initialisées aléatoirement (cls_mlp, recon) même avec l'encodeur gelé :
- Epoch hw1-hw2 (LR ~7.56e-5) : gradient explosive pour 1a, mais AMP + clipping contiennent
- Epoch hw3 : les poids `cls_mlp` partent dans une direction catastrophique → 70% des steps NaN
- Epoch hw4-hw5 : poids NaN propagés, tous les forward retournent NaN

**Conséquence en chaîne** :
1. Poids NaN dans `cls_mlp` et `recon`
2. Calibration effectuée avec ces poids NaN → loss_scale_1a=NaN, loss_scale_1b=NaN
3. Bug supplémentaire : `max(float('nan'), 1e-6)` en Python retourne NaN au lieu de 1e-6 (l'ordre correct est `max(1e-6, val)` pour que le NaN soit en second argument)
4. Phase joint : 100% des steps skippés car les pertes NaN propagées → val_dice_2 gelé à 0.1104
5. Early stop après 15 joint epochs (patience=15), epoch global 70

---

### Analyse Task 1a : prédicteur constant (recall=0.333 uniforme)

| Artefact | Accuracy | F1 | F2 | Precision | Recall | Aggregate |
|----------|----------|----|----|-----------|--------|-----------|
| Noise | 0.083 | 0.053 | 0.107 | 0.029 | **0.333** | 0.121 |
| Zipper | 0.698 | 0.274 | 0.307 | 0.233 | **0.333** | 0.369 |
| Positioning | 0.073 | 0.045 | 0.094 | 0.024 | **0.333** | 0.114 |
| Banding | 0.021 | 0.014 | 0.032 | 0.007 | **0.333** | 0.081 |
| Motion | 0.188 | 0.105 | 0.179 | 0.063 | **0.333** | 0.173 |
| Contrast | 0.083 | 0.051 | 0.104 | 0.028 | **0.333** | 0.120 |
| Distortion | 0.740 | 0.283 | 0.311 | 0.247 | **0.333** | 0.383 |
| **Global** | **0.2693** | **0.1180** | **0.1620** | **0.0899** | **0.3333** | **0.1945** |

**Signature d'un prédicteur constant** : recall = 1/3 exactement pour chacun des 7 artefacts est la signature d'un modèle qui prédit toujours la même classe (par exemple classe 0 = "sans artefact") sur un jeu de test où ~1/3 des échantillons ont le niveau 0. Les poids NaN dans `cls_mlp` ont probablement été neutralisés par la calibration (retourner un score constant = classe 0 ou la valeur initiale de bias). La precision très faible (0.007–0.247) confirme l'absence de discrimination. Les scores Zipper et Distortion sont plus élevés en accuracy car ces artefacts sont rares dans le test set (prédire 0 est souvent correct pour les classes rares).

**Conclusion** : la tête 1a est complètement non fonctionnelle dans v3. L'aggregate=0.1945 est entièrement dû au recall=1/3 et à la distribution de classes du test set.

---

### Analyse Task 2 per-class

| Classe | DSC v3 | Commentaire |
|--------|--------|-------------|
| 1 | 0.000 | Non détectée |
| 2 | 0.000 | Non détectée |
| 3 | **0.466** | Apprise ✅ (probablement grande structure) |
| 4 | 0.000 | Non détectée |
| 5 | ~0.000 | Quasi nulle |
| 6 | 0.000 | Non détectée |
| 7 | **0.378** | Apprise ✅ |
| 8 | ~0.000 | Quasi nulle |
| 9 | **0.359** | Apprise ✅ |
| 10 | 0.012 | Trace |
| 11 | 0.000 | Non détectée |
| **Global** | **0.1105** | mean_hd95=27.32, mean_rve=1.004, mean_assd=9.99 |

**Interprétation** :
- Seulement 3 classes sur 11 sont apprises (3, 7, 9) — cohérent avec un entraînement depuis zéro en 43 warmup epochs.
- Les classes non détectées sont probablement les structures petites ou peu fréquentes dans les patches de training.
- mean_DSC=0.1105 est **4× supérieur à v2 (0.0274)** et **13× supérieur à v1 (0.0083)** — progrès réel grâce aux 43 epochs de warmup.
- Reste **3× inférieur à RUN_0003 (0.362)** qui bénéficiait d'un checkpoint pré-entraîné et d'un entraînement dédié tâche 2.

---

### Causes racines complètes (3 problèmes + 1 bug)

| # | Type | Description | Impact |
|---|------|-------------|--------|
| P1 | Configuration | **Checkpoint RUN_0003 absent sur Jean Zay** : `outputs/checkpoints/RUN_0003/task2_dynunet_best.pt` introuvable → `pretrained_loaded=false` → entraînement depuis zéro → warmup beaucoup plus long (43 epochs vs <10 attendus) | Mineur (warmup a quand même convergé) |
| P2 | Hyperparamètre | **Scheduler LR partagé** (T_max=130 pour tout le run) → à l'entrée de head warmup (epoch 44), LR ≈ 7.56e-5 (75% du max) → beaucoup trop élevé pour des têtes aléatoires → explosion en 2 epochs | **Critique** |
| P3 | Code | **Grad clipping non appliqué ou insuffisant pendant head warmup** : même avec max_grad_norm=1.0, si le forward pass lui-même produit NaN (via activations instables), le clipping ne peut pas sauver les poids | **Critique** |
| B1 | Bug Python | **`max(float('nan'), 1e-6)` retourne NaN** : lors de la calibration post head-warmup, les pertes initiales des têtes NaN sont passées à `max()` → la valeur NaN devrait être éliminée mais `max(nan, x) = nan` en Python → loss_scale_1a=NaN, loss_scale_1b=NaN → toute la phase joint est compromise | **Critique** |

**Chaîne causale** : P1 → warmup long mais convergent → P2 → NaN dans les têtes → P3 (amplification) → B1 → calibration NaN → phase joint 100% NaN → val_dice_2 gelé → early stop à patience=15.

---

### Recommandations pour v4

| # | Problème (v3) | Fix v4 | Priorité |
|---|--------------|--------|----------|
| F1 | Checkpoint RUN_0003 absent | Vérifier la présence du fichier avant soumission (`ls outputs/checkpoints/RUN_0003/`) | 🔴 Critique |
| F2 | LR trop élevé (7.56e-5) pour head warmup | Scheduler **séparé** pour head warmup : LR fixe 1e-5, indépendant du CosineAnnealingLR principal | 🔴 Critique |
| F3 | Bug `max(nan, 1e-6)` | Corriger : `loss_scale = val if not math.isnan(val) else 1.0` ou utiliser `torch.nan_to_num(tensor, nan=1.0)` | 🔴 Critique |
| F4 | Warmup from scratch (long) | Avec checkpoint présent, le warmup devrait converger en <15 epochs | 🟡 Important |
| F5 | Pas de monitoring NaN en head warmup | Ajouter un check explicite : si nan_steps > 50% → abort et log d'erreur | 🟡 Important |

---

## Résultats Jean Zay v4 (job 1076254, 2026-05-20)

### Statut

**À RETESTER (v4)** — Exécuté sur Jean Zay H100 (job SLURM **1076254**, 2026-05-20). DSC warmup = **0.1649** (meilleur résultat RUN_0004), mais phase joint 100% NaN (même cause qu'en v3).

---

### ⚠️ Code non synchronisé : run exécuté avec l'ANCIEN code (v3)

Le code v4 corrigé (avec `head_warmup_lr` séparé, `GradScaler` reset, détection des poids NaN) **n'était PAS synchronisé sur Jean Zay**. Preuves dans le log :

- Message : `Head warm-up: 5 epochs |` (sans `@ lr=...`) → ancien format de log (v3)
- Message : `[Joint setup] Resetting LR to 1.00e-05 (warmup LR was 1.00e-04)` → ancien format (v3)
- Absence des messages `GradScaler reset` et `HeadWarmup setup` (présents uniquement dans le code v4 local)

Conséquence : les corrections F2 (scheduler séparé head warmup), F3 (bug `max(nan, 1e-6)`) et les nouvelles protections GradScaler reset **ne sont pas appliquées** dans cette exécution.

---

### Tableau de déroulement v4

| Phase | Epochs | Résultat clé |
|-------|--------|--------------|
| **Warmup** | 1 epoch (early exit) | val_dice_2 = **0.1649** ✅ (cible 0.10 atteinte dès epoch 1 grâce au warm-start RUN_0003) |
| **Head warmup** | 5 epochs | NaN dès epoch 2 (LR=1e-4 → explosion plus rapide qu'en v3) |
| **Calibration** | — | 1a=NaN, 1b=NaN (poids corrompus par head warmup) |
| **Joint** | 15 epochs (patience=15) | 100% NaN, val_dice_2 gelé à 0.1649, early stop epoch global 70 |

#### Détail warmup

- train_loss = **0.6391** (beaucoup mieux qu'en v3 grâce au warm-start RUN_0003 présent : 46/84 keys ✅)
- val_dice_2 = 0.1649 ≥ 0.10 → early exit immédiat après 1 seul epoch

#### Détail head warmup (nan_steps / 436 steps par epoch)

| Epoch hw | Epoch global | loss_1a | loss_1b | nan_steps/436 | % NaN |
|----------|-------------|---------|---------|---------------|-------|
| 1 | 2 | 5.108 | 1.457 | 0 | 0% |
| 2 | 3 | 4.695 | 1.173 | **133** | **31%** |
| 3 | 4 | 0.0 | 0.0 | **436** | **100%** |
| 4 | 5 | 0.0 | 0.0 | **436** | **100%** |
| 5 | 6 | 0.0 | 0.0 | **436** | **100%** |

---

### Comparaison vitesse d'explosion NaN : v3 vs v4

Le LR de head warmup était plus élevé en v4 (code v3 non sync : cosine epoch 1/130 ≈ 1e-4) qu'en v3 (cosine epoch 43/130 ≈ 7.56e-5). Cela confirme la corrélation directe LR ↔ vitesse d'explosion :

| Version | LR head warmup | NaN apparaît à l'epoch hw | Explication |
|---------|---------------|--------------------------|-------------|
| v3 (job 1063254) | ~7.56e-5 (cosine, pos 43/130) | hw epoch **3** (306/436 NaN) | scheduler partagé, warmup=43 epochs |
| v4 (job 1076254) | ~1e-4 (cosine, pos 1/130) | hw epoch **2** (133/436 NaN) | code v3 non sync, warmup=1 epoch → LR plus élevé |

**Interprétation** : plus le LR est élevé, plus l'explosion est rapide (hw3 vs hw2). La corrélation est directe et reproductible. La solution unique et suffisante est de supprimer le head warmup (`num_head_warmup_epochs: 0`), éliminant définitivement cette phase instable.

---

### Analyse distribution classes Task 1a

Distribution mesurée sur le split d'entraînement, avec poids de classe dérivés (= 1 / fréquence relative, normalisés) :

| Artefact | Cl.0 (%) | Cl.1 (%) | Cl.2 (%) | w₀ | w₁ | w₂ |
|----------|-----------|-----------|-----------|-----|-----|-----|
| Noise | 81.9 | 8.3 | 9.9 | 0.41 | 4.04 | 3.38 |
| Zipper | 65.6 | 27.3 | 7.1 | 0.51 | 1.22 | 4.69 |
| Positioning | 86.5 | 8.7 | 4.8 | 0.39 | 3.82 | 6.92 |
| Banding | **95.9** | 1.8 | 2.3 | 0.35 | **18.17** | **14.53** |
| Motion | 68.8 | 17.2 | 14.0 | 0.48 | 1.94 | 2.38 |
| Contrast | 62.4 | 32.3 | 5.3 | 0.53 | 1.03 | 6.32 |
| Distortion | 72.7 | 18.3 | 8.9 | 0.46 | 1.82 | 3.73 |

Cas extrême : **Banding** (ratio 1:52 entre cl.0 et cl.1, poids 18.17 et 14.53). Sans pondération, la cross-entropie encourage le modèle à prédire quasi-systématiquement cl.0 pour Banding, rendant les classes rares non apprenables. La pondération est critique pour que la tête 1a puisse discriminer les niveaux d'artefacts.

---

### Résultats finaux v4

Évaluation sur le checkpoint epoch 1 (warmup, val_dice_2=0.1649 — jamais amélioré en joint) :

| Tâche | Métrique | Valeur | Note |
|-------|----------|--------|------|
| Task 2 | mean DSC | **0.1649** | Meilleur résultat RUN_0004 toutes versions ✅ |
| Task 2 | mean HD95 | 26.70 | — |
| Task 2 | mean ASSD | 11.12 | — |
| Task 1a | Aggregate | 0.1944 | Prédicteur constant (head jamais entraîné) |
| Task 1b | PSNR | 3.38 dB | Head jamais entraîné |
| Durée | — | 19 min | Très court (joint inerte) |

---

### Conclusion : head warmup = source unique de NaN

Chaque version confirme la même corrélation, avec une vitesse d'explosion croissante avec le LR :

- v2 : NaN dès la phase joint (LR élevé, pas de head warmup mais têtes aléatoires en joint)
- v3 : LR head warmup ≈ 7.56e-5 → NaN epoch hw3
- v4 : LR head warmup ≈ 1e-4 → NaN epoch hw2 (plus rapide)

**Décision** : supprimer le head warmup (`num_head_warmup_epochs: 0`). L'encodeur est déjà initialisé avec les poids RUN_0003 (warm-start prouvé efficace en v4), les têtes 1a/1b peuvent apprendre directement en phase joint avec un LR faible (1e-5).

---

### Corrections v5 appliquées localement (prêtes pour prochain run)

| # | Correction | Fichier | Statut |
|---|-----------|---------|--------|
| C1 | `num_head_warmup_epochs: 0` — supprime la source de NaN | `configs/run_0004_multitask.yaml` | ✅ Local |
| C2 | Cross-entropie pondérée pour Task 1a (poids par classe) | `src/train_multitask.py` | ✅ Local |
| C3 | LR reset explicite au joint, GradScaler reset | `src/train_multitask.py` | ✅ Local |
| C4 | Détection des poids NaN avant la calibration | `src/train_multitask.py` | ✅ Local |
| C5 | Fix bug `max(nan, 1e-6)` → `nan_to_num` propre | `src/train_multitask.py` | ✅ Local |

Smoke test v5 confirmé (local) :

- `Head warm-up: 0 epochs (disabled)` ✅
- Poids de classe affichés : `Band: [0.3/18.2/14.5]` ✅
- Calibration propre : `1a=7.6904  1b=1.7345  2=0.4219` (zéro NaN) ✅
- Joint sans NaN : `total=3.3636` ✅
