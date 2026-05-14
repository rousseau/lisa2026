# RUN_0003_EXP_C — Analyse quantitative

**Date** : 2026-05-14  
**Évalué sur** : validation set, 12 sujets, 11 classes non-background  
**Script** : `evaluate_task2_dynunet.py` (version corrigée post-REVERSE_MAP)

---

## 1. Tableau récapitulatif global du cycle RUN_0003

| Run | mean_DSC | mean_HD95 | mean_RVE | mean_ASSD | Verdict |
|-----|----------|-----------|----------|-----------|---------|
| RUN_0003 (baseline) | 0.5003 | 23.17 | 0.355 | 6.14 | ✅ Baseline valide |
| RUN_0003_EXP_A | 0.0063 | 74.58 | 577.1 | 53.4 | ❌ Training divergé |
| RUN_0003_EXP_B | 0.3220 | 17.20 | 0.534 | 10.99 | ❌ Checkpoint skip → from scratch |
| **RUN_0003_EXP_C** | **0.6309** | **9.58** | **0.320** | **2.22** | **✅ WINNER** |
| RUN_0003_EXP_C_TTA | 0.0617 | 28.37 | 0.339 | 21.07 | ❌ TTA LR naïf détruit L/R |
| RUN_0003_EXP_SYM | 0.5793 | 14.94 | 0.389 | 3.57 | ⚠️ Régression vs EXP_C |
| RUN_0003_COLLAPSED† | 0.6837 | 10.29 | 0.272 | 2.67 | 🔄 Non comparable (6 classes) |

_† COLLAPSED évalué sur 6 classes fusionnées (L+R merged) — espace métrique non comparable aux runs 12-class._

---

## 2. Per-class DSC — EXP_C vs Baseline RUN_0003

| Classe | Structure    | DSC EXP_C | DSC Baseline | Delta     |
|--------|--------------|-----------|--------------|-----------|
| 1      | Hippo L      | 0.353     | 0.052        | **+0.301** |
| 2      | Hippo R      | 0.394     | 0.304        | **+0.090** |
| 3      | Caudate L    | 0.630     | 0.517        | **+0.113** |
| 4      | Caudate R    | 0.685     | 0.443        | **+0.242** |
| 5      | Lentiform L  | 0.689     | 0.576        | **+0.113** |
| 6      | Lentiform R  | 0.678     | 0.454        | **+0.224** |
| 7      | Ventricle L  | 0.656     | 0.581        | **+0.075** |
| 8      | Ventricle R  | 0.734     | 0.573        | **+0.161** |
| 9      | ExV L        | 0.704     | 0.632        | **+0.072** |
| 10     | ExV R        | 0.759     | 0.733        | +0.026     |
| 11     | Aux          | 0.659     | 0.639        | +0.020     |
| —      | **Mean**     | **0.631** | **0.500**    | **+0.131** |

### Visualisations

- `plots/run0003_expc_per_class_dsc.png` — DSC par structure (EXP_C vs baseline)
- `plots/run0003_global_metrics.png` — Métriques globales comparées sur les 6 runs
- `plots/run0003_expc_vs_baseline_delta.png` — Delta DSC par classe (EXP_C − baseline)

---

## 3. Forces et faiblesses du modèle EXP_C

### Forces

- **Structures volumineuses** : ExV R (0.759), Ventricle R (0.734), Lentiform L (0.689) et
  Caudate R (0.685) dépassent tous 0.68, niveau acceptable pour un modèle de bas champ.
- **Symétrie L/R globalement respectée** : les écarts DSC gauche/droite restent modérés
  (Lentiform : 0.011, Caudate : 0.055, Ventricle : 0.078), ce qui témoigne d'une bonne
  capacité de généralisation sur des structures pairées.
- **Amélioration spectaculaire sur l'hippocampe L** : +0.301 vs baseline (0.052 → 0.353),
  ce qui montre que le modèle baseline avait des problèmes systématiques sur cette classe.

### Faiblesses

- **Hippocampes L/R restent les classes les plus difficiles** (DSC 0.353 et 0.394),
  très en dessous des structures volumineuses. L'hippocampe est une petite structure
  (<2 cm³) dans des images de faible résolution 0.064 T, rendant la délimitation
  intrinséquement difficile.
- **Asymétrie Hippo L vs Hippo R** : l'hippocampe droit est systématiquement mieux segmenté
  (+0.041 vs gauche), suggérant un possible biais dans les annotations ou dans la
  distribution des sujets de validation.
- **Caudate L/R** : écart modéré (0.630 vs 0.685), l'hémisphère gauche étant moins bien
  capturé.

---

## 4. Explication scientifique du succès de EXP_C vs baseline

### Facteurs identifiés

**1. Filtres réduits [32,64,128,256,320] vs [48,96,192,384,512]**  
Le modèle baseline utilise des filtres larges (+50 % de paramètres), ce qui peut provoquer
un surapprentissage sur un dataset de faible taille (12 sujets d'entraînement). Les filtres
réduits imposent une régularisation implicite et améliorent la généralisation.

**2. LR réduit 1e-4 vs 2e-4**  
Un LR plus faible permet des pas de gradient plus fins, critique lors d'un entraînement from
scratch avec un petit batch. La convergence est plus stable (moins de risque d'oscillation
autour des minima locaux), ce qui explique l'écart avec EXP_B (mêmes filtres, LR=1.5e-4 →
DSC=0.322).

**3. DiceCE vs DiceFocal (γ=2.0)**  
La DiceFocal amplifie le gradient sur les exemples difficiles mais peut déstabiliser
l'entraînement sur de petits volumes (hippocampes) où le signal supervisionnel est déjà
bruité. La DiceCE offre un gradient plus uniforme et semble mieux adaptée à ce dataset.

**4. Entraînement from scratch (checkpoint skip silencieux)**  
Le checkpoint de RUN_0003 était architecturalement incompatible (filtres différents) et fut
silencieusement ignoré. Paradoxalement, partir de zéro avec les bons hyperparamètres (filtres
réduits + LR 1e-4) s'est avéré supérieur à une fine-tuning hypothétique.

**5. Absence de TTA et keep_largest**  
Le modèle n'utilise aucun post-traitement à l'inférence. L'amélioration est donc purement
issue d'un meilleur entraînement, ce qui valide la contribution intrinsèque de l'architecture
et des hyperparamètres.

---

## 5. Comparaison avec RUN_0003_COLLAPSED (6-class)

RUN_0003_COLLAPSED atteint mean_DSC=0.684 et mean_HD95=10.29 sur **6 classes fusionnées**
(L+R merged : Hippo, Caudate, Lentiform, Ventricle, ExV, Aux). Ce score n'est **pas
comparable** aux runs 12-class car :

- L'espace de prédiction est différent (7 sorties vs 12).
- L'évaluation porte sur 6 classes au lieu de 11.
- La fusion L+R masque les difficultés de distinction latérale, un objectif explicite du
  challenge (segmentation multi-structure latéralisée).

En pratique, le challenge LISA 2026 évalue les 11 structures latéralisées séparément.
**EXP_C (0.631 sur 12 classes) est donc le run de référence pour la soumission.**

---

## 6. Leçons apprises du cycle RUN_0003

### Leçon 1 — TTA avec flip LR : permuter les canaux L/R

L'expérience EXP_C_TTA (DSC=0.062) illustre un piège classique :
un flip horizontal (dim 4) sur une prédiction avec labels asymétriques (Hippo L=cl.1 vs
Hippo R=cl.2, Ventricle L/R, etc.) inverse l'anatomie sans permuter les canaux
correspondants. Le moyennage des logits produit alors des prédictions incohérentes où les
structures pairées s'annulent mutuellement (DSC≈0 pour toutes structures).

**Règle** : tout TTA avec flip LR doit s'accompagner d'une permutation explicite des canaux
de sortie (ex. échanger les logits de cl.1 et cl.2, cl.3 et cl.4, etc.) avant le moyennage.

### Leçon 2 — Symmetry consistency loss : potentiellement nuisible sur ce dataset

EXP_SYM (DSC=0.579) régresse légèrement par rapport à EXP_C (DSC=0.631). La perte de
cohérence symétrique (w=0.05) pénalise les asymétries morphologiques réelles présentes dans
le dataset ou perturbe l'optimisation de la DiceCE principale. Sur un dataset de faible
taille, contraindre la symétrie peut réduire la capacité du modèle à capturer la variabilité
inter-sujets.

### Leçon 3 — Filtres : moins = mieux sur petit dataset

La comparaison directe RUN_0003 (filters=[48,96,192,384,512]) vs EXP_C
(filters=[32,64,128,256,320]) montre que réduire la capacité du modèle améliore la
généralisation sur ce dataset. Ce résultat est cohérent avec la littérature sur la
régularisation implicite des réseaux plus petits en régime de données limitées.

### Leçon 4 — LR est le facteur discriminant entre EXP_B et EXP_C

EXP_B (mêmes filtres [32,64,128,256,320], LR=1.5e-4) obtient DSC=0.322 vs EXP_C
(LR=1e-4) → DSC=0.631. L'écart massif (+0.309 DSC) pour un seul hyperparamètre confirme
que le LR est le facteur clé sur ce dataset, notamment lors d'un entraînement from scratch
sur données limitées.
