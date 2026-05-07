# Analyse : Pourquoi les hippocampes sont à Dice nul en Task 2

## Résumé du problème

**Dice = 0.0000 pour hippocampes gauche et droit** (labels 1 et 2)

## Diagnostic

### 1. Les hippocampes EXISTENT dans les données ✓

- **79 fichiers de segmentation** avec annotations multi-structure
- **Label 1 (hippo_G)** : 92,223 voxels répartis sur 79 items (~1,170 voxels/item)
- **Label 2 (hippo_D)** : 97,307 voxels répartis sur 79 items (~1,230 voxels/item)
- **Tous les items du dataset contiennent les deux hippocampes**

### 2. Le modèle NE prédit JAMAIS les hippocampes ✗

Inspection des 14 items validation avec segmentation :

```
Tous les items :   GT_Hippo=Y (1700-3600 voxels) → Pred_Hippo=N (0 voxels)
```

Prédictions typiques :
- LISA_0002 : GT = 3,360 voxels hippocampes → Pred = **0 voxels** (à la place : L3:3252, L4:2984, L5:2638...)
- LISA_0005 : GT = 3,598 voxels hippocampes → Pred = **0 voxels** (à la place : L3:4702, L4:3959, L5:2403...)
- LISA_0029 : GT = 3,006 voxels hippocampes → Pred = **0 voxels** (à la place : L3:6325, L4:5912, L5:3082...)

**Le modèle attribue systématiquement ces voxels à d'autres labels** (principalement labels 3-6 = structures basal ganglia)

## Causes racines probables

### 1. **Localisation spatiale difficile** 
- Les hippocampes sont des structures petites et profondes (medial temporal)
- Risque de confusion avec le thalamus (label 9-10, adjacent) et putamen (label 5-6)
- L'architecture U-Net standard peut perdre de la précision spatiale pour les petites structures

### 2. **Similarité texture vs autres structures**
- Signal MRI des hippocampes vs putamen/thalamus : très similaires
- Le modèle ne discrimine pas assez les textures fine

### 3. **Données limitées pour Task 2**
- **Seulement 14 items** disponibles pour validation en Task 2
- **79 items totaux** en entraînement pour 13 classes
- Ratio très faible par classe (≈ 6 images / label) → sous-apprentissage pour les petites structures

### 4. **Loss et pondération des classes**
- Loss = CE + Soft Dice : peut être insensible aux classes petites et minoritaires
- Les hippocampes représentent ~4% des voxels labélisés (190k/4.8M)
- Soft Dice récompense principalement les gros labels (corpus callosus, thalamus)
- Pas d'explicit class weighting pour équilibrer les contributions

### 5. **Architecture insuffisante**
- La tête Task2 doit discriminer 14 classes simultanément
- Le bottleneck/feature maps peuvent ne pas capturer assez de spécificité pour 13 structures petites

## Comparaison avec ganglions de la base

| Structure | Voxels | Dice | Raison probable |
|-----------|--------|------|-----------------|
| Hippocampe | ~1,200/item | **0.0000** | ❌ Non prédit du tout |
| Caudate | ~4,300/item | 0.78 | ✓ Gros volume, bien prédit |
| Putamen | ~1,900/item | 0.69 | ✓ Taille moyenne, Dice acceptable |
| Thalamus | ~4,800/item | 0.84 | ✓ Très gros, structure claire |

**Observation** : Les hippocampes ont une taille similaire au putamen (~1.2k vs 1.9k voxels), mais leur Dice est infiniment pire (0.0 vs 0.69). Cela suggère un **problème de localisation ou d'architecture spécifique aux hippocampes**, pas juste un problème de taille.

## Corrections possibles

### Court terme (sans retraining)
1. **Ignorer les hippocampes** en Task 2 (métrique réduite à 11 structures)
2. **Adapter le threshold de segmentation** (peu probable d'aider)

### Moyen terme (optimisations)
1. **Class weighting pour Task 2** : donner plus de poids aux hippocampes dans la loss CE
2. **Augmenter la résolution spatiale** des feature maps Task2 (réduire le bottleneck)
3. **Soft Dice avec poids par classe** : pénaliser davantage les erreurs sur hippocampes

### Long terme (architecture/données)
1. **Données d'entraînement** : ajouter plus d'exemples ciso annotés
2. **Tête Task2 dédiée** : branches séparées pour petites vs gros structures
3. **Pré-entraînement anatomique** : VICReg avec classe spécifique hippocampe
4. **Attention spatiale** : ajouter un mécanisme d'attention aux hippocampes

## Conclusion

**Le Dice nul des hippocampes n'est pas dû à des données manquantes** (elles sont là, de bonne qualité), **mais à une incapacité du modèle à les localiser et discriminer**. C'est un signal que la Task 2 actuelle n'est pas bien calibrée pour les petites structures difficiles à segmenter.

Le résultat est **symptomatique d'un problème d'architecture/optimisation** plutôt qu'un problème de données brutes.
