# RUN_0003_EXP_C_TTA — Archivé

**Date d'archivage** : 2026-05-14
**Raison** : TTA avec flip LR (dim 4) appliqué sur des labels asymétriques L/R sans
permutation des canaux de sortie. Le moyennage des logits flippés avec les logits
originaux produit des prédictions incohérentes : les structures pairées (Hippo L/R,
Caudate L/R, Lentiform L/R, Ventricle L/R, ExV L/R) s'annulent mutuellement, donnant
DSC ≈ 0 pour toutes ces classes.
**Verdict** : ❌ Invalide — erreur méthodologique fondamentale dans l'implémentation TTA

## Métriques finales

| Métrique    | Valeur  |
|-------------|---------|
| mean_dsc    | 0.0617  |
| mean_hd95   | 28.37   |
| mean_rve    | 0.339   |
| mean_assd   | 21.07   |

Les structures non pairées (Aux, cl.11) conservaient un DSC >0 mais très dégradé.

## Configuration

- Checkpoint : RUN_0003_EXP_C (best epoch 28)
- TTA : flip LR [dim 4] appliqué naïvement (sans permutation canaux L/R)
- Inférence uniquement (pas de réentraînement)

## Leçon apprise

Toute augmentation avec flip LR sur une segmentation multi-classe avec labels latéralisés
**doit obligatoirement** s'accompagner d'une permutation des canaux correspondants avant
le moyennage des logits. Dans ce dataset :

| Classe originale | Classe après flip LR |
|-----------------|----------------------|
| 1 (Hippo L)     | 2 (Hippo R)          |
| 3 (Caudate L)   | 4 (Caudate R)        |
| 5 (Lentiform L) | 6 (Lentiform R)      |
| 7 (Ventricle L) | 8 (Ventricle R)      |
| 9 (ExV L)       | 10 (ExV R)           |

Sans cette permutation, moyenner les logits cl.1 flippé (= Hippo R) avec les logits cl.1
original (= Hippo L) produit un signal contradictoire → DSC effondré.

## Emplacement des artefacts

- Checkpoint utilisé : `outputs/checkpoints/RUN_0003_EXP_C/task2_dynunet_best.pt`
- Ancien dossier de résultats : `results/runs/RUN_0003_EXP_C_TTA/` (supprimé 2026-05-14)
