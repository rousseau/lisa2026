# RUN_0003_EXP_SYM — Archivé

**Date d'archivage** : 2026-05-14
**Raison** : Régression des performances vs RUN_0003_EXP_C (DSC=0.579 < 0.631).
L'ajout d'une symmetry consistency loss (w=0.05) nuit à la généralisation sur ce dataset
de faible taille.
**Verdict** : ⚠️ Rejeté — régression, hypothèse invalidée

## Métriques finales

| Métrique    | Valeur  |
|-------------|---------|
| mean_dsc    | 0.5793  |
| mean_hd95   | 14.94   |
| mean_rve    | 0.389   |
| mean_assd   | 3.57    |

Régression sur toutes les métriques vs EXP_C :
- DSC : −0.052 (−8 % relatif)
- HD95 : +5.36 mm (+56 %)
- ASSD : +1.35 mm (+61 %)

## Configuration

- Architecture : DynUNet filters=[32,64,128,256,320] (même que EXP_C)
- Initialisation : checkpoint RUN_0003_EXP_C (best epoch 28)
- Loss : DiceCE + SymmetryConsistencyLoss (w=0.05)
- LR : 1e-4

## Leçon apprise

La symmetry consistency loss (w=0.05) pénalise les asymétries morphologiques réelles
présentes dans le dataset. Sur un petit dataset (12 sujets d'entraînement), la variabilité
inter-sujets est une information utile que le réseau doit capturer — la contraindre
artificiellement vers la symétrie réduit la capacité de généralisation.

De plus, l'initialisation depuis EXP_C (fine-tune) ne suffit pas à compenser la
dégradation introduite par la loss supplémentaire : les structures pairées (hippocampes,
caudates) montrent les plus fortes régressions.

**Conclusion** : la symmetry consistency loss est contre-productive sur ce dataset à
faible taille et à forte variabilité morphologique. À ne pas réutiliser sauf sur un
dataset beaucoup plus large.

## Emplacement des artefacts

- Checkpoint : `outputs/checkpoints/RUN_0003_EXP_SYM/task2_dynunet_best.pt`
- Ancien dossier de résultats : `results/runs/RUN_0003_EXP_SYM/` (supprimé 2026-05-14)
