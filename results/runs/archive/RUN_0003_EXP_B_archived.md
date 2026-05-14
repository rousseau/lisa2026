# RUN_0003_EXP_B — Archivé

**Date d'archivage** : 2026-05-14
**Raison** : Checkpoint skip silencieux — le checkpoint de RUN_0003 utilisait les
filtres [48,96,192,384,512] alors que EXP_B configurait [32,64,128,256,320].
L'incompatibilité architecturale a provoqué un skip silencieux du checkpoint, forçant
un entraînement from scratch non prévu, avec un LR (1.5e-4) inadapté à cet usage.
**Verdict** : ❌ Rejeté — résultats issus d'une configuration involontaire

## Métriques finales

| Métrique    | Valeur  |
|-------------|---------|
| mean_dsc    | 0.3220  |
| mean_hd95   | 17.20   |
| mean_rve    | 0.534   |
| mean_assd   | 10.99   |

## Configuration effective (non celle prévue)

- Architecture effective : DynUNet filters=[32,64,128,256,320]
- Entraînement : from scratch (checkpoint RUN_0003 ignoré silencieusement)
- Loss : DiceCE
- LR : 1.5e-4 (trop élevé pour from scratch sur ce dataset)

## Leçon apprise

Le checkpoint skip silencieux est la cause principale de la mauvaise performance. Le LR
de 1.5e-4, prévu pour une fine-tune, est inadapté à un entraînement from scratch sur ce
dataset de faible taille. La comparaison directe avec EXP_C (mêmes filtres, LR=1e-4,
DSC=0.631) confirme que l'écart de +0.309 DSC est attribuable au seul LR.

Ce run a motivé l'implémentation d'un mécanisme de warning explicite lors d'un checkpoint
skip (voir `train_task2_dynunet.py`, warning ajouté post-EXP_B).

## Emplacement des artefacts

- Checkpoint : `outputs/checkpoints/RUN_0003_EXP_B/task2_dynunet_best.pt`
- Ancien dossier de résultats : `results/runs/RUN_0003_EXP_B/` (supprimé 2026-05-14)
