# RUN_0003_EXP_A — Archivé

**Date d'archivage** : 2026-05-14
**Raison** : Training divergé dès le début — le modèle a convergé vers une prédiction
de fond uniquement. Le checkpoint sauvegardé ne contient aucune information structurelle
utile (DSC moyen = 0.006).
**Verdict** : ❌ Rejeté — invalide, non récupérable

## Métriques finales

| Métrique    | Valeur  |
|-------------|---------|
| mean_dsc    | 0.0063  |
| mean_hd95   | 74.58   |
| mean_rve    | 577.1   |
| mean_assd   | 53.4    |

Toutes les classes 7–10 avaient DSC ≈ 0. Le modèle prédisait quasi-exclusivement la
classe 0 (fond).

## Configuration

- Architecture : DynUNet filters=[48,96,192,384,512] (même que RUN_0003)
- Loss : DiceFocal (γ=2.0)
- Epochs prévus : 90

## Leçon apprise

Un modèle qui diverge en prédisant uniquement le fond produit un checkpoint trompeur :
les métriques de loss semblent se stabiliser mais les DSC par classe restent à 0. Il est
essentiel de monitorer les DSC per-class pendant l'entraînement (pas seulement la loss
globale) pour détecter une divergence précoce. La loss DiceFocal avec γ=2.0 peut
amplifier les gradients sur des voxels bruités et entraîner une instabilité sur ce
dataset de faible taille.

## Emplacement des artefacts

- Checkpoint : `outputs/checkpoints/RUN_0003_EXP_A/task2_dynunet_best.pt`
- Ancien dossier de résultats : `results/runs/RUN_0003_EXP_A/` (supprimé 2026-05-14)
