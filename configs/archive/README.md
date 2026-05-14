# Archive des configurations historiques

Ce répertoire contient les fichiers de configuration des phases de conception
antérieures au pivot vers des modèles séparés par tâche (RUN_0001 et suivants).

## Contenu

| Fichier | Description |
|---------|-------------|
| train_default.yaml | Configuration originale du modèle joint multi-tâche (Task 1a + 1b + 2 simultanés) |
| train_v6.yaml | v6 : ajout de la simulation d'artefacts TorchIO en ligne |
| train_v7.yaml | v7 : augmentation d'artefacts refactorisée, paramètres UPF/Sundaresan 2024 |
| train_v8.yaml | v8 : FactorizedProjection (z_anat, z_mod, z_art), VICReg, gradient surgery |
| pretrain_v8.yaml | Pré-entraînement auto-supervisé VICReg pour v8 |
| run_0001_task1a_design.yaml | Spécification conceptuelle Task 1a pour RUN_0001 (schéma différent du code final) |
| run_0001_task2_design.yaml | Spécification conceptuelle Task 2 pour RUN_0001 (schéma différent du code final) |

## Raison de l'archivage

Le projet a pivoté d'un modèle joint unique (train_default/v6/v7/v8) vers
des modèles spécialisés par tâche (RUN_0001, RUN_0002, RUN_0003, RUN_0004).
Ces fichiers documentent l'historique de conception mais ne correspondent plus
à aucun script d'entraînement actif.

Les configs actives sont dans `configs/` (niveau supérieur).
