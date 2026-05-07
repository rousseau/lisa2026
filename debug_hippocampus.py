#!/usr/bin/env python3
"""
Diagnostic : Analyse du Dice nul pour hippocampes en Task 2.

Hypothèses à tester :
  1. Les hippocampes ne sont pas présents / sont rares dans le dataset local
  2. Le modèle ne prédît pas du tout d'hippocampes (output = zéro)
  3. Les masks des hippocampes sont complètement zéro dans les annotations
"""

import sys
from pathlib import Path
import numpy as np
import torch
import os

# Changez le répertoire de travail pour src/ pour les imports relatifs
os.chdir(Path(__file__).parent / "src")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import LISAJointDataset
from model import BackboneLISA

# SEG_LABELS défini localement (depuis evaluate_all.py)
SEG_LABELS = {
    1: "hippo_G",
    2: "hippo_D",
    3: "caude_G",
    4: "caude_D",
    5: "putamen_G",
    6: "putamen_D",
    7: "globus_G",
    8: "globus_D",
    9: "thalamus_G",
    10: "thalamus_D",
    11: "corps_calleux",
    12: "ventricule_G",
    13: "ventricule_D",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Étape 1 : Analyser la présence de labels dans le dataset
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("DIAGNOSTIC HIPPOCAMPES — DICE NUL EN TASK 2")
print("="*80)

for split_name in ["train", "val"]:
    print(f"\n{'─'*80}")
    print(f"Dataset split: {split_name}")
    print(f"{'─'*80}")
    
    ds = LISAJointDataset(
        "/home/rousseau/Data/LISA2026",
        target_size=(96, 96, 96),
        split=split_name,
        val_fraction=0.2
    )
    
    # Collecter les statistiques sur les labels
    label_counts = {i: 0 for i in range(1, 14)}
    label_present = {i: 0 for i in range(1, 14)}
    n_seg_items = 0
    
    for i, item in enumerate(ds):
        seg = item["seg"].numpy()
        has_seg = item["has_seg"].item()
        
        if has_seg and seg.max() > 0:
            n_seg_items += 1
            for label in range(1, 14):
                mask = seg == label
                count = mask.sum()
                label_counts[label] += count
                if count > 0:
                    label_present[label] += 1
    
    print(f"\nTotal items: {len(ds)}")
    print(f"Items avec segmentation: {n_seg_items}")
    print(f"\nPrésence des labels (voxels / items):")
    print(f"  Label  Structure              Voxels totaux  Items contenant")
    print(f"  ─────  ─────────────────────  ─────────────  ─────────────────")
    for label in range(1, 14):
        name = SEG_LABELS.get(label, f"unknown_{label}")
        voxels = label_counts[label]
        items = label_present[label]
        marker = "*** ZERO" if voxels == 0 else ""
        print(f"  {label:2d}    {name:18s}      {voxels:10d}      {items:3d} {marker}")


# ──────────────────────────────────────────────────────────────────────────────
# Étape 2 : Charger le modèle et inspecter les prédictions
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{'─'*80}")
print("Inspecter les PRÉDICTIONS du modèle")
print(f"{'─'*80}\n")

ckpt_path = Path("/home/rousseau/Exp/lisa2026/outputs/checkpoints/best_model.pt")
if not ckpt_path.exists():
    print(f"ERROR: checkpoint non trouvé: {ckpt_path}")
    exit(1)

ckpt = torch.load(str(ckpt_path), map_location=device)
cfg = ckpt.get("config", {})

model = BackboneLISA(
    base=cfg.get("base_channels", 16),
    c_anat=cfg.get("c_anat", 16),
    c_mod=cfg.get("c_mod", 8),
    c_art=cfg.get("c_art", 8),
    n_artifacts=cfg.get("n_artifacts", 7),
    n_severity=cfg.get("n_severity", 3),
    n_seg_classes=cfg.get("n_seg_classes", 14),
).to(device)

model.load_state_dict(ckpt["model"])
model.eval()
print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")

# Valider sur quelques cas ciso avec segmentation
val_ds = LISAJointDataset(
    "/home/rousseau/Data/LISA2026",
    target_size=(96, 96, 96),
    split="val",
    val_fraction=0.2
)

seg_items = [
    (i, item)
    for i, item in enumerate(val_ds)
    if item["has_seg"].item() and item["seg"].numpy().max() > 0
]

print(f"\nItems validation avec segmentation: {len(seg_items)}")

if seg_items:
    print(f"\nAnalyse détaillée des prédictions:")
    print(f"  Item  Subject                Has_hippo_GT  Max_pred_logit[label]     Pred_hippo_class")
    print(f"  ────  ────────────────────  ──────────────  ────────────────────────  ────────────────")
    
    for idx, (i, item) in enumerate(seg_items[:10]):  # première 10
        x = item["image"].unsqueeze(0).to(device)
        gt = item["seg"].numpy()
        
        with torch.no_grad():
            preds_dict = model(x, run_task1a=False, run_task1b=False, run_task2=True)
            seg_logits = preds_dict["task2"]  # [1, 14, D, H, W]
        
        seg_probs = torch.softmax(seg_logits, dim=1)  # [1, 14, D, H, W]
        seg_pred = seg_probs.argmax(dim=1).cpu().numpy()[0]  # [D, H, W]
        
        # Vérifier présence hippocampes dans GT
        has_hippo_1 = (gt == 1).sum() > 0
        has_hippo_2 = (gt == 2).sum() > 0
        has_hippo_gt = "Y" if (has_hippo_1 or has_hippo_2) else "N"
        
        # Vérifier présence hippocampes en prédiction
        pred_hippo_1 = (seg_pred == 1).sum() > 0
        pred_hippo_2 = (seg_pred == 2).sum() > 0
        pred_hippo_classes = "Y" if (pred_hippo_1 or pred_hippo_2) else "N"
        
        # Max logit pour labels hippocampe
        max_logit_1 = seg_logits[0, 1].max().item()
        max_logit_2 = seg_logits[0, 2].max().item()
        
        subj = item["subject"]
        print(f"  {idx:2d}    {subj:20s}  {has_hippo_gt:3s}            "
              f"[1]={max_logit_1:7.3f} [2]={max_logit_2:7.3f}       {pred_hippo_classes}")


# ──────────────────────────────────────────────────────────────────────────────
# Étape 3 : Chercher les logs d'entraînement
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{'─'*80}")
print("Historique d'entraînement")
print(f"{'─'*80}\n")

log_path = Path("/home/rousseau/Exp/lisa2026/outputs/logs/train_v8_pretrained_full.log")
if log_path.exists():
    print(f"Log trouvé: {log_path}")
    # Lire les 20 premières et les 20 dernières lignes
    with open(log_path) as f:
        lines = f.readlines()
    
    print(f"\nPremières 10 lignes d'entraînement :")
    for line in lines[:10]:
        print("  " + line.rstrip())
    
    print(f"\n... (intermédiaire omis) ...\n")
    
    print(f"Dernières 15 lignes d'entraînement :")
    for line in lines[-15:]:
        print("  " + line.rstrip())
else:
    print(f"Log non trouvé: {log_path}")

print(f"\n{'─'*80}")
print("Fin du diagnostic")
print(f"{'─'*80}\n")
