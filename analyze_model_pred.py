#!/usr/bin/env python3
"""
Diagnostic : Vérifier ce que le modèle prédit pour les hippocampes.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# Setup paths
model_dir = Path("/home/rousseau/Exp/lisa2026")
os.chdir(model_dir / "src")
sys.path.insert(0, str(model_dir / "src"))
from model import BackboneLISA
from dataset import LISAJointDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n" + "="*80)
print("DIAGNOSTIC : PRÉDICTIONS DU MODÈLE POUR HIPPOCAMPES")
print("="*80 + "\n")

# Charger modèle
ckpt_path = model_dir / "outputs" / "checkpoints" / "best_model.pt"
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
print(f"✓ Modèle chargé (epoch {ckpt.get('epoch', '?')})")

# Charger validation set
val_ds = LISAJointDataset(
    "/home/rousseau/Data/LISA2026",
    target_size=(96, 96, 96),
    split="val",
    val_fraction=0.2
)

# Trouver les items ciso avec segmentation
seg_items = [
    (i, item)
    for i, item in enumerate(val_ds)
    if item["has_seg"].item() and item["seg"].numpy().max() > 0
]

print(f"✓ Items validation avec seg: {len(seg_items)}\n")

if seg_items:
    print(f"{'Item':<5} {'Subject':<25} {'GT Hippo':<12} {'Pred Hippo':<12} {'Pred labels (raw count)':<40}")
    print(f"{'-'*5} {'-'*25} {'-'*12} {'-'*12} {'-'*40}")
    
    for idx, (i, item) in enumerate(seg_items[:15]):
        x = item["image"].unsqueeze(0).to(device)
        gt = item["seg"].numpy()
        
        with torch.no_grad():
            preds_dict = model(x, run_task1a=False, run_task1b=False, run_task2=True)
            seg_logits = preds_dict["task2"]  # [1, 14, D, H, W]
        
        seg_probs = torch.softmax(seg_logits, dim=1)
        seg_pred = seg_probs.argmax(dim=1).cpu().numpy()[0]
        
        # Compter voxels GT
        gt_hippo_1 = (gt == 1).sum()
        gt_hippo_2 = (gt == 2).sum()
        gt_hippo_any = "Y" if (gt_hippo_1 + gt_hippo_2 > 0) else "N"
        
        # Compter voxels prédits
        pred_hippo_1 = (seg_pred == 1).sum()
        pred_hippo_2 = (seg_pred == 2).sum()
        pred_hippo_any = "Y" if (pred_hippo_1 + pred_hippo_2 > 0) else "N"
        
        # Distribution de prédictions
        unique, counts = np.unique(seg_pred, return_counts=True)
        pred_dist = ", ".join([f"L{u}:{c}" for u, c in zip(unique, counts) if u > 0])
        if len(pred_dist) > 38:
            pred_dist = pred_dist[:35] + "..."
        
        subj = item["subject"]
        print(f"{idx:<5} {subj:<25} {gt_hippo_any} ({gt_hippo_1+gt_hippo_2:4d}vx) {pred_hippo_any} ({pred_hippo_1+pred_hippo_2:4d}vx)    {pred_dist:<40}")

print(f"\n{'─'*80}")
print("Hypothèses testées :")
print(f"{'─'*80}")
print("1. Si GT_Hippo=Y mais Pred_Hippo=N → modèle ne prédit pas les hippocampes")
print("2. Si Pred_Hippo > 0 → modèle CAN prédire les hippocampes sur quelques items")
print("3. Regarder la distribution Pred_labels pour voir où vont les voxels hippocampes\n")
