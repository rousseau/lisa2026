#!/usr/bin/env python3
"""
Analyse minimaliste des labels de segmentation ciso.
Charge directement les fichiers NIfTI sans dépendre de dataset.py.
"""

import nibabel as nib
import numpy as np
from pathlib import Path

data_root = Path("/home/rousseau/Data/LISA2026")

# Labels
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

print("\n" + "="*80)
print("DIAGNOSTIC : PRÉSENCE DES LABELS HIPPOCAMPES DANS LES DONNÉES")
print("="*80 + "\n")

# Trouver tous les fichiers ciso
ciso_files = sorted(data_root.glob("LISA_*_ciso.nii.gz"))
print(f"Fichiers ciso trouvés : {len(ciso_files)}\n")

# Analyser les segmentations disponibles
seg_files = sorted(data_root.glob("LISA_*_LF_seg.nii.gz"))
print(f"Fichiers de segmentation trouvés : {len(seg_files)}\n")

label_stats = {label: {"voxels": 0, "presence": 0} for label in range(1, 14)}
total_items_with_seg = 0

for seg_file in seg_files:
    try:
        img = nib.load(seg_file)
        seg_data = img.get_fdata(dtype=np.float32).astype(np.int32)
        
        if seg_data.max() > 0:
            total_items_with_seg += 1
            
            for label in range(1, 14):
                mask = seg_data == label
                count = mask.sum()
                label_stats[label]["voxels"] += count
                if count > 0:
                    label_stats[label]["presence"] += 1
    except Exception as e:
        print(f"  Erreur lecture {seg_file}: {e}")

print(f"Items avec segmentation non-vide : {total_items_with_seg}\n")

print(f"{'Label':<5} {'Structure':<20} {'Voxels totaux':<15} {'Présent dans N items':<20}")
print(f"{'-'*5} {'-'*20} {'-'*15} {'-'*20}")

for label in range(1, 14):
    name = SEG_LABELS.get(label, f"unknown_{label}")
    voxels = label_stats[label]["voxels"]
    presence = label_stats[label]["presence"]
    marker = "  ⚠ ZÉRO" if voxels == 0 else ""
    print(f"{label:<5} {name:<20} {voxels:<15} {presence:<20} {marker}")

# Détail des hippocampes
print(f"\n{'─'*80}")
print("DÉTAIL HIPPOCAMPES (labels 1 et 2)")
print(f"{'─'*80}\n")

hippo_presence_per_item = {1: [], 2: []}

for seg_file in seg_files:
    try:
        subject = seg_file.name.replace("_LF_seg.nii.gz", "")
        img = nib.load(seg_file)
        seg_data = img.get_fdata(dtype=np.float32).astype(np.int32)
        
        for label in [1, 2]:
            mask = seg_data == label
            count = mask.sum()
            hippo_presence_per_item[label].append((subject, count))
    except Exception:
        pass

print("Hippocampe gauche (label 1) :")
for subj, count in hippo_presence_per_item[1]:
    marker = "(PRÉSENT)" if count > 0 else "(absent)"
    print(f"  {subj:30s}  {count:8d} voxels  {marker}")

print("\nHippocampe droit (label 2) :")
for subj, count in hippo_presence_per_item[2]:
    marker = "(PRÉSENT)" if count > 0 else "(absent)"
    print(f"  {subj:30s}  {count:8d} voxels  {marker}")

print(f"\n{'─'*80}\n")
