#!/usr/bin/env python
"""Prepare LISA data as nnU-Net Dataset501 with custom split from task2_fixed.pkl.

Usage:
    python src/prepare_nnunet_dataset.py --data-root /home/rousseau/Data/LISA2026 \
        --split-pkl results/splits/task2_fixed.pkl \
        --output-root nnUNet_raw/Dataset501_LISA
"""

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-pkl", default="results/splits/task2_fixed.pkl")
    parser.add_argument("--output-root", default="nnUNet_raw/Dataset501_LISA")
    parser.add_argument("--image-suffix", default="_ciso.nii.gz")
    parser.add_argument("--label-suffix", default="_LF_seg.nii.gz")
    args = parser.parse_args()

    with open(args.split_pkl, "rb") as fh:
        split = pickle.load(fh)

    train_subjects = set(split.get("train_subjects", []))
    val_subjects = set(split.get("val_subjects", []))
    all_subjects = sorted(train_subjects | val_subjects)

    out_root = Path(args.output_root)
    images_tr = out_root / "imagesTr"
    labels_tr = out_root / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    num_training = 0
    for subj in all_subjects:
        img_src = Path(args.data_root) / f"{subj}{args.image_suffix}"
        lbl_src = Path(args.data_root) / f"{subj}{args.label_suffix}"
        if not img_src.exists() or not lbl_src.exists():
            print(f"[WARN] Skipping {subj}: missing files")
            continue

        # nnU-Net naming convention: case_identifier_0000.nii.gz
        img_dst = images_tr / f"{subj}_0000.nii.gz"
        lbl_dst = labels_tr / f"{subj}.nii.gz"

        # Use symlink (absolute) to save disk space
        img_dst.unlink(missing_ok=True)
        lbl_dst.unlink(missing_ok=True)
        img_dst.symlink_to(img_src.resolve())
        lbl_dst.symlink_to(lbl_src.resolve())
        num_training += 1

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {
            "background": 0,
            "L_Hippocampus": 1,
            "R_Hippocampus": 2,
            "L_Caudate": 3,
            "R_Caudate": 4,
            "L_Lentiform": 5,
            "R_Lentiform": 6,
            "L_Ventricle": 7,
            "R_Ventricle": 8,
            "L_ExV": 9,
            "R_ExV": 10,
            "Aux": 11,
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": "LISA2026_Task2",
        "description": "LISA 2026 low-field MRI multi-structure segmentation",
    }

    with open(out_root / "dataset.json", "w") as fh:
        json.dump(dataset_json, fh, indent=2)

    # Custom split to preserve comparability with RUN_0003
    splits = [{"train": sorted(train_subjects), "val": sorted(val_subjects)}]
    splits_path = out_root / "splits_final.pkl"
    with open(splits_path, "wb") as fh:
        pickle.dump(splits, fh)

    print(f"[OK] Dataset501_LISA prepared: {num_training} cases")
    print(f"     Train: {len(train_subjects)} | Val: {len(val_subjects)}")
    print(f"     Custom split written to {splits_path}")


if __name__ == "__main__":
    main()
