#!/usr/bin/env python
"""Create a fixed Task 2 patient-level split for segmentation."""
import argparse
import os
import pickle
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.datasets import build_task2_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/splits/task2_fixed.pkl")
    parser.add_argument("--manifest-output", type=str, default="results/splits/task2_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--image-suffix", type=str, default="_ciso.nii.gz")
    parser.add_argument("--label-suffix", type=str, default="_LF_seg.nii.gz")
    args = parser.parse_args()

    records = build_task2_records(
        data_root=args.data_root,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
    )

    if len(records) == 0:
        raise RuntimeError(f"No Task2 image/label pairs found in {args.data_root}")

    df = pd.DataFrame(records)
    groups = df["subject"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(df, groups=groups))

    train_subjects = sorted(df.iloc[train_idx]["subject"].unique().tolist())
    val_subjects = sorted(df.iloc[val_idx]["subject"].unique().tolist())

    split = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "method": "GroupShuffleSplit",
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "n_samples": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(split, f)

    os.makedirs(os.path.dirname(args.manifest_output), exist_ok=True)
    df.to_csv(args.manifest_output, index=False)

    print(f"Split saved to {args.output}")
    print(f"Manifest saved to {args.manifest_output}")
    print(f"Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)}")


if __name__ == "__main__":
    main()
