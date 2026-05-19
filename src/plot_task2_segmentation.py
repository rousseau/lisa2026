#!/usr/bin/env python
"""Generate qualitative segmentation plots for RUN_0003 Task2 DynUNet."""
import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colors import ListedColormap
from monai.inferers import sliding_window_inference

from src.models import Task2DynUNetModel


def to_3tuple(values):
    return tuple(int(v) for v in values)


def build_model(cfg, device):
    mcfg = cfg["model"]
    model = Task2DynUNetModel(
        in_channels=int(mcfg["in_channels"]),
        out_channels=int(mcfg["out_channels"]),
        kernel_size=tuple(tuple(int(x) for x in ks) for ks in mcfg["kernel_size"]),
        strides=tuple(tuple(int(x) for x in st) for st in mcfg["strides"]),
        upsample_kernel_size=tuple(tuple(int(x) for x in st) for st in mcfg["upsample_kernel_size"]),
        filters=tuple(int(x) for x in mcfg["filters"]),
        norm_name=mcfg.get("norm_name", "instance"),
        deep_supervision=bool(mcfg.get("deep_supervision", False)),
    ).to(device)

    ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], "task2_dynunet_best.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def find_subject_paths(data_root, subject):
    img = os.path.join(data_root, f"{subject}_ciso.nii.gz")
    lbl = os.path.join(data_root, f"{subject}_LF_seg.nii.gz")
    if not os.path.exists(img):
        raise FileNotFoundError(img)
    if not os.path.exists(lbl):
        raise FileNotFoundError(lbl)
    return img, lbl


def infer_segmentation(model, image_np, roi_size, overlap, sw_batch_size, device):
    image = torch.from_numpy(image_np[None, None]).float().to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int16)
    return pred


def pick_slice_indices(label):
    nonzero = (label > 0).astype(np.uint8)
    if nonzero.sum() == 0:
        d, h, w = label.shape
        return d // 2, h // 2, w // 2

    ax = np.argmax(nonzero.sum(axis=(1, 2)))
    cor = np.argmax(nonzero.sum(axis=(0, 2)))
    sag = np.argmax(nonzero.sum(axis=(0, 1)))
    return int(ax), int(cor), int(sag)


def get_views(volume, idxs):
    ax, cor, sag = idxs
    v_ax = volume[ax, :, :]
    v_cor = volume[:, cor, :]
    v_sag = volume[:, :, sag]
    return [np.rot90(v_ax), np.rot90(v_cor), np.rot90(v_sag)]


def make_overlay(ax, img2d, seg2d, cmap, title):
    ax.imshow(img2d, cmap="gray")
    masked = np.ma.masked_where(seg2d == 0, seg2d)
    ax.imshow(masked, cmap=cmap, alpha=0.55, interpolation="nearest", vmin=0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def subject_mean_dice(pred_csv, subject):
    df = pd.read_csv(pred_csv)
    if subject not in set(df["subject"].unique()):
        return None
    return float(df[df["subject"] == subject]["dsc"].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/run_0003_task2_dynunet.yaml")
    parser.add_argument("--pred-csv", type=str, default="results/runs/RUN_0003/predictions_val_task2.csv")
    parser.add_argument("--split-pkl", type=str, default="results/splits/task2_fixed.pkl")
    parser.add_argument("--subjects", nargs="*", default=["LISA_0018", "LISA_0050", "LISA_1001"])
    parser.add_argument("--out-dir", type=str, default="results/runs/RUN_0003/plots")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = os.getenv("LISA_DATA_ROOT", cfg["data"]["data_root"])

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.split_pkl, "rb") as f:
        split = pickle.load(f)
    val_subjects = set(split.get("val_subjects", []))

    selected = [s for s in args.subjects if s in val_subjects]
    if len(selected) == 0:
        raise RuntimeError("No selected subjects are in validation split")

    model = build_model(cfg, device)
    roi_size = to_3tuple(cfg["data"]["val_roi_size"])
    overlap = float(cfg["inference"]["overlap"])
    sw_batch_size = int(cfg["inference"]["sw_batch_size"])

    colors = plt.cm.get_cmap("tab20", int(cfg["model"]["out_channels"]))
    cmap = ListedColormap(colors(np.arange(colors.N)))

    summary = []

    for subject in selected:
        img_path, lbl_path = find_subject_paths(data_root, subject)
        img = nib.load(img_path).get_fdata().astype(np.float32)
        lbl = nib.load(lbl_path).get_fdata().astype(np.int16)

        pred = infer_segmentation(model, img, roi_size, overlap, sw_batch_size, device)

        idxs = pick_slice_indices(lbl)
        img_views = get_views(img, idxs)
        lbl_views = get_views(lbl, idxs)
        pred_views = get_views(pred, idxs)

        dsc = subject_mean_dice(args.pred_csv, subject)
        dsc_txt = "n/a" if dsc is None else f"{dsc:.3f}"

        fig, axes = plt.subplots(3, 3, figsize=(13, 12))
        view_names = ["Axial", "Coronal", "Sagittal"]

        for j in range(3):
            axes[0, j].imshow(img_views[j], cmap="gray")
            axes[0, j].set_title(f"{view_names[j]} - MRI", fontsize=10)
            axes[0, j].axis("off")

            make_overlay(axes[1, j], img_views[j], lbl_views[j], cmap, f"{view_names[j]} - GT")
            make_overlay(axes[2, j], img_views[j], pred_views[j], cmap, f"{view_names[j]} - Pred")

        fig.suptitle(f"RUN_0003 | {subject} | mean DSC={dsc_txt}", fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        out_png = os.path.join(args.out_dir, f"run0003_seg_{subject}.png")
        fig.savefig(out_png, dpi=140)
        plt.close(fig)

        summary.append({"subject": subject, "mean_dsc": dsc, "plot": out_png})

    with open(os.path.join(args.out_dir, "run0003_plot_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Generated {len(summary)} segmentation plot(s) in {args.out_dir}")
    for row in summary:
        print(f" - {row['subject']}: {row['plot']} | mean_dsc={row['mean_dsc']}")


if __name__ == "__main__":
    main()
