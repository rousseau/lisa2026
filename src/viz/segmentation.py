"""Qualitative segmentation overlays for Task 2 runs."""

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


def _build_model(cfg, device):
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


def _find_subject_paths(data_root, subject):
    img = os.path.join(data_root, f"{subject}_ciso.nii.gz")
    lbl = os.path.join(data_root, f"{subject}_LF_seg.nii.gz")
    return img, lbl


def _infer(model, image_np, roi_size, overlap, sw_batch_size, device):
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


def _pick_slice_indices(label):
    nonzero = (label > 0).astype(np.uint8)
    if nonzero.sum() == 0:
        d, h, w = label.shape
        return d // 2, h // 2, w // 2
    ax = np.argmax(nonzero.sum(axis=(1, 2)))
    cor = np.argmax(nonzero.sum(axis=(0, 2)))
    sag = np.argmax(nonzero.sum(axis=(0, 1)))
    return int(ax), int(cor), int(sag)


def _get_views(volume, idxs):
    ax, cor, sag = idxs
    return [np.rot90(volume[ax, :, :]), np.rot90(volume[:, cor, :]), np.rot90(volume[:, :, sag])]


def _make_overlay(ax, img2d, seg2d, cmap, title):
    ax.imshow(img2d, cmap="gray")
    masked = np.ma.masked_where(seg2d == 0, seg2d)
    ax.imshow(masked, cmap=cmap, alpha=0.55, interpolation="nearest", vmin=0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _subject_mean_dice(pred_csv, subject):
    df = pd.read_csv(pred_csv)
    if subject not in set(df["subject"].unique()):
        return None
    return float(df[df["subject"] == subject]["dsc"].mean())


def generate_overlays(run_id: str, out_dir: str) -> None:
    """Generate 3×3 segmentation overlay plots for a Task 2 run.

    This is a **simplified generic version** of the old plot_task2_segmentation.py.
    It reads the run config, loads the best checkpoint, and generates one overlay
    per validation subject defined in ``results/splits/task2_fixed.pkl``.

    Parameters
    ----------
    run_id:
        Full run ID, e.g. ``"RUN_0003"``.
    out_dir:
        Directory to write PNGs into.
    """
    config_path = f"configs/run_{run_id.lower().replace('run_', '')}_task2_dynunet.yaml"
    if not os.path.isfile(config_path):
        print(f"[WARN] Config not found: {config_path} — skipping qualitative plots.")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = os.getenv("LISA_DATA_ROOT", cfg["data"]["data_root"])

    split_pkl = cfg["data"].get("split_pkl", "results/splits/task2_fixed.pkl")
    with open(split_pkl, "rb") as f:
        split = pickle.load(f)
    val_subjects = set(split.get("val_subjects", []))

    if not val_subjects:
        print("[WARN] No validation subjects found — skipping overlays.")
        return

    # Limit to 3 representative subjects for speed
    selected = sorted(val_subjects)[:3]

    model = _build_model(cfg, device)
    roi_size = to_3tuple(cfg["data"]["val_roi_size"])
    overlap = float(cfg["inference"]["overlap"])
    sw_batch_size = int(cfg["inference"]["sw_batch_size"])

    colors = plt.cm.get_cmap("tab20", int(cfg["model"]["out_channels"]))
    cmap = ListedColormap(colors(np.arange(colors.N)))

    pred_csv = os.path.join(f"results/runs/{run_id}", "predictions_val_task2.csv")

    for subject in selected:
        try:
            img_path, lbl_path = _find_subject_paths(data_root, subject)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc} — skipping {subject}.")
            continue

        img = nib.load(img_path).get_fdata().astype(np.float32)
        lbl = nib.load(lbl_path).get_fdata().astype(np.int16)
        pred = _infer(model, img, roi_size, overlap, sw_batch_size, device)

        idxs = _pick_slice_indices(lbl)
        img_views = _get_views(img, idxs)
        lbl_views = _get_views(lbl, idxs)
        pred_views = _get_views(pred, idxs)

        dsc = _subject_mean_dice(pred_csv, subject) if os.path.isfile(pred_csv) else None
        dsc_txt = "n/a" if dsc is None else f"{dsc:.3f}"

        fig, axes = plt.subplots(3, 3, figsize=(13, 12))
        view_names = ["Axial", "Coronal", "Sagittal"]
        for j in range(3):
            axes[0, j].imshow(img_views[j], cmap="gray")
            axes[0, j].set_title(f"{view_names[j]} - MRI", fontsize=10)
            axes[0, j].axis("off")
            _make_overlay(axes[1, j], img_views[j], lbl_views[j], cmap, f"{view_names[j]} - GT")
            _make_overlay(axes[2, j], img_views[j], pred_views[j], cmap, f"{view_names[j]} - Pred")

        fig.suptitle(f"{run_id} | {subject} | mean DSC={dsc_txt}", fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        out_png = os.path.join(out_dir, f"{run_id}_seg_{subject}.png")
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"   Overlay: {out_png}")
