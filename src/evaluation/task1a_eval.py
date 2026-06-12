"""Task 1a evaluation — inline metric computation.

Supports both ordinal-per-task (RUN_0001) and multi-label (RUN_0002) modes.
Metrics are computed with sklearn directly; no intermediate CSV is required.
"""

from typing import Sequence

import numpy as np
import os
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import TASK_NAMES


def _compute_task_metrics(y_true, y_pred):
    """Return metric dict for a single artifact task."""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0))
    pre = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    agg = float(np.mean([acc, f1, f2, pre, rec]))
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "f2_macro": f2,
        "precision_macro": pre,
        "recall_macro": rec,
        "aggregate": agg,
    }


def aggregate_task1a_metrics(per_task: dict[str, dict]) -> dict:
    """Average per-task metrics into global Task 1a metrics."""
    keys = ["accuracy", "f1_macro", "f2_macro", "precision_macro", "recall_macro"]
    global_vals = {k: float(np.mean([per_task[t][k] for t in per_task])) for k in keys}
    global_vals["aggregate"] = float(np.mean(list(global_vals.values())))
    return global_vals


def evaluate_task1a_multilabel(
    model,
    val_loader: DataLoader,
    device: str,
    smoke_test: bool = False,
    task_name: str | None = None,
) -> dict:
    """Evaluate a multi-label Task 1a model inline.

    Parameters
    ----------
    task_name :
        If the model requires a task dispatch argument (e.g. ``"1a"`` for
        ``DynUNetMultiHeadModel``), it will be passed as ``model(imgs,
        task=task_name)``.  When ``None`` the model is called directly.

    Returns
    -------
    dict with keys ``per_task`` and ``global``.
    """
    model.eval()
    all_preds = [[] for _ in TASK_NAMES]
    all_labels = [[] for _ in TASK_NAMES]

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Eval-Task1a")):
        imgs = batch["img"].to(device).float()
        labels = batch["labels"].to(device)  # [B, 7]
        with torch.no_grad():
            if task_name is not None:
                logits = model(imgs, task=task_name)  # [B, 7, 3]
            else:
                logits = model(imgs)  # [B, 7, 3]
        preds = torch.argmax(logits, dim=-1).cpu().numpy()  # [B, 7]
        lbl_np = labels.cpu().numpy()
        for t in range(len(TASK_NAMES)):
            all_preds[t].extend(preds[:, t].tolist())
            all_labels[t].extend(lbl_np[:, t].tolist())
        if smoke_test and batch_idx >= 1:
            break

    per_task = {}
    for t, task_name in enumerate(TASK_NAMES):
        per_task[task_name] = _compute_task_metrics(all_labels[t], all_preds[t])

    return {"per_task": per_task, "global": aggregate_task1a_metrics(per_task)}


def evaluate_task1a_ordinal(
    model_factory,
    task_order: Sequence[str],
    val_loader_factory,
    ckpt_dir: str,
    device: str,
    smoke_test: bool = False,
) -> dict:
    """Evaluate a set of independent ordinal classifiers (RUN_0001 style).

    Parameters
    ----------
    model_factory : callable()
        Returns a fresh ``Task1aOrdinalModel``.
    task_order : list of str
        Artifact names in the order they appear in the CSV.
    val_loader_factory : callable(task_name)
        Returns a ``DataLoader`` for a given task.
    ckpt_dir : str
        Directory containing ``{task_name}_best.pt`` checkpoints.
    device : str
        Torch device.
    smoke_test : bool

    Returns
    -------
    dict with keys ``per_task`` and ``global``.
    """
    all_preds = [[] for _ in task_order]
    all_labels = [[] for _ in task_order]

    for t, task in enumerate(task_order):
        ckpt_path = f"{ckpt_dir}/{task}_best.pt"
        if not os.path.exists(ckpt_path):
            print(f"  ⚠ Checkpoint not found: {ckpt_path}")
            continue
        model = model_factory().to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state if "model_state_dict" not in state else state["model_state_dict"])
        model.eval()

        loader = val_loader_factory(task)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Eval-{task}")):
                img = batch["img"].to(device).float()
                labels = batch["label"].cpu().numpy()
                logits = model(img)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds[t].extend(preds.tolist())
                all_labels[t].extend(labels.tolist())
                if smoke_test and batch_idx >= 1:
                    break

    per_task = {}
    for t, task in enumerate(task_order):
        if not all_labels[t]:
            continue
        per_task[task] = _compute_task_metrics(all_labels[t], all_preds[t])

    return {"per_task": per_task, "global": aggregate_task1a_metrics(per_task)}
