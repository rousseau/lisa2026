"""Shared plotting utilities for LISA 2026 visualisations."""

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Constants ──────────────────────────────────────────────────────────────

DPI = 150
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 10)

CLASS_NAMES = {
    1: "Hippo L",
    2: "Hippo R",
    3: "Caudate L",
    4: "Caudate R",
    5: "Lentiform L",
    6: "Lentiform R",
    7: "Ventricle L",
    8: "Ventricle R",
    9: "ExV L",
    10: "ExV R",
    11: "Aux",
}


# ── Helpers ────────────────────────────────────────────────────────────────


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_figure(fig: plt.Figure, path: str, dpi: int = DPI) -> str:
    """Save *fig* to *path* and close it.

    Returns the absolute path to the saved file.
    """
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return os.path.abspath(path)


def plot_bar_comparison(
    data: dict[str, Any],
    title: str,
    ylabel: str,
    out_path: str,
    color_map: dict[str, str] | None = None,
    hline: tuple[float, str] | None = None,
) -> str:
    """Create a grouped bar chart from a dict ``label → value`` and save it.

    Parameters
    ----------
    data:
        Mapping of run/label names to scalar values.
    title:
        Plot title.
    ylabel:
        Y-axis label.
    out_path:
        Output file path (PNG).
    color_map:
        Optional dict mapping labels to colors.
    hline:
        Optional (value, label) horizontal reference line.

    Returns
    -------
    Absolute path to the saved figure.
    """
    labels = list(data.keys())
    values = list(data.values())
    colors = [color_map.get(l, "#2196F3") for l in labels] if color_map else ["#2196F3"] * len(labels)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    if hline is not None:
        val, lbl = hline
        ax.axhline(val, color="grey", linestyle="--", linewidth=0.8, alpha=0.5, label=lbl)

    ymax = max(values) if values else 1.0
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + ymax * 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()
    return save_figure(fig, out_path)


def plot_training_curves(history: list[dict], out_path: str) -> str:
    """Plot training / validation loss curves from a training history.

    Parameters
    ----------
    history:
        List of per-epoch dicts (must contain at least ``epoch`` and ``loss``
        or ``val_loss`` keys).
    out_path:
        Output PNG path.

    Returns
    -------
    Absolute path to the saved figure.
    """
    if not history:
        print("[WARN] Empty history — skipping training curves.")
        return ""

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Loss curves
    for key in ("loss", "val_loss"):
        if key in history[0]:
            axes[0].plot(epochs, [h[key] for h in history], label=key, marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Metric curves (anything matching "val_*")
    val_keys = [k for k in history[0].keys() if k.startswith("val_") and k != "val_loss"]
    for key in val_keys[:3]:  # limit to first 3 to avoid clutter
        axes[1].plot(epochs, [h[key] for h in history], label=key, marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return save_figure(fig, out_path)
