#!/usr/bin/env python
"""Generate quantitative comparison plots for the RUN_0003 cycle (Task 2)."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt

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

RUNS = {
    "RUN_0003\n(baseline)": "results/runs/RUN_0003/metrics.json",
    "EXP_B": "results/runs/RUN_0003_EXP_B/metrics.json",
    "EXP_C\n(winner ✓)": "results/runs/RUN_0003_EXP_C/metrics.json",
    "EXP_SYM": "results/runs/RUN_0003_EXP_SYM/metrics.json",
}
COLORS = {
    "RUN_0003\n(baseline)": "#888888",
    "EXP_B": "#F44336",
    "EXP_C\n(winner ✓)": "#2196F3",
    "EXP_SYM": "#FF9800",
}

out_dir = "results/runs/RUN_0003_EXP_C/plots"
os.makedirs(out_dir, exist_ok=True)

# ── Figure 1: per-class DSC ───────────────────────────────────────────────────
class_ids = list(range(1, 12))
x = np.arange(len(class_ids))
width = 0.20

fig, ax = plt.subplots(figsize=(14, 6))
for i, (label, path) in enumerate(RUNS.items()):
    with open(path) as f:
        d = json.load(f)
    pc = {e["class_id"]: e["dsc"] for e in d.get("per_class", [])}
    dscs = [pc.get(c, 0.0) for c in class_ids]
    ax.bar(
        x + i * width,
        dscs,
        width,
        label=label,
        color=COLORS[label],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

ax.set_xlabel("Structure", fontsize=12)
ax.set_ylabel("DSC", fontsize=12)
ax.set_title(
    "Task 2 — Per-class DSC (RUN_0003 cycle, 12-class, val set n=12)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(
    [CLASS_NAMES[c] for c in class_ids], rotation=35, ha="right", fontsize=10
)
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5, label="DSC=0.5")
ax.legend(fontsize=10, loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out1 = f"{out_dir}/run0003_expc_per_class_dsc.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ── Figure 2: global metrics ──────────────────────────────────────────────────
metrics_keys = ["mean_dsc", "mean_hd95", "mean_rve", "mean_assd"]
metrics_disp = ["Mean DSC (↑)", "Mean HD95 mm (↓)", "Mean RVE (↓)", "Mean ASSD mm (↓)"]

fig2, axes = plt.subplots(1, 4, figsize=(14, 4))
run_labels = list(RUNS.keys())
for ax2, mkey, mdisp in zip(axes, metrics_keys, metrics_disp):
    vals = []
    for label, path in RUNS.items():
        with open(path) as f:
            d = json.load(f)
        vals.append(d["global"][mkey])
    bar_colors = [COLORS[r] for r in run_labels]
    bars = ax2.bar(
        range(len(vals)), vals, color=bar_colors, edgecolor="white", linewidth=0.5
    )
    ax2.set_xticks(range(len(vals)))
    ax2.set_xticklabels(run_labels, fontsize=7.5)
    ax2.set_title(mdisp, fontsize=10, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ymax = max(vals)
    for bar, v in zip(bars, vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + ymax * 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

plt.suptitle(
    "Task 2 — Global metrics comparison (RUN_0003 cycle)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
out2 = f"{out_dir}/run0003_global_metrics.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ── Figure 3: EXP_C per-class detailed (radar-like bar) ──────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 5))
with open("results/runs/RUN_0003_EXP_C/metrics.json") as f:
    d_expc = json.load(f)
with open("results/runs/RUN_0003/metrics.json") as f:
    d_base = json.load(f)

pc_expc = {e["class_id"]: e["dsc"] for e in d_expc["per_class"]}
pc_base = {e["class_id"]: e["dsc"] for e in d_base["per_class"]}

x3 = np.arange(len(class_ids))
w = 0.35
b1 = ax3.bar(
    x3 - w / 2,
    [pc_base.get(c, 0) for c in class_ids],
    w,
    label="RUN_0003 baseline",
    color="#888888",
    alpha=0.7,
    edgecolor="white",
)
b2 = ax3.bar(
    x3 + w / 2,
    [pc_expc.get(c, 0) for c in class_ids],
    w,
    label="RUN_0003_EXP_C (best)",
    color="#2196F3",
    alpha=0.9,
    edgecolor="white",
)

# Delta annotation
for i, c in enumerate(class_ids):
    delta = pc_expc.get(c, 0) - pc_base.get(c, 0)
    col = "#1B5E20" if delta > 0 else "#B71C1C"
    ax3.text(
        x3[i] + w / 2,
        pc_expc.get(c, 0) + 0.02,
        f"{delta:+.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
        color=col,
        fontweight="bold",
    )

ax3.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax3.set_ylim(0, 1.05)
ax3.set_xticks(x3)
ax3.set_xticklabels(
    [CLASS_NAMES[c] for c in class_ids], rotation=35, ha="right", fontsize=10
)
ax3.set_ylabel("DSC", fontsize=12)
ax3.set_title(
    "EXP_C vs Baseline — Per-class DSC with delta (+green / -red)",
    fontsize=12,
    fontweight="bold",
)
ax3.legend(fontsize=10)
ax3.grid(axis="y", alpha=0.3)
plt.tight_layout()
out3 = f"{out_dir}/run0003_expc_vs_baseline_delta.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

print("\nAll plots generated successfully.")
