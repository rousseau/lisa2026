#!/usr/bin/env python
"""Unified visualisation entry point for LISA 2026.

Usage
-----
    python visualize.py --run RUN_0001
    python visualize.py --run RUN_0003 --qualitative
    python visualize.py --compare RUN_0001 RUN_0003 RUN_0004
    python visualize.py --run RUN_0003 --training-curves
"""

import argparse
import json
import os
import sys
from pathlib import Path

from src.evaluation.metrics_io import read_metrics
from src.viz.plot_utils import (
    CLASS_NAMES,
    plot_bar_comparison,
    plot_training_curves,
    save_figure,
    ensure_dir,
)
from src.cli.registry import discover_runs


def _auto_discover_runs() -> list[str]:
    """Return a sorted list of run IDs whose metrics.json exists locally."""
    registry = discover_runs()
    found = []
    for run_id in sorted(registry.keys()):
        results_dir = f"results/runs/RUN_{run_id}"
        if os.path.isfile(os.path.join(results_dir, "metrics.json")):
            found.append(run_id)
    return found


def _plot_per_class_dsc(run_id: str, metrics: dict, out_dir: str) -> str:
    """Bar chart of per-class DSC for Task 2 runs."""
    import matplotlib.pyplot as plt
    import numpy as np

    per_class = metrics.get("per_class", [])
    if not per_class:
        return ""

    # per_class may be a list of dicts or a dict mapping class_id → metrics
    if isinstance(per_class, list):
        pc = {e["class_id"]: e.get("dsc", 0.0) for e in per_class}
    else:
        pc = {int(k): v.get("dsc", 0.0) for k, v in per_class.items()}

    class_ids = sorted([c for c in pc.keys() if c != 0])
    if not class_ids:
        class_ids = sorted(pc.keys())
    x = np.arange(len(class_ids))
    labels = [CLASS_NAMES.get(c, f"Class {c}") for c in class_ids]
    values = [pc.get(c, 0.0) for c in class_ids]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, values, color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("DSC", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{run_id} — Per-class DSC", fontsize=12, fontweight="bold")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{run_id}_per_class_dsc.png")
    return save_figure(fig, out_path)


def cmd_single_run(run_id: str, qualitative: bool, training_curves: bool) -> None:
    """Generate standard plots for a single run."""
    results_dir = f"results/runs/{run_id}"
    if not os.path.isdir(results_dir):
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)

    metrics_path = os.path.join(results_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        print(f"[ERROR] metrics.json not found for {run_id}")
        sys.exit(1)

    metrics = read_metrics(results_dir)
    out_dir = os.path.join(results_dir, "plots")
    ensure_dir(out_dir)

    print(f"\n📊 {run_id} — Generating visuals")

    # 1. Global metrics text summary
    task = metrics.get("task", "unknown")
    status = metrics.get("status", "final")
    global_m = metrics.get("global", {})
    print(f"   Task   : {task}  (status={status})")
    for k, v in global_m.items():
        if isinstance(v, (int, float)):
            print(f"   {k:20s}: {v:.4f}")

    # 2. Per-class DSC bar chart (Task 2 only)
    if "per_class" in metrics:
        p = _plot_per_class_dsc(run_id, metrics, out_dir)
        if p:
            print(f"   Plot   : {p}")

    # 3. Training curves
    if training_curves:
        hist_path = os.path.join(results_dir, "training_history.json")
        if os.path.isfile(hist_path):
            with open(hist_path) as fh:
                history = json.load(fh)
            p = plot_training_curves(history, os.path.join(out_dir, f"{run_id}_training_curves.png"))
            if p:
                print(f"   Curves : {p}")
        else:
            print(f"   [WARN] No training_history.json found — skipping curves.")

    # 4. Qualitative segmentation overlays (Task 2 only, if requested)
    if qualitative and task in ("task2", "2"):
        # Delegate to existing visualisation logic (kept in src/viz/segmentation.py)
        from src.viz.segmentation import generate_overlays
        generate_overlays(run_id, out_dir)

    print(f"\n✅ Output directory: {out_dir}")


def cmd_compare(run_ids: list[str]) -> None:
    """Generate comparison bar charts across multiple runs."""
    if len(run_ids) < 2:
        print("[ERROR] --compare requires at least two runs.")
        sys.exit(1)

    # Collect metrics
    all_metrics = {}
    for rid in run_ids:
        results_dir = f"results/runs/{rid}"
        try:
            m = read_metrics(results_dir)
            all_metrics[rid] = m
        except FileNotFoundError:
            print(f"[WARN] {rid} metrics.json missing — skipping.")

    if not all_metrics:
        print("[ERROR] No metrics found for any of the specified runs.")
        sys.exit(1)

    out_dir = "results/plots/comparisons"
    ensure_dir(out_dir)

    # Detect common global metric keys across runs
    sample_global = next(iter(all_metrics.values()))["global"]
    common_keys = set(sample_global.keys())
    for m in all_metrics.values():
        common_keys &= set(m.get("global", {}).keys())

    print(f"\n📊 Comparing {len(all_metrics)} runs: metric keys = {common_keys}")

    for key in common_keys:
        data = {rid: m["global"][key] for rid, m in all_metrics.items() if key in m["global"]}
        if not data:
            continue
        title = f"Comparison — {key.replace('_', ' ').title()}"
        ylabel = key.replace("mean_", "").upper()
        out_path = os.path.join(out_dir, f"compare_{key}.png")
        plot_bar_comparison(data, title=title, ylabel=ylabel, out_path=out_path)

    # If all runs have per_class, generate per-class DSC comparison
    if all("per_class" in m for m in all_metrics.values()):
        import matplotlib.pyplot as plt
        import numpy as np

        # Assume list-of-dicts per_class format
        class_ids = None
        for m in all_metrics.values():
            pc = m["per_class"]
            if isinstance(pc, list) and pc:
                class_ids = sorted([e["class_id"] for e in pc if e.get("class_id", 0) != 0])
                break

        if class_ids:
            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(len(class_ids))
            width = 0.80 / max(len(all_metrics), 1)
            for i, (rid, m) in enumerate(all_metrics.items()):
                pc = {e["class_id"]: e.get("dsc", 0.0) for e in m["per_class"]}
                vals = [pc.get(c, 0.0) for c in class_ids]
                ax.bar(x + i * width, vals, width, label=rid, alpha=0.85)

            ax.set_xticks(x + width * (len(all_metrics) - 1) / 2)
            ax.set_xticklabels([CLASS_NAMES.get(c, f"Class {c}") for c in class_ids], rotation=35, ha="right")
            ax.set_ylabel("DSC")
            ax.set_ylim(0, 1.0)
            ax.set_title("Per-class DSC comparison", fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            save_figure(fig, os.path.join(out_dir, "compare_per_class_dsc.png"))

    print(f"\n✅ Comparison plots saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified visualisation entry point for LISA 2026.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", type=str, help="Run ID to visualise (e.g. RUN_0001 or 0001).")
    parser.add_argument("--compare", nargs="+", help="Compare multiple runs.")
    parser.add_argument("--qualitative", action="store_true", help="Generate qualitative overlays (Task 2 only).")
    parser.add_argument("--training-curves", action="store_true", help="Plot training loss curves if history exists.")
    parser.add_argument("--auto-compare", action="store_true", help="Compare all runs with metrics.json automatically.")
    args = parser.parse_args()

    if args.compare:
        cmd_compare(args.compare)
    elif args.auto_compare:
        runs = [f"RUN_{r}" for r in _auto_discover_runs()]
        if len(runs) < 2:
            print("[WARN] Fewer than 2 runs have metrics.json locally.")
            sys.exit(0)
        cmd_compare(runs)
    elif args.run:
        rid = args.run if args.run.upper().startswith("RUN_") else f"RUN_{args.run}"
        cmd_single_run(rid, qualitative=args.qualitative, training_curves=args.training_curves)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
