"""Utilities — dispatcher helpers, split management, and convenience re-exports."""
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------------------------
# Dispatcher helpers (shared between train.py and evaluate.py)
# ---------------------------------------------------------------------------


def normalise_run_id(raw: str) -> str:
    """Strip optional 'RUN_' prefix (case-insensitive).

    Examples::

        normalise_run_id("RUN_0003") == "0003"
        normalise_run_id("0003")     == "0003"
    """
    stripped = raw.strip()
    if stripped.upper().startswith("RUN_"):
        return stripped[4:]
    return stripped


def run_cmd(cmd: list[str]) -> None:
    """Run *cmd* via subprocess, exit the process on failure."""
    print(f"  $ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] Command failed (return code {exc.returncode}).")
        sys.exit(1)


def smoke_args(entry: dict, smoke_test: bool) -> list[str]:
    """Return ``["--smoke_test"]`` when *smoke_test* is True and the entry
    supports it, otherwise return an empty list (with a warning if needed).
    """
    if not smoke_test:
        return []
    module_key = entry.get("module") or entry.get("eval_module", "<unknown>")
    if entry.get("supports_smoke_test", True):
        return ["--smoke_test"]
    print(
        f"  [WARNING] '{module_key}' does not support --smoke_test; running without it."
    )
    return []



def create_fixed_split(csv_path, seed=42, n_splits=5):
    """Create fixed patient-level split"""
    df = pd.read_csv(csv_path)
    
    # Extract subject ID from filename (LISA_XXXX_...)
    df['subject'] = df['filename'].str.extract(r'(LISA_\d+)')[0]
    
    # Create stratification label (majority vote across all tasks)
    task_cols = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']
    df['strata'] = (df[task_cols] > 0).sum(axis=1) > 0  # Good (0) vs Artifact (1+)
    
    # StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, df['strata'], df['subject'])):
        if fold_idx == 0:  # Use first fold
            split = {
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'seed': seed,
                'method': 'StratifiedGroupKFold',
                'n_samples': len(df),
                'n_subjects': df['subject'].nunique(),
            }
            break
    
    return split


def save_split(split, output_path):
    """Save split to pickle"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(split, f)
    print(f"✓ Split saved to {output_path}")


def load_split(split_path):
    """Load split from pickle"""
    with open(split_path, 'rb') as f:
        split = pickle.load(f)
    return split


# ---------------------------------------------------------------------------
# Convenience re-exports from sub-modules
# ---------------------------------------------------------------------------

from .seed import set_seed  # noqa: E402
from .config import load_config, apply_env_overrides  # noqa: E402
