"""Utilities"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


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
