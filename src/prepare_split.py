#!/usr/bin/env python
"""
Prepare fixed patient-level split for Task 1a
"""
import argparse
import os
from src.utils import create_fixed_split, save_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='/home/rousseau/Data/LISA2026/LISA_Task1a_2026.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/splits/task1a_fixed.pkl')
    args = parser.parse_args()
    
    print(f"[1/2] Creating fixed split from {args.csv}...")
    split = create_fixed_split(args.csv, seed=args.seed, n_splits=5)
    
    print(f"[2/2] Saving split...")
    save_split(split, args.output)
    
    print(f"\n✓ Split created:")
    print(f"  Train: {len(split['train_indices'])} samples")
    print(f"  Val: {len(split['val_indices'])} samples")
    print(f"  Subjects: {split['n_subjects']}")


if __name__ == '__main__':
    main()
