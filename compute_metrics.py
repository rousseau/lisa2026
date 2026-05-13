#!/usr/bin/env python
"""
Compute metrics for Task 1a predictions
"""
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, fbeta_score


def compute_metrics(predictions_csv, ground_truth_csv):
    """Compute per-task and global metrics"""
    
    pred_df = pd.read_csv(predictions_csv)
    gt_df = pd.read_csv(ground_truth_csv)
    
    # Merge on filename
    df = pred_df.merge(gt_df, on='filename', how='inner')
    
    tasks = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']
    
    per_task = {}
    global_scores = {'accuracy': [], 'f1_macro': [], 'f2_macro': [], 'precision_macro': [], 'recall_macro': []}
    
    for task in tasks:
        if f"{task}_x" in df.columns:  # Prediction column
            y_true = df[f"{task}_y"].values  # Ground truth
            y_pred = df[f"{task}_x"].values  # Prediction
        else:
            continue
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        agg = np.mean([acc, f1, f2, prec, rec])
        
        per_task[task] = {
            'accuracy': float(acc),
            'f1_macro': float(f1),
            'f2_macro': float(f2),
            'precision_macro': float(prec),
            'recall_macro': float(rec),
            'aggregate': float(agg)
        }
        
        global_scores['accuracy'].append(acc)
        global_scores['f1_macro'].append(f1)
        global_scores['f2_macro'].append(f2)
        global_scores['precision_macro'].append(prec)
        global_scores['recall_macro'].append(rec)
    
    # Global average
    global_metrics = {
        'accuracy': float(np.mean(global_scores['accuracy'])),
        'f1_macro': float(np.mean(global_scores['f1_macro'])),
        'f2_macro': float(np.mean(global_scores['f2_macro'])),
        'precision_macro': float(np.mean(global_scores['precision_macro'])),
        'recall_macro': float(np.mean(global_scores['recall_macro'])),
    }
    global_metrics['aggregate'] = float(np.mean(list(global_metrics.values())))
    
    return per_task, global_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--ground-truth', type=str, default='/home/rousseau/Data/LISA2026/LISA_Task1a_2026.csv')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--run-id', type=str, default='0001')
    parser.add_argument('--run-date', type=str, default='2026-05-12')
    args = parser.parse_args()
    
    print(f"Computing metrics from {args.predictions}...")
    per_task, global_metrics = compute_metrics(args.predictions, args.ground_truth)
    
    result = {
        'run_id': args.run_id,
        'date': args.run_date,
        'task': '1a',
        'per_task': per_task,
        'global': global_metrics
    }
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Metrics saved to {args.output}")
    print(f"\nGlobal Score: {global_metrics['aggregate']:.4f}")
    print(f"  Accuracy: {global_metrics['accuracy']:.4f}")
    print(f"  F1: {global_metrics['f1_macro']:.4f}")
    print(f"  F2: {global_metrics['f2_macro']:.4f}")
    print(f"  Precision: {global_metrics['precision_macro']:.4f}")
    print(f"  Recall: {global_metrics['recall_macro']:.4f}")


if __name__ == '__main__':
    main()
