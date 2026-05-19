#!/usr/bin/env python
"""
Evaluate Task 1a models on validation set
"""
import argparse
import os
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets import Task1aDataset
from src.models import Task1aOrdinalModel


def evaluate(config, task_names, split='val'):
    """Evaluate all task models"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_dir = config['output']['checkpoint_dir']
    
    predictions_data = []
    
    for task in task_names:
        print(f"\nEvaluating {task}...")
        
        # Load model
        model = Task1aOrdinalModel().to(device)
        ckpt_path = os.path.join(ckpt_dir, f"{task}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠ Checkpoint not found: {ckpt_path}")
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        # Load data
        dataset = Task1aDataset(
            csv_path=config['data']['csv_path'],
            bids_root=config['data']['bids_root'],
            split_pkl=config['data']['split_pkl'],
            fold=split,
            task_name=task,
            stage='val'
        )
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
        
        # Predict
        all_filenames = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                img = batch['img'].to(device).float()
                filenames = batch['filename']
                labels = batch['label'].cpu().numpy()
                
                logits = model(img)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_filenames.extend(filenames)
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Store predictions
        if not predictions_data:
            for i, fn in enumerate(all_filenames):
                predictions_data.append({'filename': fn})
        
        for i, (fn, pred) in enumerate(zip(all_filenames, all_preds)):
            for j, data in enumerate(predictions_data):
                if data['filename'] == fn:
                    data[task] = pred
                    break
        
        print(f"  ✓ {task}: {len(all_preds)} predictions")
    
    # Save predictions CSV
    output_csv = config['output']['predictions_file']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(predictions_data)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Predictions saved to {output_csv}")
    
    return output_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/run_0001_baseline.yaml')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    task_names = config['tasks']
    evaluate(config, task_names, split='val')


if __name__ == '__main__':
    main()
