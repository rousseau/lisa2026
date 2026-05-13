#!/usr/bin/env python
"""
RUN_0002 – Multi-label Evaluation Script for Task 1a
Loads the single multi-head model and produces predictions for all 7 artifacts
in one forward pass per image.
"""
import argparse
import os
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets import Task1aMultiLabelDataset, TASK_NAMES
from src.models import Task1aMultiLabelModel


def evaluate(config: dict, split: str = 'val', smoke_test: bool = False) -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = Task1aMultiLabelModel().to(device)
    ckpt_path = os.path.join(config['output']['checkpoint_dir'], 'multilabel_best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"✓ Loaded checkpoint: {ckpt_path}")

    # Data
    data_cfg = config['data']
    dataset = Task1aMultiLabelDataset(
        csv_path=data_cfg['csv_path'],
        bids_root=data_cfg['bids_root'],
        split_pkl=data_cfg['split_pkl'],
        fold=split,
        stage='val',
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    print(f"  {len(dataset)} samples in '{split}' split")

    # Inference
    records = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if smoke_test and batch_idx >= 2:
                break
            imgs = batch['img'].to(device).float()
            logits = model(imgs)                            # [B, 7, 3]
            preds = torch.argmax(logits, dim=-1).cpu()     # [B, 7]

            for i, fn in enumerate(batch['filename']):
                row = {'filename': fn}
                for t, task in enumerate(TASK_NAMES):
                    row[task] = int(preds[i, t].item())
                records.append(row)

    # Save predictions
    output_csv = config['output']['predictions_file']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)
    print(f"✓ Predictions saved to {output_csv} ({len(records)} rows)")
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="RUN_0002 – Multi-label Task 1a Evaluation")
    parser.add_argument('--config', type=str, default='configs/run_0002_upf.yaml')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke_test:
        print("⚡ SMOKE TEST MODE")

    evaluate(config, split=args.split, smoke_test=args.smoke_test)


if __name__ == '__main__':
    main()
