#!/usr/bin/env python
"""
Train Task 1a model for a single artifact type
"""
import argparse
import os
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from src.datasets import get_dataloaders
from src.models import Task1aOrdinalModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, fbeta_score


class Trainer:
    def __init__(self, config, task_name):
        self.config = config
        self.task_name = task_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_val_f1 = 0
        self.patience_counter = 0
        
        # Model
        self.model = Task1aOrdinalModel(num_classes=3).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        
        # Data
        self.train_loader, self.val_loader, self.n_train, self.n_val = get_dataloaders(
            csv_path=config['data']['csv_path'],
            bids_root=config['data']['bids_root'],
            split_pkl=config['data']['split_pkl'],
            task_name=task_name,
            batch_size=config['training']['batch_size'],
            num_workers=2
        )
        
        # Checkpoint path
        self.ckpt_dir = config['output']['checkpoint_dir']
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.ckpt_dir, f"{task_name}_best.pt")
        
        # Log path
        self.log_file = os.path.join(config['output']['log_dir'], f"task1a_{task_name}.log")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        self.log = open(self.log_file, 'w')
        self._log(f"Task: {task_name}")
        self._log(f"Train samples: {self.n_train}, Val samples: {self.n_val}")
    
    def _log(self, msg):
        print(msg)
        self.log.write(msg + '\n')
        self.log.flush()
    
    def _compute_metrics(self, y_true, y_pred):
        """Compute metrics"""
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        agg = np.mean([acc, f1, f2, prec, rec])
        return {'acc': acc, 'f1': f1, 'f2': f2, 'prec': prec, 'rec': rec, 'agg': agg}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        for batch in tqdm(self.train_loader, desc=f"Train"):
            img = batch['img'].to(self.device).float()
            label = batch['label'].to(self.device)
            
            logits = self.model(img)
            loss = cross_entropy(logits, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())
        
        metrics = self._compute_metrics(all_labels, all_preds)
        return total_loss / len(self.train_loader), metrics
    
    def val_epoch(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val"):
                img = batch['img'].to(self.device).float()
                label = batch['label'].to(self.device)
                
                logits = self.model(img)
                loss = cross_entropy(logits, label)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(label.cpu().numpy())
        
        metrics = self._compute_metrics(all_labels, all_preds)
        return total_loss / len(self.val_loader), metrics
    
    def train(self):
        num_epochs = self.config['training']['num_epochs']
        patience = self.config['early_stopping']['patience']
        
        for epoch in range(num_epochs):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.val_epoch()
            
            msg = f"Epoch {epoch+1:3d}/{num_epochs} | "
            msg += f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            msg += f"val_f1={val_metrics['f1']:.4f} val_agg={val_metrics['agg']:.4f}"
            self._log(msg)
            
            # Early stopping
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
                self._log(f"  → Best checkpoint saved (F1={self.best_val_f1:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    self._log(f"  → Early stopping triggered at epoch {epoch+1}")
                    break
        
        self._log(f"\n✓ Training completed. Best F1: {self.best_val_f1:.4f}")
        self.log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/run_0001_baseline.yaml')
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    trainer = Trainer(config, args.task)
    trainer.train()


if __name__ == '__main__':
    main()
