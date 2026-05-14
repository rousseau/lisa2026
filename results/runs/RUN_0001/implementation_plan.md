# RUN_0001 – Implementation Plan

## Execution Overview

This document details the step-by-step execution of RUN_0001 baseline training and evaluation.

**Duration**: ~2 days (Day 1: prep 1h, Day 2: training 4-5h, Day 3: eval+consolidation 1h)

---

## Phase 1: Data Preparation (Day 1)

### Step 1.1 – Generate Fixed Patient-Level Split

**Objective**: Create a deterministic train/val split at patient level (StratifiedGroupKFold) that will be reused across all future runs for comparability.

**Command**:
```bash
cd /home/rousseau/Exp/lisa2026
python src/prepare_split.py \
  --csv data/LISA2026_labels.csv \
  --seed 42 \
  --n_splits 5 \
  --output results/splits/task1a_fixed.pkl \
  --verbose
```

**Input**:
- `data/LISA2026_labels.csv` – CSV with columns: `filename, Noise, Zipper, Positioning, Banding, Motion, Contrast, Distortion, subject_id`

**Output**:
- `results/splits/task1a_fixed.pkl` – Pickle file containing dict of splits:
  ```python
  {
    'train_indices': [...],
    'val_indices': [...],
    'subject_groups': {...},
    'seed': 42,
    'n_samples': N,
    'n_subjects': S
  }
  ```

**Validation**:
- Check no subject appears in both train and val: `assert len(train_subjects & val_subjects) == 0`
- Check label distribution preserved: stratification ratio within ±5%
- Log to: `results/runs/RUN_0001/notes.md`

### Step 1.2 – Validate Data Integrity

**Objective**: Ensure all volume files exist and labels are consistent.

**Command**:
```bash
python src/validate_data.py \
  --csv data/LISA2026_labels.csv \
  --bids_root data/BIDS \
  --output results/runs/RUN_0001/data_validation.txt
```

**Checks**:
- All `.nii.gz` files referenced in CSV exist ✓
- Label range 0–2 for all tasks ✓
- No missing values in task columns ✓
- No NaN volumes ✓

### Step 1.3 – Create Dataset Loaders (Dry Run)

**Objective**: Test data pipeline (load, preprocess, batch) without training.

**Command**:
```bash
python -c "
from src.datasets.task1a import Task1aDataset
from torch.utils.data import DataLoader

dataset = Task1aDataset(
    csv_path='data/LISA2026_labels.csv',
    bids_root='data/BIDS',
    split_pkl='results/splits/task1a_fixed.pkl',
    fold='train',
    task_name='Noise',
    stage='train'
)
loader = DataLoader(dataset, batch_size=4, num_workers=2)

# Iterate once to check for errors
for batch in loader:
    print(f'Batch shape: {batch[\"img\"].shape}')
    print(f'Label dtype: {batch[\"label\"].dtype}')
    break
print('✓ Dataset pipeline OK')
"
```

**Expected Output**:
```
Batch shape: torch.Size([4, 1, 150, 150, 150])
Label dtype: torch.int64
✓ Dataset pipeline OK
```

**Artifacts**:
- Logs to stdout (capture in `notes.md`)

---

## Phase 2: Training (Day 2)

### Step 2.1 – Training Configuration Setup

**Objective**: Finalize config file for all 7 tasks.

**File**: `configs/run_0001_baseline.yaml`

**Content** (example):
```yaml
# RUN_0001 Baseline Configuration
run_id: "0001"
date: "2026-05-12"
tasks:
  - Noise
  - Zipper
  - Positioning
  - Banding
  - Motion
  - Contrast
  - Distortion

# Data
data:
  csv_path: "data/LISA2026_labels.csv"
  bids_root: "data/BIDS"
  split_pkl: "results/splits/task1a_fixed.pkl"
  spatial_size: [150, 150, 150]

# Training
training:
  batch_size: 8
  num_epochs: 70
  learning_rate: 1.0e-4
  optimizer: "adam"
  loss_type: "ordinal_emd"  # Combined Ordinal Contrastive + EMD
  
# Early Stopping
early_stopping:
  patience: 10
  monitor: "val_f1_macro"
  mode: "max"

# Augmentation
augmentation:
  rotate_prob: 0.2
  rotate_range: [15, 15, 10]
  affine_prob: 0.2
  affine_scale: 0.05
  intensity_shift_prob: 0.2
  intensity_shift: 0.1
  contrast_prob: 0.2
  contrast_gamma: [0.8, 1.2]

# Environment
device: "cuda"
num_workers: 2
seed: 42
```

### Step 2.2 – Train Model for Each Task

**Objective**: Train 7 independent DenseNet264 models, one per artifact type.

**Command** (run sequentially):
```bash
# Create output directories
mkdir -p outputs/checkpoints/RUN_0001
mkdir -p outputs/logs/RUN_0001

# Train each task
for TASK in Noise Zipper Positioning Banding Motion Contrast Distortion; do
  echo "Training $TASK..."
  python src/train_task1a.py \
    --config configs/run_0001_baseline.yaml \
    --task $TASK \
    --output_dir outputs/checkpoints/RUN_0001 \
    --log_file outputs/logs/RUN_0001/task1a_${TASK}.log \
    2>&1 | tee -a outputs/logs/RUN_0001/task1a_${TASK}.log
  
  echo "✓ Training complete for $TASK"
done
```

**Per-task Output**:
- Checkpoint: `outputs/checkpoints/RUN_0001/{TASK}_best.pt`
- Log: `outputs/logs/RUN_0001/task1a_{TASK}.log`
- Metrics per epoch: logged to stdout + file

**Expected Duration per Task**: ~30-40 min (GPU)

**Monitoring**:
```bash
# In another terminal
tail -f outputs/logs/RUN_0001/task1a_Noise.log
```

**Expected Log Content**:
```
Epoch 1/70: loss=0.87, val_f1=0.52, early_stop_counter=0
Epoch 2/70: loss=0.72, val_f1=0.61, early_stop_counter=0
...
Epoch 63/70: loss=0.15, val_f1=0.78, early_stop_counter=0
Epoch 64/70: val_f1=0.75 (no improvement) early_stop_counter=1
...
Epoch 73 (ES triggered): Loading best model from epoch 63
✓ Training completed. Best epoch: 63, Best F1: 0.78
```

### Step 2.3 – Checkpoint Verification

**Objective**: Verify all 7 checkpoints saved and loadable.

**Command**:
```bash
python -c "
import torch
import os

checkpoint_dir = 'outputs/checkpoints/RUN_0001'
tasks = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']

for task in tasks:
    ckpt_path = os.path.join(checkpoint_dir, f'{task}_best.pt')
    if not os.path.exists(ckpt_path):
        print(f'✗ Missing checkpoint: {ckpt_path}')
    else:
        ckpt = torch.load(ckpt_path)
        print(f'✓ {task}: {ckpt.keys()}')
"
```

**Expected Output**:
```
✓ Noise: dict_keys(['model_state_dict', 'optimizer_state_dict', 'epoch', 'metrics'])
✓ Zipper: ...
...
```

---

## Phase 3: Evaluation (Day 3)

### Step 3.1 – Inference on Validation Set

**Objective**: Generate predictions for all validation samples across all 7 tasks.

**Command**:
```bash
python src/evaluate_task1a.py \
  --config configs/run_0001_baseline.yaml \
  --checkpoint_dir outputs/checkpoints/RUN_0001 \
  --split val \
  --output_csv results/runs/RUN_0001/predictions_val.csv \
  --output_probas results/runs/RUN_0001/probabilities_val.pkl
```

**Output Files**:
- `results/runs/RUN_0001/predictions_val.csv` – CSV with columns: `filename, Noise, Zipper, Positioning, Banding, Motion, Contrast, Distortion` (hard predictions)
- `results/runs/RUN_0001/probabilities_val.pkl` – Pickle dict of probability arrays per task

**Expected CSV format**:
```
filename,Noise,Zipper,Positioning,Banding,Motion,Contrast,Distortion
sub-001_ses-01_acq-axi_run-1_T1w.nii.gz,1,0,2,1,0,1,2
sub-001_ses-01_acq-cor_run-2_T1w.nii.gz,0,1,1,1,0,0,1
...
```

### Step 3.2 – Compute Metrics

**Objective**: Calculate per-task and global metrics.

**Command**:
```bash
python src/compute_metrics_task1a.py \
  --predictions results/runs/RUN_0001/predictions_val.csv \
  --ground_truth data/LISA2026_labels.csv \
  --split_pkl results/splits/task1a_fixed.pkl \
  --output_json results/runs/RUN_0001/metrics.json \
  --output_table results/runs/RUN_0001/metrics_table.txt
```

**Output Files**:
- `results/runs/RUN_0001/metrics.json` – Structured metrics for dashboard ingestion
- `results/runs/RUN_0001/metrics_table.txt` – Human-readable table

**Expected metrics.json format**:
```json
{
  "run_id": "0001",
  "date": "2026-05-12",
  "task": "1a",
  "per_task": {
    "Noise": {
      "accuracy": 0.75,
      "f1_macro": 0.72,
      "f2_macro": 0.71,
      "precision_macro": 0.70,
      "recall_macro": 0.75,
      "aggregate": 0.7260
    },
    ...
  },
  "global": {
    "accuracy": 0.745,
    "f1_macro": 0.718,
    "f2_macro": 0.715,
    "precision_macro": 0.702,
    "recall_macro": 0.745,
    "aggregate": 0.7252
  },
  "ordinal": {
    "mean_mae": 0.35,
    "mean_off_by_1_accuracy": 0.92
  }
}
```

### Step 3.3 – Generate Plots

**Objective**: Visualize per-task performance.

**Command**:
```bash
python src/plot_results.py \
  --metrics results/runs/RUN_0001/metrics.json \
  --output_dir results/runs/RUN_0001/plots
```

**Output Files**:
- `results/runs/RUN_0001/plots/per_task_comparison.png` – Bar chart of per-task aggregate scores
- `results/runs/RUN_0001/plots/metric_heatmap.png` – Heatmap of (tasks × metrics)

---

## Phase 4: Consolidation

### Step 4.1 – Capture Environment Snapshot

**Command**:
```bash
python -c "
import torch
import sys
import os

print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
" > results/runs/RUN_0001/environment.txt

conda list > results/runs/RUN_0001/environment_conda.txt
```

### Step 4.2 – Create Run Summary

**Command**:
```bash
python -c "
import json
from datetime import datetime

summary = {
    'run_id': '0001',
    'date': '2026-05-12',
    'timestamp': datetime.now().isoformat(),
    'phase': 'completed',
    'tasks': ['Task 1a'],
    'status': 'Ready for analysis'
}

with open('results/runs/RUN_0001/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
"
```

### Step 4.3 – Update Global Index

**Command**:
```bash
python src/update_runs_index.py \
  --run_id 0001 \
  --metrics_file results/runs/RUN_0001/metrics.json \
  --index_file results/RUNS_INDEX.md
```

### Step 4.4 – Final Verification

**Objective**: Ensure all required files are present.

**Checklist**:
```bash
# Required files
[ ] results/runs/RUN_0001/AGENTS.md
[ ] results/runs/RUN_0001/implementation_plan.md
[ ] results/runs/RUN_0001/metrics.json
[ ] results/runs/RUN_0001/config_snapshot.yaml
[ ] results/runs/RUN_0001/notes.md
[ ] outputs/checkpoints/RUN_0001/{TASK}_best.pt × 7
[ ] outputs/logs/RUN_0001/task1a_*.log × 7
[ ] results/runs/RUN_0001/predictions_val.csv
[ ] results/runs/RUN_0001/environment.txt

# Optional
[ ] results/runs/RUN_0001/plots/
[ ] results/runs/RUN_0001/probabilities_val.pkl
```

**Command**:
```bash
ls -lh results/runs/RUN_0001/
ls -lh outputs/checkpoints/RUN_0001/
echo "✓ All required files present"
```

---

## Success Criteria

### Training Success
- [ ] All 7 models train without errors
- [ ] Loss curves show descent (not divergence)
- [ ] Early stopping triggers between epoch 60-70
- [ ] Validation metrics improve steadily

### Evaluation Success
- [ ] Predictions generated for all validation samples
- [ ] All metrics computed (no NaN or Inf values)
- [ ] Global aggregate score ≥ 0.60 (baseline threshold)
- [ ] Per-task scores within expected range (0.50 – 0.85)

### Reproducibility Success
- [ ] All artifacts saved with correct names
- [ ] Split fixed and documented
- [ ] Environment captured
- [ ] All hyperparameters logged

---

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size (8 → 4) in config

### Issue: Data loading fails
**Solution**: Check BIDS paths in CSV; validate with `src/validate_data.py`

### Issue: Metrics NaN
**Solution**: Check for empty prediction arrays; verify label distribution

### Issue: Training diverges
**Solution**: Reduce learning rate (1e-4 → 1e-5); check loss implementation

---

## Expected Runtime

| Phase | Duration | Notes |
|-------|----------|-------|
| Data prep | 1h | Sequential, mostly I/O |
| Training | 4–5h | 7 tasks × 30–40 min each, parallel-friendly |
| Evaluation | 1h | Inference + metrics + plots |
| **Total** | **~6–7h** | Can be parallelized to ~3h |

---

## Manual Intervention Points

1. **After step 2.1**: Review config manually; confirm values match AGENTS.md
2. **After each task training (step 2.2)**: Check log for training issues
3. **After step 3.2**: Inspect metrics table; verify global score reasonable
4. **After step 4.4**: Confirm all files present before marking run "complete"

---

## Next Steps After Completion

1. Update `AGENTS.md` results section with final scores
2. Record decision (Accepted as baseline) in `AGENTS.md`
3. Create PR/commit with `results/runs/RUN_0001/`
4. Begin RUN_0002 (incremental improvements)
