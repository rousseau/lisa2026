# RUN_0001 Baseline Execution Summary

## Status: ✓ LAUNCHED AND RUNNING

### Launch Timeline
- **Launch Time**: 2026-05-12 09:12:32 CEST
- **Current Time**: 2026-05-12 09:14:32+ CEST
- **Terminal ID**: 2acb49d6-46e3-4ebb-9c68-d487053c7d6e (async, timeout 6 hours)

### Pre-Training Completed ✓
- **[1] Environment**: Conda env `lisa2026` activated
- **[2] Directories**: Created all output paths
- **[3] Split Generation**: Fixed patient-level split created
  - Train: 436 samples
  - Val: 96 samples  
  - Subjects: 244
  - File: `results/splits/task1a_fixed.pkl`

### Training Progress (Real-time)
**Task 1/7: Noise** - IN PROGRESS
- **Epoch 1**: ✓ COMPLETE
  - Duration: ~2 minutes
  - Train Loss: 0.5122
  - Val Loss: 0.3994
  - Val F1: 0.4493
  - Checkpoint: Saved
- **Epoch 2**: Currently at 75% (41/55 batches)
- **Time per batch**: ~1.49-1.56 seconds
- **Est. time per epoch**: ~1:25 (training batch time only)
- **Est. total time for Noise**: ~100 minutes

**Tasks 2-7**: Queued (Zipper, Positioning, Banding, Motion, Contrast, Distortion)

### Pipeline Architecture
```
├─ Phase 1: Data Preparation ✓
│  └─ Fixed split generation (StratifiedGroupKFold, n=5)
│
├─ Phase 2: Training IN PROGRESS →
│  ├─ Task 1: Noise (7 tasks total)
│  │  └─ 70 epochs max + early stopping (patience=10, monitor=val_f1)
│  ├─ Tasks 2-7: Sequential
│  └─ Est. duration: 7 tasks × 100 min ≈ 700 min = 11.7 hours
│
├─ Phase 3: Evaluation (after all training)
│  ├─ Inference on val set (96 samples × 7 models)
│  └─ Metrics computation (Accuracy, F1, F2, Precision, Recall)
│
└─ Phase 4: Results Consolidation
   ├─ Update AGENTS.md with results
   ├─ Update RUNS_INDEX.md
   └─ Generate final report
```

### Monitoring Commands
```bash
# Watch current task logs (real-time)
tail -f /home/rousseau/Exp/lisa2026/outputs/logs/RUN_0001/task1a_Noise.log

# Check completed checkpoints
ls -1 /home/rousseau/Exp/lisa2026/outputs/checkpoints/RUN_0001/*_best.pt

# Check overall progress
ps aux | grep run_training_full

# Last line of active log (update as tasks complete)
for task in Noise Zipper Positioning Banding Motion Contrast Distortion; do
  echo "=== $task ===" 
  tail -3 /home/rousseau/Exp/lisa2026/outputs/logs/RUN_0001/task1a_${task}.log
done
```

### Expected Timeline
| Phase | Time | Status |
|-------|------|--------|
| Data Prep | 0.5 min | ✓ Complete |
| Training (7 tasks) | ~11-12 hrs | ◐ In Progress |
| Evaluation | ~3 min | ⏳ Pending |
| Consolidation | ~5 min | ⏳ Pending |
| **Total** | **~11-12 hrs** | **◐ In Progress** |

### Key Outputs (To be Generated)
- ✓ `results/splits/task1a_fixed.pkl` - Fixed split metadata
- ◐ `outputs/checkpoints/RUN_0001/*.pt` - Trained model checkpoints (7 total)
- ◐ `outputs/logs/RUN_0001/*.log` - Training logs (7 total)
- ⏳ `results/runs/RUN_0001/predictions_val.csv` - Model predictions on val set
- ⏳ `results/runs/RUN_0001/metrics.json` - Final performance metrics
- ⏳ `results/runs/RUN_0001/config_snapshot.yaml` - Runtime configuration snapshot

### Troubleshooting
If training stalls:
```bash
# Check GPU status
nvidia-smi

# Check memory
ps aux | grep python | grep train_task1a

# Kill and restart (if needed)
kill -9 <PID>
bash /home/rousseau/Exp/lisa2026/run_training_full.sh
```

### Next Actions
1. **Wait for completion** (~11-12 hours)
2. **Check final results**: Review `results/runs/RUN_0001/metrics.json`
3. **Update documentation**: Fill in AGENTS.md results section
4. **Validate baseline**: Check if scores meet expected threshold (target > 0.60)
5. **Archive results**: Copy to results/RUNS_INDEX.md

---

**Generated**: 2026-05-12 09:14:32 CEST  
**Status**: ✓ OPERATIONAL - Training running normally  
**Estimated Completion**: ~20:30 CEST (same day)
