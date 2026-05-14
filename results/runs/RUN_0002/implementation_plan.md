# RUN_0002 — Implementation Plan

## Objective

Replace 7 independent models with one multi-head model and improve loss with EMD+Focal.

## Steps

1. **Data preparation**
   - Reuse existing split `results/splits/task1a_fixed.pkl` from RUN_0001.

2. **Training (single run, all 7 tasks simultaneously)**
   - Run `python train_task1a_multilabel.py --config configs/run_0002_upf.yaml`
   - Best checkpoint: `outputs/checkpoints/RUN_0002/multilabel_best.pt`

3. **Evaluation**
   - Run `python evaluate_task1a_multilabel.py --config configs/run_0002_upf.yaml`
   - Produces `results/runs/RUN_0002/predictions_val.csv`

4. **Metrics**
   - Run `python compute_metrics.py --predictions ... --output results/runs/RUN_0002/metrics.json`

## Launcher

```bash
bash scripts/run_0002.sh
# or on Jean Zay:
sbatch src/slurm/lisa_jeanzay.slurm --run RUN_0002
```
