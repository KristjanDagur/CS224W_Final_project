#!/bin/bash
# Learning-rate sweep for the synthetic ladder (tuning seed 0; final runs use seeds 1-3).
# Array index -> (lr, rung): rung = idx % 4, lr = LRS[idx / 4]. One lr is shared by all rungs.
# Usage (repo root):  LRS="3e-4 1e-3 3e-3" scripts/submit.sh cpu --array=0-11 scripts/slurm/synthetic_lr_sweep.sbatch
# Then:               python scripts/fit_lr.py
#SBATCH --job-name=synth_lr
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
read -r -a LR_LIST <<< "${LRS:?set LRS, e.g. LRS=\"3e-4 1e-3 3e-3\"}"
RUNGS=(R0 R1 R2 R3)
RUNG=${RUNGS[$((SLURM_ARRAY_TASK_ID % 4))]}
LR=${LR_LIST[$((SLURM_ARRAY_TASK_ID / 4))]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -m amlinterp.train --config configs/synthetic.yaml --rung "$RUNG" --seed 0 \
    --set lr="$LR" tag="lr$LR" runs_dir=runs/lr_sweep
