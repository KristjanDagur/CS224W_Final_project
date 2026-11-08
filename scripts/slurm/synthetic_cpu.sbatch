#!/bin/bash
# Synthetic ladder sweep on CPU nodes (normal partition; avoids the GPU queue).
# Array index -> (rung, seed): 0-11 = R0..R3 x seeds 1..3.
# Submit from the repo root:  sbatch scripts/slurm/synthetic_cpu.sbatch
# Resubmitting resumes unfinished runs from runs/<run_id>/last.pt.
#SBATCH --job-name=synth_ladder
#SBATCH --partition=normal
#SBATCH --array=0-11
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
RUNGS=(R0 R1 R2 R3)
RUNG=${RUNGS[$((SLURM_ARRAY_TASK_ID % 4))]}
SEED=$((SLURM_ARRAY_TASK_ID / 4 + 1))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -m amlinterp.train --config configs/synthetic.yaml --rung "$RUNG" --seed "$SEED" "$@"
