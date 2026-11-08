#!/bin/bash
# Generic job-array runner: array task i runs line i+1 of a task file as `python -m amlinterp.train <line>`.
# Partition and resources are given on the sbatch command line, e.g. for one L40S:
#   TASKS=scripts/tasks/synth_final.txt LR_FROM=runs/lr_sweep/selected_lr.json \
#     sbatch --array=0-11 --partition=gpu --qos=gpu --gpus=1 --cpus-per-task=8 --mem=16G \
#            --time=02:00:00 scripts/slurm/run_tasks.sbatch
# If LR_FROM is set, every "@LR" in the line is replaced by the "lr" field of that JSON file,
# read when the task starts (so the file can be produced by an upstream dependency job).
# Runs resume from runs/<run_id>/last.pt, so resubmitting a killed task continues it.
#SBATCH --job-name=ladder
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS:?set TASKS to a task file}")
[ -n "$LINE" ] || { echo "no line $((SLURM_ARRAY_TASK_ID + 1)) in $TASKS"; exit 1; }
# REQUIRE_VERTEX=1: skip unless LR_FROM is an lr_fit.json whose proposal is a new (vertex) lr.
if [ "${REQUIRE_VERTEX:-0}" = 1 ] && [ "$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['kind'])" "$LR_FROM")" != vertex ]; then
    echo "proposal is an already-tested lr; nothing to confirm"; exit 0
fi
if [ -n "${LR_FROM:-}" ]; then
    LR=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['lr'])" "$LR_FROM")
    LINE=${LINE//@LR/$LR}
fi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[$(date)] $(hostname) ${CUDA_VISIBLE_DEVICES:+GPU $CUDA_VISIBLE_DEVICES}: $LINE"
# shellcheck disable=SC2086
python -m amlinterp.train $LINE
