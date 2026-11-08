#!/bin/bash
# Generic job-array runner: array task i runs line i+1 of a task file as `python -m amlinterp.train <line>`.
# Resources are given on the command line; scripts/submit.sh adds the cluster's GPU flags:
#   TASKS=scripts/tasks/synth_final.txt LR=0.003 \
#     scripts/submit.sh gpu --array=0-11 --cpus-per-task=8 --mem=16G --time=04:00:00 scripts/slurm/run_tasks.sbatch
# Every "@LR" in the line is replaced by $LR, or, if LR_FROM is set, by the "lr" field of that
# JSON file, read when the task starts (so the file can be produced by an upstream dependency job).
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
fi
if [[ $LINE == *@LR* ]]; then
    LINE=${LINE//@LR/${LR:?the task line uses @LR; set LR or LR_FROM}}
fi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[$(date)] $(hostname) ${CUDA_VISIBLE_DEVICES:+GPU $CUDA_VISIBLE_DEVICES}: $LINE"
# shellcheck disable=SC2086
python -m amlinterp.train $LINE
