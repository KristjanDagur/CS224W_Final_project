#!/bin/bash
# Run one short command in the project env on a CPU node (used for pipeline glue steps).
#   scripts/submit.sh cpu --dependency=afterany:<jobid> scripts/slurm/run_cmd.sbatch python scripts/fit_lr.py
#SBATCH --job-name=glue
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
"$@"
