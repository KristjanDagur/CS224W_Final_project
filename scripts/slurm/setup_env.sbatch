#!/bin/bash
# Build the environment (scripts/setup_env.sh) on a compute node, then run the test suite.
#   scripts/submit.sh cpu --partition=dev scripts/slurm/setup_env.sbatch     # Sherlock: dev allows 4 CPUs/16 GB
#   scripts/submit.sh cpu scripts/slurm/setup_env.sbatch                     # FarmShare
#SBATCH --job-name=setup_env
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash scripts/setup_env.sh
source scripts/activate.sh
python -m pip freeze --exclude-editable
pytest -q
