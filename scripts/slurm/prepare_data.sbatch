#!/bin/bash
# Build the cached datasets once: synthetic circulant splits, then AMLworld HI-Small
# (graph.pt, patterns.pt, stats.json). RAW is the directory holding HI-Small_Trans.csv and
# HI-Small_Patterns.txt (default data/raw).
#   RAW=data/raw scripts/submit.sh cpu scripts/slurm/prepare_data.sbatch
#SBATCH --job-name=prepare_data
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
echo "[$(date)] $(hostname)"
python -m amlinterp.data.synthetic --out data/processed/synthetic
echo "[$(date)] synthetic done"
python -m amlinterp.data.aml --raw "${RAW:-data/raw}" --name HI-Small --out data/processed/aml_hi_small
echo "[$(date)] AMLworld done"
