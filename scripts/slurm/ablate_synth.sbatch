#!/bin/bash
# Mean-ablation (H2) of synthetic runs, one run per array task. For the FINAL report, not the milestone.
#   scripts/submit.sh cpu --array=0-14 scripts/slurm/ablate_synth.sbatch
# Run list: scripts/tasks/ablate_runs.txt (line i+1 = task i). Writes runs/analysis/ablation_synth/<run_id>.json.
# Extracts validation activations first if a run lacks them (unit selection uses val, never test).
#SBATCH --job-name=ablate_synth
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/activate.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" scripts/tasks/ablate_runs.txt)
OUT=runs/analysis/ablation_synth
echo "[$(date)] $(hostname): $RUN"
[ -f "$OUT/$(basename "$RUN").json" ] && { echo "already done"; exit 0; }
[ -f "$RUN/activations_val.pt" ] || python -m amlinterp.analysis.hooks --run "$RUN" --split val
python -m amlinterp.analysis.ablation --run "$RUN" --out "$OUT"
echo "[$(date)] finished"
