#!/bin/bash
# Depth experiment analysis: dissect the 7-layer R3 runs and compare them with the 6-layer ones.
# Submit after training (e.g. --dependency=afterany:<train job>):
#   scripts/submit.sh cpu --dependency=afterany:<id> scripts/slurm/analyze_depth.sbatch
# Writes runs/analysis/dissection_synth_L7_test{.json,_summary.csv}, runs/analysis/figures_L7/
# and runs/analysis/depth_R3{.md,_first_layer.csv,_f1.csv}.
#SBATCH --job-name=analyze_depth
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/activate.sh
L7_RUNS='runs/synth_R3_L7_h64_s?'
done_runs=$(ls $L7_RUNS/metrics.json 2>/dev/null | wc -l)
echo "[$(date)] $(hostname): $done_runs of 3 L7 runs finished"
[ "$done_runs" -eq 3 ] || { echo "not all L7 runs finished; resubmit training first"; exit 1; }

RUNS="$L7_RUNS" OUT=runs/analysis/dissection_synth_L7_test bash scripts/slurm/dissect_synth.sbatch
python -m amlinterp.analysis.plots --results runs/analysis/dissection_synth_L7_test.json \
    --out runs/analysis/figures_L7 --layers-rung R3
python -m amlinterp.analysis.depth --rung R3 --out runs/analysis/depth_R3 \
    --dissection L6=runs/analysis/dissection_synth_test.json L7=runs/analysis/dissection_synth_L7_test.json
echo "[$(date)] finished"
