#!/bin/bash
# Add the size-matched random control (20 random sets of K units) to existing ablation results.
#   scripts/submit.sh cpu --array=0-14 scripts/slurm/ablate_random_sets.sbatch   (K defaults to 5)
# Run list: scripts/tasks/ablate_runs.txt. Skips runs that already have the size-K control.
#SBATCH --job-name=ablate_rsets
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/activate.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
K=${K:-5}
RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" scripts/tasks/ablate_runs.txt)
OUT=runs/analysis/ablation_synth
RES="$OUT/$(basename "$RUN").json"
echo "[$(date)] $(hostname): $RUN K=$K"
if python -c "import json,sys; sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])).get('random_sets', {}) else 1)" "$RES" "$K"; then
    echo "already has the size-$K control"; exit 0
fi
python -m amlinterp.analysis.ablation --run "$RUN" --out "$OUT" --add-random-k "$K"
echo "[$(date)] finished"
