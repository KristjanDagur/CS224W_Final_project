#!/bin/bash
# Neuron dissection of the synthetic ladder: activations (analysis.hooks), then one dissection per run
# in parallel (one CPU each), then the merged rung x concept table.
#   FarmShare: scripts/submit.sh cpu scripts/slurm/dissect_synth.sbatch
#   Sherlock (normal queue was ~24 h on 2026-11-08; dev rejects sbatch, so run it interactively):
#     srun -p dev -c 4 --mem=16G -t 02:00:00 bash scripts/slurm/dissect_synth.sbatch
#   (an interactive srun dies with the shell that started it; rerunning resumes, see below)
# Env: SPLIT (default test), OUT (default runs/analysis/dissection_synth_<split>), N_BOOT, N_PERM,
#      RUNS (glob of run dirs, default 'runs/synth_R?_L6_h64_s?'; e.g. 'runs/synth_R3_L7_h64_s?').
# About 20-30 CPU-minutes per run with 100 bootstrap + 20 permutation resamples (IoU, AP, AUROC).
#SBATCH --job-name=dissect_synth
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/activate.sh
SPLIT=${SPLIT:-test}
OUT=${OUT:-runs/analysis/dissection_synth_$SPLIT}
NPROC=${SLURM_CPUS_PER_TASK:-4}
echo "[$(date)] $(hostname) split=$SPLIT out=$OUT procs=$NPROC"

runs=(${RUNS:-runs/synth_R?_L6_h64_s?})
missing=()
for r in "${runs[@]}"; do [ -f "$r/activations_$SPLIT.pt" ] || missing+=("$r"); done
if [ ${#missing[@]} -gt 0 ]; then
    OMP_NUM_THREADS=$NPROC python -m amlinterp.analysis.hooks --run "${missing[@]}" --split "$SPLIT"
fi

# Runs whose result file exists are skipped, so a killed job (time limit, closed interactive session)
# resumes where it stopped. Delete $OUT/ to recompute everything.
mkdir -p "$OUT"
todo=()
for r in "${runs[@]}"; do [ -f "$OUT/$(basename "$r").json" ] || todo+=("$r"); done
echo "${#todo[@]} of ${#runs[@]} runs to dissect"
[ ${#todo[@]} -eq 0 ] || printf '%s\n' "${todo[@]}" | xargs -P "$NPROC" -I{} bash -c '
    source scripts/activate.sh
    run=$(basename {})
    OMP_NUM_THREADS=1 python -m amlinterp.analysis.dissection --acts {}/activations_'"$SPLIT"'.pt \
        --out '"$OUT"'/$run --n-boot '"${N_BOOT:-100}"' --n-perm '"${N_PERM:-20}"' > '"$OUT"'/$run.log 2>&1 \
        && echo "[$(date +%T)] done $run" || { echo "FAILED $run (see '"$OUT"'/$run.log)"; exit 1; }'

python -m amlinterp.analysis.dissection --combine "$OUT"/synth_*.json --out "$OUT"
echo "[$(date)] finished"
