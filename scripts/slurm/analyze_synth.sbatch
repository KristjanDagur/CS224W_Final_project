#!/bin/bash
# Synthetic dissection analysis, batch (no interactive session). Submit from the repo root:
#   scripts/submit.sh cpu scripts/slurm/analyze_synth.sbatch
# Stages (each independent of the later ones, so early figures survive a later failure):
#   1. heatmaps + per-layer figures from the finished dissection (runs/analysis/dissection_synth_test.json)
#   2. val + test activations for the 12 synthetic runs (analysis.hooks; skipped if already present)
#   3. cross-check: re-dissect one run's regenerated test activations, compare with the stored dissection
#   4. PR curves: unit selected on val, evaluated on test
#SBATCH --job-name=analyze_synth
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/activate.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
RES=runs/analysis/dissection_synth_test.json
FIG=runs/analysis/figures
runs=(runs/synth_R?_L6_h64_s?)
echo "[$(date)] $(hostname): ${#runs[@]} runs"

echo "== 1. coverage + layer figures from $RES"
python -m amlinterp.analysis.plots --results "$RES" --out "$FIG" --layers-rung R0 R1 R2 R3

echo "== 2. activations (val, test)"
for split in val test; do
    todo=()
    for r in "${runs[@]}"; do [ -f "$r/activations_$split.pt" ] || todo+=("$r"); done
    [ ${#todo[@]} -eq 0 ] || python -m amlinterp.analysis.hooks --run "${todo[@]}" --split "$split"
done

echo "== 3. cross-check against the stored dissection (synth_R3_L6_h64_s1, test)"
python -m amlinterp.analysis.dissection --acts runs/synth_R3_L6_h64_s1/activations_test.pt \
    --out runs/analysis/crosscheck_R3_s1 --n-boot 0 --n-perm 0 > /dev/null
python - <<'PY'
import json
new = json.load(open("runs/analysis/crosscheck_R3_s1.json"))[0]["concepts"]
old = json.load(open("runs/analysis/dissection_synth_test/synth_R3_L6_h64_s1.json"))[0]["concepts"]
diff = max(abs(new[c][m]["value"] - old[c][m]["value"]) for c in old for m in ("iou", "ap", "auroc"))
same_unit = sum(new[c]["auroc"]["unit"] == old[c]["auroc"]["unit"] for c in old)
print(f"max |coverage difference| = {diff:.2e}; same best AUROC unit for {same_unit}/{len(old)} concepts")
PY

echo "== 4. PR curves (select on val, evaluate on test)"
python -m amlinterp.analysis.plots --results "$RES" --out "$FIG" --layers-rung R0 R1 R2 R3 \
    --acts-select 'runs/synth_R?_L6_h64_s?/activations_val.pt' \
    --acts-eval 'runs/synth_R?_L6_h64_s?/activations_test.pt'
echo "[$(date)] finished; figures and CSVs in $FIG"
