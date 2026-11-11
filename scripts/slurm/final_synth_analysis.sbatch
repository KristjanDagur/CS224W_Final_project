#!/bin/bash
# Final synthetic analysis over seeds 1-10 (pre-registered 2026-11-09, see docs/experiment_log.md).
# Run after the seed 4-10 training, ablation and dissection jobs:
#   scripts/submit.sh cpu --dependency=afterok:<ablation>:<dissect L6>:<dissect L7> scripts/slurm/final_synth_analysis.sbatch
# Outputs: runs/analysis/ablation_synth_stats_10seeds.{csv,md}, ablation_synth_10seeds_summary.csv,
#          figures_10seeds/, depth_R3_10seeds.*
#SBATCH --job-name=final_synth
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/activate.sh
A=runs/analysis
python - <<'PY'
import collections, glob, json
groups = collections.Counter()
for f in glob.glob("runs/analysis/ablation_synth/*.json"):
    r = json.load(open(f))
    assert "5" in r.get("random_sets", {}), f"{f} lacks the size-5 random control"
    groups[(r["rung"], r["num_layers"])] += 1
print("ablation results per group:", dict(groups))
assert len(groups) == 5 and all(n == 10 for n in groups.values()), "expected 10 seeds in each of 5 groups"
PY
python -m amlinterp.analysis.ablation_stats --results $A/ablation_synth/*.json --out $A/ablation_synth_stats_10seeds
python -m amlinterp.analysis.ablation --summarize $A/ablation_synth/*.json --out $A/ablation_synth_10seeds
python -m amlinterp.analysis.plots --results $A/dissection_synth_test_10seeds.json --out $A/figures_10seeds \
    --layers-rung R0 R1 R2 R3 \
    --acts-select 'runs/synth_R?_L6_h64_s*/activations_val.pt' --acts-eval 'runs/synth_R?_L6_h64_s*/activations_test.pt'
python -m amlinterp.analysis.baselines --graph data/processed/synthetic/test.pt \
    --dissection $A/dissection_synth_test_10seeds.json --out $A/figures_10seeds/degree_baseline.csv
python -m amlinterp.analysis.depth --rung R3 --out $A/depth_R3_10seeds \
    --dissection L6=$A/dissection_synth_test_10seeds.json L7=$A/dissection_synth_L7_test_10seeds.json
echo "[$(date)] finished"
