#!/usr/bin/env bash
# sbatch with this cluster's partition/GPU flags (scripts/cluster.sh) prepended.
# Usage (repo root):  scripts/submit.sh gpu|cpu [sbatch options...] <script.sbatch> [script args...]
# Options given here come after the cluster flags, so they win (e.g. --partition=dev).
# Prints the job ID only (--parsable), so calls can be chained:
#   prep=$(scripts/submit.sh cpu --time=02:00:00 scripts/slurm/prepare_data.sbatch)
#   scripts/submit.sh gpu --dependency=afterok:$prep --array=0-11 ... scripts/slurm/run_tasks.sbatch
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/cluster.sh"
kind=${1:?usage: submit.sh gpu|cpu [sbatch options...] script.sbatch [args...]}
shift
case $kind in
    gpu) flags=("${GPU_FLAGS[@]}") ;;
    cpu) flags=("${CPU_FLAGS[@]}") ;;
    *) echo "first argument must be gpu or cpu, got $kind" >&2; exit 1 ;;
esac
exec sbatch --parsable "${flags[@]}" "$@"
