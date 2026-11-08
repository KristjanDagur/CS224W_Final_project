# Source this file to activate the project environment: `source scripts/activate.sh`
# (every session, and inside every Slurm job). Cluster settings come from scripts/cluster.sh.
#
# FarmShare: activates .venv/, which uses the system /usr/bin/python3.12 (Ubuntu 24.04 on all
# nodes), so no `module load` is needed; some compute nodes lack the python modules.
# Sherlock: defines `python` and `pytest` shell functions that run the venv's interpreter inside
# the Apptainer image (GPUs are passed through with --nv when the node has them).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cluster.sh"

if [ -z "$SIF" ]; then
    source "$VENV/bin/activate"
else
    python() {
        local nv=()
        command -v nvidia-smi >/dev/null 2>&1 && nv=(--nv)
        apptainer exec "${nv[@]}" "$SIF" "$VENV/bin/python" "$@"
    }
    pytest() { python -m pytest "$@"; }
fi
