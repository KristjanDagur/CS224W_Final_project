#!/usr/bin/env bash
# Build the project environment from requirements.txt (pinned), then install amlinterp itself.
# Usage: bash scripts/setup_env.sh        then: source scripts/activate.sh
# On Sherlock, run it inside a job (scripts/slurm/setup_env.sbatch): it pulls the container
# image and installs about 3 GB of wheels. Cluster settings come from scripts/cluster.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster.sh
echo "cluster: $CLUSTER  venv: $VENV  image: ${SIF:-none}"

if [ -z "$SIF" ]; then
    python3 -m venv "$VENV"
    PY=("$VENV/bin/python")
else
    mkdir -p "$(dirname "$SIF")" "$PIP_CACHE_DIR" "$APPTAINER_CACHEDIR"
    if [ ! -f "$SIF" ]; then
        # On Sherlock, the default `apptainer pull` segfaulted in mksquashfs (exit 139, 2026-11-08);
        # building with node-local temp space and a bounded thread count worked.
        export APPTAINER_TMPDIR=${L_SCRATCH_JOB:-${TMPDIR:-/tmp}}
        apptainer build --mksquashfs-args "-processors ${SLURM_CPUS_PER_TASK:-4}" "$SIF" "$SIF_SOURCE"
    fi
    [ -x "$VENV/bin/python" ] || apptainer exec "$SIF" python3 -m venv "$VENV"
    nv=()
    command -v nvidia-smi >/dev/null 2>&1 && nv=(--nv)  # GPU passthrough, so the final check sees CUDA
    PY=(apptainer exec ${nv[@]+"${nv[@]}"} "$SIF" "$VENV/bin/python")
fi
"${PY[@]}" -m pip install --upgrade pip
"${PY[@]}" -m pip install -r requirements.txt
"${PY[@]}" -m pip install --no-deps -e .

# Large artifacts go to cluster storage, linked into the repo so configs can use data/ and runs/.
if [ -n "$STORAGE_ROOT" ]; then
    for d in data runs logs; do
        mkdir -p "$STORAGE_ROOT/$d"
        if [ -L "$d" ] || [ ! -e "$d" ]; then
            ln -sfn "$STORAGE_ROOT/$d" "$d"
        else
            echo "warning: $d/ exists and is not a symlink; leaving it in place" >&2
        fi
    done
fi

"${PY[@]}" -c "import sys, torch, torch_geometric; print('python', sys.version.split()[0], '| torch', torch.__version__, '| pyg', torch_geometric.__version__, '| cuda available:', torch.cuda.is_available())"
