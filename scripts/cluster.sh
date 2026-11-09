# Cluster-specific settings, in one place. Sourced by scripts/activate.sh, scripts/setup_env.sh
# and scripts/submit.sh; job scripts themselves contain no partition/GPU/node flags.
#
# FarmShare is the primary cluster (default). Sherlock is the fallback; it is detected through
# $SHERLOCK, which Sherlock sets on every node. Override with AMLINTERP_CLUSTER=farmshare|sherlock.
#
#   farmshare  venv in .venv/ built from the system python3.12 (Ubuntu 24.04); data/, runs/
#              and logs/ are plain directories in the repo (gitignored).
#   sherlock   the host is CentOS 7 (glibc 2.17), which cannot run the pinned torch 2.13 wheels
#              (manylinux_2_28), and has no PyG module. Python therefore runs inside an Apptainer
#              image (python:3.12-bookworm) with the same pinned venv. Image and venv live in
#              $HOME/envs/amlinterp (6.7 GB; environments may not go on the purged $SCRATCH, and
#              the user does not want $GROUP_HOME/$GROUP_SCRATCH used). data/, runs/ and logs/ are
#              symlinks into $SCRATCH. Override the env location with AMLINTERP_ENV_ROOT.
#
# Exposes: CLUSTER, REPO, VENV, SIF (empty = no container), SIF_SOURCE, STORAGE_ROOT
# (empty = keep data/runs/logs in the repo), and the sbatch flag arrays GPU_FLAGS and CPU_FLAGS.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -n "${AMLINTERP_CLUSTER:-}" ]; then
    CLUSTER=$AMLINTERP_CLUSTER
elif [ -n "${SHERLOCK:-}" ]; then
    CLUSTER=sherlock
else
    CLUSTER=farmshare
fi

case $CLUSTER in
    farmshare)
        VENV=$REPO/.venv
        SIF=""
        SIF_SOURCE=""
        STORAGE_ROOT=""
        # L40S nodes oat-01..06; barley-04 could not see /home/users (2026-11-07).
        GPU_FLAGS=(--partition=gpu --qos=gpu --gpus=1)
        CPU_FLAGS=(--partition=normal --exclude=barley-04)
        ;;
    sherlock)
        ENV_ROOT=${AMLINTERP_ENV_ROOT:-$HOME/envs/amlinterp}  # personal env; user decision: no group folders
        VENV=$ENV_ROOT/venv
        SIF=$ENV_ROOT/py312-bookworm.sif
        SIF_SOURCE=docker://python:3.12-bookworm
        STORAGE_ROOT=$SCRATCH/CS224W
        # Public gpu partition; restrict to the Ampere/Ada/Hopper cards (FarmShare runs used L40S).
        GPU_FLAGS=(--partition=gpu -G 1 -C "GPU_SKU:L40S|GPU_SKU:H100_SXM5|GPU_SKU:RTX_3090")
        CPU_FLAGS=(--partition=normal)
        export PIP_CACHE_DIR=$SCRATCH/.cache/pip APPTAINER_CACHEDIR=$SCRATCH/.cache/apptainer
        ;;
    *)
        echo "unknown AMLINTERP_CLUSTER=$CLUSTER (expected farmshare or sherlock)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
