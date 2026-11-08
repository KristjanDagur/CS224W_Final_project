# Source this file to activate the project environment: `source scripts/activate.sh`
# The venv uses the system /usr/bin/python3.12 (Ubuntu 24.04 on all FarmShare nodes), so no
# `module load` is needed. Some compute nodes lack the python modules.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin/activate"
