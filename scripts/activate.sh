# Source this file to activate the project environment: `source scripts/activate.sh`
if command -v module >/dev/null 2>&1; then module load python/3.12.3; fi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin/activate"
