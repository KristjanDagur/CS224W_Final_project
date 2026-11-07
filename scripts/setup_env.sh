#!/usr/bin/env bash
# Create the project virtualenv in .venv/ (FarmShare: uses the python/3.12.3 module).
# Usage: bash scripts/setup_env.sh        then: source scripts/activate.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v module >/dev/null 2>&1; then module load python/3.12.3; fi
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch, torch_geometric; print('torch', torch.__version__, '| pyg', torch_geometric.__version__, '| cuda available:', torch.cuda.is_available())"
