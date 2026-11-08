#!/usr/bin/env bash
# Create the project virtualenv in .venv/ from the system python3 (>= 3.10).
# Usage: bash scripts/setup_env.sh        then: source scripts/activate.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# Uses the system python3 (3.12 on FarmShare Ubuntu 24.04); no module needed.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch, torch_geometric; print('torch', torch.__version__, '| pyg', torch_geometric.__version__, '| cuda available:', torch.cuda.is_available())"
