#!/bin/bash
# Quick sanity check that the venv sees the GPU on a FarmShare gpu node.
#SBATCH --job-name=gpu_check
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
nvidia-smi
python - <<'PY'
import torch, torch_geometric
from torch_geometric.nn import GINEConv
print("torch", torch.__version__, "pyg", torch_geometric.__version__)
print("cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
x = torch.randn(1000, 64, device="cuda"); ei = torch.randint(0, 1000, (2, 5000), device="cuda")
ea = torch.randn(5000, 64, device="cuda")
conv = GINEConv(torch.nn.Linear(64, 64), edge_dim=64).cuda()
print("GINEConv forward OK:", tuple(conv(x, ei, ea).shape))
PY
