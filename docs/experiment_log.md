# Experiment log

Append-only. One line per run or result: date | run ID / item | what changed | headline numbers | artifacts.

| Date | Run / item | What changed | Headline | Artifacts |
|---|---|---|---|---|
| 2026-11-07 | env | venv: Python 3.12.3 (FarmShare module), torch 2.13.0+cu126, PyG 2.8.1, pyg_lib 0.9.0 | imports + LinkNeighborLoader OK on CPU; GPU check job 1952070 queued | `requirements.txt`, `scripts/setup_env.sh` |
| 2026-11-07 | synthetic data v1 | circulant generator n=8192, d=6, r=11.1; split seeds train/val/test = 0/1/2; self-loops kept | C4–C6, S-G, B-C 2–5 pts below Egressy Table 3; superseded | n/a |
| 2026-11-07 | synthetic data v2 | self-loops re-drawn (m = 24,576 non-loop edges) | positive rates match Table 3 to within ~1 sd (train: deg-in .348, C2 .200, C6 .784, S-G .322, B-C .322) | `data/processed/synthetic/{train,val,test}.pt`, `stats.json` |
