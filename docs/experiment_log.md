# Experiment log

Append-only. One line per run or result: date | run ID / item | what changed | headline numbers | artifacts.

| Date | Run / item | What changed | Headline | Artifacts |
|---|---|---|---|---|
| 2026-11-07 | env | venv: Python 3.12.3 (FarmShare module), torch 2.13.0+cu126, PyG 2.8.1, pyg_lib 0.9.0 | imports + LinkNeighborLoader OK on CPU; GPU check job 1952070 queued | `requirements.txt`, `scripts/setup_env.sh` |
| 2026-11-07 | synthetic data v1 | circulant generator n=8192, d=6, r=11.1; split seeds train/val/test = 0/1/2; self-loops kept | C4–C6, S-G, B-C 2–5 pts below Egressy Table 3; superseded | n/a |
| 2026-11-07 | synthetic data v2 | self-loops re-drawn (m = 24,576 non-loop edges) | positive rates match Table 3 to within ~1 sd (train: deg-in .348, C2 .200, C6 .784, S-G .322, B-C .322) | `data/processed/synthetic/{train,val,test}.pt`, `stats.json` |
| 2026-11-07 | synth_R{0,3}_L6_h64_s0_smoke | smoke test, 2 epochs on CPU (login node, 4 threads); lr 1e-3, batch 256, ego_mode exact (disjoint full 6-hop subgraphs) | R0 val mean F1 38.6 (deg-in 98, deg-out 0, as theory predicts); R3 70.0 after 2 epochs; ~9 s/epoch R0, ~28 s/epoch R3 | `runs/smoke/` (not real runs) |
| 2026-11-07 | lr sweep (jobs 1952376, 1952411) | first two submissions failed: `module load python` missing on compute nodes + barley-04 lacks /home; then YAML read `lr=3e-4` as a string | no results; fixed `scripts/activate.sh` and `train.py` override parsing | `logs/` |
| 2026-11-07 | lr sweep (job 1952452) | synthetic, seed 0, lr ∈ {3e-4, 1e-3, 3e-3} × R0–R3, 100 epochs, else `configs/synthetic.yaml`; objective = val mean F1 averaged over rungs, parabola in log10(lr) (`scripts/fit_lr.py`) | running | `runs/lr_sweep/` |
