# AGENTS.md: instructions for coding agents

Read `README.md` first for the project goal, ladder and layout. `proposal.pdf` is the source of truth for
the research question and hypotheses. This file covers how to work in the repo and the technical facts that
are easy to get wrong.

## Core principle: this is a controlled experiment

The four rungs (R0–R3) must differ **only** in the adaptation being added:
- Same hyperparameters, data split, number of epochs, model-selection rule and seeds across rungs.
- If you must change something (e.g. the batch size to fit in GPU memory), change it for **every** rung and record it in `docs/experiment_log.md`.
- Never tune on the test split. Select checkpoints and decision thresholds on validation only.

## Technical facts (verified against IBM Multi-GNN code and Egressy et al., 2026-11-07)

1. **Ego IDs are batch-level in IBM's code.**
   - `AddEgoIds` (`train_util.py`) sets the ego bit on the endpoints of **all seed edges in the batch** (default `batch_size=8192`), not only the edge being scored.
   - Our code must support `ego_mode ∈ {batch, exact}`:
     - `batch` reproduces the benchmark.
     - `exact` marks only the target edge's endpoints, so each analysis edge needs its own subgraph or a small batch.
   - Activations of ego models depend on the batch. **Always record `ego_mode` and the batch protocol with any saved activations.**

2. **Port numbers are per neighbor, ordered by time.**
   - All edges between the same pair of accounts share one port number.
   - Ports are assigned in order of the first transaction timestamp with each counterparty.
   - Both the incoming and outgoing port go into the edge features.
   - For reverse edges, the last two edge-feature columns are swapped.
   - The IBM implementation is a slow Python loop over 5M edges. Compute it once, cache it, and vectorize if possible.

3. **Depth limits which cycles can be detected.**
   - With L GNN layers and the edge readout `MLP([h_u ‖ h_v ‖ e_uv])`, an ego model can only close cycles of length ≤ L+1 through the target edge. With L=2 that is length 2–3.
   - AML runs use **2 layers** (benchmark). Synthetic runs use **6 layers** (Egressy F.3), so 6-cycles are detectable.
   - Stratify any cycle analysis by cycle length.

4. **Synthetic graphs: use the random circulant generator. Do not plant patterns.**
   - Parameters: n=8192 nodes, average degree d=6, radius r=11.1. Each edge picks u uniformly and v = round(N(u, r)) mod n. Generate an independent graph for train, val and test.
   - m = ⌊nd/2⌋ = 24,576 edges, with self-loops **re-drawn**. Keeping them undershoots Egressy Table 3 by 2–5 points on C4–C6, S-G and B-C (verified 2026-11-07). Synthetic edges have no features; a random `edge_time` orders the ports.
   - Egressy F.3: synthetic runs use the full neighborhood (no sampling) and 6 layers.
   - Label every pattern by **enumerating** it in the graph.
   - Planting patterns creates degree shortcuts, which would make degree neurons look like pattern detectors.
   - Label definitions (Egressy App. D.2):
     - deg-in/out > 3 (counting multi-edges).
     - fan-in/out > 3 (counting unique neighbors).
     - Ck: the node lies on a directed k-cycle, k = 2..6.
     - Scatter-gather: sink node with at least 2 intermediate nodes on length-2 paths from one source.
     - Biclique: sink of a directed K2,2.
   - It is one multi-task node classification problem with 11 outputs.

5. **AMLworld facts.**
   - Node features are constant; all signal comes from structure and edge features (timestamp, amounts, currencies, payment format).
   - The chronological split builds three nested graphs: train ⊂ train+val ⊂ all. Evaluation uses only the indices belonging to each split.
   - The patterns file covers about 62% of laundering transactions (3,209 of 5,177, in 370 attempts). The remainder is "none", and models have near-zero recall on it.
   - Pattern types: FAN-IN, FAN-OUT, GATHER-SCATTER, SCATTER-GATHER, CYCLE, BIPARTITE, STACK, RANDOM. Verify the exact header format by reading the file; don't assume it.

6. **Concept masks are confounded with the label.** Every pattern member is a laundering edge, so a generic "laundering" unit scores some IoU with every pattern. Always:
   - include structural concepts computed for **all** edges/nodes regardless of label;
   - report selectivity against the laundering-label mask baseline;
   - use average precision as well as IoU, because concepts are rare;
   - compute bootstrap CIs at the **attempt** level, not the transaction level.

7. **IBM GINe architecture details to preserve.**
   - Linear node/edge embeddings come first.
   - Each layer is `x = (x + relu(BN(GINEConv(x, e)))) / 2`.
   - Reverse MP uses `to_hetero(..., aggr='mean')` over the `to` and `rev_to` edge types.
   - The decoder is `MLP(3h → 50 → 25 → 2)`.
   - `model_settings.json` hidden size is about 66; the paper reports 64. Pick one, use it for every rung, and log it.

8. **Ablation protocol.** Mean-ablate using means from the training split. Fix the decision threshold on validation *before* ablating. Always compare against random-unit ablations.

## Reference numbers for sanity checks

**Egressy Table 1:** synthetic, minority-class F1 (%), mean of 5 runs.

| Model | deg-in | deg-out | fan-in | fan-out | C2 | C3 | C4 | C5 | C6 | S-G | B-C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GIN | 99.77 | 43.58 | 95.57 | 35.91 | 34.67 | 58.00 | 50.80 | 43.12 | 48.59 | 69.31 | 63.12 |
| +ReverseMP | 98.87 | 99.08 | 94.99 | 95.25 | 35.96 | 63.85 | 69.09 | 67.44 | 71.23 | 65.83 | 66.18 |
| +Ports | 98.41 | 98.35 | 98.51 | 99.16 | 39.15 | 63.58 | 69.00 | 70.35 | 75.04 | 67.84 | 65.78 |
| +EgoIDs (Multi-GIN) | 99.48 | 99.09 | 99.62 | 99.32 | 98.97 | 98.73 | 97.46 | 91.60 | 84.23 | 97.42 | 94.33 |

**Egressy Table 2:** AML Small HI, minority-class F1 (%), 5 seeds.

| Model | F1 |
|---|---|
| GIN | 28.70 ± 1.13 |
| +ReverseMP | 46.79 ± 4.97 |
| +Ports | 56.85 ± 2.64 |
| +EgoIDs (Multi-GIN) | 57.15 ± 4.99 |
| Multi-GIN+EU | 64.79 ± 1.22 |

Our reproduction should land within roughly the reported standard deviations. If it doesn't, investigate before
building analyses on top.

## Working conventions

- **Run IDs:** `{dataset}_{rung}_L{layers}_h{hidden}_s{seed}[_{tag}]`, e.g. `amlhis_R3_L2_h64_s1`.
- **Every run writes** to `runs/<run_id>/`:
  - `config.json`, the fully resolved config including `ego_mode`, batch size and git commit hash;
  - the best checkpoint by validation F1;
  - metrics as JSON;
  - the decision threshold chosen on validation.
- **Seeds:** seed Python, NumPy and torch (including CUDA). Use at least 3 seeds per rung for any claim about whether a detector emerges.
- **Experiment log:** append one line per run or result to `docs/experiment_log.md`: date, run ID, what changed, headline metrics, and where the artifacts are. Never rewrite past entries.
- **Do not commit** anything under `data/` or `runs/`, raw CSVs, checkpoints, or activation dumps. Keep the repo small; the final Colab downloads checkpoints from external storage.
- **Do not add wandb** as a hard dependency (IBM's code requires it). Log locally; wandb may be optional behind a flag.
- **Attribution:** files adapted from IBM Multi-GNN keep a header comment naming the source file and the Apache-2.0 license.
- **Code quality matters for grading.** The Colab is graded on correctness, design and documentation. Write docstrings for public functions and keep modules focused. Code snippets also appear in the blog.
- **Report honestly.** If a run fails or a reproduction misses the reference numbers, record it in the log as-is.

## Cluster usage (FarmShare preferred, Sherlock fallback)

- **Never** train on login nodes. Use Slurm.
- FarmShare GPU jobs use `--partition=gpu --qos=gpu --gpus=1`; GPUs are not allocated by default.
- Sherlock's public `gpu` partition has P100 (16 GB) and P40 (24 GB) cards. Select one with `-C GPU_SKU:...`. Pascal GPUs may need an older CUDA build of PyTorch.
- FarmShare GPUs (checked 2026-11-07 with `sinfo`/`scontrol show node`/`sacctmgr show qos`):
  - `gpu` partition: 6 nodes (`oat-01`…`oat-06`), each 4× **NVIDIA L40S 48 GB** (Ada, CC 8.9), 64 CPUs, 250 GB RAM. 24 GPUs total.
  - QOS `gpu`: max 4 GPUs and 4 running jobs per user, 32 submitted jobs per user, 2-day wall limit.
  - `MaxMemPerCPU=4000` MB, so a 32 GB job must request 8 CPUs (`--cpus-per-task=8 --mem=32G`).
  - Heavily contended: on 2026-11-07, 22 of 24 GPUs were allocated and about 95 jobs were waiting on priority. Submit early; short `--time` requests backfill faster.
  - Any current PyTorch CUDA 12.x wheel supports CC 8.9.
- Scratch: `/scratch/users/$USER` (shared filesystem, 81% full on 2026-11-07). Home is `/home/users/$USER` (NFS).
- CPU jobs: the `normal` partition (barley/wheat nodes, 64+ CPUs each) has no GPU queue. Use it for synthetic runs.
- Gotchas found 2026-11-07:
  - Do **not** `module load python/...` in job scripts. It is missing on some compute nodes (barley-03), and the venv uses the system `/usr/bin/python3.12` anyway.
  - `barley-04` could not see `/home/users`, so jobs there die without writing a log. Submit with `--exclude=barley-04` until it is fixed.
- Request roughly 1 GPU, 8 CPUs, 32 GB RAM, and a time limit of 16 hours or less. Submit rung × seed as one job array.
- Keep data and runs on cluster scratch, not in home quota. Copy checkpoints and metrics back for analysis.
- Make training resumable: checkpoint every epoch, so a job killed at its time limit can continue.

## Project status / next steps

Milestone due 2026-11-10:

- [ ] Pinned env (`requirements.txt` + `scripts/setup_env.sh`): verified on FarmShare login node (CPU); GPU check pending; laptop not yet verified
- [ ] AMLworld download + preprocessing + cached ports; patterns parser + stats (counts per type per split, cycle-length distribution)
- [x] Synthetic circulant generator + labelers (`data/synthetic.py`, tests vs brute force; rates match Table 3)
- [x] Vectorized port numbering (`data/ports.py`, tested against IBM's loop)
- [x] GIN ladder model + `train.py` (resumable) + synthetic task; smoke-tested on CPU
- [ ] Synthetic training R0–R3 × 3 seeds (6 layers; `scripts/slurm/synthetic_cpu.sbatch`); reproduce Table 1 rows
- [ ] AML training R0–R3 (cluster); compare to Table 2
- [ ] Activation hooks with `ego_mode`; dissection (IoU + AP) on synthetic; first rung × concept heatmap
- [ ] Milestone report draft (problem, data and processing, model and metrics, preliminary results)

After the milestone: AML dissection, the exact-vs-batch ego experiment, cycle-length stratification, SAEs, ablations,
probes, extra seeds. Then the blog figures and the Colab. The Ethereum check is the first thing to cut.

Update this checklist as items are completed.
