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
   - Verified 2026-11-07 (`data/processed/aml_hi_small/stats.json`):
     - IBM's day split is 64/19/17%, not 60/20/20 (days 0–5 / 6–7 / 8–17).
     - The illicit ratio rises across splits: 0.078% → 0.107% → 0.186%. Thresholds tuned on val will see a shifted base rate on test.
     - 147 of the 370 attempts span a split boundary. Attempt-level statistics must count each attempt's transactions per split.
     - Cycles have 2–12 hops. With 2 layers only 2–3 hop cycles are reachable, and test has just 6 such attempts (3 + 3), so AML cycle claims will have wide CIs.
     - 11.6% of edges are self-loops and 80% are parallel edges. The max out-port is 14,229, so raw ports are heavy-tailed (IBM z-normalizes them).

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
   - **Known IBM bugs that we reproduce on purpose** (user decision, 2026-11-07: follow IBM so Table 2 is comparable; don't "fix" these without asking):
     - *Per-split normalization.* `z_norm` is applied separately to each split graph's edge features (train, train+val, all), each with its own mean/std. Test features therefore use statistics that include test edges, and the same edge gets different feature values in different splits. The clean alternative is train-only statistics.
     - *Dropped seed edges.* IBM finds the seed edges in a sampled batch by matching an `EdgeID` column (`add_arange_ids`, then `mask = isin(batch IDs, seed IDs)`). A seed edge that the neighbor sampler did not include is silently dropped from that batch's loss and evaluation. It is only re-added for the Small_J/Q datasets, not HI-Small. Evaluation therefore covers a subset of val/test that is biased against hub accounts. Log how many seeds are dropped per split, so the effect can be reported.

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
- **One code path for both clusters.** All cluster-specific settings live in `scripts/cluster.sh` (FarmShare is the default; Sherlock is detected through `$SHERLOCK`, override with `AMLINTERP_CLUSTER`). Job scripts carry no partition/GPU/node flags. Submit with `scripts/submit.sh gpu|cpu [sbatch options] <script>`, which prepends the cluster's flags and prints the job ID. `source scripts/activate.sh` gives a working `python` on either cluster.
- FarmShare GPU jobs use `--partition=gpu --qos=gpu --gpus=1`; GPUs are not allocated by default.
- FarmShare GPUs (checked 2026-11-07 with `sinfo`/`scontrol show node`/`sacctmgr show qos`):
  - `gpu` partition: 6 nodes (`oat-01`…`oat-06`), each 4× **NVIDIA L40S 48 GB** (Ada, CC 8.9), 64 CPUs, 250 GB RAM. 24 GPUs total.
  - QOS `gpu`: max 4 GPUs and 4 running jobs per user, 32 submitted jobs per user, 2-day wall limit.
  - `MaxMemPerCPU=4000` MB, so a 32 GB job must request 8 CPUs (`--cpus-per-task=8 --mem=32G`).
  - Heavily contended: on 2026-11-07, 22 of 24 GPUs were allocated and about 95 jobs were waiting on priority. Submit early; short `--time` requests backfill faster.
  - Any current PyTorch CUDA 12.x wheel supports CC 8.9.
- Scratch: `/scratch/users/$USER` (shared filesystem, 81% full on 2026-11-07). Home is `/home/users/$USER` (NFS).
- CPU jobs: the `normal` partition (barley/wheat nodes, 64+ CPUs each) has no GPU queue. Use it for synthetic runs.
  - Those nodes are Xeon Gold 6148 (2017, hyperthreaded: `--cpus-per-task=8` gives 4 physical cores). They are shared with other users' heavy jobs, so the same run's epoch time varies up to 4× by node (40 s vs 209 s on 2026-11-07). Budget time limits for the slow case and rely on resume.
- Gotchas found 2026-11-07:
  - Do **not** `module load python/...` in job scripts. It is missing on some compute nodes (barley-03), and the venv uses the system `/usr/bin/python3.12` anyway.
  - `barley-04` could not see `/home/users`, so jobs there die without writing a log. Submit with `--exclude=barley-04` until it is fixed.
  - Priority vs backfill (checked with `sprio`):
    - **Priority** favors large requests (`PriorityFavorSmall=no`, `PriorityWeightTRES=cpu=10000,mem=30000,GRES/gpu=50000`). The memory term, about 30,000 × requested MB, dominates; age and fairshare are small next to it. A job asking for 8 GB therefore sits at the bottom, while whole-node 256 GB requests sit at the top.
    - **Backfill** (`sched/backfill`) still starts small, *short* jobs early in gaps the top job can't use. Short `--time` limits plus resumable training are the way to start sooner.
    - Request what a job really needs (about 8 CPUs and 32 GB for AML). Don't pad requests to jump the queue.
- Request roughly 1 GPU, 8 CPUs, 32 GB RAM, and a time limit of 16 hours or less. Submit rung × seed as one job array.
- **Sherlock (fallback; used 2026-11-08 while FarmShare was down).** Read `/etc/agents/AGENTS.md` and its topic files first; they are binding there.
  - Host OS is CentOS 7 (glibc 2.17): the pinned torch 2.13 wheels (`manylinux_2_28`) cannot install, and there is no PyG module (newest `py-pytorch` is 2.10 on Python 3.14). Python therefore runs in an Apptainer image (`docker://python:3.12-bookworm`, which also provides a modern git for `config.json`) with the same pinned venv. Image and venv: `$HOME/envs/amlinterp/` (6.7 GB of the 15 GB home quota). Build them inside a job: `scripts/slurm/setup_env.sbatch`.
  - **Do not use `$GROUP_HOME` or `$GROUP_SCRATCH`** (user decision, 2026-11-08). Environments may not go on `$SCRATCH` (Sherlock rule 8: purged), so the environment is in `$HOME`; everything else (data, runs, logs, pip/Apptainer caches) goes on `$SCRATCH`.
  - `data/`, `runs/`, `logs/` are symlinks into `$SCRATCH/CS224W/` (purged after 90 days unmodified: copy results you need to keep elsewhere).
  - `gpu` partition: L40S, H100, RTX 3090, V100, 2080 Ti, P100, Titan Xp; `cluster.sh` restricts to L40S/H100/3090. QOS `gpu`: 16 GPUs and 128 CPUs per user, 100 submitted jobs, 2-day limit, up to 32 GB per CPU.
  - `--exclude` is not supported on Sherlock; select hardware with `-C` only. No `--qos`/`--account` needed.
  - `dev` partition rejects `sbatch`; it is for interactive `srun`/`sh_dev` only (4 CPUs, 16 GB, 2 GPUs as A30 MIG 6 GB slices, 2 h).
  - Queues on 2026-11-08: about 1,500 GPUs' worth of jobs pending; `--test-only` estimated about 2 days to start a GPU job and about 11 h for a CPU job on `normal`.
- Keep data and runs on cluster scratch, not in home quota. Copy checkpoints and metrics back for analysis.
- Make training resumable: checkpoint every epoch, so a job killed at its time limit can continue.

## Project status / next steps

Milestone due 2026-11-10:

- [x] Pinned env (`requirements.txt` + `scripts/setup_env.sh`): verified on FarmShare CPU nodes and an L40S GPU node (driver 595.91, CUDA 13.2); Sherlock via Apptainer (2026-11-08); laptop not yet verified
- [x] AMLworld preprocessing + cached ports (`data/aml.py`); patterns parser + stats (`data/patterns.py`, `stats.json`)
- [x] AML task (`tasks/aml.py`): LinkNeighborLoader [100,100], batch ego IDs, IBM per-split z-norm; smoke-tested on L40S
- [x] Synthetic circulant generator + labelers (`data/synthetic.py`, tests vs brute force; rates match Table 3)
- [x] Vectorized port numbering (`data/ports.py`, tested against IBM's loop)
- [x] GIN ladder model + `train.py` (resumable) + synthetic task; smoke-tested on CPU
- [x] Synthetic training R0–R3 × 3 seeds (6 layers, lr 3e-3; `scripts/tasks/synth_final.txt`): Table 1 reproduced (Sherlock job 51581520, 2026-11-08; deviations in the experiment log)
- [ ] AML training R0–R3 × 3 seeds (`scripts/tasks/aml_final.txt`); compare to Table 2. Running on Sherlock (job 51581524, started 2026-11-08)
- [x] Activation hooks with `ego_mode` (`analysis/hooks.py`) and dissection code (`analysis/dissection.py`: rate-matched IoU + AP, signed units, permutation null, group bootstrap); tested on planted units and tiny graphs
- [ ] Dissection on trained synthetic runs (`scripts/slurm/dissect_synth.sbatch`); first rung × concept heatmap (`analysis/plots.py`)
- [ ] Milestone report draft (problem, data and processing, model and metrics, preliminary results): `docs/milestone_report.md`

After the milestone: AML dissection, the exact-vs-batch ego experiment, cycle-length stratification, SAEs, ablations,
probes, extra seeds. Then the blog figures and the Colab. The Ethereum check is the first thing to cut.

Update this checklist as items are completed.
