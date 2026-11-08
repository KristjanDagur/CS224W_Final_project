# What Do Provably Powerful GNNs Learn?

**Interpreting money-laundering detectors on an expressiveness ladder.**
CS224W (Stanford, Autumn 2026) final project, Project type 1 (Application). Solo project by Kristjan Egilsson.
The full proposal is in [`proposal.pdf`](proposal.pdf).

## Research question

Egressy et al. (AAAI 2024) proved that three adaptations let a message-passing GNN detect any directed
subgraph pattern in a directed multigraph:

1. **Reverse message passing:** an account also aggregates messages from the accounts it sends money to.
2. **Port numbering:** each account numbers its counterparties, so repeated transactions with one counterparty can be recognized.
3. **Ego IDs:** a binary marker on the endpoints of the transaction being scored, so the model can detect money returning to them (cycles).

These adaptations also greatly improve laundering detection. This project asks:

> When an adaptation provably unlocks a pattern, does the trained model develop internal features that
> detect it, and does it rely on them?

## Approach

**The ladder.** Four models that differ only by one adaptation each, all built on GIN with edge features:

| Rung | Model | IBM Multi-GNN flags | Patterns it should newly detect (theory + Egressy Table 1) |
|---|---|---|---|
| R0 | GIN | *(none)* | in-degree, edge statistics |
| R1 | + reverse MP | `--reverse_mp` | out-degree, fan-out |
| R2 | + ports | `--reverse_mp --ports` | fan-in / fan-out (unique counterparties) |
| R3 | + ego IDs (Multi-GIN) | `--reverse_mp --ports --ego` | directed cycles, scatter-gather, bicliques |

Optional extra rungs: R3 + edge updates (`--emlps`), and R3 with 4 layers (to test the depth limit on cycle length).

**Data.**
- **Synthetic graphs:** random circulant directed multigraphs following Egressy et al. Appendix D. Every node has exact labels for degree-in/out, fan-in/out, directed cycles of length 2–6, scatter-gather and biclique.
- **AMLworld HI-Small** (Altman et al., NeurIPS 2023 D&B): about 515K accounts and 5M transactions. The task is binary edge classification with IBM's chronological day split (target 60/20/20; actual 64/19/17%). `HI-Small_Patterns.txt` gives ground-truth pattern membership for about 62% of laundering transactions.
- **Optional:** the Ethereum phishing network (node classification).

**Analyses, run on each rung:**
1. **Neuron dissection:** IoU and average precision between unit activations and concept masks.
2. **TopK sparse autoencoders** on hidden states, followed by the same dissection on SAE latents.
3. **Causal mean-ablation:** per-pattern recall drop.
4. **Linear probes** as a decodability baseline.

**Hypotheses.**
- **H1:** concept coverage grows along the ladder.
- **H2:** the new detectors are causally used, with pattern-specific ablation effects.
- **H3:** SAE latents recover patterns that single neurons miss.

**Metrics:** minority-class F1, PR-AUC, and per-pattern recall.

## Repository layout

> **Status:** data, model and training code are in place; `analysis/` is still planned. Keep this layout
> in sync as code lands.

```
proposal.pdf
README.md               this file
AGENTS.md               instructions for coding agents (read before changing anything)
requirements.txt        pinned pip environment (scripts/setup_env.sh builds .venv/)
src/amlinterp/
  data/aml.py           AMLworld loading, formatting, chronological split            (done)
  data/patterns.py      parser for HI-Small_Patterns.txt -> per-edge (attempt_id, pattern_type)  (done)
  data/synthetic.py     random circulant generator + exact pattern labelers          (done)
  data/ports.py         vectorized port numbering (IBM semantics)                    (done)
  data/transforms.py    rung -> input graph (ports, reverse edges), ego IDs per batch (done)
  models/gin.py         GIN ladder + RUNGS (adapted from IBM Multi-GNN, Apache-2.0)  (done)
  tasks/synthetic.py    synthetic node task: loaders, batch step, metrics            (done)
  tasks/aml.py          AMLworld edge task, IBM protocol (adapted, Apache-2.0)       (done)
  metrics.py            minority-class F1, PR-AUC, validation threshold tuning       (done)
  train.py              training/eval entry point (one config = one run, resumable)  (done)
  analysis/hooks.py     activation extraction per scored item + batch protocol / ego mode (done)
  analysis/dissection.py  unit x concept IoU + AP, coverage, permutation null, bootstrap CIs (done)
  analysis/sae.py
  analysis/ablation.py
  analysis/probes.py
configs/                one YAML per dataset (rung and seed are CLI args)
scripts/cluster.sh      per-cluster settings (FarmShare default, Sherlock fallback)
scripts/submit.sh       sbatch + the cluster's partition/GPU flags
scripts/slurm/          cluster-agnostic job scripts (job arrays over rung x seed)
scripts/tasks/          task files: one train.py command line per array index
tests/                  pytest: labelers vs brute force, ports vs IBM loop, subgraph == full-graph forward
notebooks/              exploration + the final Colab
docs/experiment_log.md  append-only log of every run and result
data/                   NOT committed (raw/ and processed/)
runs/                   NOT committed (checkpoints, logs, activations)
```

## Setup

Python 3.12 virtualenv in `.venv/` (not committed). Versions are pinned in `requirements.txt`: torch 2.13.0 (CUDA 12.6), pyg_lib 0.9.0 and torch-geometric 2.8.1. PyG no longer needs `torch-scatter` or `torch-sparse`.

```bash
bash scripts/setup_env.sh        # once; FarmShare: system python3.12. Sherlock: run inside a job (see below)
source scripts/activate.sh       # every session (also used inside Slurm jobs)
```

Works on a laptop with any Python ≥3.10. Do not `module load python` on FarmShare: some compute nodes lack those modules, and the venv does not need them.

**Clusters.** FarmShare is the primary cluster; Sherlock is the fallback. `scripts/cluster.sh` detects which one
you are on and holds everything that differs (environment, storage, Slurm flags). Submit through
`scripts/submit.sh`, which adds the right partition/GPU flags:

```bash
scripts/submit.sh cpu scripts/slurm/setup_env.sbatch                       # build env + run tests
RAW=data/raw scripts/submit.sh cpu scripts/slurm/prepare_data.sbatch       # synthetic + AMLworld caches
TASKS=scripts/tasks/aml_final.txt scripts/submit.sh gpu --array=0-11 --cpus-per-task=8 --mem=32G \
    --time=06:00:00 scripts/slurm/run_tasks.sbatch                         # AML ladder, R0-R3 x 3 seeds
```

On Sherlock (CentOS 7, glibc 2.17) the pinned torch wheels cannot run on the host, so `setup_env.sh` builds the
same pinned venv inside an Apptainer image (`python:3.12-bookworm`) in `$GROUP_HOME/$USER/amlinterp/`, and
`activate.sh` makes `python` run through it. `data/`, `runs/` and `logs/` are symlinks into `$SCRATCH/CS224W/`.

**AMLworld data.** Download `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` from Kaggle
(`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`) into `data/raw/`. Then preprocess once and cache
the result as a `.pt` file in `data/processed/`. Port computation is slow, so never redo it inside each training job.

## Compute

- **Laptop** (RTX 2000 Ada, 8 GB): development, synthetic experiments, and all analysis on saved activations.
- **Cluster** (FarmShare preferred, Sherlock as fallback): AML training runs. Each needs one GPU (at least 16 GB VRAM preferred), about 32 GB RAM, about 8 CPUs, and up to about 14 hours.
  - Egressy report about 0.5 hours (R0) and about 3.5 hours (R1–R3) on an RTX 3090.
  - Submit as one Slurm job array over rung × seed.

## Deadlines (11:59pm PT)

| Item | Date | Notes |
|---|---|---|
| Proposal | 2026-10-20 | done |
| Milestone | 2026-11-10 | credit/no-credit: code (data processing, train/eval) + report draft |
| Final | 2026-12-10 | Medium blog draft link + Google Colab + Medium account name |

## Key references

- Egressy et al., *Provably Powerful Graph Neural Networks for Directed Multigraphs*, AAAI 2024. [arXiv:2306.11586](https://arxiv.org/abs/2306.11586)
- Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023 D&B. [arXiv:2306.16424](https://arxiv.org/abs/2306.16424)
- IBM Multi-GNN code: <https://github.com/IBM/Multi-GNN> (Apache-2.0)
- Xuanyuan et al., *Global Concept-Based Interpretability for GNNs via Neuron Analysis*, AAAI 2023
- Gao et al., *Scaling and Evaluating Sparse Autoencoders*, 2024 (TopK SAE)
- Full bibliography: `proposal.pdf`

## License and attribution

Model, loader and port-numbering code adapted from IBM Multi-GNN (Apache-2.0) must keep attribution in the file
headers. AMLworld is released under CDLA-Sharing-1.0.
