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
- **AMLworld HI-Small** (Altman et al., NeurIPS 2023 D&B): about 515K accounts and 5M transactions. The task is binary edge classification with a chronological 60/20/20 split. `HI-Small_Patterns.txt` gives ground-truth pattern membership for about 62% of laundering transactions.
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

> **Status:** the repo currently holds only the proposal and these docs. The layout below is the *planned*
> structure; keep it in sync as code lands.

```
proposal.pdf
README.md               this file
AGENTS.md               instructions for coding agents (read before changing anything)
requirements.txt        pinned pip environment (scripts/setup_env.sh builds .venv/)
src/amlinterp/
  data/aml.py           AMLworld loading, formatting, port numbers, chronological split
  data/patterns.py      parser for HI-Small_Patterns.txt -> per-edge (attempt_id, pattern_type)
  data/synthetic.py     random circulant generator + exact pattern labelers
  models/gin.py         GIN ladder (adapted from IBM Multi-GNN, Apache-2.0)
  train.py              training/eval entry point (one config = one run)
  analysis/hooks.py     activation extraction (records ego mode)
  analysis/dissection.py
  analysis/sae.py
  analysis/ablation.py
  analysis/probes.py
configs/                one YAML per rung/dataset
scripts/slurm/          cluster job scripts (job arrays over rung x seed)
notebooks/              exploration + the final Colab
docs/experiment_log.md  append-only log of every run and result
data/                   NOT committed (raw/ and processed/)
runs/                   NOT committed (checkpoints, logs, activations)
```

## Setup

Python 3.12 virtualenv in `.venv/` (not committed). Versions are pinned in `requirements.txt`: torch 2.13.0 (CUDA 12.6), pyg_lib 0.9.0 and torch-geometric 2.8.1. PyG no longer needs `torch-scatter` or `torch-sparse`.

```bash
bash scripts/setup_env.sh        # once; on FarmShare it loads the python/3.12.3 module
source scripts/activate.sh       # every session (also used inside Slurm jobs)
```

On a laptop without `module`, use any Python ≥3.10 and run the same script.

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
