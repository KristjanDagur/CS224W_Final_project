# What Do Provably Powerful GNNs Learn? Interpreting Money-Laundering Detectors on an Expressiveness Ladder

**CS224W Project Milestone (Project 1: Application).** Kristjan Egilsson, kde2804@stanford.edu

> DRAFT (2026-11-08). Numbers marked TODO are filled in when the corresponding job finishes. Every number
> here is traceable to `docs/experiment_log.md`.

## 1. Problem and motivation

Money launderers move funds through recognizable transaction patterns: fan-ins, fan-outs,
scatter-gathers, and cycles that return money to its origin. Message-passing GNNs are at most as
expressive as 1-WL and provably cannot count many such substructures. Egressy et al. (AAAI 2024) showed
that three adaptations (reverse message passing, port numbering and ego IDs) make a GNN able to detect
any directed subgraph pattern in a directed multigraph, and that they greatly improve laundering
detection on IBM's AMLworld benchmark.

Higher F1 does not show *how* the improved models work, and a compliance analyst needs to know whether a
flag reflects a laundering pattern or a shortcut. Our question:

> When an adaptation provably unlocks a pattern, does the trained model develop internal features that
> detect it, and does it rely on them?

We train an *expressiveness ladder* of four models that differ by exactly one adaptation each, and test
three hypotheses:

- **H1:** concept coverage (how well the best single unit detects a pattern) grows along the ladder as the
  theory predicts. For example, cycle detectors appear only once ego IDs are added.
- **H2:** the new detectors are causally used, so mean-ablating them hurts that pattern specifically.
- **H3:** where no single neuron detects a provably detectable pattern, sparse-autoencoder latents
  recover it.

This milestone reports the data pipelines, the trained ladder on synthetic data (a reproduction of
Egressy et al. Table 1), the AMLworld runs in progress, and the first dissection results.

## 2. Datasets and processing

### 2.1 Synthetic graphs with exact pattern labels

We follow Egressy et al. App. D. Each graph is a random circulant directed multigraph:
- n = 8,192 nodes and m = ⌊nd/2⌋ = 24,576 edges (average degree d = 6);
- each edge picks u uniformly and v = round(N(u, r)) mod n with r = 11.1;
- self-loops are re-drawn;
- a random edge time orders the port numbers.

Train, validation and test are three independently generated graphs.

Every node gets 11 binary labels, computed by *enumerating* each pattern in the graph:
- deg-in/out > 3 (counting multi-edges);
- fan-in/out > 3 (counting unique neighbors);
- lies on a directed k-cycle, k = 2..6 (C2–C6);
- scatter-gather (S-G) sink;
- biclique (B-C, the sink of a directed K2,2).

The labelers are unit-tested against brute force on small graphs.

**Change from the proposal.** The proposal said patterns would be *planted*. We instead label the patterns
that occur naturally in random graphs, as Egressy et al. do. Planting adds extra edges, so pattern nodes
would have higher degree than other nodes. A degree neuron would then look like a cycle detector. This
"degree shortcut" is exactly the confound an interpretability study has to avoid.

Re-drawing self-loops matters: keeping them undershoots Egressy's positive rates by 2–5 points on C4–C6,
S-G and B-C. With re-drawing, the train-graph rates match their Table 3 within about one sd (deg-in .348,
C2 .200, C6 .784, S-G .322, B-C .322). The base rates range from 0.19 (C2) to 0.78 (C6). That range is
why we use rate-matched thresholds and chance-corrected scores (Section 4).

### 2.2 AMLworld HI-Small

AMLworld (Altman et al., NeurIPS 2023) is a synthetic transaction dataset from an agent-based simulator. We
use HI-Small:
- 515,088 accounts (nodes) and 5,078,345 transactions (directed, timestamped edges with amount,
  currency and payment format);
- 5,177 laundering transactions (0.10%).

The task is binary edge classification. We follow IBM's Multi-GNN preprocessing exactly, so our results
are comparable to Egressy Table 2:
- **Chronological day split.** It comes out at 64/19/17%, not the nominal 60/20/20. The illicit ratio
  rises across splits (0.078% → 0.107% → 0.186%), so thresholds tuned on validation see a shifted base
  rate on test.
- **Nested split graphs.** Train ⊂ train+val ⊂ all; each split is evaluated only on its own edges.
- **Per-split z-normalization** of edge features.
- **Ports.** Each account numbers its counterparties in order of their first transaction. All edges
  with the same counterparty share a port, and reverse edges swap the in- and out-port columns. IBM
  computes ports with a Python loop over 5M edges. We vectorized it (tested equal to IBM's loop) and
  cache the result.

`HI-Small_Patterns.txt` lists the transactions of 370 laundering *attempts* of 8 pattern types:
- 3,209 rows, all of which we match to edges;
- they cover 62% of laundering transactions;
- 147 attempts span a split boundary.

These pattern labels are the ground-truth concepts for the AML dissection. Two properties of the graph
matter for interpretation:
- 11.6% of edges are self-loops and 80% are parallel edges;
- with the benchmark depth of 2 layers, only cycles of length 2–3 through the target edge are
  detectable, and the test split has just 6 such cycle attempts.

## 3. Models

All rungs share the GIN-with-edge-features backbone of IBM Multi-GNN (PyG):
- linear node and edge embeddings;
- residual layers x ← (x + ReLU(BN(GINEConv(x, e)))) / 2;
- hidden size 64;
- the readout MLP(→50→25→2) on [h_u ‖ h_v ‖ e_uv] for edges (AML) or on h_v for nodes (synthetic,
  11 outputs).

| Rung | Adds | Theory: newly detectable |
|---|---|---|
| R0 | GIN | in-degree, edge statistics |
| R1 | + reverse message passing (separate `to`/`rev_to` convolutions, mean-combined) | out-degree, fan-out |
| R2 | + port numbers (in- and out-port as edge features) | fan-in, fan-out (unique counterparties) |
| R3 | + ego IDs (a 0/1 feature on the target's endpoints) = Multi-GIN | cycles, scatter-gather, bicliques |

**Controlled experiment.** Every rung uses the same hyperparameters, split, epoch budget,
model-selection rule (best validation mean F1) and seeds (1, 2, 3). Decision thresholds and checkpoints
are chosen on validation only.

- **Synthetic:** 6 layers (Egressy F.3, so 6-cycles are in range), full neighborhoods, Adam with
  lr 3e-3, 100 epochs, batch 256. The lr is the best of {3e-4, 1e-3, 3e-3} by validation F1 averaged
  over rungs.
- **AML:** IBM's GIN settings (lr 6.2e-3, class-weighted cross-entropy, [100, 100] neighbor sampling,
  batch 8,192, 2 layers, 100 epochs).

**Ego IDs.** IBM's code marks the endpoints of *all* seed edges in a batch, not just the scored edge. We
implement both versions:
- `batch` reproduces the benchmark and is used for AML training;
- `exact` marks only the target and is used for synthetic training and for all synthetic activations.

R3's activations depend on this choice, so it is recorded with every saved activation file.

## 4. Interpretability methods and metrics

**Task metrics.** Minority-class F1 (the benchmark standard) at the 0.5 threshold and at a
validation-tuned threshold, plus PR-AUC.

**Activations.** For each trained model we record, for every test item, all hidden layers h0..h6 and both
decoder layers. On synthetic data an item is a node, giving 523 units. On AML an item is an edge, and
the representation is [h_u ‖ h_v] plus the decoder layers.

**Neuron dissection (H1).** For unit k and binary concept c we compute two scores:
- **IoU** = |M_k ∩ C_c| / |M_k ∪ C_c|, where the unit's mask M_k is its top-p_c fraction of items.
  Matching the concept's own base rate p_c is needed because the concepts are common (0.19–0.78) and a
  fixed top quantile would cap IoU.
- **Average precision (AP)** of the unit's activation as a score for the concept. It is
  threshold-free, and chance is p_c. We report it chance-corrected, (AP − p_c)/(1 − p_c), so 0 means a
  random unit and 1 a perfect detector.

Hidden states can be negative, so each unit is scored in both directions. Concept *coverage* is the best
score over units, overall and per layer.

The best of hundreds of units beats single-unit chance easily, so we test it against a **permutation
null**: concept labels are shuffled across items 20 times and the max over all units recomputed. A
**bootstrap** (100 resamples, re-selecting the best unit each time) gives CIs. On AML, resampling is at
the level of laundering attempts.

Coverage is reported over the GNN layers h1..h6. The decoder layers sit directly before the output, so
any accurate model "covers" its own targets there.

**Planned for H2/H3** (after the milestone):
- mean-ablation of detector units, with training-split means and validation thresholds fixed beforehand,
  measuring per-concept recall drops against random-unit ablations;
- TopK sparse autoencoders (Gao et al. 2024) on the hidden states, followed by the same dissection on
  their latents;
- linear probes as a decodability ceiling.

## 5. Preliminary results

### 5.1 The synthetic ladder reproduces Egressy et al. Table 1

Test minority-class F1 (%), mean over 3 seeds at the 0.5 threshold; sd in parentheses where it is ≥ 2 points (and for the row means and R3 C6). Egressy Table 1 (mean of 5
runs) in brackets.

| Rung | deg-in | deg-out | fan-in | fan-out | C2 | C3 | C4 | C5 | C6 | S-G | B-C | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 GIN | 99.9 [99.8] | 20.9 (10.8) [43.6] | 95.3 [95.6] | 14.9 (9.3) [35.9] | 27.0 (6.3) [34.7] | 53.3 (4.6) [58.0] | 64.3 (3.3) [50.8] | 52.7 [43.1] | 52.3 [48.6] | 68.3 [69.3] | 64.9 (2.2) [63.1] | 55.8 (2.6) [58.4] |
| R1 +reverse | 99.3 [98.9] | 99.3 [99.1] | 95.2 [95.0] | 96.0 [95.3] | 49.0 (2.2) [36.0] | 64.4 (2.0) [63.9] | 74.1 [69.1] | 73.6 [67.4] | 77.6 [71.2] | 68.7 [65.8] | 67.5 (2.2) [66.2] | 78.6 (0.4) [75.3] |
| R2 +ports | 99.8 [98.4] | 99.2 [98.4] | 99.8 [98.5] | 99.7 [99.2] | 47.9 [39.2] | 63.5 [63.6] | 75.7 [69.0] | 73.4 [70.4] | 76.5 [75.0] | 67.5 [67.8] | 66.3 [65.8] | 79.0 (0.2) [76.8] |
| R3 +ego IDs | 99.2 [99.5] | 99.8 [99.1] | 100.0 [99.6] | 99.9 [99.3] | 100.0 [99.0] | 100.0 [98.7] | 99.7 [97.5] | 98.2 [91.6] | 78.3 (0.6) [84.2] | 98.9 [97.4] | 98.3 [94.3] | 97.5 (0.1) [96.4] |

Each adaptation unlocks what the theory predicts:
- **R1 (reverse):** out-degree jumps from 21 to 99 and fan-out from 15 to 96.
- **R2 (ports):** fan-in and fan-out reach ≥ 99.7.
- **R3 (ego IDs):** C2–C5, scatter-gather and biclique jump to 98–100. On average the ladder matches
  or slightly exceeds the paper's.

Deviations, logged as found:
- **R0 out-degree and fan-out** are below the paper (20.9 vs 43.6, 14.9 vs 35.9) with high variance
  across seeds. A plain GIN cannot see outgoing edges, so these scores are near chance by construction.
  The paper's higher values likely reflect threshold or seed noise more than real signal.
- **R1/R2 2-cycles** are about 10 points above the paper.
- **R3 6-cycles** are 6 points below (78.3 vs 84.2).
- **Training length:** 8 of 12 runs selected a checkpoint from epoch ≥ 87, so 100 epochs is slightly
  short. This applies equally to all rungs, so the comparison stays controlled.

### 5.2 Dissection: which concepts get a dedicated unit, and where

TODO (job running): Figure 1, the rung × concept coverage heatmap (`runs/analysis/figures/coverage_ap.png`);
Figure 2, the R3 concept × layer heatmap.

A pilot on one R3 model (synth_R3_s1, few resamples) shows two things:
- Nearly every concept has a single dedicated unit (chance-corrected AP > 0.95 for degree, fan and
  C2–C5).
- **The layer at which the best k-cycle unit appears grows with k:** C2 at layer 3, C3 at layer 4, C4 at
  layer 5, C5 and C6 at layer 6. This is consistent with the ego bit having to travel around the cycle
  before the node can detect it.

TODO: confirm across seeds and rungs.

### 5.3 AMLworld ladder

TODO (running on Sherlock GPUs since 2026-11-08). Compare to Egressy Table 2: GIN 28.70 ± 1.13,
+ReverseMP 46.79 ± 4.97, +Ports 56.85 ± 2.64, +EgoIDs 57.15 ± 4.99.

## 6. Plan for the rest of the quarter

- **Nov 11–20:** AML dissection with the confound controls below; exact-vs-batch ego comparison;
  ablations (H2).
  - Pattern masks are subsets of the laundering label, so every result is reported against a
    laundering-label baseline.
  - Structural concepts are computed for all edges, not just laundering ones.
  - CIs are bootstrapped at the attempt level.
- **Nov 20–28:** sparse autoencoders and probes (H3) on both datasets; a degree-matched control for
  cycle detectors (cycle membership correlates with degree even in random graphs).
- **Dec 1–10:** blog post and Colab. The optional Ethereum check is the first item to cut.
