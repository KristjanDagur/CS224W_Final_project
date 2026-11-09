# Handoff: Sherlock → FarmShare (2026-11-09)

Written by the coding agent on Sherlock for the agent picking up on FarmShare. Read `AGENTS.md` first
(it is binding), then this file. `docs/experiment_log.md` has every run and decision with numbers;
this file is the short version plus what to do next.

**Deadline:** milestone (code + report draft) due **2026-11-10 23:59 PT**.

## 1. What happened

- FarmShare went down on 2026-11-08, and its `data/`, `runs/` and `logs/` became unreachable.
- Everything was rebuilt and rerun on Sherlock. The code is cluster-agnostic (`scripts/cluster.sh`,
  `scripts/submit.sh`), and FarmShare is the default cluster again.
- On Sherlock, Python runs inside an Apptainer image because the OS is too old for the pinned torch
  wheels. None of that applies on FarmShare: use the normal `.venv` via `source scripts/activate.sh`.
- The synthetic learning rate (3e-3) was chosen from the FarmShare lr sweep's 11 of 12 finished runs
  (see the 2026-11-08 decision in the log).

## 2. Results so far (all produced on Sherlock)

**Synthetic ladder:** Egressy Table 1 reproduced. Test mean F1 over 3 seeds:

| Rung | Ours | Paper |
|---|---|---|
| R0 | 55.8 | 58.4 |
| R1 | 78.6 | 75.3 |
| R2 | 79.0 | 76.8 |
| R3 | 97.5 | 96.4 |

Each adaptation unlocks the predicted concepts. Deviations: R0 out-degree and fan-out are low and
noisy; R3 C6 is 78.3 vs 84.2; most runs selected a checkpoint from epoch ≥ 87, so they were still
improving at 100 epochs. The full per-concept table is in `docs/milestone_report.md` §5.1.

**AML ladder:** test F1 at 0.5, mean ± sd over 3 seeds, Egressy Table 2 in brackets:

| Rung | Ours | Paper |
|---|---|---|
| R0 | 35.2 ± 7.9 | [28.70 ± 1.13] |
| R1 | 53.0 ± 4.7 | [46.79 ± 4.97] |
| R2 | 60.5 ± 3.6 | [56.85 ± 2.64] |
| R3 | 55.7 ± 9.3 | [57.15 ± 4.99] |

- The ladder shape matches the paper: large gains from reverse MP and ports, and ego IDs add nothing
  on top of ports on HI-Small, as in the paper.
- R1–R3 are within about 1.4 of the paper's sd, so by the decision rule in the log the pipeline counts
  as validated.
- **R0 gap is not a bug.** We checked our code against IBM's line by line. The paper's GIN row is
  copied from Altman et al. 2023, which used its own hyperparameter search and 4 seeds; we use IBM's
  `gin` config for every rung, by design.
- **R3 seed variance is large** (46.7 / 65.3 / 55.0). More seeds would help before making claims about
  R3 vs R2.

**Synthetic dissection: computed, NOT yet looked at.** The user asked to do the analysis together, so
don't present conclusions before they ask.
- Covers all 12 runs: test split, IoU, AP and AUROC, 100 group-bootstrap and 20 permutation resamples.
- Outputs: `runs/analysis/dissection_synth_test.json` (per run, concept and layer),
  `runs/analysis/dissection_synth_test_summary.csv` (rung × concept; `gnn_mean` = GNN layers h1..h6
  only), and per-run files in `runs/analysis/dissection_synth_test/`.
- Pilot on synth_R3_s1 (few resamples): the best k-cycle unit appears one layer deeper as k grows
  (C2 h3, C3 h4, C4 h5, C5/C6 h6). This still needs to be confirmed across seeds.

## 3. Artifacts are NOT in git: copy them from Sherlock

On Sherlock, `runs/`, `data/` and `logs/` are symlinks into `/scratch/users/kde2804/CS224W/`. That
filesystem is purged after 90 days without modification, so FarmShare should hold the kept copy.

| What | Size | Needed on FarmShare? |
|---|---|---|
| `runs/`: 24 runs (`best.pt`, `last.pt`, `metrics.json`, `config.json`, `history.jsonl`, `thresholds.json`; synth runs also `activations_test.pt`) + `runs/analysis/` | 246 MB | yes, all |
| `logs/` | 0.3 MB | yes (provenance) |
| `data/processed/` | 493 MB | optional: rebuild instead with `scripts/slurm/prepare_data.sbatch`. It is deterministic; Sherlock and FarmShare v2 gave identical stats |
| `/scratch/users/kde2804/refs/`: IBM Multi-GNN clone, Egressy + Altman paper texts | small | no: `git clone --depth 1 https://github.com/IBM/Multi-GNN`; arXiv 2306.11586 and 2306.16424 |

The user runs the copy from FarmShare, pulling through Sherlock's transfer node (the login needs
Sherlock 2FA, so the agent can't do it alone):

```bash
mkdir -p ~/sherlock_import
rsync -avP kde2804@dtn.sherlock.stanford.edu:/scratch/users/kde2804/CS224W/runs/ ~/sherlock_import/runs/
rsync -avP kde2804@dtn.sherlock.stanford.edu:/scratch/users/kde2804/CS224W/logs/ ~/sherlock_import/logs/
```

**Do not merge blindly into FarmShare's `runs/`.** FarmShare may still hold its own runs from
2026-11-07 with the *same run IDs*:
- `runs/lr_sweep/`;
- the synthetic finals pipeline (jobs 1952881–1952884);
- the AML ladder (`runs/aml_pipeline_jobs.txt`).

Those were interrupted by the outage. The Sherlock runs are the canonical finals, and the log
documents them. Recommended:
1. Look at what FarmShare's `runs/` contains.
2. Move it aside, for example to `runs_farmshare_20261107/`. Don't delete it.
3. Put the Sherlock `runs/` in place.

Otherwise `train.py` could resume or skip runs based on the wrong `last.pt`/`metrics.json`.

Also on FarmShare:
- Check whether any of those old FarmShare jobs are still queued (`squeue --me`). Cancel them only
  after asking the user.
- **Bonus check:** if `runs/lr_sweep/` survived, verify the lr = 3e-3 decision. The missing 12th run
  (R3 at lr 1e-3) was filled with a placeholder; see the caveat in the log on 2026-11-08.
- Log in `docs/experiment_log.md` that the Sherlock artifacts were imported.

## 4. Code added on Sherlock (all committed by the user, up to `8a1b48e`)

- `analysis/hooks.py`: activation extraction for the synthetic and AML tasks; records `ego_mode` and
  the protocol.
- `analysis/dissection.py`:
  - rate-matched IoU, AP and AUROC (ties count 1/2; matches sklearn, tested), signed units;
  - permutation null of max-over-units, group bootstrap;
  - `--combine` merges per-run JSONs; the summary includes `gnn_mean`.
- `analysis/plots.py` (**tested only on fake data so far**):
  - rung × concept coverage heatmap, with the rung where theory unlocks each concept outlined;
  - concept × layer heatmap for one rung;
  - PR-curve small multiples. Each concept's unit is **selected on val and plotted on test**, to avoid
    selection bias. Needs `activations_val.pt`.
  - Colours follow the dataviz skill's palette (validated).
- `scripts/slurm/dissect_synth.sbatch`: extraction + parallel dissection + combine. Resumable: it
  skips runs that are already dissected. On FarmShare: `scripts/submit.sh cpu scripts/slurm/dissect_synth.sbatch`.
- `tests/test_analysis.py`: 8/8 pass on Sherlock (2026-11-08). The full suite has not been re-run
  since the AUROC test was added, so run `pytest -q` first on FarmShare.

## 5. Next steps, in order

1. Import the Sherlock artifacts (§3), check `pytest -q`, and log the import.
2. Validation activations:
   `python -m amlinterp.analysis.hooks --run runs/synth_R*_L6_h64_s* --split val`. This took 3 min on
   4 CPUs; run it in a job, not on the login node.
3. Figures:
   ```bash
   python -m amlinterp.analysis.plots --results runs/analysis/dissection_synth_test.json \
       --out runs/analysis/figures --layers-rung R3 \
       --acts-select 'runs/synth_R*_L6_h64_s*/activations_val.pt' \
       --acts-eval 'runs/synth_R*_L6_h64_s*/activations_test.pt'
   ```
   Look at the PNGs before using them (dataviz skill, step 7).
4. **With the user:** go through the dissection results.
   - H1: does coverage jump at the outlined rungs?
   - Does the cycle-depth pattern hold across seeds?
   - In the PR-curve table, compare val-selected AP with test AP.
5. Fill `docs/milestone_report.md`:
   - §5.2: figures and their reading;
   - §5.3: the AML table above, plus the R0 provenance note and R3 variance;
   - remove the TODOs.

   Ask the user about the course's milestone format (page limit, template, whether a Colab is
   required now).
6. After the milestone (see `AGENTS.md`):
   - AML dissection, with structural concepts, the laundering-label baseline and attempt-level CIs;
   - exact-vs-batch ego comparison;
   - degree-matched control for cycle units;
   - linear probes, SAEs, ablations;
   - more seeds, especially for AML R3.

## 6. Working with this user

- **The user commits and pushes themselves.** Never commit or push unless asked.
- **The user wants momentum** (the deadline is close), but asks conceptual questions along the way.
  Answer them plainly and concretely; they often lead to improvements (AUROC and the PR curves came
  from their questions).
- **Ask before deleting anything**, including old runs. The user once denied a deletion.
- **Interactive jobs:** an `srun` started from the agent's shell dies when the session ends. For
  anything longer than a few minutes, prefer batch jobs. Save outputs incrementally (resumable
  scripts).

## 7. Sherlock leftovers (for the user)

- `/home/groups/mberger1/kde2804/amlinterp`: an old copy of the environment in the group folder. The
  user does not want group folders used. Delete it when convenient; the agent's `rm` was declined.
- `$HOME/envs/amlinterp/` (6.7 GB): the Sherlock environment. Keep it if Sherlock might be needed
  again.
- Back on Sherlock, the agent must first read `/etc/agents/AGENTS.md` (Sherlock rule 12) and keep
  following its rules: no work on login nodes, no `--exclude`, `dev` only via interactive `srun`.
