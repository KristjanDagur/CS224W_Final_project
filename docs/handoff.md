# Handoff: Sherlock → FarmShare (2026-11-09)

Written by the Sherlock agent for the agent on FarmShare. It covers only what is **new since the FarmShare
commit `dbb71a6` ("checkpoint", 2026-11-08 21:20)**. Everything before that is already in the repo and
in `docs/experiment_log.md`.

**Deadline:** milestone (code + report draft) due **2026-11-10 23:59 PT**.

## 1. AML seeds 2–3 finished on Sherlock, and they are the canonical copy

The Sherlock array 51581524 ran tasks 4–11 (R0–R3 × seeds 2, 3):
- started 2026-11-09 01:19–01:45;
- all completed by 05:27, exit 0, on L40S/H100/3090-class nodes;
- test seed coverage ≥ 99.90%.

FarmShare job **1957135** queued the same 8 runs, with a start estimate of 06:12. It is therefore the
copy that started second. Under the rule logged on 2026-11-08 ("cancel whichever copy starts second"),
**the Sherlock runs are canonical**. To do on FarmShare:
1. Check `squeue --me` and `sacct -j 1957135`.
2. If it is still pending, ask the user before cancelling it.
3. If it already ran, move its outputs aside as replicates (as was done with `runs/farmshare_replicates/`)
   before copying the Sherlock runs in under the canonical names. Otherwise `train.py` could skip or
   resume from the wrong files.

Test minority F1 at 0.5, mean ± sd over seeds 1–3, Egressy Table 2 in brackets:

| Rung | Seeds 1 / 2 / 3 | Mean ± sd | Paper |
|---|---|---|---|
| R0 | 38.85 / 40.65 / 26.17 | 35.23 ± 7.89 | [28.70 ± 1.13] |
| R1 | 56.49 / 55.00 / 47.64 | 53.04 ± 4.74 | [46.79 ± 4.97] |
| R2 | 58.38 / 58.37 / 64.66 | 60.47 ± 3.63 | [56.85 ± 2.64] |
| R3 | 46.71 / 65.28 / 54.98 | 55.66 ± 9.30 | [57.15 ± 4.99] |

- Val-tuned threshold F1 (means): 36.0 / 54.0 / 62.0 / 61.1.
- Test PR-AUC (means): 27.9 / 52.9 / 60.1 / 58.5.
- Best epochs: R0 88–98; R1–R3 28–72.

Reading:
- The ladder shape matches the paper.
- R1–R3 are within about 1.4 of the paper's sd, so the AML pipeline counts as validated by the
  2026-11-08 rule.
- R0's seed-to-seed spread (26–41), together with the FarmShare R0 replicate (31.83), supports the
  FarmShare log entry: the R0 gap is largely run-to-run variance, on top of the borrowed Altman baseline.
- R3 varies a lot between seeds, so make no R3-vs-R2 claim without more seeds.

## 2. Copy the Sherlock runs to FarmShare (the user runs this; it needs Sherlock 2FA)

Only seeds 2–3 are new; seed 1 and the synthetic artifacts were already copied. Pull into a staging
folder first:

```bash
mkdir -p ~/sherlock_import/runs ~/sherlock_import/logs
rsync -avP kde2804@dtn.sherlock.stanford.edu:'/scratch/users/kde2804/CS224W/runs/amlhis_R*_L2_h64_s[23]' ~/sherlock_import/runs/
rsync -avP kde2804@dtn.sherlock.stanford.edu:/scratch/users/kde2804/CS224W/logs/ ~/sherlock_import/logs/
```

Then, after the check in §1, move them into `runs/` and log the import.

On Sherlock, `$SCRATCH` is purged after 90 days without modification, so FarmShare should hold the kept
copy of everything.

## 3. Next steps for FarmShare

1. §1 and §2: resolve the duplicate, import the runs, log it.
2. Rerun `analysis/aml_table.py` on seeds 1–3. Update milestone report §5.3, which currently has the
   seed-1 table only, with the 3-seed table and the reading above.
3. Continue the FarmShare plan already in the log: the R3 L7 depth test (jobs 1957261–1957263), then
   finishing the report. Ask the user about the course's milestone format if it is still unknown.

## 4. Working with this user

- **The user commits and pushes themselves.** Never commit or push unless asked.
- **The user wants momentum** but asks conceptual questions along the way. Answer them plainly; their
  questions led to the AUROC metric and the PR curves.
- **Ask before deleting or cancelling anything.** The user once denied a deletion.
- **Interactive jobs:** an interactive `srun` from the agent's shell dies when the session ends. Use
  batch jobs and resumable scripts for anything long.

## 5. Sherlock leftovers (for the user)

- `/home/groups/mberger1/kde2804/amlinterp`: an old copy of the environment in the group folder. The
  user does not want group folders used. Delete it when convenient; the agent's `rm` was declined.
- `$HOME/envs/amlinterp/` (6.7 GB): the Sherlock environment. Keep it if Sherlock may be needed again.
- `/scratch/users/kde2804/refs/`: the IBM Multi-GNN clone and the paper texts. Re-download elsewhere if
  needed.
- Back on Sherlock, an agent must first read `/etc/agents/AGENTS.md` (Sherlock rule 12) and follow its
  rules: no compute on login nodes, no `--exclude`, `dev` only via interactive `srun`.
