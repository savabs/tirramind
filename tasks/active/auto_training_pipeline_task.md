---
title: "Task: Fully Automated GNN Training Pipeline"
tags:
  - doc/task
  - phase/auto-pipeline
  - topic/training
  - topic/kaggle
  - topic/autonomic-workflow
  - layer/learning
  - status/active
---

# Task: Fully Automated GNN Training Pipeline

Status: active
Research: [[auto_training_pipeline]]
Spec: [[auto_training_pipeline_spec]]

## Exit Condition

Starting a Kaggle training run, waiting for it to finish, and triggering the next run all happen
without human action. The human's only interaction is:
1. Reviewing GitHub Issues opened on structural halts.
2. Applying the fix via `apply-training-fix` skill.
3. Pushing — which automatically triggers the next Kaggle run.

---

## Steps

### Phase 0: Verification (no code yet)

- [ ] **AP.0.1** — Verify GitHub token write access from Kaggle
  - Add `Contents: write` scope to `GITHUB_TOKEN` Kaggle secret (or create `GITHUB_WRITE_TOKEN`)
  - Test `git push origin HEAD:training-state --force` from inside a Kaggle cell
  - **Exit condition:** push succeeds with 0 exit code. Log result in runbook.
  - **If fails:** document which token scope is needed and add new Kaggle secret.

- [ ] **AP.0.2** — Pull current kernel metadata from Kaggle API
  - `kaggle kernels pull deeperisbetter/tirramind-h-g --path notebooks/tirramind-h-g/ --metadata`
  - Inspect `kernel-metadata.json` — confirm `enable_gpu`, `dataset_sources`, `id` fields
  - Commit to repo at `notebooks/tirramind-h-g/kernel-metadata.json`
  - **Exit condition:** file committed, `kaggle kernels push --path notebooks/tirramind-h-g/` succeeds (creates new version visible in Kaggle UI)

- [ ] **AP.0.3** — Confirm session time budget detection approach
  - In a Kaggle cell, print `os.environ` — identify which env vars are available
  - Confirm `time.monotonic()` is reliable for 11-hour session tracking
  - **Exit condition:** know exactly how orchestrator will track elapsed time

---

### Phase 1: `pipeline_orchestrator.py`

- [ ] **AP.1.1** — Write `find_latest_checkpoint(checkpoint_dir)` helper
  - Glob `epoch_*.pt`, parse numbers, return max or 0 if none
  - Unit test: empty dir → 0, dir with epoch_022.pt → 22

- [ ] **AP.1.2** — Write `is_collapsed(metrics_path)` helper
  - Load last 3 records from metrics.jsonl
  - Returns True if return_loss mean < -0.05 for all 3 OR total_loss > 10× first epoch total
  - Unit test: normal metrics → False, 3 negative-loss records → True

- [ ] **AP.1.3** — Write `build_retrain_cmd(epochs, resume, config_file, **kwargs)` helper
  - Returns `["python", "scripts/retrain_gnn.py", "--epochs", str(epochs), ...]`
  - Applies all standard flags (device, gdelt-frac, max-windows, skip-eval, backup, etc.)
  - If `config_file` provided and exists: append `["--config-file", str(config_file)]`

- [ ] **AP.1.4** — Write `write_halt_record(reason, epoch, details, knowledge_dir)` helper
  - Writes `knowledge/session_summary_{ts}.md` with frontmatter `state: {reason}_halt`
  - Includes: epoch, last pattern, reason, details

- [ ] **AP.1.5** — Write `sync_state_to_github(state, epoch, files, message)` helper
  - `git add` the given files
  - `git commit -m "training: epoch {epoch} | state={state}"`
  - `git push origin HEAD:training-state --force`
  - Returns success bool (logs error but does NOT raise on failure)

- [ ] **AP.1.6** — Assemble main loop with `--total-budget-hours` time gate
  - Runs `build_retrain_cmd` → `subprocess.call` for each block
  - Runs `auto_improve.py --no-watch` after each block
  - Applies next_config.json on exit_code 1
  - Breaks on exit_code 2 (structural) or collapse or crash
  - Falls through to sync on budget exhaustion
  - **Exit condition:** `python3 -m py_compile scripts/pipeline_orchestrator.py` passes

- [ ] **AP.1.7** — Add CLI (argparse) and `__main__` block
  - Flags: `--checkpoint-dir`, `--db-path`, `--knowledge-dir`, `--block-size` (default 5),
    `--total-budget-hours` (default 11), `--device` (default cuda), `--config-file` (optional)
  - **Exit condition:** `python3 scripts/pipeline_orchestrator.py --help` prints usage

- [ ] **AP.1.8** — Write unit tests for AP.1.1–AP.1.5 helpers
  - Use `tempfile.mkdtemp()` for all file I/O — no real filesystem mutations
  - Mock `subprocess.call` and `subprocess.run`
  - 1 happy path + 2 failure cases per helper (per leaf-node test rule)
  - **Exit condition:** `python3 -m pytest tests/test_pipeline_orchestrator.py -x -q` passes

---

### Phase 2: GitHub Actions workflow

- [ ] **AP.2.1** — Create `.github/workflows/training_monitor.yml`
  - Trigger: `push` to branch `training-state`
  - Job 1: parse `session_summary_*.md` frontmatter → extract `state` and `epoch`
  - Job 2 (if structural/collapse): open GitHub Issue with session summary body
  - Job 3 (if session_end/config_changed/improving): trigger Kaggle kernel push
  - Cap at 3 auto-triggers per 24h (check recent workflow runs count)
  - **Exit condition:** manually push a fake `training-state` commit, verify correct job runs

- [ ] **AP.2.2** — Add GitHub Secrets for Kaggle API
  - Add `KAGGLE_USERNAME` = `deeperisbetter` to repo secrets
  - Add `KAGGLE_KEY` = value from `~/.kaggle/kaggle.json` to repo secrets
  - **Exit condition:** GH Actions job can run `kaggle kernels pull` without auth error

- [ ] **AP.2.3** — Test end-to-end trigger
  - Push a mock `session_summary_*.md` with `state: session_end` to `training-state`
  - Verify: GitHub Actions runs, Kaggle API push fires, new kernel version appears on Kaggle
  - **Exit condition:** new kernel version visible on `deeperisbetter/tirramind-h-g` page

---

### Phase 3: Kaggle notebook update

- [ ] **AP.3.1** — Update notebook cell 5: pull training-state branch files
  - Add git fetch + checkout of `metrics.jsonl`, `next_config.json`, `improvement_history.jsonl`
  - If branch doesn't exist: log "fresh start" and continue
  - **Exit condition:** run in notebook manually, verify files appear in `/kaggle/working/`

- [ ] **AP.3.2** — Update notebook cell 11: replace retrain command with orchestrator
  - Replace `retrain_gnn.py` subprocess with `pipeline_orchestrator.py` subprocess
  - Pass `--config-file` if `next_config.json` exists in working dir
  - **Exit condition:** cell runs without error on Kaggle (can test with `--total-budget-hours 0.1 --block-size 1`)

- [ ] **AP.3.3** — Update notebook cell 13: adjust download filter
  - Cell 13 currently copies `if epoch_num > N`. This should now be dynamic:
    read `metrics.jsonl` to find last epoch, copy all newer than last downloaded.
  - **Exit condition:** no manual cell editing needed between runs

---

### Phase 4: State persistence (optional, after AP.0.1 resolution)

- [ ] **AP.4.1** — Create `scripts/sync_training_state.py`
  - Uses `kaggle datasets version` to push state files to `tirramind-data`
  - Called from orchestrator at session end if GitHub push succeeds (belt + suspenders)
  - CLI: `python scripts/sync_training_state.py --checkpoint-dir ... --message "..."`
  - **Priority:** P2 — only needed if GitHub state branch is insufficient for Kaggle session pickup

---

### Phase 5: Update runbook

- [ ] **AP.5.1** — Update `docs/memory/kaggle_runbook.md`
  - Add new section: "Automated Pipeline Workflow"
  - Document: which secrets are needed, what to do on structural halt GitHub Issues,
    how to manually trigger if GitHub Actions fails, how to override orchestrator flags
  - Remove: instructions for manual cell 11 editing (now obsolete)

---

## Human Interaction Points (the only times human acts)

| Event | GitHub signal | Human action |
|---|---|---|
| Structural halt | GitHub Issue opened with tag `training-halt` | Read Issue → run `apply-training-fix` skill → push fix → Actions auto-resumes |
| Model collapse | Same Issue flow | Diagnose with `gnn_attention_diagnostic.py` → fix → push |
| Kaggle OOM | Session summary in repo with `state: crash` | Reduce `--max-windows` in orchestrator defaults or add flag override |
| Dataset corruption | Manual detection (no automation here) | Manual diagnosis |
| First-ever run | Manual | Start Kaggle session once; after that it's self-triggering |

---

## Prerequisites (before AP.1 coding begins)

- [x] `trainer.py` writes `metrics.jsonl` ✅
- [x] `auto_improve.py` 7-pattern decision tree ✅
- [x] `retrain_gnn.py --config-file` flag ✅
- [ ] AP.0.1 — GitHub token write access verified
- [ ] AP.0.2 — kernel-metadata.json committed

AP.1 coding should not begin until AP.0.1 and AP.0.2 are complete.
The orchestrator's sync function depends on knowing the exact push command that works.

---

## Notes

- `pipeline_orchestrator.py` is a **leaf node**: it calls other scripts but nothing imports it.
  Safe to develop and test in isolation with mocked subprocesses.
- The GitHub Actions `training_monitor.yml` loop has a 3-trigger-per-24h cap to prevent
  runaway billing or infinite loops from a bad pattern that keeps returning non-structural codes.
- All new code is Python stdlib only (`subprocess`, `json`, `time`, `pathlib`, `argparse`).
  No new dependencies.
- The `training-state` git branch is a special single-commit branch (force-pushed each session).
  It does not pollute `main` history. The training timeline is readable via GitHub UI.

---

## Related

- [[auto_training_pipeline]] — research
- [[auto_training_pipeline_spec]] — spec
- [[auto_ml_researcher_task]] — slow-path research loop (structural halts)
- [[kaggle_runbook]] — Kaggle infrastructure
- [[quant_training_ground]] — roadmap (where this fits in Phase 41 → 48 journey)
