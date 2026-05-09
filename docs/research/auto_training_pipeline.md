---
title: "Research: Fully Automated GNN Training Pipeline"
tags:
  - doc/research
  - phase/auto-pipeline
  - topic/training
  - topic/autonomic-workflow
  - topic/kaggle
  - layer/learning
  - status/active
---

# Research: Fully Automated GNN Training Pipeline

**Problem:** The self-improvement loop (trainer → metrics.jsonl → auto_improve → next_config.json)
works, but the human still performs 4–6 manual steps between every training block:
download checkpoints, run auto_improve locally, edit Kaggle notebook cell 11, click "Run All".

**Goal:** The human's only job is to ensure things don't break (structural escalations,
data corruption, model collapse). Every config-change iteration runs without human touch.

---

## Current State: What Exists

### What's built and working
| Component | Location | Status |
|---|---|---|
| `trainer.py` writes `metrics.jsonl` per epoch | `agent/models/gnn/trainer.py` | ✅ Done |
| `auto_improve.py` decision tree (7 patterns) | `scripts/auto_improve.py` | ✅ Done |
| `retrain_gnn.py --config-file` flag | `scripts/retrain_gnn.py` | ✅ Done |
| `retrain_gnn.py --auto-improve` trigger | `scripts/retrain_gnn.py` | ✅ Done |
| `auto_research.py` slow-path trigger | `scripts/auto_research.py` | ✅ Done |
| `research-training-issue` skill | `.claude/skills/research-training-issue/` | ✅ Done |
| `apply-training-fix` skill | `.claude/skills/apply-training-fix/` | ✅ Done |

### What's still manual
| Manual step | Friction | Automation target |
|---|---|---|
| Download checkpoint from Kaggle | Requires human browser action | Kaggle output auto-upload to dataset |
| Run `auto_improve.py` locally | Must remember to do it | Runs inside Kaggle session |
| Edit Kaggle notebook cell 11 (flags) | Error-prone, requires notebook version bump | Replaced by `pipeline_orchestrator.py` |
| Start new Kaggle session | Requires human click | GitHub Actions → Kaggle API |
| Upload new epoch checkpoints to `tirramind-data` | Zip, upload, wait | Automated sync script |

---

## Infrastructure Constraints (hard limits that shape design)

1. **Laptop = code only.** 7.6 GB RAM, GTX 1650 (4 GB VRAM). Cannot run training.
2. **Kaggle session budget.** 12-hour GPU sessions. P100 or T4. After 12h, session dies.
3. **Kaggle state is ephemeral.** Working directory (`/kaggle/working/`) is lost after session.
   Only `/kaggle/input/` (datasets) and outputs explicitly downloaded persist.
4. **Code delivery via GitHub.** Notebook clones `savabs/tirramind` on startup via `GITHUB_TOKEN` secret.
5. **Data delivery via `tirramind-data`.** Kaggle dataset `savabs/tirramind-data` (private).
   Currently contains: `pipeline.db`, `epoch_*.pt` checkpoints.
6. **Kaggle API.** `kaggle kernels push` creates a new kernel version and queues it for execution.
   This is the mechanism for cross-session triggering.
7. **No server.** No always-on process. GitHub Actions is the only persistent compute outside Kaggle.

---

## The Two Loops

The full automation problem decomposes into two independent loops:

### Loop 1: Intra-Session (Within One Kaggle 12h Session)

This loop runs ENTIRELY on Kaggle, needs no external trigger mid-session.

```
pipeline_orchestrator.py (replaces Kaggle cell 11)
├── SETUP: load next_config.json from tirramind-data (if exists)
├── LOOP (while GPU budget remaining):
│   ├── run retrain_gnn.py for BLOCK_SIZE epochs (e.g. 5)
│   ├── run auto_improve.py → pattern + exit_code
│   ├── exit_code 0 (improving): continue with same config
│   ├── exit_code 1 (config_change): apply next_config.json → continue
│   └── exit_code 2 (structural): break → STOP, alert
└── FINALLY: sync state to tirramind-data + commit metrics to GitHub
```

**BLOCK_SIZE = 5 epochs.** Each block takes ~20–30 min on P100.
In a 12h session with 5-epoch blocks: up to ~20 decision cycles.

**Why not run all epochs in one shot?**
Because we want auto_improve to react to patterns mid-session (e.g., catch divergence at epoch 3,
not waste 7 more epochs). 5-epoch blocks balance responsiveness vs. overhead.

### Loop 2: Inter-Session (Kaggle → GitHub → Kaggle)

This loop bridges Kaggle sessions. State must persist.

```
Kaggle session ends
↓
sync_training_state.py:
  - uploads metrics.jsonl, next_config.json, improvement_history.jsonl
    as new version of tirramind-data dataset
  - commits metrics.jsonl summary to GitHub (lightweight)

GitHub Actions (training_monitor.yml):
  - triggered by commit push from Kaggle
  - reads latest metrics.jsonl
  - if structural → opens GitHub issue → STOP (human reads, applies fix, pushes)
  - if exit_code 0 or 1 → triggers new Kaggle session via Kaggle API
    (kaggle kernels push → new version → queued run)

New Kaggle session:
  - cell 5 copies next_config.json from tirramind-data to working dir
  - pipeline_orchestrator.py reads it at start
  - resumes from highest epoch_*.pt in tirramind-data
```

---

## Design Decisions

### Decision 1: `pipeline_orchestrator.py` replaces Kaggle notebook cell 11

**Option A** (chosen): A single orchestrator script on Kaggle handles the full training loop.
The notebook becomes: setup (cells 1–9) → `python scripts/pipeline_orchestrator.py` → done.

**Option B** (rejected): Keep retrain_gnn.py as entry point, add an outer bash loop in the notebook.
Rejected because: bash loops in Jupyter are fragile, can't handle complex state, hard to test locally.

**Option C** (rejected): GitHub Actions orchestrates each epoch individually.
Rejected because: inter-epoch latency would be 2–5 minutes per epoch (GH Actions startup time).
At 5 min/epoch overhead, this is 50% wasted GPU time.

### Decision 2: Kaggle → GitHub commit for state sync

After each session, `pipeline_orchestrator.py` runs:
```bash
git add knowledge/metrics.jsonl knowledge/next_config.json knowledge/improvement_history.jsonl
git commit -m "training: epoch {N} metrics ({pattern})"
git push
```
Using the `GITHUB_TOKEN` secret already in the notebook.

**Why commit to GitHub (not just Kaggle dataset)?**
- GitHub commit is the trigger for GitHub Actions → enables cross-session monitoring.
- Lightweight files (JSON) are fine in git. Binary checkpoints stay in Kaggle dataset.
- Gives a visible audit trail of training progress in the repo.

**Risk:** repo history fills with training commits. Mitigate: use `--force-with-lease` on a
`training-state` branch, rebased to main. This keeps one training-state commit, not thousands.

### Decision 3: Cross-session trigger via `kaggle kernels push` from GitHub Actions

**Mechanism:**
1. GitHub Actions workflow checks out repo.
2. Reads `kernel-metadata.json` (stored at `notebooks/tirramind-h-g/kernel-metadata.json`).
3. Updates `title` field with current epoch count and date.
4. Runs `kaggle kernels push --path notebooks/tirramind-h-g/`.
5. Kaggle queues the kernel for execution (new version).

**Requires:** `KAGGLE_USERNAME` and `KAGGLE_KEY` as GitHub Actions secrets.

**Alternative:** Kaggle scheduled notebooks (built-in scheduling UI).
Rejected because: schedule doesn't read dynamic parameters (epoch count, config).
The Kaggle API push approach gives us full parametric control.

### Decision 4: State files in `tirramind-data` + GitHub

State persistence layer:
| File | In `tirramind-data`? | In GitHub? | Why |
|---|---|---|---|
| `metrics.jsonl` | No (large over time) | Yes (branch) | GitHub trigger, lightweight |
| `next_config.json` | Yes | Yes | Kaggle reads on startup |
| `improvement_history.jsonl` | Yes | Yes | Decision context across sessions |
| `epoch_*.pt` | Yes | No | Binary, too large for git |
| `pipeline.db` | Yes | No | Binary, 309 MB |

`tirramind-data` dataset version is bumped at end of each session.
The notebook's cell 5 already copies from `/kaggle/input/tirramind-data/` — we add the new files.

### Decision 5: Human escalation = GitHub Issue

When `auto_improve.py` returns exit code 2 (structural):
1. `pipeline_orchestrator.py` writes `knowledge/structural_halt_<ts>.md` to explain the halt.
2. Commits to GitHub.
3. GitHub Actions opens a GitHub Issue with the trigger content.
4. User reviews → applies fix (via `apply-training-fix` skill) → pushes → Actions triggers next run.

The user's prompt is literally the GitHub Issue body. The Copilot skill is invoked in the VS Code
context. This closes the loop: Kaggle signals → GitHub Issue → Copilot fixes → Kaggle resumes.

---

## Safety Gates (Human Checkpoints)

The pipeline runs autonomously EXCEPT at these four gates:

| Gate | Trigger | Human action |
|---|---|---|
| **Structural halt** | `auto_improve` exit code 2 (all config options exhausted) | Review GitHub Issue, run `apply-training-fix` skill, push fix |
| **Model collapse** | Mean IC drops below -0.05 for 3+ consecutive epochs | Auto-detected by orchestrator; session stops with GitHub Issue |
| **Dataset corruption** | `pipeline.db` integrity check fails at session start | Manual diagnosis, can't be automated |
| **Kaggle OOM** | `retrain_gnn.py` exits with non-zero (CUDA OOM) | Reduce `--max-windows` or `--batch-size`; orchestrator logs the flag to try |

All other decisions (LR changes, weight rebalancing, GDELT fraction, ListNet tau) are autonomous.

---

## State Machine: Full Pipeline Lifecycle

```
State: TRAINING_BLOCK
  → (block complete, improving)       → TRAINING_BLOCK (same config)
  → (block complete, config_changed)  → TRAINING_BLOCK (new config from next_config.json)
  → (block complete, structural)      → STRUCTURAL_HALT
  → (collapse detected)               → COLLAPSE_HALT
  → (GPU budget < threshold)          → SESSION_END

State: SESSION_END
  → sync state to tirramind-data
  → commit metrics to GitHub
  → GitHub Actions triggers: new Kaggle session
  → transitions to TRAINING_BLOCK in new session

State: STRUCTURAL_HALT
  → write structural_halt_*.md
  → commit to GitHub
  → GitHub Actions opens Issue
  → WAITING_FOR_HUMAN

State: WAITING_FOR_HUMAN
  → human applies fix via skill
  → push to main
  → Actions triggers new Kaggle session
  → transitions to TRAINING_BLOCK

State: COLLAPSE_HALT
  → same as STRUCTURAL_HALT but with collapse metadata
```

---

## Components to Build

| Component | Type | Priority |
|---|---|---|
| `scripts/pipeline_orchestrator.py` | New script (Kaggle entry point) | P0 — core loop |
| `scripts/sync_training_state.py` | New script (state persistence) | P0 — cross-session |
| `.github/workflows/training_monitor.yml` | New GH Actions workflow | P0 — cross-session trigger |
| `notebooks/tirramind-h-g/kernel-metadata.json` | New file (kernel config for API push) | P0 — GH Actions needs it |
| Kaggle notebook cell 5 update | Edit notebook | P1 — add new state files to setup |
| Kaggle notebook cell 11 update | Edit notebook | P1 — replace retrain with orchestrator |
| Collapse detection in orchestrator | Logic | P1 — safety gate |
| `training-state` branch setup | Git | P2 — clean history |

---

## Open Questions (needs verification before implementation)

1. **Kaggle API push from GH Actions:** Does `kaggle kernels push` from a CI environment
   correctly queue the kernel for execution? Needs a test push.

2. **Git push from Kaggle session:** Can the Kaggle notebook do `git push` to GitHub
   using the `GITHUB_TOKEN` secret? The token currently only has `Contents: read`.
   Need to add `Contents: write` or use a separate `GITHUB_WRITE_TOKEN` secret.

3. **tirramind-data dataset write from Kaggle:** Can a Kaggle kernel write a new dataset
   version from within the session? The `kaggle datasets version` command requires kaggle.json,
   which the notebook doesn't currently have. Alternative: push state files to GitHub only
   (since dataset version is uploaded manually or from the laptop after downloading).

4. **Session time budget detection:** How do we know how much GPU time is remaining?
   Kaggle doesn't expose this directly. Can probe via `os.environ.get("KAGGLE_KERNEL_RUN_TYPE")`
   and start timer at session start. If we started at time T and 11h have elapsed, stop.

5. **Kernel-metadata.json format for `kaggle kernels push`:** Need to verify the exact
   fields required (`id`, `title`, `code_file`, `language`, `kernel_type`, `is_private`,
   `enable_gpu`, `dataset_sources`, `kernel_sources`). Must match the existing kernel.

---

## Related Prior Art in This Repo

- [[auto_ml_researcher]] — slow-path research loop (arXiv + GitHub search)
- [[autonomic_workflow_system]] — meta-level autonomous workflow maintenance
- [[phase41b_gnn_signal_extraction]] — the training problem this pipeline addresses
- [[quant_training_ground]] — roadmap context (Phase 41 → 47 → 40 → 48)

---

## Related

- [[auto_training_pipeline_spec]] — spec
- [[auto_training_pipeline_task]] — task file
- [[auto_ml_researcher_task]] — parallel slow-path research loop
- [[kaggle_runbook]] — Kaggle infrastructure details
