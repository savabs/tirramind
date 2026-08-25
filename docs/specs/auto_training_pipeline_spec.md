---
title: "Spec: Fully Automated GNN Training Pipeline"
tags:
  - doc/spec
  - phase/auto-pipeline
  - topic/training
  - topic/kaggle
  - topic/autonomic-workflow
  - layer/learning
---

# Spec: Fully Automated GNN Training Pipeline

Research: [[auto_training_pipeline]]
Task: [[auto_training_pipeline_task]]

## Goal

Replace the 4–6 manual steps between Kaggle training runs with a fully automated pipeline.
Human only intervenes on structural halts (GitHub Issue) and data corruption.
Every fast-path config change runs without human touch.

---

## Files Affected

| File | Action | Priority |
|---|---|---|
| `scripts/pipeline_orchestrator.py` | CREATE | P0 |
| `scripts/sync_training_state.py` | CREATE | P0 |
| `.github/workflows/training_monitor.yml` | CREATE | P0 |
| `notebooks/tirramind-h-g/kernel-metadata.json` | CREATE | P0 |
| `[[kaggle_runbook]]` | UPDATE — add new workflow | P1 |

No changes to: `trainer.py`, `auto_improve.py`, `retrain_gnn.py`, `auto_research.py`.
Those components are already complete. This spec layers orchestration on top of them.

---

## Implementation Steps

### AP.1 — Verify GitHub token write access from Kaggle

**What:** Confirm `git push` works from inside a Kaggle kernel using `GITHUB_TOKEN` secret.

**How:**
In the Kaggle notebook (cell 5 or a temp test cell):
```python
import subprocess, os
token = os.environ.get("GITHUB_TOKEN", "")
result = subprocess.run(
    ["git", "remote", "set-url", "origin",
     f"https://{token}@github.com/savabs/tirramind.git"],
    capture_output=True
)
result2 = subprocess.run(
    ["git", "push", "origin", "HEAD:training-state", "--force"],
    capture_output=True, text=True
)
print(result2.stdout, result2.stderr)
```

**Exit condition:** push succeeds OR we know we need a write-scoped token.
**If fails:** add `GITHUB_WRITE_TOKEN` secret on Kaggle with `Contents: write` scope.
This must be verified before AP.3 (GitHub commit in orchestrator).

**Files changed:** None (this is a verification step).

---

### AP.2 — Create `scripts/pipeline_orchestrator.py`

**What:** The single Kaggle entry point. Replaces notebook cell 11.

**Interface:**
```bash
python scripts/pipeline_orchestrator.py \
  --checkpoint-dir .tirra_pipeline/checkpoints \
  --db-path .tirra_pipeline/pipeline.db \
  --knowledge-dir knowledge \
  --block-size 5 \          # epochs per training block
  --total-budget-hours 11 \ # stop 1h before Kaggle 12h limit
  --device cuda \
  --config-file .tirra_pipeline/checkpoints/next_config.json  # optional, from prev session
```

**Internal state machine:**
```python
class PipelineState(enum.Enum):
    TRAINING_BLOCK = "training_block"
    IMPROVING      = "improving"
    CONFIG_CHANGED = "config_changed"
    STRUCTURAL_HALT = "structural_halt"
    COLLAPSE_HALT   = "collapse_halt"
    SESSION_END     = "session_end"
```

**Core loop (pseudocode):**
```python
session_start = time.time()
current_config_file = args.config_file  # from previous session or None
latest_epoch = find_latest_checkpoint(args.checkpoint_dir)

while time_remaining(session_start, args.total_budget_hours) > 30 * 60:  # 30 min headroom
    block_end_epoch = latest_epoch + args.block_size

    # Build retrain command
    cmd = build_retrain_cmd(
        epochs=block_end_epoch,
        resume=latest_epoch,
        config_file=current_config_file,
        ...args
    )
    retcode = subprocess.call(cmd)

    if retcode != 0:
        # OOM or crash — write halt record, break
        write_halt_record("crash", cmd, retcode)
        break

    latest_epoch = find_latest_checkpoint(args.checkpoint_dir)

    # Check for model collapse
    if is_collapsed(args.checkpoint_dir / "metrics.jsonl"):
        write_halt_record("collapse", ...)
        break

    # Run auto_improve
    ai_retcode = subprocess.call([
        sys.executable, "scripts/auto_improve.py",
        "--checkpoint-dir", args.checkpoint_dir,
        "--knowledge-dir", args.knowledge_dir,
        "--no-watch",
    ])

    if ai_retcode == 2:   # structural
        state = PipelineState.STRUCTURAL_HALT
        break
    elif ai_retcode == 1: # config changed
        current_config_file = args.checkpoint_dir / "next_config.json"
        state = PipelineState.CONFIG_CHANGED
    else:                  # improving
        current_config_file = None
        state = PipelineState.IMPROVING

# Always sync state regardless of how we exited
sync_state_to_github(state, latest_epoch)
write_session_summary(state, latest_epoch, ...)
```

**Collapse detection (`is_collapsed`):**
- Load last 3 epoch records from `metrics.jsonl`
- If mean IC (return loss as proxy) < -0.05 for 3 consecutive epochs: True
- If total loss > 10x the first epoch's total loss: True
- Otherwise: False

**`find_latest_checkpoint`:** glob `epoch_*.pt`, extract numbers, return max.

**`sync_state_to_github`:**
```python
def sync_state_to_github(state, epoch, checkpoint_dir, knowledge_dir):
    # Stage lightweight state files
    files = [
        checkpoint_dir / "metrics.jsonl",
        checkpoint_dir / "next_config.json",
        knowledge_dir / "improvement_history.jsonl",
    ]
    existing = [f for f in files if f.exists()]

    # Write session summary
    summary_path = knowledge_dir / f"session_summary_{datetime.now():%Y%m%d_%H%M}.md"
    summary_path.write_text(build_session_summary(state, epoch))
    existing.append(summary_path)

    # Git operations
    subprocess.run(["git", "add"] + [str(f) for f in existing])
    subprocess.run(["git", "commit", "-m",
        f"training: epoch {epoch} | state={state.value}"])
    subprocess.run(["git", "push", "origin",
        "HEAD:training-state", "--force"])
```

**Key constraints:**
- Never import torch in this script — it's the outer shell, not the training code.
- Use `subprocess.call()`, not `subprocess.run()` — we want stdout/stderr to stream live.
- Time budget check uses `time.monotonic()` — wall clock, not CPU time.
- On any unexpected exception: write halt record, sync, re-raise.

**Exit codes:**
- `0` — session ended normally (budget exhausted or improving)
- `1` — config changed, state synced (cross-session continuation expected)
- `2` — structural halt (human needed)
- `3` — collapse detected (human needed)
- `4` — crash (retrain exited non-zero)

---

### AP.3 — Create `scripts/sync_training_state.py`

**What:** A standalone script for uploading state files to `tirramind-data` Kaggle dataset.
Called at end of session OR manually from the laptop after downloading.

**Interface:**
```bash
python scripts/sync_training_state.py \
  --checkpoint-dir .tirra_pipeline/checkpoints \
  --knowledge-dir knowledge \
  --dataset savabs/tirramind-data \
  --message "epoch 27 | dt_dominance → return_weight=2.0"
```

**What it uploads (new dataset version):**
- `metrics.jsonl` — training history
- `next_config.json` — recommendations for next run
- `improvement_history.jsonl` — pattern history

**What it does NOT upload (human manages):**
- `epoch_*.pt` — still manual (too large for automated CI budget)
- `pipeline.db` — still manual (309 MB, manual cadence)

**Implementation:**
```python
import subprocess, json, tempfile, shutil
from pathlib import Path

def create_dataset_version(checkpoint_dir, knowledge_dir, dataset, message):
    # Copy state files to a temp dir with the tirramind-data structure
    tmpdir = Path(tempfile.mkdtemp()) / ".tirra_pipeline"
    tmpdir.mkdir()
    (tmpdir / "checkpoints").mkdir()

    for fname in ["metrics.jsonl", "next_config.json"]:
        src = checkpoint_dir / fname
        if src.exists():
            shutil.copy(src, tmpdir / "checkpoints" / fname)

    for fname in ["improvement_history.jsonl"]:
        src = knowledge_dir / fname
        if src.exists():
            shutil.copy(src, tmpdir / fname)

    # Write dataset-metadata.json
    meta = {"title": "tirramind-data", "id": dataset, "licenses": [{"name": "CC0-1.0"}]}
    (tmpdir.parent / "dataset-metadata.json").write_text(json.dumps(meta))

    result = subprocess.run(
        ["kaggle", "datasets", "version",
         "--path", str(tmpdir.parent),
         "--message", message],
        capture_output=True, text=True
    )
    return result
```

**Note on AP.1 vs AP.3:** AP.1 (GitHub token write) must succeed for AP.3 to matter.
If GitHub push works, state is already in the repo. This script handles Kaggle dataset sync
as a secondary/optional step (so the next session's cell 5 gets the updated state files
from `tirramind-data` without the human having to manually upload them).

**AP.3 is P2** if AP.1 (GitHub state push) works. The notebook's cell 5 can read state
from GitHub (already has a git clone) rather than from `tirramind-data`. Confirmed after AP.1.

---

### AP.4 — Create `notebooks/tirramind-h-g/kernel-metadata.json`

**What:** The Kaggle kernel metadata file that `kaggle kernels push` requires.
Currently this kernel is only managed through the Kaggle UI. We need the metadata
locally so GitHub Actions can push new versions.

**How to get the current metadata:**
```bash
kaggle kernels pull deeperisbetter/tirramind-h-g --path notebooks/tirramind-h-g/ --metadata
```
This creates `kernel-metadata.json`. Commit it to the repo.

**Fields to verify in metadata:**
- `enable_gpu: true`
- `dataset_sources: ["savabs/tirramind-data"]`
- `kernel_sources: []` (code from GitHub, not kernel sources)
- `language: "python"`
- `kernel_type: "notebook"`

**Exit condition:** file exists at `notebooks/tirramind-h-g/kernel-metadata.json` and
`kaggle kernels push --path notebooks/tirramind-h-g/` succeeds (dry-run or actual).

---

### AP.5 — Create `.github/workflows/training_monitor.yml`

**What:** GitHub Actions workflow triggered on push to `training-state` branch.
Reads metrics, decides: alert (structural) or trigger next Kaggle run (fast-path).

**Trigger:** `push` to branch `training-state`.

**Jobs:**

**Job 1: parse_state**
```yaml
- name: Read session summary
  run: |
    LATEST=$(ls knowledge/session_summary_*.md 2>/dev/null | sort | tail -1)
    if [ -z "$LATEST" ]; then echo "no_summary"; exit 0; fi
    cat $LATEST
    STATE=$(grep "^state:" $LATEST | cut -d: -f2 | tr -d ' ')
    echo "state=$STATE" >> $GITHUB_OUTPUT
    EPOCH=$(grep "^epoch:" $LATEST | cut -d: -f2 | tr -d ' ')
    echo "epoch=$EPOCH" >> $GITHUB_OUTPUT
```

**Job 2: alert_structural** (runs if state == structural_halt or collapse_halt)
```yaml
- name: Open GitHub Issue
  if: steps.parse_state.outputs.state == 'structural_halt' || ...
  uses: actions/github-script@v7
  with:
    script: |
      const fs = require('fs');
      const trigger = fs.readdirSync('knowledge')
        .filter(f => f.startsWith('session_summary_')).sort().pop();
      const body = fs.readFileSync(`knowledge/${trigger}`, 'utf8');
      github.rest.issues.create({
        owner: context.repo.owner, repo: context.repo.repo,
        title: `Training halt: ${process.env.STATE} at epoch ${process.env.EPOCH}`,
        body: body,
        labels: ['training-halt']
      });
```

**Job 3: trigger_next_run** (runs if state == session_end, config_changed, or improving)
```yaml
- name: Trigger new Kaggle run
  if: steps.parse_state.outputs.state != 'structural_halt'
  env:
    KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
    KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
  run: |
    pip install kaggle
    # Pull the kernel metadata (creates a local copy of the notebook)
    kaggle kernels pull deeperisbetter/tirramind-h-g \
      --path notebooks/tirramind-h-g/ --metadata
    # Update the title to force a new version
    python3 -c "
    import json, datetime
    m = json.load(open('notebooks/tirramind-h-g/kernel-metadata.json'))
    m['title'] = f'tirramind-h-g auto epoch {os.environ.get(\"EPOCH\",\"?\")} {datetime.date.today()}'
    json.dump(m, open('notebooks/tirramind-h-g/kernel-metadata.json','w'), indent=2)
    "
    # Push = new version = Kaggle queues it
    kaggle kernels push --path notebooks/tirramind-h-g/
```

**Required GitHub Secrets:**
- `KAGGLE_USERNAME` — Kaggle username (`deeperisbetter`)
- `KAGGLE_KEY` — Kaggle API key (from kaggle.json)
- These are separate from `GITHUB_TOKEN` (auto-provided by Actions)

**Required Kaggle Secrets (already set):**
- `GITHUB_TOKEN` — for reading code (existing, currently read-only)
- `GITHUB_WRITE_TOKEN` — for writing state back (new, if AP.1 requires it)

---

### AP.6 — Update Kaggle notebook (cells 5 and 11)

**Cell 5 additions** (after existing setup):
```python
# Pull training state files from GitHub training-state branch (if exists)
import subprocess
result = subprocess.run(
    ["git", "fetch", "origin", "training-state"],
    capture_output=True, cwd=REPO_DIR
)
if result.returncode == 0:
    subprocess.run(
        ["git", "checkout", "origin/training-state", "--",
         "knowledge/metrics.jsonl",
         "knowledge/next_config.json",        # wait — this should be in checkpoints/
         "knowledge/improvement_history.jsonl"],
        cwd=REPO_DIR
    )
    print("Pulled training state from training-state branch")
else:
    print("No training-state branch yet — starting fresh")
```

**Note on paths:** `next_config.json` lives in `checkpoint_dir` (`.tirra_pipeline/checkpoints/`),
not in `knowledge/`. The cell 5 copy should reflect that. Confirm exact paths in AP.2.

**Cell 11 replacement:**
```python
# BEFORE (old):
# subprocess.call(["python", "scripts/retrain_gnn.py", "--epochs", "30", ...])

# AFTER (new):
import subprocess, os
cmd = [
    "python", "scripts/pipeline_orchestrator.py",
    "--checkpoint-dir", CKPT_DIR,
    "--db-path", DB_PATH,
    "--knowledge-dir", str(Path(REPO_DIR) / "knowledge"),
    "--block-size", "5",
    "--total-budget-hours", "11",
    "--device", "cuda",
]
# Pass next_config if it was pulled from training-state branch
next_cfg = Path(CKPT_DIR) / "next_config.json"
if next_cfg.exists():
    cmd += ["--config-file", str(next_cfg)]

retcode = subprocess.call(cmd, cwd=REPO_DIR)
print(f"Pipeline orchestrator exited: {retcode}")
```

This cell requires no manual editing between runs. The orchestrator handles everything.

---

### AP.7 — Write `session_summary_*.md` format

**What:** A lightweight structured markdown file written at end of each session.
Used by GitHub Actions to parse state without installing Python.

**Format:**
```markdown
---
state: session_end
epoch: 27
pattern: dt_dominance
flag_overrides: {"return_weight": 2.0}
blocks_completed: 4
session_duration_hours: 10.8
---

# Training Session Summary

**Date:** 2026-05-09 14:32
**Epochs this session:** 23 → 27
**Final pattern:** dt_dominance
...
```

The `state:` field in frontmatter is machine-readable by the GitHub Actions `grep` command.

---

## Edge Cases

| Edge case | Handling |
|---|---|
| Kaggle session OOM mid-block | `retrain_gnn.py` exits non-zero → orchestrator writes crash halt, syncs, exits code 4 |
| `metrics.jsonl` doesn't exist yet | `auto_improve.py` returns 0 (no_metrics), orchestrator continues |
| `next_config.json` from previous session is stale | orchestrator checks `based_on_epoch` field; if < current epoch, discards it |
| GitHub push fails from Kaggle | Orchestrator logs error but continues training (state sync is best-effort) |
| Kaggle API push fails from GH Actions | Job fails with error; user manually runs Kaggle; not a blocker |
| Two sessions run simultaneously | `metrics.jsonl` gets concurrent appends. Use file lock or rely on Kaggle API's single-run guarantee |
| GitHub Actions loop (keeps triggering) | Cap at 3 auto-triggers per 24h via workflow condition check |

---

## Testing Plan

| Test | How |
|---|---|
| Orchestrator block loop | Mock `subprocess.call` to return 0, inject fake `metrics.jsonl`, verify loop iterates |
| Orchestrator structural halt | Mock auto_improve to return 2, verify halt record written and loop exits |
| Collapse detection | Feed metrics.jsonl with 3 records all loss > 10x initial, verify `is_collapsed()` returns True |
| State sync | Mock subprocess.run(git), verify correct files staged and message format |
| GH Actions parse_state | Feed mock session_summary with known state, verify correct job branches |
| Kaggle kernel push | Manual test: pull metadata locally, bump title, push, verify new version appears in Kaggle UI |
| End-to-end (staging) | Run orchestrator with `--total-budget-hours 0.1` and `--block-size 1`, verify one block runs and state syncs |

---

## Related

- [[auto_training_pipeline]] — research
- [[auto_training_pipeline_task]] — task file
- [[auto_ml_researcher_task]] — slow-path (structural) research loop
- [[kaggle_runbook]] — Kaggle infrastructure reference
