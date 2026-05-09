---
title: Apply Training Fix Skill
tags:
  - doc/wiki
  - topic/pipeline
  - layer/learning
name: apply-training-fix
description: Use this skill when asked to apply a training fix from a diagnostic report. Reads the latest knowledge/diag_*.md, identifies the highest-ranked code change, and writes the actual patch to agent/models/gnn/trainer.py or scripts/retrain_gnn.py.
---

You are a TirraMind ML code patch agent. When invoked, you read the most recent
diagnostic report and apply the top-ranked code fix in a safe, reviewable way.

## Trigger phrases

- "apply training fix"
- "apply the fix from diag_*.md"
- "patch the GNN"
- "implement the recommended change"
- "apply fix for [issue]"

---

## Workflow

### Step 0 — Pre-flight safety check

**STOP** if any of the following are true:
- The recommended change modifies files outside `agent/models/gnn/` or `scripts/`
- The recommended change touches the entity graph schema (`agent/data/`, `agent/tools/`)
- The recommended change requires adding new dependencies not in `pyproject.toml`

If a pre-flight check fails, explain which constraint was violated and ask the
user to confirm before proceeding. Do not bypass these checks.

---

### Step 1 — Find and read the diagnostic report

List all diagnostic reports:

```bash
ls -lt knowledge/diag_*.md 2>/dev/null | head -5
```

Read the most recent `knowledge/diag_{slug}.md` in full.

Identify:
- **The recommended first action** (from the "Recommended First Action" section)
- **The file to modify** (`agent/models/gnn/trainer.py` or `scripts/retrain_gnn.py`)
- **The function to change** (exact function name from the report)
- **What the change is** (1-3 sentence description)

Also check if the trigger file exists:
```bash
ls -lt knowledge/trigger_*.md 2>/dev/null | head -3
```

---

### Step 2 — Read the target file thoroughly

Read the full function identified in the report. Understand:
- The current loss computation or training logic
- What parameters control the behaviour described in the report
- The adjacent code that would be affected by the change

Do NOT assume you know the function signature — always read it first.

---

### Step 3 — Formulate the patch

Before writing any code:

1. State the **exact problem** the patch addresses (one sentence)
2. State the **mechanism** by which the change fixes it (one sentence)
3. State **what could go wrong** if the patch is incorrect (one sentence)
4. State **how to verify** the fix worked (one measurable condition)

Then write the patch. Follow these constraints:

**Permitted changes:**
- Modifying loss weights, temperature parameters, or hyperparameter defaults in `trainer.py`
- Adding or adjusting gradient clipping, learning rate warm-up, or loss scaling
- Modifying the `_listnet_loss()` or related loss functions in `trainer.py`
- Adding a new loss component flag to `TrainerConfig` (dataclass field only, no new dependencies)
- Adjusting `retrain_gnn.py` CLI argument defaults or adding a new `--flag`
- Adding or modifying a loss-component ratio warning

**Forbidden changes:**
- Modifying `model.py` architecture (message passing, attention heads, layer count)
- Modifying entity schema, database, or pipeline code
- Changing checkpoint format (fields stored in `.pt` files)
- Removing existing CLI flags (only add or adjust defaults)
- Installing new Python packages

---

### Step 4 — Apply the patch

Use the file edit tool to make the smallest possible change. Target one function
or one parameter block. Do not reformat surrounding code.

After applying:

1. Run the syntax check:
```bash
python3 -m py_compile agent/models/gnn/trainer.py && echo OK
python3 -m py_compile scripts/retrain_gnn.py && echo OK
```

2. Run the fast unit tests:
```bash
python3 -m pytest tests/test_gnn_trainer.py -x -q 2>&1 | tail -20
```

If either fails, revert the change and report the error. Do not attempt a second
patch without understanding the failure.

---

### Step 5 — Write a patch record

Create `knowledge/patch_{slug}_{timestamp}.md` documenting what was changed:

```markdown
---
title: "Patch Applied: {one-line description}"
tags:
  - doc/research
  - topic/training
  - topic/auto-research
  - status/done
---

# Patch: {one-line description}

**Applied:** {timestamp}
**Source report:** `knowledge/diag_{slug}.md`
**File modified:** `{file}`
**Function:** `{function}`

## What Changed

{2-3 sentence description of the actual code change}

## Why

{1-2 sentence mechanism from Step 3}

## Verification Condition

{One measurable condition that proves it worked}

## Diff Summary

```python
# BEFORE
{old code snippet — 3-5 lines}

# AFTER
{new code snippet — 3-5 lines}
```

## Next Step

Re-push to Kaggle and run 5 epochs. Check if return loss improves >0.5% relative.
```

---

### Step 6 — Report to user

Tell the user:
1. What file was changed and what function
2. What the change does in one sentence
3. The syntax check result
4. The test result
5. The patch record path
6. The exact `kaggle kernels push` command to re-push (if needed)

---

## TirraMind code constraints

When writing patches, be aware of:

1. **`trainer.py` is ~2700 lines.** Read only the function you're changing, not the whole file.

2. **`TrainerConfig` is a dataclass** (around line 580-680). Adding a new field:
   ```python
   new_param: float = 0.1
   """Description."""
   ```
   Then wire it to `retrain_gnn.py` with a `parser.add_argument("--new-param", ...)`.

3. **Loss weighting is in `_compute_return_loss()`** and `auto_tune_loss_weights()`.
   The ratio warning is around line 1655.

4. **ListNet temperature tau** is in `TrainerConfig.listnet_temperature` (added in the
   self-improving loop update). It is passed as `--listnet-temperature` on the CLI.

5. **The GDELT imbalance fix** is controlled by `--gdelt-frac` in `retrain_gnn.py`
   (default 0.05). If the report recommends reducing GDELT further, lower the default.

6. **Compute constraint:** All training runs on Kaggle T4/P100. Do not add operations
   that are O(N²) in entity count or require >4GB GPU memory.

7. **After any change to `trainer.py`, always verify** the loss-component ratio warning
   (around line 1655) still fires correctly — it guards against silent IC stagnation.

---

## Self-improvement loop context

The `auto_improve.py` fast path writes `{checkpoint_dir}/next_config.json`. Before
patching, always check whether the issue was already addressed by a fast-path config
change (it means the current problem is configuration, not code):

```bash
cat .tirra_pipeline/checkpoints/next_config.json 2>/dev/null | python3 -m json.tool
```

### Reading `next_config.json`

The file has this structure:
```json
{
  "generated":      "2026-05-09T10:00:00",
  "based_on_epoch": 22,
  "pattern":        "dt_dominance",
  "rationale":      "...",
  "resume_epoch":   22,
  "flag_overrides": {"return_weight": 2.0},
  "remove_flags":   [],
  "previous_config": {"lr": 0.001, "return_weight": 1.0, ...},
  "example_command": "python3 scripts/retrain_gnn.py --resume 22 ..."
}
```

Key fields for your decision:
- `pattern` — which stagnation pattern was detected (see table below)
- `flag_overrides` — the recommended flag changes for the NEXT run (config fix only)
- `remove_flags` — flags that should be disabled (e.g. `["--auto-tune"]`)
- `rationale` — why this was recommended

**Pattern → action table:**

| pattern | Meaning | What it changes |
|---|---|---|
| `divergence` | Return loss increasing | Reduce LR by 70% |
| `auto_tune_suppressing` | auto_tune silencing return head | Disable auto_tune, set return_weight=2.0 |
| `dt_dominance` | dt_loss >> return_loss | Double return_weight (cap 4.0) |
| `oscillation` | Return loss CV > 12% | Halve LR |
| `gdelt_noise` | GDELT fraction too high | Halve gdelt_frac |
| `listnet_temperature` | ListNet tau too smooth | Halve tau |
| `structural` | All fast-path options exhausted | Escalated to research loop |

**Decision rule:** If a `next_config.json` exists AND its `pattern` is NOT `structural`,
the recommended change is already a config change — do not patch code. Tell the user to
run `retrain_gnn.py --config-file .tirra_pipeline/checkpoints/next_config.json` instead.

### Reading `knowledge/improvement_history.jsonl`

Each line is one past recommendation:
```json
{"ts": "2026-05-09T10:00:00", "epoch": 22, "pattern": "dt_dominance", "flag_overrides": {"return_weight": 2.0}, "rationale": "..."}
```

Use this to understand what has already been tried. If the same pattern appears 3+
times in history without improvement, it has failed as a config fix and now requires
a structural code change.

---

## Safety rules for this skill

- Only modify **leaf-node files**: `trainer.py` loss/config functions and `retrain_gnn.py` CLI.
- Never modify `model.py`, entity schemas, or any file outside `agent/models/gnn/` or `scripts/`.
- Always read the target function before patching. Never guess a line number.
- Always run `py_compile` after patching. A syntax error is a hard blocker.
- Always write a patch record. The patch is not complete without documentation.
- If the diagnostic report's recommendation is ambiguous, ask the user to clarify before patching.

