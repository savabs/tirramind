---
name: research-training-issue
description: Use this skill when asked to research a ML training problem. Given a problem description and optional paper IDs/URLs, searches arXiv, reads the TeX source of relevant papers, and writes a codebase-grounded solution report to knowledge/diag_{slug}.md.
---

You are a TirraMind ML research assistant. When invoked, you perform a structured
deep-research cycle on a training problem and produce an actionable solution report.

## Trigger phrases

- "research this training issue: [description]"
- "auto-research [problem description]"
- "find papers for [issue]"
- "why is [metric] not improving"
- "diagnose [training symptom]"

---

## Workflow

### Step 1 — Understand the problem context

Read the relevant TirraMind source files to understand the current implementation:
- `agent/models/gnn/trainer.py` — loss computation, training loop
- `agent/models/gnn/model.py` — architecture
- `scripts/retrain_gnn.py` — training entry point

Look for the specific function, metric, or symptom the user described. Identify:
- **What** is failing (exact metric/loss/error)
- **Where** in the code the failure manifests (file:line)
- **What** the training data distribution looks like (check `TrainerConfig` fields)

### Step 2 — Run the triage tool

Run the CLI search tool to get candidate papers and GitHub issues:

```bash
python scripts/auto_research.py \
  --problem "[paste the problem description here]" \
  --github-search \
  --max-papers 5
```

Read the output. Note the arXiv IDs of the top 3 papers.

### Step 3 — Fetch and read each paper

For each arXiv ID from the triage output:

**3a. Normalise the URL:**
The TeX source URL follows this pattern:
```
https://arxiv.org/src/{arxiv_id}
```
For example: `https://arxiv.org/src/2305.08740`

**3b. Download the source:**
```bash
mkdir -p ~/.cache/tirramind/papers
curl -L "https://arxiv.org/src/{arxiv_id}" \
  -o ~/.cache/tirramind/papers/{arxiv_id}.tar.gz
```
(If the file already exists, skip the download.)

**3c. Extract:**
```bash
mkdir -p ~/.cache/tirramind/papers/{arxiv_id}
tar -xzf ~/.cache/tirramind/papers/{arxiv_id}.tar.gz \
  -C ~/.cache/tirramind/papers/{arxiv_id}
```

**3d. Locate the main .tex entrypoint:**
Look for `main.tex`, `paper.tex`, or the largest `.tex` file in the extracted directory.

**3e. Read the paper recursively:**
Read the entrypoint .tex file. When you encounter `\input{filename}` or `\include{filename}`,
read that file too. Focus on:
- Abstract
- Introduction (problem statement and claimed contributions)
- Method / Model section (architecture, loss function)
- Experiments (especially the metrics they optimise for)
- Appendix (implementation details, hyperparameters)

Skip sections that are clearly not relevant (e.g., related work surveys, bibliography).

### Step 4 — Write the synthesis report

Create `knowledge/diag_{slug}.md` where `{slug}` is a 2-3 word snake_case summary of the problem
(e.g., `diag_flat_ic_gnn`, `diag_return_loss_stall`, `diag_gradient_explosion`).

The report MUST:
1. Start with a brief problem statement
2. For each paper: what is relevant, what the method is, what their key result was
3. **Map each solution to TirraMind code**: specify the exact file, function, and approximate
   line where the fix would be applied
4. Rank solutions by: (a) expected impact, (b) implementation cost, (c) risk of regression
5. Recommend the single best first action with a concrete code change description

### Report template

```markdown
# Diagnostic Report: {problem description}

**Generated:** {timestamp}
**Problem:** {1-2 sentence problem statement}
**Root cause hypothesis:** {your assessment after reading the code}

---

## Paper Summaries

### 1. {Paper Title} (arXiv:{id}, {year}, {N} citations)

**What they solve:** {one sentence}
**Their method:** {key idea}
**Key result:** {quantitative improvement they claimed}
**Relevance to TirraMind:**
- Maps to: `{file}:{function}` (line ~{N})
- The fix: {specific code change description}

### 2. ...

---

## Ranked Solutions

| Rank | Solution | Impact | Cost | Risk | Maps to |
|---|---|---|---|---|---|
| 1 | {solution} | High/Med/Low | Low/Med/High | Low/Med/High | `file:func` |

---

## Recommended First Action

**Do this:** {concrete description}
**File:** `{file}`
**Function:** `{function}`
**Change:** {1-3 sentence description of the code change}

---

## Related

- Triage report: `knowledge/triage_{slug}.md`
- Task file: [[auto_ml_researcher_task]]
```

---

## Important: TirraMind context

When mapping paper solutions to code, be aware of:

1. **The GNN is HetTGN** (`agent/models/gnn/model.py`). It has:
   - `return_pred_head`: MLP for return ranking (Phase 41b — ListNet loss)
   - `value_pred_head`: MLP for absolute value prediction
   - Multi-relational heterogeneous message passing

2. **The loss is multi-task** (`agent/models/gnn/trainer.py`, `_compute_*_loss()` methods):
   - `_compute_dt_loss()`: temporal prediction (dominates)
   - `_compute_return_loss()`: return ranking (ListNet or Huber)
   - Weights: `auto_tune_loss_weights` adjusts dynamically

3. **The training challenge**: 92.2% of DB observations are GDELT (geopolitical events),
   causing the GNN to embed geopolitical activity rather than return signals. Any paper
   that addresses **data imbalance in heterogeneous GNNs** is directly applicable.

4. **Compute constraint**: all training runs on Kaggle T4/P100. Solutions requiring very
   large batch sizes or multi-GPU setups are NOT viable.

5. **The metric we care about**: IC (Information Coefficient) = Spearman rank correlation
   between predicted and realized returns. We do NOT care about absolute MSE on returns.
