---
title: "Research: Automated ML Problem Researcher"
tags:
  - doc/research
  - phase/auto-research
  - topic/tooling
  - topic/training
  - status/active
---

# Research: Automated ML Problem Researcher

**Problem being solved:** When training problems arise (flat loss, IC=0, loss divergence, OOM, etc.)
the engineer must manually search arXiv, GitHub, Kaggle, and Stack Overflow for solutions.
This wastes context-window budget, is slow, and produces inconsistent research quality.
We want an automated system that, given a training issue description, produces a ranked, 
codebase-grounded report of solutions within seconds.

---

## Inspiration: Karpathy's nanochat autoresearch rounds

**Source:** `github.com/karpathy/nanochat` (MIT License — implementations may be reused)

Karpathy uses a **Claude agent skill** (`.claude/skills/read-arxiv-paper/SKILL.md`) to automate
research within the nanochat training workflow. The skill:
1. Takes an arXiv URL
2. Downloads the TeX source (not PDF — richer content, no layout noise)
3. Reads the full paper recursively (main.tex + all `\input{}` includes)
4. Writes `knowledge/summary_{tag}.md` connecting the paper to the current codebase

This skill powered two leaderboard improvements labelled "autoresearch round 1" (Mar 9 2026)
and "autoresearch round 2" (Mar 14 2026), cutting training time by ~41% (3.04h → 1.80h).

The key principle: **the skill operates within the repo's context** — it reads relevant
source files before writing the summary, so the output is grounded in actual code, not generic.

---

## What We Need (TirraMind-Specific)

TirraMind's training problems fall into three categories:

### Category A: Loss/Signal Failures
- Flat return loss (confirmed in Phase 41b: 92.2% GDELT dominance)
- IC=0 despite training (ranking loss vs. MSE confusion — solved by ListNet in Phase 41b)
- Gradient explosion/vanishing (EWC lambda too high)
- Validation loss divergence

### Category B: Architecture Deficiencies
- Insufficient attention heads for multi-domain signal
- Message-passing depth too shallow for cross-domain propagation
- Node embedding collapse (all embeddings converge to same point)

### Category C: Data / Training Dynamics
- OOM on Kaggle T4 GPU
- Slow convergence due to wrong learning rate schedule
- EWC weight blowup when continually learning new domains

---

## Research: Existing Approaches

### 1. Sakana AI "AI Scientist" (2024)
- **Paper:** arXiv:2408.06292 — "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery"
- **Approach:** Full automated research loop: idea generation → experiment → write paper → review
- **Key limitation:** Requires GPU compute per research iteration (expensive for our setup)
- **License:** Apache 2.0 — concepts reusable
- **Relevance to us:** The "idea generation from failure analysis" component is directly applicable.
  We do NOT need the full pipeline — just the search + triage step.

### 2. Karpathy nanochat `.claude/skills/read-arxiv-paper`
- **Source:** `github.com/karpathy/nanochat/.claude/skills/read-arxiv-paper/SKILL.md`
- **License:** MIT — code patterns directly reusable
- **Approach:** Claude skill that downloads arXiv TeX source, reads it, connects to codebase
- **Key insight:** Uses `/src/` URL pattern for TeX source (free, richer than PDF)
- **Our adaptation:** Add multi-source search (arXiv + GitHub + Kaggle discussions) and
  automatically trigger based on training diagnostics

### 3. LitSearch / Semantic Scholar API
- **Source:** `api.semanticscholar.org/graph/v1/paper/search` — FREE, no key required
- **Approach:** Full-text semantic search across 200M+ papers
- **Key fields:** `title`, `abstract`, `year`, `citationCount`, `openAccessPdf.url`
- **Relevance:** Better than raw arXiv search for finding the most-cited/relevant papers
- **Verified:** `curl "https://api.semanticscholar.org/graph/v1/paper/search?query=GNN+heterogeneous+return+prediction&fields=title,abstract,year,citationCount&limit=5"`

### 4. GitHub Code Search API
- **Source:** `api.github.com/search/code` — free tier: 10 req/min unauthenticated
- **Approach:** Search for code patterns, issues, and discussions
- **Use case:** "flat IC in GNN training" → find GitHub issues in stock prediction repos
- **Relevance:** Finds practical implementations and known bug fixes

### 5. Kaggle Discussions API
- **Source:** `kaggle.com/discussions` (available via `kaggle.json` auth)
- **Use case:** Search for Kaggle notebooks/discussions mentioning the problem
- **Note:** Limited API support — better to use GitHub search targeting Kaggle notebooks

---

## Design Decision: Skill vs. Script vs. Daemon

| Option | Pros | Cons |
|---|---|---|
| **Claude skill only** (`.claude/skills/`) | Zero code, works in VS Code Copilot | Manual trigger only, no CLI |
| **CLI script** (`scripts/auto_research.py`) | Scriptable, can be called from trainer callback | Requires API calls, may be slow |
| **Daemon** (watches training logs) | Fully automatic | Complex, hard to test, running process on laptop |

**Decision: Skill + CLI script.** The Claude skill handles the deep-reading and synthesis.
The CLI script handles search and triage (finding the right papers to read). The daemon
is rejected — the laptop cannot sustain background processes during Kaggle training.

**Auto-trigger mechanism:** Not a daemon. Instead, add a `--research` flag to `retrain_gnn.py`
that, when IC stays flat for N consecutive evaluation windows, writes a trigger file
`knowledge/trigger_{timestamp}.md` with the diagnostic context. The user then runs
`python scripts/auto_research.py --from-trigger knowledge/trigger_*.md` to kick off the search.

---

## Optimization Target Validation

The user's concern: "let assure that goal or the outcome which we are optimising for is a good one."

**Current state (post Phase 41b):**
- Training loss: ListNet cross-entropy (ranking loss) ✅ — directly optimizes what IC measures
- Eval metric: Spearman IC ✅ — correctly measures return ranking ability
- **Alignment confirmed:** ListNet's per-window loss = softmax KL divergence over return ranks.
  Minimizing this IS equivalent to maximizing Spearman IC (up to monotonic transformation).

**Residual concern:**
- The `_compute_dt_loss()` (temporal prediction head) still dominates total loss weighting
- `auto_tune_loss_weights` corrects this dynamically, but we should log loss component ratios
  per epoch to verify return head is receiving meaningful gradient

**Recommended: Add a loss-component ratio assertion** in the trainer: if
`dt_loss / return_loss > 50` for more than 3 consecutive epochs, emit a WARNING so the
user knows the balance is off.

---

## Depth Roadmap (Signal Depth Doctrine)

This feature is Layer 7 (LLM Support) in the computation stack — it helps humans and
the agent understand training failures. It is NOT a new data source or prediction component.

- **L1:** Single-source search (arXiv only). Returns paper titles and abstracts.
- **L2:** Multi-source search (arXiv + GitHub + Kaggle). Entity = problem type.
- **L3:** Codebase-grounded synthesis. Links each paper/solution to the exact file/function
  in TirraMind where the fix would be applied, with code diffs.

We target **L2** for the initial implementation (multi-source, entity-level problem tracking).
L3 synthesis is done by Claude via the skill, not the CLI script.

---

## Files to Create

| File | Purpose |
|---|---|
| `.claude/skills/research-training-issue/SKILL.md` | Claude skill: triggered on a problem, searches arXiv+GitHub, reads papers, writes synthesis |
| `scripts/auto_research.py` | CLI tool: semantic search, returns ranked URLs for the skill to read |
| `knowledge/` directory | Where skill outputs are written (per Karpathy pattern) |

---

## Related

- [[phase41b_gnn_signal_extraction]] — the problem that triggered this feature
- [[auto_ml_researcher_spec]] — spec
- [[auto_ml_researcher_task]] — task file
