---
title: Claude Rules for TirraMind
tags:
  - doc/governance
  - layer/meta
---

# Claude Rules for TirraMind

Operating guide for Claude Code working on this codebase. Complements `RulesForAI.md` (architecture/workflow) and `LESSONS.md` (failure modes).

---

## 1. Layer Discipline Is Absolute

**The 7-layer architecture is load-bearing.** Code in the wrong layer = invisible debt.

- **Layer 1 (Surveillance):** `agent/tools/` — data fetching only. Free APIs, HTTP clients, no feature logic.
- **Layer 2 (Feature Engineering):** `agent/quant/` — BOCPD, HMM, spectral, scoring. Stateless math.
- **Layer 3 (World Model):** `agent/models/` — Bayesian graphs, causal inference, beliefs.
- **Layer 4 (Signal Fusion):** `agent/fusion/` — Kalman, particle filters, multi-source combo.
- **Layer 5 (RL Policy):** `agent/learning/` — model-based RL, bandit arms, portfolio optimization.
- **Layer 6 (Adversarial):** `agent/adversarial/` — edge decay, manipulation resistance, robustness.
- **Layer 7 (LLM Support):** `agent/reasoning/` — text parsing, narration. **LLM does not decide.**

Before writing code:
- Identify which layer your change belongs to.
- If it touches multiple layers, check if you're collapsing boundaries. If yes, refactor.
- If you're adding a new file, verify it belongs in its intended layer. Ask if unsure.

---

## 2. LESSONS.md Before Every Coding Session

**Read `LESSONS.md` fuckup log every session.** Each entry is a hard-won bug that will return if you're not watching.

Critical patterns to watch:
- **F-01 (Embedding Collapse):** Contrastive loss must have true negatives. Verify embedding diversity before training: `torch.std(emb, dim=0).mean() > 0.1`.
- **F-02 (GNN Bypass):** Check that model embeddings are in the computational graph for your loss. Print active branch at training start.
- **F-03 (History Column Shift):** History arrays must stay length-aligned on checkpoint resume. Front-pad with NaN when adding mid-training losses.
- **F-04 (Data Leakage):** IC eval must use future-blind lookback windows. Verify all test splits are time-ordered, not random.

If you modify training loops, loss functions, checkpoint resume logic, or eval metrics: re-read the relevant fuckup first.

---

## 3. Research → Spec → Task Before Code

This is in `RulesForAI.md` but critical enough to repeat: **do not write implementation code for non-trivial work without these artifacts:**

```
docs/research/<feature>.md
docs/specs/<feature>_spec.md
tasks/active/<task>.md
```

- **Research:** facts, OSS patterns, design space. No code.
- **Spec:** ordered, atomic steps. Each step is independently verifiable.
- **Task:** action checklist tied to spec steps. Mark done as you go.

"Non-trivial" = anything that touches multiple files, changes behavior, modifies architecture, or touches training logic.

If the request is vague, write research in chat instead of guessing code. Stay in planning mode until the path is clear.

---

## 4. Training Logic Is Sacred

Changes to `trainer.py`, loss functions, model forward passes, or checkpoint resume:
- Always diff against the last known-good checkpoint.
- Add a specific test for your change before merging (not "smoke test", specific reproduction).
- If adding a new loss component, add `LESSONS.md` entry FIRST with prevention rule.
- Always print the active branch/mode at training start so you can grep the log.
- If you change how history is tracked, pad and align all arrays, then test checkpoint resume → resume → verify arrays align.

---

## 5. Evidence Graph Is Your North Star

The evidence/knowledge graph (`agent/evidence/`, model checkpoints in `docs/memory/`) is the source of truth for learned patterns.

- Before making changes to feature engineering or model architecture, check `docs/memory/checkpoint_*_evidence_graph.md` for what the last training run discovered.
- If you change feature extraction, you're invalidating learned weights. Document why.
- If you change the feature set, retrain from scratch or justify why old weights transfer.
- Checkpoints are immutable once created. Create a new one for each major run.

---

## 6. Session Boundaries

Complete a feature or major sub-phase, then start a fresh chat.

**At the end of a meaningful increment:**
- Commit with a clear message (cite the task/spec).
- Write or update `docs/memory/checkpoint_<date>_<topic>.md` with:
  - What was learned (embedding diversity? loss convergence? eval insight?)
  - Known issues or next steps
  - How to resume from this checkpoint
- Leave `tasks/active/<task>.md` with "✓ DONE" and a one-line summary of the outcome.

Future sessions will have the checkpoint + completed task as the handoff point, not chat history.

---

## 7. Cost Discipline

**$0 until proven edge.** (This is in README, but Claude should know it deeply.)

- Do not add external APIs, cloud services, or paid subscriptions without explicit approval.
- Do not change any `TIRRA_*` env var defaults without adding research first.
- When benchmarking, use the free tier or local compute.
- If you're about to run something expensive (cloud training, API quota), call it out explicitly before proceeding.

---

## 8. Wiki and Knowledge Base

The `wiki/` directory is the compiled knowledge base. It's **LLM-maintained** (not sacred, but useful).

- Before major changes, check `wiki/pages/` for existing synthesis on the topic.
- After a major run, consider adding or updating a wiki page so future sessions have context.
- Use `tirra-wiki-catalog` to regenerate the index when you add new pages.
- Wiki entries are organized by topic, not chronologically. Search by concept, not date.

---

## 9. Critical Files: Read Before Editing

Before you edit any of these, **read them first**:
- `trainer.py` — loss computation, forward pass, history tracking. Changes here echo through training.
- `agent/models/gnn.py` — embedding architecture. Changes affect layer discipline and F-01/F-02.
- `agent/quant/scoring.py` — feature engineering. Changes affect layer 2 and validity of prior signals.
- `agent/pipeline/dag.py` — execution scheduler. Changes affect scheduling and ordering guarantees.
- `pyproject.toml` — dependencies. Changes to versions or constraints can break training (see pytorch 2.5.1 fix).

For any change to these files:
1. Read the file first.
2. Write a spec documenting why and what you're changing.
3. Add a targeted test.
4. Run full test suite before committing.

---

## 10. Failure Mode Triage

When something breaks:

1. **Is it already in LESSONS.md?** If yes, apply the prevention rule immediately.
2. **Can you reproduce it in isolation?** Write a test, don't guess.
3. **Is it a training convergence issue?** Check embedding diversity, loss history alignment, data leakage.
4. **Is it a data issue?** Verify time-ordering, window causality, future-blindness.
5. **Is it layer discipline?** Check which layer the code is in; it might belong elsewhere.

If it's novel, add it to LESSONS.md (symptom, root cause, fix, prevention rule) so it doesn't return.

---

## 11. Commit Hygiene

Commits should be atomic and citable:
- ✓ `feat: add cross-sectional ranking contrastive loss (F-01 fix)`
- ✓ `fix: pad history arrays on checkpoint resume (F-03 prevention)`
- ✗ `updates and fixes` (not atomic, not citable)

Link commits to task/spec: `Closes tasks/active/xyz.md` in the message.

---

## 12. The Agent Team: Triage Before Dispatch

There are 20 specialists in `.claude/agents/`, each with an exclusive domain and
a documented `## Boundaries` section.

**"Ask the team" NEVER means "run every agent."**

- All team requests route through **`principal-architect`** (or `/team`). It
  triages and dispatches only the specialists that actually own the work.
- Every dispatch MUST produce a triage block naming who was **DISPATCHED**, who
  was **EXCLUDED**, and why. The EXCLUDED line is what forces real triage — a
  block without it is incomplete.
- **Default 1–3 agents.** 4–6 needs a stated reason. **7+ needs the owner's
  explicit approval — ask first.**
- Dispatching *nobody* is a valid outcome. If a `grep` settles it, do that.
- Prefer sequential dispatch when later work depends on earlier findings —
  running a specialist on a premise a cheaper agent could have falsified is waste.

A full-roster fan-out once cost 1.6M tokens and 40 minutes to answer questions
three agents owned. Breadth is not thoroughness.

---

## 13. When to Ask / When to Decide

**Ask before:**
- Adding a new external dependency or paid service
- Changing the core model architecture (anything in layer 3 or above)
- Modifying how training history is tracked or checkpointed
- Touching GPU/CUDA assumptions (especially after the PyTorch 2.5.1 compat fix)
- Merging datasets or changing the feature set in a way that invalidates prior checkpoints

**Decide and ship:**
- Bug fixes with tests
- Refactoring within a layer (no behavior change)
- Adding a new feature to an existing tool (Layer 1)
- Documentation, checkpoint writes, memory updates
- Anything already covered by a completed spec
