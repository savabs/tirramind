---
title: "Feature: Reviewed Memory — Lesson Promotion Pipeline + Episodic Decay"
tags:
  - doc/research
  - phase/29
  - topic/memory
  - layer/learning
---

# Feature: Reviewed Memory — Lesson Promotion Pipeline + Episodic Decay

## Goal
- Prevent the autonomous loop from accumulating false lessons from noisy runs.
- Currently, `LearningEntry` objects are stored directly by the autonomous loop with no validation gate. One bad run can poison the lesson pool that feeds reflection and goal generation.
- Add a **candidate → validated** promotion pipeline where lessons must accumulate statistical evidence across multiple runs before they influence future agent behavior.
- Add **episodic decay** so `episodic.jsonl` does not grow unbounded.

## Search Log
- GitHub keywords: "agent memory governance", "lesson promotion pipeline", "agentic-stack memory"
- Documentation: agentic-stack architecture.md, auto_dream.py, promote.py, review_state.py
- Other: Direct README and source review of https://github.com/codejunkie99/agentic-stack

## External Repositories Reviewed
- Repository: codejunkie99/agentic-stack (v0.5.0)
  - Why relevant: Implements a full candidate lifecycle (staged → graduated/rejected) for coding-agent lessons. Has content clustering, heuristic prefilter, anti-churn rejection tracking, and episodic decay.
  - Useful implementation idea: Candidate lifecycle with rejection_count to prevent recurring junk from looking fresh. Separating mechanical staging from subjective review. Anti-churn: tracking which specific lessons blocked a candidate so unrelated changes don't trigger re-staging.
  - License: MIT
  - Reuse conclusion: **concept only** — their implementation is markdown/CLI-oriented for human coding-agent review. TirraMind needs statistical promotion rules for autonomous operation, structured data clustering (not text Jaccard), and inline post-run processing instead of cron-based dream cycles.

## Current Architecture
- `agent/memory/store.py`: EpisodicMemory (append-only JSONL), SemanticMemory (Fact store + LearningEntry list), WorkingMemory (LLM message buffer)
- `agent/core/autonomous.py`: After each run, stores LearningEntry directly via `self._semantic.store_learning()` — no validation gate.
- `agent/learning/reflection.py`: Reflector.reflect() receives `semantic_facts` (all facts) and `attempted_goals` (all past goals). No filtering by lesson quality.
- `agent/config/settings.py`: AgentConfig (frozen dataclass, `TIRRA_` env vars). Currently no memory governance params.
- Module: `agent/memory/` contains only `__init__.py` and `store.py`.

## Observations
- LearningEntry has: goal, score, success, dead_end, lessons (list[str]), arm, reward, timestamp
- No `validated` or `status` field — every stored learning is treated equally by reflection
- Reflector receives `semantic_facts` (Fact objects, not LearningEntries) — but the autonomous loop calls `get_attempted_goals()` and `get_dead_ends()` which read from unvalidated learnings
- Episodic memory grows without bound — `episodic.jsonl` will become large over hundreds of iterations
- The reflection prompt feeds up to 15 facts and 20 episodes — so volume is capped in the prompt, but the underlying data is not curated
- SemanticMemory.store_learning() is a direct append with no dedup or quality check

## Risks
- **False lesson accumulation**: A noisy run produces a lesson like "macro_data always fails" which is really a transient API issue. Without validation, this poisons future reflection.
- **Lesson contradiction**: Two runs produce opposite lessons on the same topic. Both persist. Reflector sees both and gets confused.
- **Episodic bloat**: Over 500+ iterations, loading and scanning episodic.jsonl at startup becomes slow.
- **Over-engineering risk**: Promotion thresholds that are too strict will prevent the agent from learning anything useful. Need sensible defaults.
- **Backward compatibility**: Existing `learnings.jsonl` files have no `validated` field. Migration needed.

## Data Requirements
- Input: LearningEntry stream from autonomous loop (already exists)
- Clustering key: (arm, tool set, outcome direction). These fields already exist on LearningEntry.
- Evidence: episode IDs (need to add episode UUID or use timestamp+step as composite key)

## Math/Algorithm Survey

### Candidate clustering
- **Approach**: Group by (arm name × primary tool × reward sign). Not text similarity — TirraMind has typed structured data.
- **Why**: arms and tools are categorical. Reward sign (positive/negative) captures directional consistency. This is simpler and more correct than Jaccard on text for our use case.
- **Alternative rejected**: Text embedding similarity (over-engineered, adds vector dependency for a simple grouping task).

### Promotion rules (deterministic, no LLM)
- **Accept when ALL of**:
  - support_count ≥ `TIRRA_LESSON_MIN_SUPPORT` (default 3) episodes
  - spanning ≥ `TIRRA_LESSON_MIN_RUNS` (default 2) distinct runs
  - reward direction consistent: sign agreement ≥ 80% across supporting episodes
  - not a duplicate of an existing accepted lesson (same cluster key, Jaccard on lesson text > 0.7)
- **Reject when ANY of**:
  - contradicts an accepted lesson with higher support count
  - avg reward < `TIRRA_LESSON_REJECT_THRESHOLD` (default 0.2) — low-value pattern
  - stale: no new evidence in 14 days while still staged
- **Everything else**: stays staged, awaiting more evidence

### Episodic decay
- **Approach**: Episodes older than `TIRRA_EPISODE_TTL_DAYS` (default 30) get removed from in-memory list. Raw JSONL file gets rotated (archived to `episodic_archive/`).
- **Compression**: Before removal, compute per-tool summary stats (count, avg success rate, avg reward) and store as a single summary Fact in SemanticMemory. This preserves aggregate knowledge while releasing raw data.
- **Why not keep everything**: Cold-start cost, memory footprint, and diminishing value of old individual episodes.

## Implementation Intent
- **Approved**:
  - `CandidateStore` class in new `agent/memory/candidates.py`
  - `validated` field on LearningEntry (default False, backward compatible)
  - `EpisodicMemory.decay()` method
  - `SemanticMemory.get_validated_learnings()` filter
  - Config params: TIRRA_LESSON_MIN_SUPPORT, TIRRA_LESSON_MIN_RUNS, TIRRA_EPISODE_TTL_DAYS
  - Wire CandidateStore.process() into autonomous loop post-run
  - Filter Reflector inputs to validated-only learnings
- **Rejected**:
  - CLI review tools (no human in loop during autonomous execution)
  - Markdown rendering of lessons (consumer is LLM, not human)
  - Cron-based dream cycle (post-run inline processing is simpler)
  - Text embedding clustering (typed fields are sufficient)

## Related

- [[reviewed_memory_spec]]
- [[store]]
- [[autonomous]]
- [[reflection]]
- [[bandit]]
