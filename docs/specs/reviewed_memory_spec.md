---
title: "Spec: Reviewed Memory — Lesson Promotion Pipeline + Episodic Decay"
tags:
  - doc/spec
  - phase/29
  - topic/memory
  - layer/learning
---

# Spec: Reviewed Memory

## Goal

Add a statistical lesson-promotion pipeline so the autonomous loop only learns from patterns with sufficient cross-run evidence. Add episodic decay to prevent unbounded memory growth.

## Files Affected

| Action | File | Description |
|--------|------|-------------|
| **Create** | `agent/memory/candidates.py` | CandidateStore: cluster, stage, promote, reject lessons |
| **Modify** | `agent/memory/store.py` | Add `validated` field to LearningEntry, add `EpisodicMemory.decay()`, add `SemanticMemory.get_validated_learnings()` |
| **Modify** | `agent/config/settings.py` | Add 3 new TIRRA_ env vars for promotion thresholds |
| **Modify** | `agent/core/autonomous.py` | Wire CandidateStore.process() after each run, filter reflection inputs |
| **Create** | `tests/test_candidates.py` | Full test suite for CandidateStore |
| **Modify** | `tests/test_memory.py` | Tests for decay(), validated field, get_validated_learnings() |

## Implementation Steps

### Step 29.1: Add `validated` field to LearningEntry + backward compat

File: `agent/memory/store.py`

- Add `validated: bool = False` to `LearningEntry` dataclass (default False = backward compatible with existing JSONL files)
- Add `run_id: str = ""` to `LearningEntry` (identifies which autonomous run produced it — needed for "spanning ≥ 2 runs" check)
- Add `SemanticMemory.get_validated_learnings(n)` → returns only entries where `validated is True`
- Add `SemanticMemory.get_unvalidated_learnings()` → returns entries where `validated is False`

**Test**: Create learnings with/without validated flag, confirm get_validated_learnings filters correctly. Load a JSONL file without the `validated` field, confirm it defaults to False.

### Step 29.2: Add `EpisodicMemory.decay()`

File: `agent/memory/store.py`

- Add `EpisodicMemory.decay(max_age_days: int = 30, archive_dir: Path | None = None)`
  - Remove episodes older than `max_age_days` from in-memory list
  - If `archive_dir` is provided, write removed episodes to `archive_dir/episodic_YYYYMMDD.jsonl` before deletion
  - Rewrite the main `episodic.jsonl` with only the retained episodes
  - Return count of decayed episodes
- Add `Episode.id` field: `f"{timestamp}_{step}"` composite key (computed property, no JSONL change needed)

**Test**: Create 50 episodes spanning 60 days, decay with 30-day TTL, confirm only recent 30 days remain. Confirm archive file written. Confirm empty-list edge case.

### Step 29.3: Add config params

File: `agent/config/settings.py`

- Add to `AgentConfig`:
  - `lesson_min_support: int = 3` ← `TIRRA_LESSON_MIN_SUPPORT`
  - `lesson_min_runs: int = 2` ← `TIRRA_LESSON_MIN_RUNS`
  - `episode_ttl_days: int = 30` ← `TIRRA_EPISODE_TTL_DAYS`

**Test**: Confirm env var parsing, confirm defaults.

### Step 29.4: Create CandidateStore

File: `agent/memory/candidates.py`

Core class: `CandidateStore`

```python
@dataclass
class LessonCandidate:
    cluster_key: str          # "{arm}:{primary_tool}:{reward_sign}"
    claim: str                # first lesson text that seeded this candidate
    evidence: list[dict]      # [{run_id, timestamp, reward, lesson_text}, ...]
    support_count: int
    distinct_runs: set[str]
    avg_reward: float
    reward_sign_agreement: float  # fraction of evidence with same sign
    first_seen: float
    last_seen: float
    status: str               # "staged" | "accepted" | "rejected"
    rejection_count: int = 0
    rejection_reason: str = ""

class CandidateStore:
    def __init__(self, persist_path: Path, config: AgentConfig):
        ...

    def process(self, new_learnings: list[LearningEntry], run_id: str) -> ProcessResult:
        """Main entry point. Called after each autonomous run.
        1. Cluster new learnings by (arm, primary_tool, reward_sign)
        2. Merge into existing candidates (update evidence)
        3. Run promotion rules on all staged candidates
        4. Return which candidates were promoted/rejected
        """

    def _cluster_key(self, entry: LearningEntry) -> str:
        """Generate cluster key from structured fields."""

    def _check_promotion(self, candidate: LessonCandidate, config: AgentConfig) -> str:
        """Returns 'accept', 'reject', or 'staged' (no change)."""

    def _is_duplicate(self, candidate: LessonCandidate, accepted: list[LessonCandidate]) -> bool:
        """Jaccard on lesson text tokens, threshold 0.7."""

    def get_accepted(self) -> list[LessonCandidate]:
        ...

    def get_staged(self) -> list[LessonCandidate]:
        ...
```

Persistence: `candidates.jsonl` in the memory directory. One line per candidate, full state.

**Promotion rules** (deterministic, per research doc):
- Accept: support_count ≥ min_support AND distinct_runs ≥ min_runs AND sign_agreement ≥ 0.8 AND not duplicate
- Reject: contradicts accepted with higher support OR avg_reward < 0.2 OR stale (no evidence in 14 days)
- Otherwise: stays staged

**Test**: See step 29.6 for full test suite.

### Step 29.5: Wire into autonomous loop

File: `agent/core/autonomous.py`

Changes:
1. Import CandidateStore
2. In `__init__`: create `self._candidates = CandidateStore(persist_path=mem_dir / "candidates.jsonl", config=config)`
3. Generate a `run_id` per `run()` call: `f"run_{int(time.time())}"`
4. Pass `run_id` to `LearningEntry` when storing
5. After step 8 (store_learning), call `self._candidates.process([learning_entry], run_id=run_id)`
6. After the loop ends, call `self._episodic.decay(max_age_days=config.episode_ttl_days, archive_dir=mem_dir / "episodic_archive")`
7. Change reflection inputs: replace `self._semantic.get_attempted_goals()` with goals from validated learnings only (use `get_validated_learnings()` for dead-end checks)
8. Keep `all_facts()` in reflection — Facts are a separate channel from LearningEntries and don't need the promotion gate

**When a candidate is promoted**: Set the corresponding LearningEntry's `validated = True` and re-persist semantic memory.

**Test**: Integration test in test_candidates.py — simulate 3+ runs, confirm promotion triggers on the 3rd run.

### Step 29.6: Edge case test suite

File: `tests/test_candidates.py`

Must cover:
1. **Cold start**: No existing candidates, first run produces learnings → all staged, none promoted
2. **Promotion threshold**: Exactly min_support entries from exactly min_runs → promoted
3. **Below threshold**: 2 entries from 1 run → stays staged
4. **Contradicting lessons**: Accepted lesson with support=5, new candidate contradicts with support=3 → rejected
5. **Duplicate detection**: Same lesson text re-clustered → merged into existing candidate, not new
6. **Sign disagreement**: 3 entries, 2 positive reward, 1 negative → sign_agreement = 0.67 < 0.8 → stays staged
7. **Stale rejection**: Candidate staged 15+ days ago with no new evidence → rejected
8. **Anti-churn**: Rejected candidate, new evidence arrives → re-staged, tracks rejection_count
9. **Low reward reject**: 3 entries across 2 runs, avg reward = 0.1 → rejected
10. **Backward compat**: Load old LearningEntry JSONL without `validated` or `run_id` fields → defaults work
11. **Decay**: 100 episodes, 30-day TTL, confirm correct retention count + archive
12. **Empty state**: Process with no learnings → no crash, no candidates
13. **Persistence round-trip**: Create candidates → save → reload → state identical
14. **Multiple lessons per run**: Single run produces 5 learnings across 3 clusters → 3 candidates

## Edge Cases

- Old `learnings.jsonl` files without `validated`/`run_id` → both default to False/""
- `candidates.jsonl` does not exist on first run → created on first write
- Empty lesson text in LearningEntry → skip (don't create candidates from empty lessons)
- All candidates rejected in a run → no promotions, no error
- Candidate with support from 10+ runs → promoted long ago, but new evidence still updates avg_reward

## Testing Plan

1. Unit tests for each CandidateStore method
2. Unit tests for decay()
3. Integration test: simulate full 5-iteration autonomous loop, verify promotion lifecycle
4. Backward compatibility test: load pre-existing JSONL without new fields
5. Persistence round-trip for CandidateStore
6. All 14 edge cases listed above

## Related

- [[reviewed_memory]]
- [[store]]
- [[autonomous]]
- [[reflection]]
- [[bandit]]
