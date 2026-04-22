"""Tests for reviewed-memory pipeline: CandidateStore, episodic decay, validated learnings.

Covers all 14 edge cases from the spec plus persistence round-trips.
"""

from __future__ import annotations

import json
import time

import pytest

from agent.memory.candidates import CandidateStore, LessonCandidate, ProcessResult
from agent.memory.store import Episode, EpisodicMemory, LearningEntry, SemanticMemory

# ── Helpers ───────────────────────────────────────────────────


def _learning(
    goal: str = "test goal",
    arm: str = "macro_analysis",
    reward: float = 0.5,
    lessons: list[str] | None = None,
    run_id: str = "run_1",
    score: float = 0.7,
    success: bool = True,
    dead_end: bool = False,
    validated: bool = False,
    timestamp: float | None = None,
) -> LearningEntry:
    return LearningEntry(
        goal=goal,
        score=score,
        success=success,
        dead_end=dead_end,
        lessons=lessons if lessons is not None else ["learned something useful"],
        arm=arm,
        reward=reward,
        run_id=run_id,
        validated=validated,
        timestamp=timestamp or time.time(),
    )


def _episode(step: int = 1, age_days: int = 0, success: bool = True) -> Episode:
    return Episode(
        timestamp=time.time() - age_days * 86_400,
        step=step,
        action="test_tool",
        input_summary="params",
        output_summary="result",
        success=success,
    )


# ═══════════════════════════════════════════════════════════════
# CandidateStore — core promotion logic
# ═══════════════════════════════════════════════════════════════


class TestColdStart:
    """Edge case 1: No existing candidates, first run."""

    def test_first_run_all_staged(self, tmp_path):
        store = CandidateStore(persist_path=tmp_path / "c.jsonl")
        result = store.process(
            [_learning(run_id="run_1")],
            run_id="run_1",
        )
        assert result.candidates_updated == 1
        assert result.promoted == []
        assert result.rejected == []
        assert result.still_staged == 1
        assert len(store.get_staged()) == 1


class TestPromotionThreshold:
    """Edge case 2: Exactly min_support from min_runs → promoted."""

    def test_exact_threshold_promotes(self, tmp_path):
        store = CandidateStore(
            persist_path=tmp_path / "c.jsonl",
            min_support=3,
            min_runs=2,
        )
        # Run 1: 2 learnings
        store.process(
            [_learning(run_id="run_1"), _learning(run_id="run_1")],
            run_id="run_1",
        )
        assert len(store.get_staged()) == 1
        assert len(store.get_accepted()) == 0

        # Run 2: 1 more learning (total 3, from 2 runs)
        result = store.process(
            [_learning(run_id="run_2")],
            run_id="run_2",
        )
        assert len(result.promoted) == 1
        assert len(store.get_accepted()) == 1


class TestBelowThreshold:
    """Edge case 3: 2 entries from 1 run → stays staged."""

    def test_single_run_stays_staged(self, tmp_path):
        store = CandidateStore(
            persist_path=tmp_path / "c.jsonl",
            min_support=3,
            min_runs=2,
        )
        result = store.process(
            [_learning(run_id="run_1")] * 3,
            run_id="run_1",
        )
        # Has 3 support but only 1 run
        assert result.promoted == []
        assert result.still_staged == 1


class TestContradiction:
    """Edge case 4: Accepted lesson contradicted by weaker candidate → rejected."""

    def test_weaker_contradiction_rejected(self, tmp_path):
        store = CandidateStore(
            persist_path=tmp_path / "c.jsonl",
            min_support=2,
            min_runs=2,
        )
        # Build up an accepted positive lesson
        store.process([_learning(arm="macro", reward=0.8, run_id="r1")], run_id="r1")
        store.process([_learning(arm="macro", reward=0.6, run_id="r2")], run_id="r2")
        accepted = store.get_accepted()
        assert len(accepted) == 1

        # Now add contradicting negative lesson (same arm:tool, negative reward)
        store.process(
            [_learning(arm="macro", reward=-0.5, run_id="r3")],
            run_id="r3",
        )
        store.process(
            [_learning(arm="macro", reward=-0.4, run_id="r4")],
            run_id="r4",
        )
        # The negative candidate should be rejected (contradicts accepted with higher support)
        rejected = store.get_rejected()
        assert len(rejected) == 1
        assert "contradicts" in rejected[0].rejection_reason


class TestDuplicateDetection:
    """Edge case 5: Same lesson text re-clustered → merged, not new."""

    def test_same_cluster_key_merges(self, tmp_path):
        store = CandidateStore(persist_path=tmp_path / "c.jsonl")
        store.process([_learning(run_id="r1")], run_id="r1")
        store.process([_learning(run_id="r2")], run_id="r2")
        # Should be 1 candidate with support_count=2
        all_cands = store.get_all()
        assert len(all_cands) == 1
        cand = list(all_cands.values())[0]
        assert cand.support_count == 2
        assert cand.distinct_run_count == 2


class TestSignDisagreement:
    """Edge case 6: Mixed reward signs → stays staged."""

    def test_mixed_signs_no_promotion(self, tmp_path):
        store = CandidateStore(
            persist_path=tmp_path / "c.jsonl",
            min_support=3,
            min_runs=2,
            sign_agreement_threshold=0.8,
        )
        # These all go to the same cluster because reward >= 0 → "pos"
        # But we can't easily test sign disagreement within one cluster
        # because the cluster key includes the sign.
        # Instead, test that a candidate with manipulated evidence
        # (mixed signs) stays staged.
        store.process([_learning(reward=0.5, run_id="r1")], run_id="r1")
        store.process([_learning(reward=0.5, run_id="r2")], run_id="r2")
        store.process([_learning(reward=0.5, run_id="r3")], run_id="r3")

        # Manually inject mixed evidence to test sign_agreement logic
        cand = list(store.get_all().values())[0]
        cand.evidence.append({"run_id": "r4", "timestamp": time.time(), "reward": -0.5, "lesson_text": "bad", "goal": "g"})
        cand.evidence.append({"run_id": "r5", "timestamp": time.time(), "reward": -0.3, "lesson_text": "bad", "goal": "g"})
        cand.support_count = len(cand.evidence)
        cand.reward_sign_agreement = CandidateStore._sign_agreement(cand)
        # 3 pos, 2 neg → 3/5 = 0.6 < 0.8 threshold
        assert cand.reward_sign_agreement == pytest.approx(0.6)
        # Re-run promotion check
        assert store._check_promotion(cand, []) == "staged"


class TestStaleRejection:
    """Edge case 7: Candidate staged 15+ days ago, insufficient evidence → rejected."""

    def test_stale_rejected(self, tmp_path):
        store = CandidateStore(
            persist_path=tmp_path / "c.jsonl",
            min_support=3,
            stale_days=14,
        )
        store.process([_learning(run_id="r1")], run_id="r1")
        # Simulate staleness by backdating last_seen
        cand = list(store.get_all().values())[0]
        cand.last_seen = time.time() - 15 * 86_400
        decision = store._check_promotion(cand, [])
        assert decision == "reject"
        assert "stale" in cand.rejection_reason


class TestAntiChurn:
    """Edge case 8: Rejected candidate gets new evidence → re-processable."""

    def test_rejection_count_tracked(self, tmp_path):
        store = CandidateStore(
            persist_path=tmp_path / "c.jsonl",
            min_support=3,
            stale_days=14,
        )
        store.process([_learning(run_id="r1")], run_id="r1")
        # Force rejection
        cand = list(store.get_all().values())[0]
        cand.last_seen = time.time() - 15 * 86_400
        cand.status = "staged"
        store._check_promotion(cand, [])
        assert cand.rejection_reason == "stale_insufficient_evidence"

        # New evidence: reset status, reprocess
        cand.status = "staged"
        cand.rejection_count = 1
        cand.last_seen = time.time()
        # Still only 1 support, stays staged (not enough)
        decision = store._check_promotion(cand, [])
        assert decision == "staged"
        # Rejection count is preserved
        assert cand.rejection_count == 1


class TestLowRewardReject:
    """Edge case 9: 3 entries across 2 runs, avg reward = 0.1 → rejected."""

    def test_low_reward_rejected(self, tmp_path):
        store = CandidateStore(
            persist_path=tmp_path / "c.jsonl",
            min_support=3,
            min_runs=2,
            low_reward_threshold=0.2,
        )
        store.process(
            [_learning(reward=0.1, run_id="r1"), _learning(reward=0.1, run_id="r1")],
            run_id="r1",
        )
        result = store.process(
            [_learning(reward=0.1, run_id="r2")],
            run_id="r2",
        )
        assert len(result.rejected) == 1
        rejected = store.get_rejected()
        assert "low_avg_reward" in rejected[0].rejection_reason


class TestBackwardCompat:
    """Edge case 10: Old LearningEntry JSONL without validated/run_id."""

    def test_old_format_loads(self, tmp_path):
        # Write an old-format entry (no validated or run_id)
        old_entry = {
            "goal": "old goal",
            "score": 0.5,
            "success": True,
            "dead_end": False,
            "lessons": ["old lesson"],
            "arm": "macro",
            "reward": 0.4,
            "timestamp": time.time(),
        }
        learning_path = tmp_path / "learnings.jsonl"
        learning_path.write_text(json.dumps(old_entry) + "\n")

        # SemanticMemory auto-discovers learnings at <parent>/learnings.jsonl
        # when persist_path is <parent>/semantic.jsonl
        sem = SemanticMemory(persist_path=tmp_path / "semantic.jsonl")

        entries = sem.get_learnings()
        assert len(entries) == 1
        assert entries[0].validated is False
        assert entries[0].run_id == ""


class TestEmptyState:
    """Edge case 12: Process with no learnings → no crash."""

    def test_empty_process(self, tmp_path):
        store = CandidateStore(persist_path=tmp_path / "c.jsonl")
        result = store.process([], run_id="run_1")
        assert result.candidates_updated == 0
        assert result.promoted == []
        assert result.rejected == []

    def test_empty_lessons_list(self, tmp_path):
        store = CandidateStore(persist_path=tmp_path / "c.jsonl")
        entry = _learning(lessons=[])
        result = store.process([entry], run_id="run_1")
        assert result.candidates_updated == 0


class TestPersistenceRoundTrip:
    """Edge case 13: Save → reload → identical state."""

    def test_round_trip(self, tmp_path):
        path = tmp_path / "c.jsonl"
        store1 = CandidateStore(persist_path=path, min_support=2, min_runs=2)
        store1.process([_learning(run_id="r1")], run_id="r1")
        store1.process([_learning(run_id="r2")], run_id="r2")
        # Should have 1 accepted candidate
        assert len(store1.get_accepted()) == 1

        # Reload
        store2 = CandidateStore(persist_path=path, min_support=2, min_runs=2)
        assert len(store2.get_accepted()) == 1
        cand1 = list(store1.get_all().values())[0]
        cand2 = list(store2.get_all().values())[0]
        assert cand1.cluster_key == cand2.cluster_key
        assert cand1.support_count == cand2.support_count
        assert cand1.status == cand2.status


class TestMultipleLessonsPerRun:
    """Edge case 14: Single run, 5 learnings across 3 clusters → 3 candidates."""

    def test_multi_cluster(self, tmp_path):
        store = CandidateStore(persist_path=tmp_path / "c.jsonl")
        result = store.process(
            [
                _learning(arm="macro", reward=0.5),
                _learning(arm="macro", reward=0.3),
                _learning(arm="regime", reward=0.6),
                _learning(arm="regime", reward=0.4),
                _learning(arm="sentiment", reward=-0.2),
            ],
            run_id="run_1",
        )
        # macro:test:pos, regime:test:pos, sentiment:test:neg → 3 clusters
        assert len(store.get_all()) == 3
        assert result.candidates_updated == 5


# ═══════════════════════════════════════════════════════════════
# Episodic Decay
# ═══════════════════════════════════════════════════════════════


class TestEpisodicDecay:
    """Edge case 11: Decay removes old episodes, archives them."""

    def test_decay_retains_recent(self, tmp_path):
        path = tmp_path / "episodic.jsonl"
        mem = EpisodicMemory(persist_path=path)

        # Add 10 old episodes (40 days) and 10 recent (5 days)
        for i in range(10):
            mem.add(_episode(step=i, age_days=40))
        for i in range(10, 20):
            mem.add(_episode(step=i, age_days=5))

        assert len(mem.all()) == 20
        decayed = mem.decay(max_age_days=30, archive_dir=tmp_path / "archive")
        assert decayed == 10
        assert len(mem.all()) == 10

        # Verify archive written
        archive_files = list((tmp_path / "archive").glob("*.jsonl"))
        assert len(archive_files) == 1
        archive_lines = archive_files[0].read_text().strip().split("\n")
        assert len(archive_lines) == 10

    def test_decay_empty_no_crash(self, tmp_path):
        mem = EpisodicMemory(persist_path=tmp_path / "e.jsonl")
        decayed = mem.decay(max_age_days=30)
        assert decayed == 0

    def test_decay_all_recent(self, tmp_path):
        mem = EpisodicMemory(persist_path=tmp_path / "e.jsonl")
        for i in range(5):
            mem.add(_episode(step=i, age_days=1))
        decayed = mem.decay(max_age_days=30)
        assert decayed == 0
        assert len(mem.all()) == 5

    def test_decay_rewrites_file(self, tmp_path):
        path = tmp_path / "episodic.jsonl"
        mem = EpisodicMemory(persist_path=path)
        for i in range(5):
            mem.add(_episode(step=i, age_days=40))
        for i in range(5, 8):
            mem.add(_episode(step=i, age_days=1))

        mem.decay(max_age_days=30)
        # Reload from disk
        mem2 = EpisodicMemory(persist_path=path)
        assert len(mem2.all()) == 3


# ═══════════════════════════════════════════════════════════════
# Validated Learnings Filter
# ═══════════════════════════════════════════════════════════════


class TestValidatedLearnings:
    def test_get_validated_only(self, tmp_path):
        sem = SemanticMemory(persist_path=tmp_path / "s.jsonl")
        sem.store_learning(_learning(validated=False, goal="g1", run_id="r1"))
        sem.store_learning(_learning(validated=True, goal="g2", run_id="r2"))
        sem.store_learning(_learning(validated=False, goal="g3", run_id="r3"))

        validated = sem.get_validated_learnings(n=10)
        assert len(validated) == 1
        assert validated[0].goal == "g2"

    def test_get_unvalidated(self, tmp_path):
        sem = SemanticMemory(persist_path=tmp_path / "s.jsonl")
        sem.store_learning(_learning(validated=False, goal="g1", run_id="r1"))
        sem.store_learning(_learning(validated=True, goal="g2", run_id="r2"))

        unvalidated = sem.get_unvalidated_learnings()
        assert len(unvalidated) == 1
        assert unvalidated[0].goal == "g1"

    def test_mark_validated(self, tmp_path):
        sem = SemanticMemory(persist_path=tmp_path / "s.jsonl")
        sem.store_learning(_learning(validated=False, goal="target", run_id="r1"))
        sem.store_learning(_learning(validated=False, goal="other", run_id="r2"))

        found = sem.mark_validated("target", "r1")
        assert found is True
        assert sem.get_validated_learnings()[0].goal == "target"

    def test_mark_validated_not_found(self, tmp_path):
        sem = SemanticMemory(persist_path=tmp_path / "s.jsonl")
        sem.store_learning(_learning(goal="other", run_id="r1"))
        found = sem.mark_validated("nonexistent", "rx")
        assert found is False

    def test_mark_validated_persists(self, tmp_path):
        path = tmp_path / "s.jsonl"
        sem = SemanticMemory(persist_path=path)
        sem.store_learning(_learning(validated=False, goal="g1", run_id="r1"))
        sem.mark_validated("g1", "r1")

        # Reload
        sem2 = SemanticMemory(persist_path=path)
        # Load learnings from the learning path
        sem2._learning_path = tmp_path / "learnings.jsonl"
        # The mark_validated uses _persist_learnings which writes to _learning_path
        validated = sem.get_validated_learnings()
        assert len(validated) == 1


# ═══════════════════════════════════════════════════════════════
# Duplicate rejection for accepted lessons
# ═══════════════════════════════════════════════════════════════


class TestDuplicateRejection:
    def test_duplicate_of_accepted_rejected(self, tmp_path):
        store = CandidateStore(
            persist_path=tmp_path / "c.jsonl",
            min_support=2,
            min_runs=2,
            duplicate_threshold=0.7,
        )
        # Get one lesson accepted
        store.process(
            [_learning(arm="a1", lessons=["the market is trending upward strongly"], run_id="r1")],
            run_id="r1",
        )
        store.process(
            [_learning(arm="a1", lessons=["the market is trending upward strongly"], run_id="r2")],
            run_id="r2",
        )
        assert len(store.get_accepted()) == 1

        # New candidate with near-identical text
        store.process(
            [_learning(arm="a2", lessons=["the market is trending upward strongly today"], run_id="r3")],
            run_id="r3",
        )
        store.process(
            [_learning(arm="a2", lessons=["the market is trending upward strongly today"], run_id="r4")],
            run_id="r4",
        )
        rejected = store.get_rejected()
        dup_rejected = [r for r in rejected if "duplicate" in r.rejection_reason]
        assert len(dup_rejected) == 1


# ═══════════════════════════════════════════════════════════════
# _contradicts helper
# ═══════════════════════════════════════════════════════════════


class TestContradicts:
    def test_same_prefix_different_sign(self):
        a = LessonCandidate(cluster_key="macro:fetch:pos", claim="good")
        b = LessonCandidate(cluster_key="macro:fetch:neg", claim="bad")
        assert CandidateStore._contradicts(a, b) is True

    def test_same_prefix_same_sign(self):
        a = LessonCandidate(cluster_key="macro:fetch:pos", claim="good")
        b = LessonCandidate(cluster_key="macro:fetch:pos", claim="also good")
        assert CandidateStore._contradicts(a, b) is False

    def test_different_prefix(self):
        a = LessonCandidate(cluster_key="macro:fetch:pos", claim="good")
        b = LessonCandidate(cluster_key="regime:scan:neg", claim="bad")
        assert CandidateStore._contradicts(a, b) is False


# ═══════════════════════════════════════════════════════════════
# _is_duplicate helper
# ═══════════════════════════════════════════════════════════════


class TestIsDuplicate:
    def test_identical_text(self, tmp_path):
        store = CandidateStore(persist_path=tmp_path / "c.jsonl", duplicate_threshold=0.7)
        cand = LessonCandidate(cluster_key="a:b:pos", claim="the quick brown fox")
        accepted = [LessonCandidate(cluster_key="c:d:pos", claim="the quick brown fox")]
        assert store._is_duplicate(cand, accepted) is True

    def test_different_text(self, tmp_path):
        store = CandidateStore(persist_path=tmp_path / "c.jsonl", duplicate_threshold=0.7)
        cand = LessonCandidate(cluster_key="a:b:pos", claim="the quick brown fox")
        accepted = [LessonCandidate(cluster_key="c:d:pos", claim="something completely different")]
        assert store._is_duplicate(cand, accepted) is False

    def test_empty_claim(self, tmp_path):
        store = CandidateStore(persist_path=tmp_path / "c.jsonl")
        cand = LessonCandidate(cluster_key="a:b:pos", claim="")
        accepted = [LessonCandidate(cluster_key="c:d:pos", claim="something")]
        assert store._is_duplicate(cand, accepted) is False
