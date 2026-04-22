"""
TirraMind Agent — Lesson Candidate Store

Statistical lesson-promotion pipeline.  New LearningEntries are *staged* as
candidates.  A candidate is only promoted (``status='accepted'``) when it
accumulates enough cross-run evidence.  This prevents noisy single-run
observations from polluting the lesson pool that feeds reflection and goal
generation.

Pipeline (called after each autonomous run):
  1. Cluster new learnings by (arm, primary_tool, reward_sign).
  2. Merge into existing candidates (append evidence).
  3. Evaluate each staged candidate against promotion / rejection rules.
  4. Return a ProcessResult describing what changed.

Inspired by the candidate lifecycle in codejunkie99/agentic-stack (MIT), but
adapted for autonomous statistical operation: quantitative thresholds replace
human CLI review.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.memory.store import LearningEntry

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class LessonCandidate:
    """A candidate lesson awaiting sufficient evidence for promotion."""

    cluster_key: str  # "{arm}:{primary_tool}:{reward_sign}"
    claim: str  # first lesson text that seeded this candidate
    evidence: list[dict[str, Any]] = field(default_factory=list)
    support_count: int = 0
    distinct_runs: list[str] = field(default_factory=list)
    avg_reward: float = 0.0
    reward_sign_agreement: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    status: str = "staged"  # staged | accepted | rejected
    rejection_count: int = 0
    rejection_reason: str = ""

    @property
    def distinct_run_count(self) -> int:
        return len(set(self.distinct_runs))


@dataclass
class ProcessResult:
    """Summary of a single process() call."""

    candidates_updated: int = 0
    promoted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    still_staged: int = 0


# --------------------------------------------------------------------------- #
# CandidateStore
# --------------------------------------------------------------------------- #


class CandidateStore:
    """Manages the lifecycle of lesson candidates on disk + in memory."""

    def __init__(
        self,
        persist_path: Path | None = None,
        *,
        min_support: int = 3,
        min_runs: int = 2,
        sign_agreement_threshold: float = 0.8,
        duplicate_threshold: float = 0.7,
        low_reward_threshold: float = 0.2,
        stale_days: int = 14,
    ) -> None:
        self._persist_path = persist_path
        self._min_support = min_support
        self._min_runs = min_runs
        self._sign_threshold = sign_agreement_threshold
        self._dup_threshold = duplicate_threshold
        self._low_reward = low_reward_threshold
        self._stale_days = stale_days
        self._candidates: dict[str, LessonCandidate] = {}
        if persist_path and persist_path.exists():
            self._load()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process(
        self, new_learnings: list[LearningEntry], run_id: str
    ) -> ProcessResult:
        """Main entry point — cluster, merge, evaluate."""
        result = ProcessResult()

        # 1. Cluster new learnings and merge into candidates
        for entry in new_learnings:
            if not entry.lessons:
                continue
            key = self._cluster_key(entry)
            if key not in self._candidates:
                self._candidates[key] = LessonCandidate(
                    cluster_key=key,
                    claim=entry.lessons[0],
                    first_seen=entry.timestamp,
                )
            cand = self._candidates[key]
            cand.evidence.append(
                {
                    "run_id": run_id,
                    "timestamp": entry.timestamp,
                    "reward": entry.reward,
                    "lesson_text": "; ".join(entry.lessons),
                    "goal": entry.goal,
                }
            )
            if run_id and run_id not in cand.distinct_runs:
                cand.distinct_runs.append(run_id)
            cand.support_count = len(cand.evidence)
            cand.last_seen = entry.timestamp
            cand.avg_reward = (
                sum(e["reward"] for e in cand.evidence) / cand.support_count
            )
            cand.reward_sign_agreement = self._sign_agreement(cand)
            result.candidates_updated += 1

        # 2. Evaluate all staged candidates
        accepted_list = self.get_accepted()
        for cand in list(self._candidates.values()):
            if cand.status != "staged":
                continue
            decision = self._check_promotion(cand, accepted_list)
            if decision == "accept":
                cand.status = "accepted"
                result.promoted.append(cand.cluster_key)
                log.info(
                    "Lesson promoted: %s (support=%d, runs=%d, sign=%.2f)",
                    cand.cluster_key,
                    cand.support_count,
                    cand.distinct_run_count,
                    cand.reward_sign_agreement,
                )
            elif decision == "reject":
                cand.status = "rejected"
                cand.rejection_count += 1
                result.rejected.append(cand.cluster_key)
                log.info(
                    "Lesson rejected: %s reason=%s",
                    cand.cluster_key,
                    cand.rejection_reason,
                )
            else:
                result.still_staged += 1

        self._persist()
        return result

    def get_accepted(self) -> list[LessonCandidate]:
        return [c for c in self._candidates.values() if c.status == "accepted"]

    def get_staged(self) -> list[LessonCandidate]:
        return [c for c in self._candidates.values() if c.status == "staged"]

    def get_rejected(self) -> list[LessonCandidate]:
        return [c for c in self._candidates.values() if c.status == "rejected"]

    def get_all(self) -> dict[str, LessonCandidate]:
        return dict(self._candidates)

    # ------------------------------------------------------------------ #
    # Clustering
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cluster_key(entry: LearningEntry) -> str:
        arm = entry.arm or "unknown"
        tool = entry.goal.split()[0] if entry.goal else "unknown"
        sign = "pos" if entry.reward >= 0 else "neg"
        return f"{arm}:{tool}:{sign}"

    # ------------------------------------------------------------------ #
    # Promotion rules
    # ------------------------------------------------------------------ #

    def _check_promotion(
        self,
        candidate: LessonCandidate,
        accepted: list[LessonCandidate],
    ) -> str:
        """Returns 'accept', 'reject', or 'staged'."""
        # Rejection: low average reward
        if (
            candidate.support_count >= self._min_support
            and candidate.avg_reward < self._low_reward
        ):
            candidate.rejection_reason = (
                f"low_avg_reward ({candidate.avg_reward:.3f})"
            )
            return "reject"

        # Rejection: stale — no new evidence in N days
        age_since_last = time.time() - candidate.last_seen
        if age_since_last > self._stale_days * 86_400 and candidate.support_count < self._min_support:
            candidate.rejection_reason = "stale_insufficient_evidence"
            return "reject"

        # Rejection: contradicts accepted lesson with higher support
        for acc in accepted:
            if self._contradicts(candidate, acc) and acc.support_count > candidate.support_count:
                candidate.rejection_reason = (
                    f"contradicts_accepted:{acc.cluster_key}"
                )
                return "reject"

        # Promotion check
        if candidate.support_count < self._min_support:
            return "staged"
        if candidate.distinct_run_count < self._min_runs:
            return "staged"
        if candidate.reward_sign_agreement < self._sign_threshold:
            return "staged"
        if self._is_duplicate(candidate, accepted):
            candidate.rejection_reason = "duplicate_of_accepted"
            return "reject"

        return "accept"

    @staticmethod
    def _sign_agreement(candidate: LessonCandidate) -> float:
        if not candidate.evidence:
            return 0.0
        pos = sum(1 for e in candidate.evidence if e["reward"] >= 0)
        neg = len(candidate.evidence) - pos
        return max(pos, neg) / len(candidate.evidence)

    @staticmethod
    def _contradicts(a: LessonCandidate, b: LessonCandidate) -> bool:
        """Two candidates contradict if they share arm:tool but differ in sign."""
        parts_a = a.cluster_key.rsplit(":", 1)
        parts_b = b.cluster_key.rsplit(":", 1)
        if len(parts_a) != 2 or len(parts_b) != 2:
            return False
        prefix_a, sign_a = parts_a
        prefix_b, sign_b = parts_b
        return prefix_a == prefix_b and sign_a != sign_b

    def _is_duplicate(
        self, candidate: LessonCandidate, accepted: list[LessonCandidate]
    ) -> bool:
        tokens_a = set(candidate.claim.lower().split())
        if not tokens_a:
            return False
        for acc in accepted:
            tokens_b = set(acc.claim.lower().split())
            if not tokens_b:
                continue
            intersection = tokens_a & tokens_b
            union = tokens_a | tokens_b
            jaccard = len(intersection) / len(union)
            if jaccard >= self._dup_threshold:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _persist(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "w") as f:
            for cand in self._candidates.values():
                f.write(json.dumps(asdict(cand)) + "\n")

    def _load(self) -> None:
        try:
            for line in self._persist_path.read_text().strip().split("\n"):
                if not line:
                    continue
                d = json.loads(line)
                cand = LessonCandidate(**d)
                self._candidates[cand.cluster_key] = cand
        except Exception as exc:
            log.warning("Failed to load candidate store: %s", exc)
