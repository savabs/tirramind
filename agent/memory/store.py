"""
TirraMind Agent — Memory Systems

Three tiers:
  - Episodic: append-only log of actions and outcomes (the audit trail)
  - Semantic: key facts and knowledge indexed for retrieval
  - Working: rolling context window for the current task
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# =====================================================================
# Episodic Memory — what happened
# =====================================================================


@dataclass
class Episode:
    """A single step in the agent's history."""

    timestamp: float
    step: int
    action: str  # tool name or "plan" / "think"
    input_summary: str  # what was asked / parameters
    output_summary: str  # what came back
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """Append-only event log. Persists to JSONL file."""

    def __init__(self, persist_path: Path | None = None) -> None:
        self._episodes: list[Episode] = []
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

    def add(self, episode: Episode) -> None:
        self._episodes.append(episode)
        if self._persist_path:
            self._append_to_disk(episode)

    def recent(self, n: int = 10) -> list[Episode]:
        return self._episodes[-n:]

    def all(self) -> list[Episode]:
        return list(self._episodes)

    def decay(self, max_age_days: int = 30, archive_dir: Path | None = None) -> int:
        """Remove episodes older than *max_age_days*.

        If *archive_dir* is provided, the removed episodes are written there
        before deletion.  Returns the number of decayed episodes.
        """
        cutoff = time.time() - max_age_days * 86_400
        keep: list[Episode] = []
        remove: list[Episode] = []
        for ep in self._episodes:
            (keep if ep.timestamp >= cutoff else remove).append(ep)
        if not remove:
            return 0
        # Archive before deleting
        if archive_dir is not None:
            archive_dir.mkdir(parents=True, exist_ok=True)
            ts_label = time.strftime("%Y%m%d")
            archive_path = archive_dir / f"episodic_{ts_label}.jsonl"
            with open(archive_path, "a") as f:
                for ep in remove:
                    f.write(json.dumps(asdict(ep)) + "\n")
        self._episodes = keep
        # Rewrite the main file with only retained episodes
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "w") as f:
                for ep in keep:
                    f.write(json.dumps(asdict(ep)) + "\n")
        return len(remove)

    def summary(self, n: int = 10) -> str:
        """Human-readable summary of recent episodes."""
        lines = []
        for ep in self.recent(n):
            status = "✓" if ep.success else "✗"
            lines.append(
                f"  [{status}] Step {ep.step}: {ep.action}({ep.input_summary[:60]}) → {ep.output_summary[:80]}"
            )
        return "\n".join(lines) or "(no history)"

    def _append_to_disk(self, episode: Episode) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "a") as f:
            f.write(json.dumps(asdict(episode)) + "\n")

    def _load(self) -> None:
        try:
            for line in self._persist_path.read_text().strip().split("\n"):
                if line:
                    d = json.loads(line)
                    self._episodes.append(Episode(**d))
        except Exception as exc:
            log.warning("Failed to load episodic memory: %s", exc)


# =====================================================================
# Semantic Memory — what things mean (lightweight in-memory for now)
# =====================================================================


@dataclass
class Fact:
    """A piece of knowledge extracted from agent activity."""

    key: str
    content: str
    source: str  # which episode / tool produced this
    confidence: float  # 0-1
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    tainted: bool = False  # True if sourced from untrusted content (web, etc.)


@dataclass
class LearningEntry:
    """Structured record of a goal attempt and its outcome."""

    goal: str
    score: float  # 0-1 evaluation score
    success: bool
    dead_end: bool = False
    lessons: list[str] = field(default_factory=list)
    arm: str = ""  # bandit arm that was chosen (RL layer)
    reward: float = 0.0  # scalar reward fed to bandit (RL layer)
    timestamp: float = field(default_factory=time.time)
    validated: bool = False  # True only after promotion pipeline accepts
    run_id: str = ""  # identifies which autonomous run produced this


class SemanticMemory:
    """Key-value fact store with tag-based retrieval.

    Phase 1: in-memory dictionary + JSONL persistence.
    Phase 2+: upgrade to ChromaDB / FAISS for vector similarity search.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._facts: dict[str, Fact] = {}
        self._learnings: list[LearningEntry] = []
        self._persist_path = persist_path
        self._learning_path = persist_path.parent / "learnings.jsonl" if persist_path else None
        if persist_path and persist_path.exists():
            self._load()
        if self._learning_path and self._learning_path.exists():
            self._load_learnings()

    def store(self, fact: Fact) -> None:
        self._facts[fact.key] = fact
        self._persist()

    def get(self, key: str) -> Fact | None:
        return self._facts.get(key)

    def search(self, query: str, limit: int = 5) -> list[Fact]:
        """Simple substring match. Upgrade to vector search later."""
        query_lower = query.lower()
        scored = []
        for fact in self._facts.values():
            text = f"{fact.key} {fact.content} {' '.join(fact.tags)}".lower()
            if query_lower in text:
                scored.append(fact)
        return scored[:limit]

    def all_facts(self) -> list[Fact]:
        return list(self._facts.values())

    def summary(self) -> str:
        if not self._facts:
            return "(no facts stored)"
        lines = [f"  [{f.key}] {f.content[:80]} (conf={f.confidence:.1f})" for f in self._facts.values()]
        return "\n".join(lines[-10:])

    # ------------------------------------------------------------------
    # Learning entries
    # ------------------------------------------------------------------

    def store_learning(self, entry: LearningEntry) -> None:
        """Record a goal attempt and its outcome."""
        self._learnings.append(entry)
        if self._learning_path:
            self._learning_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._learning_path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")

    def get_attempted_goals(self) -> list[str]:
        """Return descriptions of all previously attempted goals."""
        return [le.goal for le in self._learnings]

    def get_dead_ends(self) -> list[str]:
        """Return goals that were marked as dead ends."""
        return [le.goal for le in self._learnings if le.dead_end]

    def get_learnings(self, n: int = 10) -> list[LearningEntry]:
        """Return the most recent learning entries."""
        return self._learnings[-n:]

    def get_validated_learnings(self, n: int = 10) -> list[LearningEntry]:
        """Return only lessons that passed the promotion pipeline."""
        validated = [le for le in self._learnings if le.validated]
        return validated[-n:]

    def get_unvalidated_learnings(self) -> list[LearningEntry]:
        """Return lessons still awaiting promotion review."""
        return [le for le in self._learnings if not le.validated]

    def mark_validated(self, goal: str, run_id: str) -> bool:
        """Set validated=True on the matching LearningEntry and re-persist.

        Matches by (goal, run_id) pair.  Returns True if a match was found.
        """
        for le in self._learnings:
            if le.goal == goal and le.run_id == run_id:
                le.validated = True
                self._persist_learnings()
                return True
        return False

    def _persist_learnings(self) -> None:
        """Rewrite the full learnings file (used after in-place mutation)."""
        if not self._learning_path:
            return
        self._learning_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._learning_path, "w") as f:
            for le in self._learnings:
                f.write(json.dumps(asdict(le)) + "\n")

    def _load_learnings(self) -> None:
        try:
            for line in self._learning_path.read_text().strip().split("\n"):
                if line:
                    d = json.loads(line)
                    self._learnings.append(LearningEntry(**d))
        except Exception as exc:
            log.warning("Failed to load learnings: %s", exc)

    def _persist(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "w") as f:
            for fact in self._facts.values():
                f.write(json.dumps(asdict(fact)) + "\n")

    def _load(self) -> None:
        try:
            for line in self._persist_path.read_text().strip().split("\n"):
                if line:
                    d = json.loads(line)
                    fact = Fact(**d)
                    self._facts[fact.key] = fact
        except Exception as exc:
            log.warning("Failed to load semantic memory: %s", exc)


# =====================================================================
# Working Memory — current conversation context for the LLM
# =====================================================================


class WorkingMemory:
    """Manages the rolling message list sent to the LLM.

    Keeps a system prompt + recent messages within a token budget.
    Simple strategy: keep the most recent N messages.
    """

    def __init__(self, system_prompt: str, max_messages: int = 40) -> None:
        self._system_prompt = system_prompt
        self._max_messages = max_messages
        self._messages: list[dict[str, Any]] = []

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def add(self, message: dict[str, Any]) -> None:
        self._messages.append(message)
        # Trim if over budget — keep system prompt + recent messages
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages :]

    def add_user(self, content: str) -> None:
        self.add({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.add({"role": "assistant", "content": content})

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.add({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def get_messages(self) -> list[dict[str, Any]]:
        """Full message list with system prompt prepended."""
        return [{"role": "system", "content": self._system_prompt}] + self._messages

    def clear(self) -> None:
        self._messages.clear()
