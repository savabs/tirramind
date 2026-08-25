"""SkillLibrary — learned "what works" memory for the signal runtime.

Adapted from AWOS into the learning subpackage. After a verified-successful
signal action (a tender match, a fetch, a scoring approach, an alert), the
runtime records which approach won, so future runs can reuse it.

Key facts are method/strategy patterns, not just keywords:
  - strategy: the winning approach (scoring formula, fetch mode, fusion policy)
  - keyword hints: the signal/tool context it applied to
  - win_rate: how often this approach worked historically
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_DIR = Path(".awos") / "skills"

# Approach families for the signal domain (mirrors AWOS task types, adapted).
_APPROACH_SIGNALS: dict[str, tuple[str, ...]] = {
    "scoring": ("score", "rank", "order", "win", "probability", "margin", "risk", "expected"),
    "fetch": ("fetch", "pull", "load", "retrieve", "query", "scrape", "collect", "download"),
    "fusion": ("fuse", "merge", "combine", "join", "overlay", "correlate", "link", "match"),
    "alert": ("alert", "notify", "digest", "report", "flag", "watch", "monitor", "threshold"),
    "clean": ("normalize", "parse", "dedup", "clean", "validate", "resolve", "transform"),
}


@dataclass
class SkillEntry:
    """One learned winning approach."""

    approach: str  # scoring / fetch / fusion / alert / clean / other
    keywords: list[str]
    strategy: str  # the winning method, e.g. "zscore>2 + setaside filter"
    source_tool: str
    attempts: int
    success_count: int = 1
    total_count: int = 1
    last_seen: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d")
    )

    @property
    def win_rate(self) -> float:
        return round(self.success_count / max(self.total_count, 1), 2)

    @property
    def avg_attempts(self) -> float:
        return round(self.attempts / max(self.success_count, 1), 1)


class SkillLibrary:
    """Records winning signal approaches and provides context for future actions."""

    MAX_ENTRIES = 500
    MAX_CONTEXT_SKILLS = 3

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._dir = Path(skills_dir) if skills_dir is not None else _DEFAULT_SKILLS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._entries: list[SkillEntry] = []
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────
    def record(
        self,
        signal_name: str,
        source_tool: str,
        strategy: str,
        attempts: int = 1,
        operation: str = "",
    ) -> None:
        """Record a winning approach for a signal/action."""
        approach = _classify(f"{operation} {signal_name} {strategy} {source_tool}")
        keywords = _extract_keywords(f"{source_tool} {signal_name} {strategy}")

        existing = self._find_match(approach, keywords, strategy, source_tool)
        if existing is not None:
            existing.success_count += 1
            existing.total_count += 1
            existing.attempts += attempts
            existing.last_seen = datetime.now(UTC).strftime("%Y-%m-%d")
        else:
            entry = SkillEntry(
                approach=approach,
                keywords=keywords[:5],
                strategy=strategy,
                source_tool=source_tool,
                attempts=attempts,
            )
            self._entries.append(entry)
            if len(self._entries) > self.MAX_ENTRIES:
                self._entries = self._entries[-self.MAX_ENTRIES:]

        self._save()
        self._write_skill_file(approach)

    def get_context(self, signal_name: str, source_tool: str) -> str:
        """Return a short 'what works here' block for injection/routing."""
        keywords = set(_extract_keywords(source_tool + " " + signal_name))
        scored: list[tuple[float, SkillEntry]] = []
        for e in self._entries:
            overlap = len(keywords & set(e.keywords))
            score = overlap + e.win_rate * 0.5
            if score > 0:
                scored.append((score, e))
        if not scored:
            return ""
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [e for _, e in scored[: self.MAX_CONTEXT_SKILLS]]
        lines = [f"SKILL CONTEXT (past winning approaches for '{signal_name}'):"]
        for e in top:
            kw = ", ".join(e.keywords[:3]) if e.keywords else "general"
            lines.append(
                f"  • [{e.approach}:{e.source_tool}] → '{e.strategy}' "
                f"won {e.success_count}× on tasks like '{kw}' "
                f"(avg {e.avg_attempts} attempt(s), {e.win_rate} rate)"
            )
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        by_approach: dict[str, list[dict[str, Any]]] = {}
        for e in self._entries:
            by_approach.setdefault(e.approach, []).append(
                {
                    "strategy": e.strategy,
                    "source_tool": e.source_tool,
                    "win_rate": e.win_rate,
                    "success_count": e.success_count,
                }
            )
        return by_approach

    def total_entries(self) -> int:
        return len(self._entries)

    # ── Skill file writer ──────────────────────────────────────────────────
    def _write_skill_file(self, approach: str) -> None:
        entries = [e for e in self._entries if e.approach == approach]
        if not entries:
            return
        entries.sort(key=lambda e: (e.win_rate, e.success_count), reverse=True)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        lines = [
            f"# AWOS Skills: {approach}",
            "",
            f"Auto-generated from successful signal actions. Last updated: {today}.",
            "Do not edit manually — overwritten on each new success.",
            "",
            f"## Summary ({len(entries)} pattern(s))",
            "",
        ]
        for e in entries[:20]:
            kw = ", ".join(e.keywords[:4]) if e.keywords else "(general)"
            lines += [
                f"### Strategy: {e.strategy}",
                f"- **Source tool**: {e.source_tool}",
                f"- **Win rate**: {e.win_rate * 100:.0f}% ({e.success_count}/{e.total_count})",
                f"- **Avg attempts**: {e.avg_attempts}",
                f"- **Keywords**: {kw}",
                f"- **Last seen**: {e.last_seen}",
                "",
            ]
        try:
            (self._dir / f"{approach}.md").write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            logger.warning("[skill_library] could not write skill file: %s", exc)

    # ── Persistence ────────────────────────────────────────────────────────
    def _save(self) -> None:
        try:
            self._index_path.write_text(
                json.dumps([asdict(e) for e in self._entries], indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("[skill_library] save failed: %s", exc)

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            self._entries = [SkillEntry(**d) for d in data]
        except Exception as exc:
            logger.warning("[skill_library] load failed: %s", exc)
            self._entries = []

    # ── Internal helpers ───────────────────────────────────────────────────
    def _find_match(
        self,
        approach: str,
        keywords: list[str],
        strategy: str,
        source_tool: str,
    ) -> SkillEntry | None:
        top_kw = keywords[0] if keywords else ""
        for e in self._entries:
            if (
                e.approach == approach
                and e.strategy == strategy
                and e.source_tool == source_tool
                and (top_kw in e.keywords or not top_kw)
            ):
                return e
        return None


# ── Helpers ────────────────────────────────────────────────────────────────
def _classify(text: str) -> str:
    t = text.lower()
    for approach, signals in _APPROACH_SIGNALS.items():
        if any(sig in t for sig in signals):
            return approach
    return "other"


_STOPWORDS = frozenset(
    {
        "a", "an", "the", "to", "in", "for", "of", "and", "or", "is",
        "it", "this", "that", "with", "add", "update", "fix", "make",
        "into", "from", "using", "when", "should", "will",
    }
)


def _extract_keywords(text: str) -> list[str]:
    words = re.findall(r"[a-z_][a-z0-9_]*", text.lower())
    filtered = [w for w in words if len(w) > 3 and w not in _STOPWORDS]
    seen: dict[str, None] = {}
    for w in filtered:
        seen[w] = None
    return sorted(seen.keys(), key=len, reverse=True)[:8]


__all__ = ["SkillEntry", "SkillLibrary"]
