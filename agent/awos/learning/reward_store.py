"""RewardStore — outcome ledger + reward math for the learning runtime.

Adapted from AWOS reward_store.py. Every completed signal action (a fetch,
a scoring attempt, a fusion, an alert) is stored as an Episode. The learned
router / policy layer reads these to improve.

Reward math is preserved from AWOS (asymmetric cost-proportional reward):
    R(success) = 1 - eta * (cost/max_cost)          opportunity cost of overspend
    R(fail)    = -kappa * log(1 + spread*cost/max)/log(1+spread)   wasted spend
Effective range: [-kappa, 1]  (default [-0.30, 1.0])
"""

from __future__ import annotations

import json
import logging
import math
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_STORE_PATH = Path(".awos") / "reward_store.jsonl"

# Cost ladder for the signal domain (per-action cost in $). Kept cheap-first.
_MIN_COST = 0.001
_MAX_COST = 0.050
_COST_SPREAD = _MAX_COST / _MIN_COST  # 50.0

_ETA = 0.05    # opportunity-cost weight for successes
_KAPPA = 0.30  # max failure-penalty magnitude

_TIER_DEFAULT_COST = {0: 0.001, 1: 0.001, 2: 0.003, 3: 0.017, 4: 0.050}

# Action ids for the signal domain (mirrors AWOS escalation ladder)
N_ACTIONS = 6
ACTION_NAMES = [
    "heuristic",   # 0 — rule-based, cheap (default)
    "cheap_llm",   # 1 — cheap model assist
    "rich_llm",    # 2 — richer reasoning
    "statistical", # 3 — quant/stat method
    "ml_model",    # 4 — learned model (GNN/router)
    "ensemble",    # 5 — combination / escalation
]


@dataclass
class Episode:
    """One completed signal action attempt with its outcome."""

    episode_id: str
    task_id: str
    action_id: int
    features: list[float]
    success: bool
    cost_usd: float
    latency_ms: float = 0.0
    reward: float = 0.0
    signal_name: str = ""
    source_tool: str = ""
    model_name: str = ""
    task_action_text: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    reward_breakdown: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Episode:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _cost_ratio(cost_usd: float, action_id: int) -> float:
    cost = cost_usd if cost_usd > 0 else _TIER_DEFAULT_COST.get(action_id, _MIN_COST)
    return min(cost / _MAX_COST, 1.0)


def compute_reward(success, action_id: int, cost_usd: float) -> float:
    """Asymmetric cost-proportional reward (preserved from AWOS)."""
    ratio = _cost_ratio(cost_usd, action_id)
    opportunity_cost = _ETA * ratio

    if isinstance(success, float):
        pass_rate = max(0.0, min(1.0, success))
        r_success = compute_reward(True, action_id, cost_usd)
        r_failure = compute_reward(False, action_id, cost_usd)
        return round(r_success * pass_rate + r_failure * (1.0 - pass_rate), 4)

    if success:
        base_value = 1.0
        failure_penalty = 0.0
    else:
        base_value = 0.0
        raw = math.log(1.0 + _COST_SPREAD * ratio)
        norm = math.log(1.0 + _COST_SPREAD)
        failure_penalty = _KAPPA * (raw / norm)

    return round(base_value - opportunity_cost - failure_penalty, 4)


def compute_reward_breakdown(success: bool, action_id: int, cost_usd: float) -> dict[str, float]:
    ratio = _cost_ratio(cost_usd, action_id)
    opportunity_cost = _ETA * ratio
    base_value = 1.0 if success else 0.0
    if success:
        failure_penalty = 0.0
    else:
        raw = math.log(1.0 + _COST_SPREAD * ratio)
        norm = math.log(1.0 + _COST_SPREAD)
        failure_penalty = _KAPPA * (raw / norm)
    total = round(base_value - opportunity_cost - failure_penalty, 4)
    return {
        "base_value": round(base_value, 4),
        "opportunity_cost": round(-opportunity_cost, 4),
        "failure_penalty": round(-failure_penalty, 4),
        "total_reward": total,
        "cost_ratio": round(ratio, 4),
        "params": {"eta": _ETA, "kappa": _KAPPA, "spread": _COST_SPREAD},
    }


class ReplayGate:
    """Filters episodes before the learned router uses them (AWOS-adapted)."""

    def __init__(self, threshold: float = 0.4, novelty_window: int = 50) -> None:
        self._base_threshold = threshold
        self._novelty_window = novelty_window
        self._recent_features: list[list[float]] = []
        self._action_counts: list[int] = [0] * N_ACTIONS
        self._total_episodes = 0
        self._admitted_count = 0

    @property
    def threshold(self) -> float:
        if self._total_episodes == 0:
            return self._base_threshold
        return max(0.15, self._base_threshold - 0.01 * self._total_episodes**0.5)

    def admit_rate(self) -> float:
        if self._total_episodes == 0:
            return 0.0
        return self._admitted_count / self._total_episodes

    def score(self, episode: Episode) -> float:
        r = abs(episode.reward)
        informativeness = min(r / 1.0, 1.0)
        if self._recent_features:
            dists = [
                math.sqrt(sum((a - b) ** 2 for a, b in zip(episode.features, ref)))
                for ref in self._recent_features[-self._novelty_window:]
            ]
            min_dist = min(dists)
            novelty = min(min_dist / 0.75, 1.0)
        else:
            novelty = 1.0
        total_acts = sum(self._action_counts) + 1
        majority_frac = max(self._action_counts) / total_acts if total_acts > 0 else 0
        a = episode.action_id
        own_frac = (
            self._action_counts[a] / total_acts
            if (total_acts > 0 and 0 <= a < len(self._action_counts))
            else 0.0
        )
        boundary = 1.0 - own_frac / max(majority_frac, 1e-6)
        boundary = max(0.0, min(boundary, 1.0))
        return informativeness * 0.4 + novelty * 0.2 + boundary * 0.4

    def admit(self, episode: Episode) -> bool:
        s = self.score(episode)
        self._recent_features.append(list(episode.features))
        if len(self._recent_features) > self._novelty_window * 2:
            self._recent_features = self._recent_features[-self._novelty_window:]
        a = episode.action_id
        if 0 <= a < len(self._action_counts):
            self._action_counts[a] += 1
        self._total_episodes += 1
        admitted = s >= self.threshold
        if admitted:
            self._admitted_count += 1
        return admitted


class RewardStore:
    """Append-only JSONL log of signal-action outcomes."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else _DEFAULT_STORE_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        task: dict[str, Any],
        action_id: int,
        features: list[float],
        success: bool,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        model_name: str = "",
        signal_name: str = "",
        source_tool: str = "",
    ) -> Episode:
        reward = compute_reward(success, action_id, cost_usd)
        breakdown = compute_reward_breakdown(success, action_id, cost_usd)
        episode = Episode(
            episode_id=str(uuid.uuid4()),
            task_id=str(task.get("task_id", "unknown")),
            action_id=action_id,
            features=list(features),
            success=success,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            reward=reward,
            signal_name=signal_name,
            source_tool=source_tool,
            model_name=model_name,
            task_action_text=str(task.get("action", ""))[:200],
            reward_breakdown=breakdown,
        )
        self._append(episode)
        return episode

    def get_recent(self, n: int = 200) -> list[Episode]:
        lines = self._read_lines()
        recent_lines = lines[-n:] if len(lines) > n else lines
        out: list[Episode] = []
        for line in recent_lines:
            try:
                out.append(Episode.from_dict(json.loads(line)))
            except Exception as exc:
                logger.warning("[reward_store] skipping malformed line: %s", exc)
        return out

    def total_episodes(self) -> int:
        if not self._path.exists():
            return 0
        return sum(1 for _ in self._path.open(encoding="utf-8"))

    def success_rate(self, action_id: int | None = None, window: int = 50) -> float:
        episodes = self.get_recent(window)
        if action_id is not None:
            episodes = [e for e in episodes if e.action_id == action_id]
        if not episodes:
            return 0.0
        return sum(1 for e in episodes if e.success) / len(episodes)

    def summary(self) -> dict[str, Any]:
        episodes = self.get_recent(200)
        if not episodes:
            return {"total": 0, "message": "no data yet"}
        total = len(episodes)
        by_action: dict[int, dict[str, Any]] = {}
        for e in episodes:
            a = e.action_id
            if a not in by_action:
                by_action[a] = {"count": 0, "success": 0, "total_reward": 0.0}
            by_action[a]["count"] += 1
            by_action[a]["success"] += int(e.success)
            by_action[a]["total_reward"] += e.reward
        stats = {}
        for a_id, data in sorted(by_action.items()):
            name = ACTION_NAMES[a_id] if a_id < len(ACTION_NAMES) else str(a_id)
            stats[name] = {
                "count": data["count"],
                "success_rate": round(data["success"] / data["count"], 3),
                "avg_reward": round(data["total_reward"] / data["count"], 3),
            }
        return {
            "total_episodes": total,
            "overall_success_rate": round(sum(e.success for e in episodes) / total, 3),
            "by_action": stats,
        }

    def get_prioritized(self, n: int = 32, alpha: float = 0.6) -> list[Episode]:
        episodes = self.get_recent(min(n * 10, 500))
        if not episodes:
            return []
        weights = [abs(e.reward) ** alpha for e in episodes]
        total_w = sum(weights)
        if total_w == 0:
            return random.sample(episodes, min(n, len(episodes)))
        probs = [w / total_w for w in weights]
        population = list(range(len(episodes)))
        indices = random.choices(population, weights=probs, k=min(n, len(episodes)))
        seen: set[int] = set()
        chosen: list[Episode] = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                chosen.append(episodes[i])
        return chosen[:n]

    # ── Internals ───────────────────────────────────────────────────────────
    def _append(self, episode: Episode) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")

    def _read_lines(self) -> list[str]:
        if not self._path.exists():
            return []
        with self._path.open(encoding="utf-8") as f:
            return [l for l in f.read().splitlines() if l.strip()]


__all__ = [
    "Episode",
    "RewardStore",
    "ReplayGate",
    "compute_reward",
    "compute_reward_breakdown",
    "ACTION_NAMES",
    "N_ACTIONS",
]
