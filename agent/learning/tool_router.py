"""TirraMind — Learned Tool Routing (Change 12)

Contextual Thompson Sampling bandit that decides which surveillance tools
to invoke each DAG cycle.  Replaces the fixed schedule where all tools
run unconditionally.

Each optional tool is an arm with Beta(α, β) distribution.  At each
decision point the bandit samples from each arm's posterior and selects
tools whose sample exceeds a threshold:

    θ_i ~ Beta(α_i, β_i)
    run tool i  iff  θ_i > threshold

After execution, the tool's signal contribution (how much its data
contributed to downstream entity alerts) is used as the reward to update
the arm.

Always-on tools (e.g. ``fetch_instruments``) bypass the bandit entirely.

Reference: Thompson 1933 (Thompson Sampling), Agrawal & Goyal 2013
    (Analysis of Thompson Sampling for the Multi-armed Bandit Problem).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolContext:
    """Context features for tool routing decisions.

    Parameters
    ----------
    regime_id : int
        Current HMM regime index (0-based).
    day_of_week : int
        0=Monday .. 4=Friday.
    tool_staleness : dict[str, float]
        Hours since last successful fetch per tool.
    """

    regime_id: int = 0
    day_of_week: int = 0
    tool_staleness: dict[str, float] = field(default_factory=dict)


# Tools that must always run regardless of bandit decision
ALWAYS_ON_TOOLS: frozenset[str] = frozenset({"fetch_instruments"})

# Default optional tools in the daily collection DAG
DEFAULT_OPTIONAL_TOOLS: tuple[str, ...] = (
    "fetch_cftc",
    "fetch_finra_scan",
    "fetch_power_demand",
    "fetch_power_fuel",
    "fetch_gdelt",
    "fetch_polymarket",
)


class ToolRoutingBandit:
    """Contextual Thompson Sampling for surveillance tool selection.

    Each optional tool has a Beta(α, β) distribution.  The bandit samples
    from each posterior and decides whether to run the tool.

    Parameters
    ----------
    tool_names : sequence of str
        Names of optional tools (arms).  Always-on tools are excluded.
    threshold : float
        Minimum Thompson sample to activate a tool.  Lower = more tools
        run.  Default 0.3 gives moderate exploration.
    min_exploration_rate : float
        Minimum probability any tool runs (prevents total starvation).
        Implemented as: if random() < rate, force tool on.
    persist_path : Path, optional
        JSON file for persisting bandit state across sessions.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        tool_names: tuple[str, ...] | list[str] = DEFAULT_OPTIONAL_TOOLS,
        threshold: float = 0.3,
        min_exploration_rate: float = 0.1,
        persist_path: Path | None = None,
        seed: int | None = None,
    ) -> None:
        self._tool_names = tuple(tool_names)
        self._threshold = threshold
        self._min_exploration = min_exploration_rate
        self._persist_path = persist_path
        self._rng = random.Random(seed)

        # Beta distribution parameters (uniform prior)
        self._alpha: dict[str, float] = {t: 1.0 for t in self._tool_names}
        self._beta: dict[str, float] = {t: 1.0 for t in self._tool_names}
        self._pulls: dict[str, int] = {t: 0 for t in self._tool_names}
        self._total_reward: dict[str, float] = {t: 0.0 for t in self._tool_names}

        if persist_path and persist_path.exists():
            self._load()

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return the names of optional tools managed by this bandit."""
        return self._tool_names

    def decide(self, context: ToolContext | None = None) -> dict[str, bool]:
        """Decide which tools to run this cycle.

        Parameters
        ----------
        context : ToolContext, optional
            Current regime/time context.  Currently used for logging;
            future versions may condition the threshold on context.

        Returns
        -------
        dict mapping tool_name → bool (True = run, False = skip).
        Always-on tools are always True.
        """
        decisions: dict[str, bool] = {}

        for tool in self._tool_names:
            # Sample from Beta posterior
            sample = self._rng.betavariate(self._alpha[tool], self._beta[tool])

            # Minimum exploration: force on with probability min_exploration
            if self._rng.random() < self._min_exploration:
                decisions[tool] = True
                log.debug(
                    "Tool '%s': forced ON by exploration (sample=%.3f)", tool, sample
                )
            elif sample > self._threshold:
                decisions[tool] = True
                log.debug(
                    "Tool '%s': ON (sample=%.3f > threshold=%.3f)",
                    tool,
                    sample,
                    self._threshold,
                )
            else:
                decisions[tool] = False
                log.debug(
                    "Tool '%s': OFF (sample=%.3f <= threshold=%.3f)",
                    tool,
                    sample,
                    self._threshold,
                )

        # Always-on tools
        for tool in ALWAYS_ON_TOOLS:
            decisions[tool] = True

        log.info(
            "Tool routing: %d/%d optional tools enabled. Decisions: %s",
            sum(1 for t in self._tool_names if decisions.get(t, False)),
            len(self._tool_names),
            {t: decisions.get(t, True) for t in self._tool_names},
        )

        return decisions

    def record_outcome(self, tool_name: str, signal_contribution: float) -> None:
        """Update the bandit after observing a tool's signal contribution.

        Parameters
        ----------
        tool_name : str
            The tool that was executed.
        signal_contribution : float
            Reward in [0, 1].  Fraction of downstream entity alerts that
            used this tool's data.

        Raises
        ------
        ValueError
            If tool_name is not a known optional tool.
        """
        if tool_name not in self._alpha:
            raise ValueError(
                f"Unknown tool '{tool_name}'. Known: {sorted(self._alpha)}"
            )

        reward = max(0.0, min(1.0, signal_contribution))
        self._alpha[tool_name] += reward
        self._beta[tool_name] += 1.0 - reward
        self._pulls[tool_name] += 1
        self._total_reward[tool_name] += reward

        log.debug(
            "Tool '%s' outcome: reward=%.3f, α=%.1f, β=%.1f, pulls=%d",
            tool_name,
            reward,
            self._alpha[tool_name],
            self._beta[tool_name],
            self._pulls[tool_name],
        )

        if self._persist_path:
            self._save()

    def add_tool(self, tool_name: str) -> None:
        """Add a new tool with uniform prior Beta(1, 1).

        No-op if tool already exists.
        """
        if tool_name in self._alpha:
            return
        self._tool_names = (*self._tool_names, tool_name)
        self._alpha[tool_name] = 1.0
        self._beta[tool_name] = 1.0
        self._pulls[tool_name] = 0
        self._total_reward[tool_name] = 0.0
        log.info("Added new tool arm: '%s' (uniform prior)", tool_name)
        if self._persist_path:
            self._save()

    # Alias for add_tool — used by discovery pipeline
    add_arm = add_tool

    def remove_arm(self, tool_name: str) -> None:
        """Remove a tool arm.  No-op if tool does not exist."""
        if tool_name not in self._alpha:
            return
        self._tool_names = tuple(t for t in self._tool_names if t != tool_name)
        del self._alpha[tool_name]
        del self._beta[tool_name]
        del self._pulls[tool_name]
        del self._total_reward[tool_name]
        log.info("Removed tool arm: '%s'", tool_name)
        if self._persist_path:
            self._save()

    def stats(self) -> dict[str, dict[str, float]]:
        """Return per-tool statistics for diagnostics."""
        result: dict[str, dict[str, float]] = {}
        for tool in self._tool_names:
            a = self._alpha[tool]
            b = self._beta[tool]
            result[tool] = {
                "alpha": a,
                "beta": b,
                "mean": a / (a + b),
                "pulls": float(self._pulls[tool]),
                "total_reward": self._total_reward[tool],
            }
        return result

    def _save(self) -> None:
        """Persist bandit state to JSON."""
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "tool_names": list(self._tool_names),
            "alpha": self._alpha,
            "beta": self._beta,
            "pulls": self._pulls,
            "total_reward": self._total_reward,
            "threshold": self._threshold,
            "min_exploration": self._min_exploration,
        }
        self._persist_path.write_text(json.dumps(state, indent=2))

    def _load(self) -> None:
        """Load bandit state from JSON."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            state = json.loads(self._persist_path.read_text())
            for tool in self._tool_names:
                if tool in state.get("alpha", {}):
                    self._alpha[tool] = float(state["alpha"][tool])
                    self._beta[tool] = float(state["beta"][tool])
                    self._pulls[tool] = int(state.get("pulls", {}).get(tool, 0))
                    self._total_reward[tool] = float(
                        state.get("total_reward", {}).get(tool, 0.0)
                    )
            log.info("Loaded tool routing state from %s", self._persist_path)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning("Failed to load tool routing state: %s", exc)

    def save(self, path: Path | None = None) -> None:
        """Explicitly save bandit state to a given path or the configured path."""
        old = self._persist_path
        if path is not None:
            self._persist_path = path
        self._save()
        self._persist_path = old

    def load(self, path: Path | None = None) -> None:
        """Explicitly load bandit state from a given path or the configured path."""
        old = self._persist_path
        if path is not None:
            self._persist_path = path
        self._load()
        self._persist_path = old
