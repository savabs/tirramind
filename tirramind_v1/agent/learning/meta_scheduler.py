"""
TirraMind — Meta-Learned Scheduling

Per-component Thompson Sampling bandit that learns optimal refit intervals,
replacing all hardcoded scheduling constants.

Math — Thompson Sampling over discrete interval arms:
    Each schedulable component (cpd_fit, structure_refine, gnn_epochs,
    history_window) has K discrete arms representing candidate intervals.
    Each arm maintains a Beta(α, β) posterior.  At each decision point,
    we sample from each arm's Beta distribution and select the arm with
    the highest sample.

    After a refit using the chosen arm, we observe a reward ∈ [0, 1]
    (information gain from the refit, normalized via sigmoid) and update:
        α += reward
        β += (1 - reward)

    Over time, arms that consistently yield high information gain accumulate
    higher α and dominate sampling.  Arms that waste compute (no improvement)
    accumulate β and are sampled less often.

Trusted source: Thompson 1933, Chapelle & Li 2011 ("An Empirical Evaluation
of Thompson Sampling").  The application to scheduling intervals is
repo-specific engineering.

Spec: docs/specs/tier7_self_modifying_structure_spec.md (steps 14.1–14.2)
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


@dataclass(frozen=True)
class ComponentConfig:
    """Configuration for one schedulable component."""

    arms: tuple[int, ...]
    default: int


# ── Default component definitions ─────────────────────────────

DEFAULT_COMPONENTS: dict[str, ComponentConfig] = {
    "cpd_fit": ComponentConfig(arms=(3, 5, 7, 14, 30), default=7),
    "structure_refine": ComponentConfig(arms=(30, 60, 90, 180), default=90),
    "gnn_epochs": ComponentConfig(arms=(5, 10, 20, 40), default=10),
    "history_window": ComponentConfig(arms=(30, 60, 90, 180), default=90),
}


class MetaScheduler:
    """Per-component Thompson Sampling bandit for learning refit schedules.

    Parameters
    ----------
    components : dict mapping component_name → ComponentConfig, or None
        to use DEFAULT_COMPONENTS.
    persist_path : Path for JSON state persistence.
    seed : int for reproducibility.
    """

    def __init__(
        self,
        components: dict[str, ComponentConfig] | None = None,
        persist_path: Path | None = None,
        seed: int | None = None,
    ) -> None:
        self._components = dict(components or DEFAULT_COMPONENTS)
        self._persist_path = persist_path
        self._rng = random.Random(seed)

        # Beta(α, β) per arm per component
        self._alpha: dict[str, dict[int, float]] = {}
        self._beta: dict[str, dict[int, float]] = {}
        self._pulls: dict[str, dict[int, int]] = {}
        self._total_reward: dict[str, dict[int, float]] = {}

        for comp_name, cfg in self._components.items():
            self._alpha[comp_name] = {arm: 1.0 for arm in cfg.arms}
            self._beta[comp_name] = {arm: 1.0 for arm in cfg.arms}
            self._pulls[comp_name] = {arm: 0 for arm in cfg.arms}
            self._total_reward[comp_name] = {arm: 0.0 for arm in cfg.arms}

        if persist_path and persist_path.exists():
            self._load()

    @property
    def component_names(self) -> list[str]:
        return list(self._components.keys())

    def suggest(self, component: str) -> int:
        """Suggest the best interval/param for a component via Thompson Sampling.

        Returns the arm (interval value) with the highest Thompson sample.
        If the component has never been pulled, returns the default arm
        with boosted probability (prior centered on default).
        """
        if component not in self._components:
            raise ValueError(f"Unknown component: {component}")

        cfg = self._components[component]
        best_arm = cfg.default
        best_sample = -1.0

        for arm in cfg.arms:
            alpha = self._alpha[component][arm]
            beta = self._beta[component][arm]
            sample = self._rng.betavariate(alpha, beta)
            if sample > best_sample:
                best_sample = sample
                best_arm = arm

        total_pulls = sum(self._pulls[component].values())
        log.debug(
            "MetaScheduler suggest(%s): arm=%d (sample=%.3f, total_pulls=%d)",
            component,
            best_arm,
            best_sample,
            total_pulls,
        )
        return best_arm

    def record_outcome(self, component: str, arm: int, reward: float) -> None:
        """Update the Beta posterior after observing a refit outcome.

        Parameters
        ----------
        component : which component was refit
        arm : which interval/param was used
        reward : information gain in [0, 1]
        """
        if component not in self._components:
            raise ValueError(f"Unknown component: {component}")
        if arm not in self._alpha[component]:
            raise ValueError(
                f"Unknown arm {arm} for component {component}. Valid arms: {self._components[component].arms}"
            )

        reward = max(0.0, min(1.0, reward))
        self._alpha[component][arm] += reward
        self._beta[component][arm] += 1.0 - reward
        self._pulls[component][arm] += 1
        self._total_reward[component][arm] += reward

        log.debug(
            "MetaScheduler record(%s, arm=%d, reward=%.3f): α=%.1f, β=%.1f, pulls=%d",
            component,
            arm,
            reward,
            self._alpha[component][arm],
            self._beta[component][arm],
            self._pulls[component][arm],
        )

    def diagnostics(self) -> dict[str, Any]:
        """Return per-component, per-arm diagnostics."""
        result: dict[str, Any] = {}
        for comp_name, cfg in self._components.items():
            arms_info = {}
            for arm in cfg.arms:
                alpha = self._alpha[comp_name][arm]
                beta = self._beta[comp_name][arm]
                pulls = self._pulls[comp_name][arm]
                total_r = self._total_reward[comp_name][arm]
                arms_info[arm] = {
                    "alpha": alpha,
                    "beta": beta,
                    "pulls": pulls,
                    "mean_reward": total_r / pulls if pulls > 0 else 0.0,
                    "posterior_mean": alpha / (alpha + beta),
                }
            result[comp_name] = {
                "default": cfg.default,
                "arms": arms_info,
                "total_pulls": sum(self._pulls[comp_name].values()),
            }
        return result

    def save(self) -> None:
        """Persist bandit state to JSON."""
        if self._persist_path is None:
            return
        data = self.to_dict()
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps(data, indent=2))
        log.debug("MetaScheduler saved to %s", self._persist_path)

    def _load(self) -> None:
        """Load bandit state from JSON."""
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            self._restore_from_dict(data)
            log.debug("MetaScheduler loaded from %s", self._persist_path)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning("MetaScheduler load failed, using fresh state: %s", exc)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scheduler state."""
        return {
            "components": {
                name: {
                    "arms": list(cfg.arms),
                    "default": cfg.default,
                }
                for name, cfg in self._components.items()
            },
            "alpha": {comp: {str(arm): v for arm, v in arms.items()} for comp, arms in self._alpha.items()},
            "beta": {comp: {str(arm): v for arm, v in arms.items()} for comp, arms in self._beta.items()},
            "pulls": {comp: {str(arm): v for arm, v in arms.items()} for comp, arms in self._pulls.items()},
            "total_reward": {
                comp: {str(arm): v for arm, v in arms.items()} for comp, arms in self._total_reward.items()
            },
        }

    def _restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore state from serialized dict."""
        for comp_name in self._components:
            if comp_name not in data.get("alpha", {}):
                continue
            for arm in self._components[comp_name].arms:
                arm_key = str(arm)
                if arm_key in data["alpha"].get(comp_name, {}):
                    self._alpha[comp_name][arm] = float(data["alpha"][comp_name][arm_key])
                if arm_key in data["beta"].get(comp_name, {}):
                    self._beta[comp_name][arm] = float(data["beta"][comp_name][arm_key])
                if arm_key in data.get("pulls", {}).get(comp_name, {}):
                    self._pulls[comp_name][arm] = int(data["pulls"][comp_name][arm_key])
                if arm_key in data.get("total_reward", {}).get(comp_name, {}):
                    self._total_reward[comp_name][arm] = float(data["total_reward"][comp_name][arm_key])

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> MetaScheduler:
        """Create a MetaScheduler from a serialized dict."""
        components = {}
        for name, cfg_data in data.get("components", {}).items():
            components[name] = ComponentConfig(
                arms=tuple(cfg_data["arms"]),
                default=cfg_data["default"],
            )
        scheduler = cls(components=components or None, **kwargs)
        scheduler._restore_from_dict(data)
        return scheduler


# ── Reward computation ────────────────────────────────────────


def compute_refit_reward(
    component: str,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
) -> float:
    """Compute [0, 1] reward from before/after metrics of a refit.

    Centered so that no-change gives ≈ 0.5, improvements > 0.5,
    degradations < 0.5.

    Parameters
    ----------
    component : which component was refit
    before_metrics : metrics measured before refit
    after_metrics : metrics measured after refit

    Returns
    -------
    float in [0, 1]
    """
    if component == "cpd_fit":
        # Reward = sigmoid(ΔBIC), positive BIC improvement → reward > 0.5
        delta = after_metrics.get("total_bic", 0.0) - before_metrics.get("total_bic", 0.0)
        return _sigmoid(delta)

    elif component == "structure_refine":
        # Reward = sigmoid(n_confident_changes)
        n_changes = after_metrics.get("n_confident_changes", 0.0)
        return _sigmoid(n_changes)

    elif component == "gnn_epochs":
        # Reward = sigmoid(-Δval_loss), loss decrease → positive reward
        delta_loss = after_metrics.get("val_loss", 0.0) - before_metrics.get("val_loss", 0.0)
        return _sigmoid(-delta_loss)

    elif component == "history_window":
        # Same as cpd_fit: reward from held-out BIC improvement
        delta = after_metrics.get("held_out_bic", 0.0) - before_metrics.get("held_out_bic", 0.0)
        return _sigmoid(delta)

    else:
        log.warning("compute_refit_reward: unknown component '%s'", component)
        return 0.5
