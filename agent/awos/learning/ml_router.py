"""LinUCB learned router — contextual bandit for signal-operation method choice.

Adapted from AWOS ml_router.py into the learning subpackage.

Learns which method tier (heuristic / cheap-llm / rich-llm / statistical /
ml-model / ensemble) works best for which kind of signal operation
(fetch / score / fuse / alert / clean), using real outcomes.

Reference: Li et al., 2010. "A Contextual-Bandit Approach to Personalized
News Article Recommendation." WWW 2010.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
N_FEATURES = 10
N_ACTIONS = 6
ACTION_NAMES = ["heuristic", "cheap_llm", "rich_llm", "statistical", "ml_model", "ensemble"]

_DEFAULT_WEIGHTS_PATH = Path(".awos") / "linucb_weights.pkl"

# Keyword sets for signal-operation feature extraction
_FETCH_KW = {"fetch", "pull", "load", "retrieve", "query", "scrape", "collect", "download"}
_SCORE_KW = {"score", "rank", "order", "win", "probability", "margin", "risk", "expected"}
_FUSE_KW = {"fuse", "merge", "combine", "join", "overlay", "correlate", "link", "match"}
_ALERT_KW = {"alert", "notify", "digest", "report", "flag", "watch", "monitor", "threshold"}
_CLEAN_KW = {"normalize", "parse", "dedup", "clean", "validate", "resolve", "transform"}


# ── Feature Extractor ──────────────────────────────────────────────────────
class OperationFeatureExtractor:
    """Converts a signal operation descriptor → fixed 10-dim feature vector."""

    def extract(self, operation: str, failure_count: int = 0) -> np.ndarray:
        text = str(operation or "").lower()
        words = re.findall(r"\w+", text)
        word_set = set(words)

        features = np.zeros(N_FEATURES, dtype=np.float64)
        features[0] = min(len(words) / 50.0, 1.0)      # operation length norm
        features[1] = 1.0 if word_set & _FETCH_KW else 0.0
        features[2] = 1.0 if word_set & _SCORE_KW else 0.0
        features[3] = 1.0 if word_set & _FUSE_KW else 0.0
        features[4] = 1.0 if word_set & _ALERT_KW else 0.0
        features[5] = 1.0 if word_set & _CLEAN_KW else 0.0
        features[6] = 0.5  # default complexity placeholder
        features[7] = 1.0  # is_core (signal ops are always core)
        features[8] = min(failure_count / 3.0, 1.0)     # failure count norm
        features[9] = 1.0  # bias term
        return features


# ── LinUCB Router ──────────────────────────────────────────────────────────
class LinUCBRouter:
    """Linear Upper Confidence Bound contextual bandit for signal methods."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_actions: int = N_ACTIONS,
        alpha: float = 1.0,
        min_samples: int = 20,
        weights_path: Path | None = None,
        min_trials_forced: int = 3,
    ) -> None:
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.min_samples = min_samples
        self.min_trials_forced = min_trials_forced
        self._weights_path = Path(weights_path) if weights_path is not None else None
        self._total_updates = 0

        # Per-action LinUCB accumulators (disjoint model)
        self._A: list[np.ndarray] = [np.eye(n_features) for _ in range(n_actions)]
        self._b: list[np.ndarray] = [np.zeros(n_features) for _ in range(n_actions)]
        # Per-action trial counts (drives forced exploration during warm-up)
        self._trial_counts: list[int] = [0] * n_actions

        if weights_path is not None:
            self._load()

    # ── Selection ────────────────────────────────────────────────────────────
    def select(
        self,
        features: np.ndarray,
        budget_mask: list[bool] | None = None,
    ) -> int:
        """Select the best method tier for the operation (LinUCB UCB).

        Fixes cold-start collapse: before every action has been tried
        ``min_trials_signal`` times, explore the least-tried actions first so
        the empty Bayesian priors do not cause the router to lock onto action 0.
        """
        x = np.asarray(features, dtype=np.float64)

        # ── Forced exploration during warm-up ────────────────────────────────
        # Any action not yet tried ``min_trials_forced`` times globally is
        # selected uniformly among the least-tried *available* actions. This
        # guarantees the bandit sees each tier before trusting UCB estimates.
        available = [
            a
            for a in range(self.n_actions)
            if not (
                budget_mask is not None
                and a < len(budget_mask)
                and not budget_mask[a]
            )
        ]
        if available and self._total_updates < self.min_samples:
            least_tried = min(
                available,
                key=lambda a: self._trial_counts[a] if a < len(self._trial_counts) else 0,
            )
            if self._trial_counts[least_tried] < self.min_trials_forced:
                return int(least_tried)

        ucb_scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            blocked = (
                budget_mask is not None
                and a < len(budget_mask)
                and not budget_mask[a]
            )
            if blocked:
                ucb_scores[a] = -np.inf
                continue
            A_inv = np.linalg.inv(self._A[a])
            theta = A_inv @ self._b[a]
            exploitation = float(theta @ x)
            exploration = self.alpha * float(np.sqrt(x @ A_inv @ x))
            ucb_scores[a] = exploitation + exploration
        return int(np.argmax(ucb_scores))

    # ── Update ───────────────────────────────────────────────────────────────
    def update(self, features: np.ndarray, action_id: int, reward: float) -> None:
        """Online LinUCB update after observing reward (Li et al. 2010)."""
        x = np.asarray(features, dtype=np.float64)
        a = int(action_id)
        while a >= len(self._A):
            self._A.append(np.eye(self.n_features))
            self._b.append(np.zeros(self.n_features))
        self._A[a] += np.outer(x, x)
        self._b[a] += reward * x
        while a >= len(self._trial_counts):
            self._trial_counts.append(0)
        self._trial_counts[a] += 1
        self._total_updates += 1
        if self._total_updates % 10 == 0:
            self._save()

    # ── Warm-start from RewardStore ─────────────────────────────────────────
    def warm_start(self, reward_store, max_episodes: int = 100) -> int:
        episodes = reward_store.get_prioritized(max_episodes)
        if not episodes:
            return 0
        replayed = 0
        for ep in episodes:
            action_id = getattr(ep, "action_id", -1)
            if action_id < 0 or action_id >= self.n_actions:
                continue
            features = np.asarray(getattr(ep, "features", []), dtype=np.float64)
            if features.shape[0] != self.n_features:
                continue
            self.update(features, action_id, float(getattr(ep, "reward", 0.0)))
            replayed += 1
        return replayed

    # ── State ────────────────────────────────────────────────────────────────
    def is_ready(self) -> bool:
        return self._total_updates >= self.min_samples

    def total_updates(self) -> int:
        return self._total_updates

    def predict_success(self, features: np.ndarray) -> float:
        """Best-guess P(success) for each action, returns max over actions."""
        x = np.asarray(features, dtype=np.float64)
        best = -np.inf
        for a in range(self.n_actions):
            theta = np.linalg.inv(self._A[a]) @ self._b[a]
            best = max(best, float(theta @ x))
        return max(0.0, min(1.0, best))

    def learned_weights(self) -> dict[str, np.ndarray]:
        return {ACTION_NAMES[a]: np.linalg.inv(self._A[a]) @ self._b[a] for a in range(self.n_actions)}

    def summary(self) -> dict[str, Any]:
        return {
            "total_updates": self._total_updates,
            "is_ready": self.is_ready(),
            "per_action_theta_norm": {
                ACTION_NAMES[a]: float(np.linalg.norm(np.linalg.inv(self._A[a]) @ self._b[a]))
                for a in range(self.n_actions)
            },
        }

    # ── Persistence ─────────────────────────────────────────────────────────
    def _save(self) -> None:
        if self._weights_path is None:
            return
        try:
            self._weights_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._weights_path, "wb") as f:
                pickle.dump(
                    {"A": self._A, "b": self._b, "updates": self._total_updates, "trials": self._trial_counts},
                    f,
                )
        except Exception as exc:
            logger.warning("[linucb] save failed: %s", exc)

    def _load(self) -> None:
        if not self._weights_path.exists():
            return
        try:
            with self._weights_path.open("rb") as f:
                data = pickle.load(f)  # noqa: S301 — local trusted weight file
            self._A = data["A"]
            self._b = data["b"]
            self._total_updates = int(data.get("updates", 0))
            self._trial_counts = list(data.get("trials", [0] * self.n_actions))
            self.n_actions = max(self.n_actions, len(self._A))
            while len(self._trial_counts) < self.n_actions:
                self._trial_counts.append(0)
        except Exception as exc:
            logger.warning("[linucb] load failed: %s", exc)


__all__ = ["LinUCBRouter", "OperationFeatureExtractor", "N_FEATURES", "N_ACTIONS", "ACTION_NAMES"]
