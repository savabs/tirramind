"""LearningCore — ties the self-learning modules into one object.

The embedded runtime holds one LearningCore. It provides:
  - record_outcome(success, operation, action, cost, ...)  → reward/augmented memory
  - reflect_failure(operation, error_type, error, attempt) → optional helper synthesis
  - context_for(operation)                                  → skill + guideline context
  - route_method(operation, budget_mask)                    → learned method tier
  - evolve_guidelines(session_count)                        → prompt evolution
  - summary()                                               → self-learning observability

All modules are intentionally importable standalone too; LearningCore is the
convenience composition layer for the runtime.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.awos.learning.error_pattern_store import ErrorPattern, ErrorPatternStore
from agent.awos.learning.live_tool_synth import LiveToolSynthesizer, SynthesizedTool
from agent.awos.learning.ml_router import LinUCBRouter, OperationFeatureExtractor
from agent.awos.learning.prompt_evolver import PromptEvolver
from agent.awos.learning.reward_store import ReplayGate, RewardStore
from agent.awos.learning.skill_library import SkillLibrary

logger = logging.getLogger(__name__)


class LearningCore:
    """Composition of the self-improving signal-runtime learning modules."""

    def __init__(
        self,
        state_dir: str = ".awos",
        cheap_call: Callable[[str], str] | None = None,
    ) -> None:
        self._state_dir = Path(state_dir)
        base = self._state_dir

        self.errors = ErrorPatternStore(str(base / "error_patterns.jsonl"))
        self.skills = SkillLibrary(base / "skills")
        self.rewards = RewardStore(base / "reward_store.jsonl")
        self.gate = ReplayGate()
        self.evolver = PromptEvolver(store_path=str(base), cheap_call=cheap_call)
        self.tool_synth = LiveToolSynthesizer(
            tools_dir=str(base / "tools"), cheap_call=cheap_call
        )
        self.router = LinUCBRouter(weights_path=base / "linucb_weights.pkl")
        self._extractor = OperationFeatureExtractor()
        self._cheap_call = cheap_call

    # ── Outcome recording (the learning hook) ──────────────────────────────
    def record_outcome(
        self,
        *,
        task_id: str,
        operation: str,
        action_id: int,
        success: bool,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        model_name: str = "",
        signal_name: str = "",
        source_tool: str = "",
        failure_count: int = 0,
        error_type: str = "",
        error_msg: str = "",
        critique: str = "",
        attempts: int = 1,
    ) -> dict[str, Any]:
        """Record one outcome across the learning layers; returns metrics."""
        features = self._extractor.extract(operation, failure_count).tolist()

        episode = self.rewards.store(
            {"task_id": task_id, "action": operation},
            action_id=action_id,
            features=features,
            success=success,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            model_name=model_name,
            signal_name=signal_name,
            source_tool=source_tool,
        )

        admitted = self.gate.admit(episode)
        # The router must learn from EVERY observed outcome. Gating the router
        # update by the replay gate starved it: the gate scored ~37% of
        # episodes and systematically favoured cheap actions (reward magnitude
        # tracks cost), rejecting the very episodes that teach which method is
        # correct-but-costlier. The gate stays as an observability metric
        # (admit_rate), but never as a hard filter on learning signal.
        self.router.update(features, action_id, episode.reward)

        if success:
            strategy = f"method_tier_{action_id}"
            self.skills.record(
                signal_name=signal_name or operation,
                source_tool=source_tool or "default",
                strategy=strategy,
                attempts=max(attempts, 1),
                operation=operation,
            )

        if not success and (error_type or error_msg):
            self.errors.save(
                ErrorPattern(
                    task_id=task_id,
                    signal_name=signal_name or operation,
                    source_tool=source_tool or "default",
                    error_type=error_type or "UNKNOWN",
                    error_msg=error_msg,
                    critique=critique or (error_msg[:200]),
                )
            )

        return {
            "episode": episode,
            "admitted": admitted,
            "reward": episode.reward,
            "router_ready": self.router.is_ready(),
        }

    # ── Failure reflection → helper synthesis ──────────────────────────────
    def reflect_failure(
        self, operation: str, error_type: str, error: str, attempt: int
    ) -> SynthesizedTool | None:
        return self.tool_synth.reflect(operation, error_type, error, attempt)

    def find_helper(self, operation: str) -> SynthesizedTool | None:
        return self.tool_synth.find_relevant_tool(operation)

    # ── Context injection ───────────────────────────────────────────────────
    def context_for(self, operation: str, signal_name: str = "", source_tool: str = "") -> str:
        parts: list[str] = []
        skctx = self.skills.get_context(signal_name or operation, source_tool or "default")
        if skctx:
            parts.append(skctx)
        evolved = self.evolver.load_evolved_guidelines()
        if evolved:
            parts.append(f"EVOLVED GUIDELINES:\n{evolved}")
        return "\n\n".join(parts)

    # ── Method routing (learned escalation) ────────────────────────────────
    def route_method(self, operation: str, budget_mask: list[bool] | None = None) -> int:
        features = self._extractor.extract(operation)
        return self.router.select(features, budget_mask=budget_mask)

    # ── Guideline evolution ─────────────────────────────────────────────────
    def evolve_guidelines(self, session_count: int) -> str:
        if not self.evolver.should_evolve(session_count):
            return ""
        guidelines = self.evolver.evolve()
        if guidelines:
            self.evolver.persist(guidelines, session_count)
        return guidelines

    # ── Observability ───────────────────────────────────────────────────────
    def summary(self) -> dict[str, Any]:
        return {
            "rewards": self.rewards.summary(),
            "skills": self.skills.summary(),
            "errors": self.errors.summary(),
            "router": self.router.summary(),
            "total_episodes": self.rewards.total_episodes(),
        }

    def warm_start_router(self, max_episodes: int = 100) -> int:
        return self.router.warm_start(self.rewards, max_episodes=max_episodes)


__all__ = ["LearningCore"]
