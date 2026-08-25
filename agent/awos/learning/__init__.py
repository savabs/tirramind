"""AWOS learning subpackage — self-improving signal-runtime memory and policy.

Brings the best self-learning capabilities of AWOS into the embedded
TirraMind runtime:
  - ErrorPatternStore : failure-memory (Reflexion-style verbal critiques)
  - SkillLibrary      : success-memory ("what approach works")
  - RewardStore       : outcome ledger + asymmetric cost-proportional reward
  - PromptEvolver     : OPRO-style guideline evolution from outcomes
  - LiveToolSynthesizer: on-the-fly helper synthesis on recurring failures
  - LinUCBRouter      : learned method-tier selection (contextual bandit)
  - LearningCore      : composition object for the runtime
"""

from agent.awos.learning.error_pattern_store import ErrorPattern, ErrorPatternStore
from agent.awos.learning.learning_core import LearningCore
from agent.awos.learning.live_tool_synth import LiveToolSynthesizer, SynthesizedTool
from agent.awos.learning.ml_router import LinUCBRouter, OperationFeatureExtractor
from agent.awos.learning.prompt_evolver import PromptEvolver
from agent.awos.learning.reward_store import (
    Episode,
    ReplayGate,
    RewardStore,
    compute_reward,
    compute_reward_breakdown,
)
from agent.awos.learning.skill_library import SkillEntry, SkillLibrary

__all__ = [
    "ErrorPattern",
    "ErrorPatternStore",
    "SkillEntry",
    "SkillLibrary",
    "Episode",
    "RewardStore",
    "ReplayGate",
    "compute_reward",
    "compute_reward_breakdown",
    "PromptEvolver",
    "LiveToolSynthesizer",
    "SynthesizedTool",
    "LinUCBRouter",
    "OperationFeatureExtractor",
    "LearningCore",
]
