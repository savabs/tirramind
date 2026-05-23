"""TirraMind — Adversarial Intelligence Layer (Layer 6)

Monitors signal health, detects information asymmetry, and estimates
crowding risk.  Feeds ``AdversarialFlag`` objects into the RL policy
reward function and state assembler.

Components:
    - ``AdversarialFlag``   — output contract (frozen dataclass).
    - ``EdgeDecayMonitor``  — BOCPD on rolling per-signal Sharpe.
    - ``VPINEstimator``     — daily Bulk Volume Classification VPIN.
    - ``CrowdingEstimator`` — cluster density × position / liquidity.
    - ``AdversarialScanner``— orchestrator that runs all detectors.

References:
    - Research: docs/research/adversarial.md
    - Spec: docs/specs/adversarial_spec.md
"""

from agent.adversarial.config import (
    AdversarialConfig,
    CrowdingConfig,
    EdgeDecayConfig,
    VPINConfig,
)
from agent.adversarial.flags import AdversarialFlag

__all__ = [
    "AdversarialConfig",
    "AdversarialFlag",
    "CrowdingConfig",
    "EdgeDecayConfig",
    "VPINConfig",
]
