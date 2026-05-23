"""TirraMind — Adversarial Layer Configuration

Frozen dataclasses that parameterize each adversarial detector.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeDecayConfig:
    """Parameters for the BOCPD-based signal health monitor.

    rolling_window : periods used for rolling Sharpe calculation.
    bocpd_hazard_lambda : expected run length for BOCPD (mean time between
        changepoints).  Hazard = 1 / hazard_lambda.
    decay_threshold : changepoint probability above which a signal is flagged.
    min_history : minimum observation count before monitoring begins.
    periods_per_year : annualization factor (52 for weekly, 252 for daily).
    """

    rolling_window: int = 52
    bocpd_hazard_lambda: float = 100.0
    decay_threshold: float = 0.5
    min_history: int = 52
    periods_per_year: int = 52


@dataclass(frozen=True)
class VPINConfig:
    """Parameters for the Volume-synchronized PIN estimator.

    n_buckets : number of volume buckets for averaging order imbalance.
    sigma_window : rolling window for return volatility (BVC denominator).
    spike_threshold : VPIN value above which a spike flag is produced.
    """

    n_buckets: int = 50
    sigma_window: int = 20
    spike_threshold: float = 0.7


@dataclass(frozen=True)
class CrowdingConfig:
    """Parameters for convergence cluster crowding risk.

    cluster_size_threshold : minimum cluster members to trigger crowding flag.
    correlation_threshold : intra-cluster correlation above which crowding
        is considered elevated.
    volume_lookback : days of volume history used for the liquidity proxy.
    """

    cluster_size_threshold: int = 5
    correlation_threshold: float = 0.7
    volume_lookback: int = 20


@dataclass(frozen=True)
class AdversarialConfig:
    """Composite configuration for the full adversarial scanner."""

    edge_decay: EdgeDecayConfig = EdgeDecayConfig()
    vpin: VPINConfig = VPINConfig()
    crowding: CrowdingConfig = CrowdingConfig()
