"""TirraMind — Adversarial Scanner (Orchestrator)

Runs all adversarial detectors and produces a unified list of
``AdversarialFlag`` objects.  This is the single entry point consumed by
the pipeline DAG and the RL integration layer.

Execution order:
    1. EdgeDecayMonitor  — per-signal Sharpe health
    2. VPINEstimator     — market-level informed trading
    3. CrowdingEstimator — cluster-level position crowding

References:
    - Spec step 22b.1
    - Research: docs/research/adversarial.md
"""

from __future__ import annotations

import numpy as np

from agent.adversarial.config import AdversarialConfig
from agent.adversarial.crowding import CrowdingEstimator
from agent.adversarial.edge_decay import EdgeDecayMonitor
from agent.adversarial.flags import AdversarialFlag
from agent.adversarial.vpin import VPINEstimator
from agent.fusion.convergence import ConvergenceCluster


class AdversarialScanner:
    """Run all adversarial detectors and return a unified flag list."""

    def __init__(self, config: AdversarialConfig | None = None) -> None:
        self._cfg = config or AdversarialConfig()
        self._edge_decay = EdgeDecayMonitor(self._cfg.edge_decay)
        self._vpin = VPINEstimator(self._cfg.vpin)
        self._crowding = CrowdingEstimator(self._cfg.crowding)

    def scan(
        self,
        signal_returns: dict[str, np.ndarray],
        market_returns: np.ndarray,
        market_volumes: np.ndarray,
        clusters: list[ConvergenceCluster],
        position_weights: dict[str, float],
        volume_history: dict[str, np.ndarray],
        *,
        timestamp: float | None = None,
    ) -> list[AdversarialFlag]:
        """Execute all detectors and merge results.

        Parameters
        ----------
        signal_returns : {signal_name → per-period return array} for edge decay.
        market_returns : aggregate daily return series for VPIN.
        market_volumes : aggregate daily volume series for VPIN.
        clusters : current convergence clusters for crowding.
        position_weights : {entity_id → signed weight} for crowding unwind.
        volume_history : {entity_id → daily volume array} for liquidity proxy.
        timestamp : optional timestamp propagated to all flags.

        Returns
        -------
        Merged, deduplicated list of ``AdversarialFlag``.
        """
        flags: list[AdversarialFlag] = []

        # 1. Edge Decay — per signal
        for sig_name, rets in signal_returns.items():
            flags.extend(self._edge_decay.update(sig_name, rets, timestamp=timestamp))

        # 2. VPIN — market-level
        market_returns = np.asarray(market_returns, dtype=np.float64).ravel()
        market_volumes = np.asarray(market_volumes, dtype=np.float64).ravel()

        if len(market_returns) > 0 and len(market_volumes) > 0:
            try:
                vpin_series = self._vpin.compute(market_returns, market_volumes)
                flags.extend(self._vpin.flag_spikes(vpin_series, timestamp=timestamp))
            except ValueError:
                pass  # insufficient data or all-zero volumes → skip

        # 3. Crowding — cluster-level
        flags.extend(
            self._crowding.assess(
                clusters, position_weights, volume_history, timestamp=timestamp
            )
        )

        return flags
