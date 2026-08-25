"""
TirraMind — Vine Copula Tail-Dependence Encoder (Idea 7)

Computes **lower and upper tail dependence coefficients** (λ_L, λ_U) for
every linked entity pair, using bivariate Clayton and Gumbel copulas fitted
by the method of moments from Kendall's τ.

Problem
-------
The ``GraphBuilder`` encodes entity-link strength as a single scalar:
``confidence ∈ [0,1]``.  This captures *average* relationship strength but
is blind to **tail dependence** — the tendency for entities to move together
during extreme events (crashes, spikes, regime breaks).

Linear correlation has the same blind spot.  Two pairs may have identical
ρ = 0.3, but:
  - Pair A: co-crash probability λ_L = 0.60  →  dangerous!
  - Pair B: co-crash probability λ_L = 0.03  →  benign

A crisis-unaware graph will weight both edges equally.  Under stress the
GNN's attention mechanism will misattribute risk and fail to propagate
contagion signals through the high-tail-dependence subgraph.

Solution — Bivariate Copulas via Method of Moments
----------------------------------------------------
Sklar's theorem (1959): Any joint distribution F(x,y) can be decomposed as

    F(x,y) = C(F_X(x), F_Y(y))

where C: [0,1]² → [0,1] is the copula (the dependence structure alone,
marginals removed).

For a pair of entity observation series:
  1. Build aligned daily bins (mean value per day).
  2. Rank-transform → pseudo-observations u_i, v_i ∈ (0,1).
  3. Estimate Kendall's τ (rank correlation, invariant to monotone transforms).
  4. Fit **Clayton** and **Gumbel** copulas by MOM from τ.
  5. Compute tail dependence coefficients analytically.

Clayton copula (lower tail):
    C(u,v) = (u^{−θ} + v^{−θ} − 1)^{−1/θ}   (θ > 0)
    MOM:     θ = 2τ / (1 − τ)
    λ_L    = 2^{−1/θ}
    λ_U    = 0

Gumbel copula (upper tail):
    C(u,v) = exp(−[(−ln u)^θ + (−ln v)^θ]^{1/θ})   (θ ≥ 1)
    MOM:     θ = 1 / (1 − τ)
    λ_U    = 2 − 2^{1/θ}
    λ_L    = 0

For negative τ (counter-moving pairs), the **Survival Clayton** gives the
upper-tail coefficient under opposite movements:
    θ' = 2|τ| / (1 − |τ|)
    λ_L_survival = 2^{−1/θ'}   (stored under λ_U for counter-crash risk)

Output
------
``VineCopulaEncoder.run(store)`` returns
``dict[pair_key, CopulaResult]`` and stores four signals per entity pair:

    copula.<pair_hash>.tau          — Kendall's τ
    copula.<pair_hash>.lambda_lower — lower tail dependence (co-crash)
    copula.<pair_hash>.lambda_upper — upper tail dependence (co-spike)
    copula.<pair_hash>.copula_theta — fitted Clayton/Gumbel parameter

The pair_hash is a short deterministic hash of (entity_a, entity_b) so
signal names stay readable.  Full entity IDs are stored in signal metadata.

GNN Integration
---------------
The trainer calls ``VineCopulaEncoder`` in ``build_model()`` when
``use_vine_copula=True``.  Copula signals are available in the pipeline store
for the ConvergenceDetector to use as tail-risk edge priors.

Optionally, ``vine_copula_reweight=True`` updates link confidence in the
store:  ``confidence_new = confidence × (1 + copula_boost × λ_L)``.
This lets the GNN's HGTConv attention naturally upweight high-tail-risk edges.

References
----------
    Nelsen, R.B. (2006). An Introduction to Copulas. 2nd ed.
        Chapter 5.1 — Tail dependence.
        Springer Series in Statistics.

    Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges.
        Publ. Inst. Statist. Univ. Paris 8, 229–231.

    Joe, H. (1997). Multivariate Models and Dependence Concepts.
        Chapter 2 — Bivariate copula families and MOM estimation.
        Chapman & Hall.

    Bedford, T. & Cooke, R.M. (2002). Vines — A New Graphical Model for
        Dependent Random Variables. Ann. Statist. 30(4):1031–1068.
        (Vine / pair-copula construction for multivariate case.)
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_DAY: float = 86_400.0
_EPS: float = 1e-9

# Value extraction priority (mirrors other encoders)
_VALUE_KEYS = (
    "close", "usd_amount", "value", "estimated_value",
    "goldstein_scale", "btc_amount", "log_return", "num_articles",
)


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CopulaResult:
    """Bivariate copula fit result for one entity pair.

    Attributes
    ----------
    entity_id_a, entity_id_b : str
        Entity IDs of the pair (canonically sorted: a < b lexicographically).
    pair_key : str
        Short deterministic hash ``sha256(a + '|' + b)[:12]`` used in
        signal names to keep them under 64 chars.
    tau : float
        Kendall's τ rank correlation in [−1, 1].
    lambda_L : float
        Lower tail dependence coefficient (co-crash probability) in [0, 1].
        From Clayton copula when τ > 0; from survival Clayton when τ < 0.
    lambda_U : float
        Upper tail dependence coefficient (co-spike probability) in [0, 1].
        From Gumbel copula when τ > 0; 0 when τ < 0 (no upper-tail fit).
    copula_theta : float
        Fitted copula parameter (Clayton θ when τ > 0, Gumbel θ when τ ≥ 0).
    best_family : str
        ``"Clayton"`` (positive τ), ``"Gumbel"`` (positive τ),
        or ``"SurvivalClayton"`` (negative τ).
    n_joint_obs : int
        Number of aligned bins used in the fit.
    computed_at : float
        Unix timestamp.
    """

    entity_id_a: str
    entity_id_b: str
    pair_key: str
    tau: float
    lambda_L: float
    lambda_U: float
    copula_theta: float
    best_family: str
    n_joint_obs: int
    computed_at: float
    details: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# VineCopulaEncoder
# ═══════════════════════════════════════════════════════════════════════════


class VineCopulaEncoder:
    """Fit bivariate Clayton / Gumbel copulas to all linked entity pairs.

    Parameters
    ----------
    min_joint_obs : int
        Minimum aligned bins required to attempt a copula fit.  Default 20.
    lookback_days : int
        History window length.  Default 365.
    n_bins : int
        Number of bins (time buckets) to divide the lookback window into.
        With lookback_days=365 and n_bins=60 each bin ≈ 6 days.  Default 60.
    copula_boost : float
        Scale factor for link-confidence reweighting (only when
        ``vine_copula_reweight=True`` is passed to run()).
        ``confidence_new = confidence × (1 + copula_boost × λ_L)``  Default 0.5.
    obs_limit : int
        Max observations to load per entity.  Default 2000.
    """

    def __init__(
        self,
        min_joint_obs: int = 20,
        lookback_days: int = 365,
        n_bins: int = 60,
        copula_boost: float = 0.5,
        obs_limit: int = 2000,
    ) -> None:
        self.min_joint_obs = min_joint_obs
        self.lookback_days = lookback_days
        self.n_bins = n_bins
        self.copula_boost = copula_boost
        self.obs_limit = obs_limit

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        store: Any,
        as_of: float | None = None,
    ) -> dict[str, CopulaResult]:
        """Compute tail-dependence copulas for all linked entity pairs.

        Args:
            store: PipelineStore instance.
            as_of: Reference time.  Defaults to ``time.time()``.

        Returns:
            Dict mapping pair_key → CopulaResult.
        """
        if as_of is None:
            as_of = time.time()

        t_start = as_of - self.lookback_days * _DAY

        # Load all entity links
        links = store.query_all_entity_links()
        if not links:
            log.info("VineCopulaEncoder: no entity links found.")
            return {}

        # Deduplicate pairs (a→b and b→a are the same pair)
        seen_pairs: set[tuple[str, str]] = set()
        unique_pairs: list[tuple[str, str]] = []
        for link in links:
            eid_a = link.get("entity_id_a", "")
            eid_b = link.get("entity_id_b", "")
            if not eid_a or not eid_b:
                continue
            pair = tuple(sorted([eid_a, eid_b]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_pairs.append(pair)

        log.info("VineCopulaEncoder: scoring %d unique entity pairs.", len(unique_pairs))

        results: dict[str, CopulaResult] = {}

        for eid_a, eid_b in unique_pairs:
            result = self._score_pair(store, eid_a, eid_b, t_start, as_of)
            if result is not None:
                results[result.pair_key] = result

        log.info(
            "VineCopulaEncoder: fitted copulas for %d / %d pairs.",
            len(results), len(unique_pairs),
        )
        return results

    def store_results(
        self,
        results: dict[str, CopulaResult],
        store: Any,
    ) -> int:
        """Persist copula coefficients as pipeline signals.

        Signals stored per pair:
            ``copula.<pair_key>.tau``
            ``copula.<pair_key>.lambda_lower``
            ``copula.<pair_key>.lambda_upper``
            ``copula.<pair_key>.theta``

        Returns:
            Number of signals stored.
        """
        count = 0
        for pair_key, res in results.items():
            meta = {
                "entity_id_a": res.entity_id_a,
                "entity_id_b": res.entity_id_b,
                "pair_key": pair_key,
                "best_family": res.best_family,
                "n_joint_obs": res.n_joint_obs,
                "computed_at": res.computed_at,
            }
            signals = {
                f"copula.{pair_key}.tau": res.tau,
                f"copula.{pair_key}.lambda_lower": res.lambda_L,
                f"copula.{pair_key}.lambda_upper": res.lambda_U,
                f"copula.{pair_key}.theta": res.copula_theta,
            }
            for name, value in signals.items():
                try:
                    store.store_signal(signal_name=name, value=value, metadata=meta)
                    count += 1
                except Exception:
                    log.warning(
                        "VineCopulaEncoder: failed to store %r", name, exc_info=True
                    )

        log.info("VineCopulaEncoder: stored %d signals.", count)
        return count

    # ── Internal: pair scoring ─────────────────────────────────────────────

    def _score_pair(
        self,
        store: Any,
        eid_a: str,
        eid_b: str,
        t_start: float,
        as_of: float,
    ) -> CopulaResult | None:
        """Query, align, and fit copula for one entity pair."""
        obs_a = store.query_entity_observations(
            eid_a, since=t_start, until=as_of, limit=self.obs_limit
        )
        obs_b = store.query_entity_observations(
            eid_b, since=t_start, until=as_of, limit=self.obs_limit
        )

        if not obs_a or not obs_b:
            return None

        ser_a = self._bin_series(obs_a, t_start, as_of)
        ser_b = self._bin_series(obs_b, t_start, as_of)

        # Only keep bins where both series have observations
        mask = (~np.isnan(ser_a)) & (~np.isnan(ser_b))
        n_joint = int(mask.sum())

        if n_joint < self.min_joint_obs:
            log.debug(
                "VineCopulaEncoder: pair (%s, %s) only %d joint bins < %d — skip",
                eid_a[:12], eid_b[:12], n_joint, self.min_joint_obs,
            )
            return None

        x = ser_a[mask]
        y = ser_b[mask]

        tau, lambda_L, lambda_U, theta, family = _fit_bivariate_copula(x, y)

        pair_key = _pair_hash(eid_a, eid_b)
        return CopulaResult(
            entity_id_a=eid_a,
            entity_id_b=eid_b,
            pair_key=pair_key,
            tau=float(tau),
            lambda_L=float(lambda_L),
            lambda_U=float(lambda_U),
            copula_theta=float(theta),
            best_family=family,
            n_joint_obs=n_joint,
            computed_at=as_of,
        )

    # ── Internal: time series extraction ──────────────────────────────────

    def _bin_series(
        self,
        obs: list[dict[str, Any]],
        t_start: float,
        t_end: float,
    ) -> np.ndarray:
        """Build binned mean-value series from an observation list.

        Returns ndarray shape (n_bins,) with NaN for empty bins.
        """
        span = max(t_end - t_start, 1.0)
        bin_dur = span / self.n_bins

        sums = np.zeros(self.n_bins, dtype=np.float64)
        counts = np.zeros(self.n_bins, dtype=np.float64)

        for o in obs:
            t = float(o.get("observed_at", t_start))
            b = min(int((t - t_start) / bin_dur), self.n_bins - 1)
            v = _extract_value(o.get("value", {}))
            if v is not None and math.isfinite(v):
                sums[b] += v
                counts[b] += 1.0

        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(counts > 0, sums / counts, np.nan)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Pure-numpy copula fitting (no external deps)
# ═══════════════════════════════════════════════════════════════════════════


def _pair_hash(eid_a: str, eid_b: str) -> str:
    """Deterministic 12-char hash for a sorted entity pair."""
    a, b = sorted([eid_a, eid_b])
    digest = hashlib.sha256(f"{a}|{b}".encode()).hexdigest()
    return digest[:12]


def _extract_value(value_dict: Any) -> float | None:
    """Extract a numeric scalar from an observation value dict."""
    if not isinstance(value_dict, dict):
        return None
    for k in _VALUE_KEYS:
        if k in value_dict:
            try:
                v = float(value_dict[k])
                return v if math.isfinite(v) else None
            except (TypeError, ValueError):
                pass
    return None


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Kendall's τ-b for two 1D arrays.

    Uses scipy if available, else falls back to the O(n²) definition:
        τ = (concordant − discordant) / sqrt((n0−n1)(n0−n2))

    where n0=n(n-1)/2, n1=tied in x, n2=tied in y.  (τ-b correction.)
    """
    try:
        from scipy.stats import kendalltau as _kt  # noqa: PLC0415

        result = _kt(x, y, nan_policy="omit")
        return float(result.statistic)
    except ImportError:
        pass

    n = len(x)
    if n < 2:
        return 0.0

    concordant = discordant = tied_x = tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            else:
                if dx == 0:
                    tied_x += 1
                if dy == 0:
                    tied_y += 1

    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - tied_x) * (n0 - tied_y))
    if denom < _EPS:
        return 0.0
    return (concordant - discordant) / denom


def _pseudo_observations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank-transform x and y to pseudo-observations in (0,1).

    u_i = rank(x_i) / (n+1) — avoids 0 and 1 at boundaries.
    """
    n = len(x)
    u = np.argsort(np.argsort(x)).astype(float) + 1.0
    v = np.argsort(np.argsort(y)).astype(float) + 1.0
    return u / (n + 1), v / (n + 1)


def _clayton_lambda_L(theta: float) -> float:
    """Lower tail dependence for Clayton(θ).

    λ_L = 2^{−1/θ}   (Nelsen 2006, eq. 5.4)
    Valid for θ > 0.
    """
    if theta <= _EPS:
        return 0.0
    return float(2.0 ** (-1.0 / theta))


def _gumbel_lambda_U(theta: float) -> float:
    """Upper tail dependence for Gumbel(θ).

    λ_U = 2 − 2^{1/θ}   (Nelsen 2006, eq. 5.6)
    Valid for θ ≥ 1.
    """
    if theta < 1.0:
        return 0.0
    return float(2.0 - 2.0 ** (1.0 / theta))


def _fit_bivariate_copula(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, float, str]:
    """Fit bivariate Clayton + Gumbel copulas by MOM from Kendall's τ.

    Returns (tau, lambda_L, lambda_U, theta, family).

    For τ > 0:
        - Clayton MOM: θ = 2τ/(1−τ),  λ_L = 2^{−1/θ},  λ_U = 0
        - Gumbel MOM:  θ = 1/(1−τ),   λ_U = 2−2^{1/θ}, λ_L = 0
        We report both λ_L (Clayton) and λ_U (Gumbel).

    For τ ≤ 0:
        Counter-moving pairs: survival Clayton gives upper-tail under
        opposite movement risk.
        - θ' = 2|τ|/(1−|τ|), λ_survival = 2^{−1/θ'} stored as λ_U.
        - λ_L = 0 (no lower-tail co-crash structure).
    """
    tau = _kendall_tau(x, y)

    if tau > _EPS:
        # Positive dependence — fit Clayton (lower) and Gumbel (upper)
        # Clayton MOM: θ = 2τ/(1−τ)  (Joe 1997, Table 5.1)
        theta_c = 2.0 * tau / max(1.0 - tau, _EPS)
        lambda_L = _clayton_lambda_L(theta_c)

        # Gumbel MOM: θ = 1/(1−τ)  (Nelsen 2006, Table 5.1)
        theta_g = 1.0 / max(1.0 - tau, _EPS)
        lambda_U = _gumbel_lambda_U(theta_g)

        # Use Clayton theta as primary theta (lower-tail focus)
        return tau, lambda_L, lambda_U, theta_c, "Clayton+Gumbel"

    elif tau < -_EPS:
        # Negative dependence — survival Clayton for counter-movement risk
        abs_tau = abs(tau)
        theta_sc = 2.0 * abs_tau / max(1.0 - abs_tau, _EPS)
        lambda_survival = _clayton_lambda_L(theta_sc)
        # Survival Clayton λ is effectively an upper-tail risk for counter-movers
        return tau, 0.0, lambda_survival, theta_sc, "SurvivalClayton"

    else:
        # τ ≈ 0 — independence, no tail structure
        return tau, 0.0, 0.0, 0.0, "Independence"
