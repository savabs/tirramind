"""False discovery rate control for convergence detection.

Implements a four-level statistical filter that reduces false convergence
detections across thousands of simultaneous pairwise tests:

1. **Benjamini-Hochberg** (BH) — controls the expected proportion of
   false discoveries among all pairwise coincidence tests at FDR level *q*.
   Reference: Benjamini & Hochberg (1995), *J. R. Statist. Soc. B* 57(1).

2. **Fisher's combined test** — merges the per-edge p-values within a
   convergence clique into a single aggregate p-value.
   Reference: Fisher (1925), *Statistical Methods for Research Workers*.

3. **Persistence filter** — requires a convergence pattern to appear in
   ≥ *min_periods* consecutive detection cycles before emission.

4. **Cross-category filter** — requires ≥ *min_categories* distinct
   taxonomy buckets in a clique (defence-in-depth; also enforced in
   :mod:`agent.convergence.graph`).

All functions are deterministic and LLM-free (Pipeline Layer contract).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from agent.convergence.coincidence import CoincidenceResult
from agent.convergence.graph import (
    ConvergenceClique,
    build_coincidence_graph,
    detect_convergence_cliques,
)

log = logging.getLogger(__name__)

# Smallest p-value allowed before clipping (prevents log(0) in Fisher's).
_P_FLOOR = 1e-300


# ── Level 1: Benjamini-Hochberg FDR on pairwise p-values ──────


def apply_bh_correction(
    p_values: dict[tuple[str, str], float],
    q: float = 0.05,
) -> dict[tuple[str, str], bool]:
    """Apply Benjamini-Hochberg FDR correction to pairwise p-values.

    Parameters
    ----------
    p_values : dict[(sig_a, sig_b), float]
        Raw p-values for every tested pair.
    q : float
        Target false-discovery rate (default 0.05).

    Returns
    -------
    dict[(sig_a, sig_b), bool]
        ``True`` if the pair is significant after BH correction.
    """
    if not p_values:
        return {}

    pairs = list(p_values.keys())
    raw_p = np.array([p_values[k] for k in pairs], dtype=np.float64)

    # Clip to valid range to protect against numerical noise.
    raw_p = np.clip(raw_p, _P_FLOOR, 1.0)

    reject, _, _, _ = multipletests(raw_p, alpha=q, method="fdr_bh")

    return {pair: bool(rej) for pair, rej in zip(pairs, reject)}


# ── Level 2: Fisher's combined test ───────────────────────────


def fisher_combined_test(p_values: list[float]) -> float:
    """Combine multiple p-values via Fisher's method.

    .. math::

        \\chi^2 = -2 \\sum_{i=1}^k \\ln(p_i), \\quad
        p_{\\text{combined}} = \\text{sf}(\\chi^2,\\; 2k)

    Parameters
    ----------
    p_values : list[float]
        Individual p-values to combine.

    Returns
    -------
    float
        Combined p-value from the chi-squared survival function.
        Returns 1.0 for empty input.
    """
    if not p_values:
        return 1.0

    k = len(p_values)
    if k == 1:
        return float(np.clip(p_values[0], _P_FLOOR, 1.0))

    clipped = [max(p, _P_FLOOR) for p in p_values]
    chi2_stat = -2.0 * sum(math.log(p) for p in clipped)
    df = 2 * k

    combined_p = float(sp_stats.chi2.sf(chi2_stat, df))
    return max(combined_p, _P_FLOOR)


# ── Level 3: Persistence filter ───────────────────────────────


def persistence_filter(
    events: list[ConvergenceClique],
    history: dict[tuple[str, ...], int],
    min_periods: int = 2,
) -> list[ConvergenceClique]:
    """Keep only cliques that have persisted for *min_periods* consecutive cycles.

    Parameters
    ----------
    events : list[ConvergenceClique]
        Cliques detected in the current cycle.
    history : dict[tuple[str, ...], int]
        Mutable mapping of clique fingerprint → consecutive-detection count.
        Updated **in place**: new cliques are added, disappeared cliques
        are removed, surviving cliques have their count incremented.
    min_periods : int
        Minimum number of consecutive detection cycles required to emit.

    Returns
    -------
    list[ConvergenceClique]
        Cliques that satisfy the persistence requirement.
    """
    current_fps: set[tuple[str, ...]] = set()
    survivors: list[ConvergenceClique] = []

    for ev in events:
        fp = ev.fingerprint()
        current_fps.add(fp)
        history[fp] = history.get(fp, 0) + 1

        if history[fp] >= min_periods:
            survivors.append(ev)

    # Reset counters for cliques that disappeared this cycle.
    disappeared = [fp for fp in list(history.keys()) if fp not in current_fps]
    for fp in disappeared:
        del history[fp]

    return survivors


# ── Level 4: Cross-category filter ────────────────────────────


def cross_category_filter(
    events: list[ConvergenceClique],
    min_categories: int = 2,
) -> list[ConvergenceClique]:
    """Defence-in-depth: reject cliques with too few taxonomy categories.

    This is intentionally redundant with the ``min_categories`` parameter
    in :func:`~agent.convergence.graph.detect_convergence_cliques` — it
    guards against callers that bypass graph-level filtering.
    """
    return [ev for ev in events if len(ev.categories) >= min_categories]


# ── Full pipeline ──────────────────────────────────────────────


def apply_all_controls(
    pairs_p: dict[tuple[str, str], float],
    scores: dict[tuple[str, str], CoincidenceResult],
    categories: dict[str, str],
    history: dict[tuple[str, ...], int],
    *,
    q: float = 0.05,
    min_persist: int = 2,
    min_cats: int = 2,
    min_clique_size: int = 3,
) -> list[ConvergenceClique]:
    """Run the full four-level FDR pipeline.

    1. BH correction on *pairs_p* → keep significant pairs only.
    2. Rebuild coincidence graph from surviving pairs (using *scores*).
    3. Detect cliques with cross-category constraint.
    4. Fisher's combined p-value per clique (set on ``clique.score``
       metadata — does not replace ``score`` but is used for ranking).
    5. Persistence filter (updates *history* in place).
    6. Cross-category filter (defence-in-depth).

    Parameters
    ----------
    pairs_p : dict[(sig_a, sig_b), float]
        Raw pairwise p-values.
    scores : dict[(sig_a, sig_b), CoincidenceResult]
        Full coincidence results (needed for graph weights).
    categories : dict[str, str]
        Signal ID → taxonomy category.
    history : dict[tuple[str, ...], int]
        Mutable persistence state (updated in place).
    q, min_persist, min_cats, min_clique_size
        Control thresholds.

    Returns
    -------
    list[ConvergenceClique]
        Final surviving convergence events.
    """
    # Step 1: BH correction.
    significant = apply_bh_correction(pairs_p, q=q)

    # Step 2: Filter scores to only BH-surviving pairs.
    surviving_scores: dict[tuple[str, str], CoincidenceResult] = {}
    for pair, is_sig in significant.items():
        if is_sig and pair in scores:
            surviving_scores[pair] = scores[pair]

    if not surviving_scores:
        # Nothing survived BH — reset disappeared cliques and return.
        disappeared = list(history.keys())
        for fp in disappeared:
            del history[fp]
        return []

    # Step 3: Build graph and detect cliques.
    G = build_coincidence_graph(
        surviving_scores,
        categories,
        p_threshold=1.0,  # Already BH-filtered; accept all remaining.
    )
    cliques = detect_convergence_cliques(
        G,
        min_size=min_clique_size,
        min_categories=min_cats,
    )

    # Step 4: Fisher's combined p-value per clique.
    for clique in cliques:
        if clique.p_values:
            combined_p = fisher_combined_test(clique.p_values)
        else:
            combined_p = 1.0
        # Attach to detail; leave clique.score (graph score) untouched.
        clique.p_values_combined = combined_p  # type: ignore[attr-defined]

    # Step 5: Persistence filter.
    cliques = persistence_filter(cliques, history, min_periods=min_persist)

    # Step 6: Cross-category filter (defence-in-depth).
    cliques = cross_category_filter(cliques, min_categories=min_cats)

    return cliques
