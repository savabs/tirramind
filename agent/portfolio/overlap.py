"""Find the hidden same-bet inside a portfolio.

Two questions, both answered as *facts about the past*, never as advice:

1. ``correlation_clusters`` — which of these holdings actually moved together?
   A portfolio of twelve names that splits into three correlation groups is
   three bets wearing twelve tickers.
2. ``factor_exposure`` — which public yardsticks (index / sector / FX /
   commodity / vol) explain the portfolio's day-to-day movement, and how much
   of it is left over?

WHAT THIS MODULE WILL NOT DO
----------------------------
It emits no recommendation, no forecast, and no judgement. Every string it
produces describes a realised, dated, sample-sized observation about prices
that have already happened. "Eight of your holdings moved together over
2024-05-02..2026-09-22 (n=573)" is a fact. "You are over-concentrated" is an
opinion and "trim it" is regulated advice. Callers rendering this output must
keep that line; the dataclasses here are deliberately free of any verb that
implies a future.

WHERE IT MISLEADS (read this before trusting a number)
------------------------------------------------------
* **Correlation is not causation and not co-movement forever.** These are
  realised sample correlations over one specific window. A different window
  gives different numbers. Every object here carries its window and its n so
  that this is impossible to hide.
* **One common window for everything.** Pairwise-complete correlations quietly
  compute each cell on a different sample; a matrix built that way is not even
  guaranteed to be a valid correlation matrix. We instead intersect calendars
  so that *every* number in a report shares one window and one n, and we
  report by name each holding dropped to get there.
* **Constant weights.** The portfolio series is ``sum(w_i * r_i)`` using the
  weights you pass, held fixed across the whole window. That is a "what if I
  had held today's book the whole time", not your realised P&L. You never
  traded like that.
* **Currency.** Reference factors are quoted in their own currency (INDA and
  SPY in USD, USDINR in USD/INR). An INR-quoted holding regressed on a
  USD-quoted factor absorbs the exchange rate too. Each factor reports its
  ``quote_currency`` for exactly this reason.
* **89 candidate factors is a garden of forking paths.** Fitting all of them
  and reporting the prettiest would be indefensible on a public page. See
  ``factor_exposure`` for the selection discipline and the holdout that exists
  to catch it lying.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

__all__ = [
    "Cluster",
    "ClusterSet",
    "Exclusion",
    "Window",
    "correlation_clusters",
    "threshold_sweep",
    "factor_exposure",
    "format_overlap_report",
    "load_reference_returns",
]

# --------------------------------------------------------------------------
# Defaults. Every one of these is a judgement call, so each is a named
# constant with a reason rather than a magic number buried in a signature.
# --------------------------------------------------------------------------

#: Below this many shared trading days we refuse to report a correlation at
#: all. At n=120 the 95% interval on a single correlation is about +/-0.18
#: wide near rho=0.5 — already generous. At n=20 it spans nearly the whole
#: [-1, 1] range, which is why 20 days of history yields no statistic here.
MIN_TRADING_DAYS = 120

#: Average-linkage cut on 1 - rho. 0.6 is high enough that a cluster is a
#: genuine same-bet rather than "both are equities".
DEFAULT_CORRELATION_THRESHOLD = 0.6

#: Above this, two holdings are not merely correlated, they are near enough
#: the same instrument (share classes, an index fund and its index, two funds
#: tracking one benchmark) to call out separately.
NEAR_DUPLICATE_THRESHOLD = 0.98

#: Hard cap on regressors. With a few hundred days, four or more factors
#: chosen out of 89 is curve-fitting with extra steps.
DEFAULT_MAX_FACTORS = 3

#: Fraction of the window held back, untouched by factor selection.
DEFAULT_HOLDOUT_FRACTION = 0.4

#: A factor must cover this share of the portfolio window to be a candidate.
#: Every admitted candidate then shares one common sample, so a factor with a
#: ragged calendar costs *every* number days. 0.90 is calibrated, not guessed:
#: US and Indian exchange holidays differ enough that US futures cover only
#: ~93% of NSE trading days, so a 0.95 gate silently threw away all 20
#: commodities, all 4 equity-index futures and both crypto series — for the
#: sake of 29 days out of 813 on a real Indian book. Losing crude oil as a
#: candidate to save 3.6% of the sample is the wrong trade.
MIN_FACTOR_COVERAGE = 0.90

#: A selected factor must account for at least this much of daily variance in
#: absolute terms to be reported as material. A factor can clear a
#: significance gate while explaining ~1% of the movement; on a public page
#: that line reads as a finding when it is a rounding error. Immaterial
#: factors are still returned, flagged, and still counted in R-squared — they
#: are labelled, never deleted.
MIN_MATERIAL_VARIANCE_SHARE = 0.02


# --------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """The dated sample a statistic was computed on. Never optional."""

    start: dt.date
    end: dt.date
    n_days: int

    def __str__(self) -> str:
        return f"{self.start.isoformat()}..{self.end.isoformat()} (n={self.n_days} trading days)"


@dataclass(frozen=True)
class Exclusion:
    """Something we did not use, and why. Silent exclusion is how you produce
    a confident wrong answer, so nothing is ever dropped without one of these.
    """

    name: str
    reason: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.name}: {self.reason}" + (f" ({self.detail})" if self.detail else "")


@dataclass(frozen=True)
class Cluster:
    """A group of holdings whose returns moved together over one window.

    ``mean_correlation`` is the mean of the group's off-diagonal pairwise
    correlations, which flatters big groups: a group of eight has 28 pairs and
    the mean hides the worst of them. ``min_correlation`` is the weakest pair
    in the group and is the number to quote if you only quote one —
    ``min_correlation_ci`` is its 95% interval under a Fisher-z approximation
    (which assumes joint normality and independent days; daily equity returns
    are neither, so treat the interval as indicative, not exact).
    """

    members: tuple[str, ...]
    weight: float
    mean_correlation: float
    min_correlation: float
    min_correlation_ci: tuple[float, float]
    window: Window
    surprise: float

    @property
    def size(self) -> int:
        return len(self.members)

    def statement(self) -> str:
        """A factual sentence. No verb that implies a future."""
        names = ", ".join(self.members)
        return (
            f"{self.size} holdings ({names}) carrying {self.weight:.1%} of the book "
            f"moved together over {self.window}: mean pairwise correlation "
            f"{self.mean_correlation:.2f}, weakest pair {self.min_correlation:.2f} "
            f"(95% CI {self.min_correlation_ci[0]:.2f}..{self.min_correlation_ci[1]:.2f})."
        )


class ClusterSet(list):
    """``list[Cluster]`` that also carries what was left out.

    It is a real list, so ``correlation_clusters`` honours its ``list[Cluster]``
    contract and callers can iterate it without knowing this type exists. The
    extra attributes exist because a bare list has nowhere to put the
    exclusions, and an exclusion that has nowhere to go becomes a silent one.
    """

    def __init__(
        self,
        clusters: Sequence[Cluster] = (),
        *,
        window: Window | None = None,
        holdings_used: tuple[str, ...] = (),
        singletons: tuple[str, ...] = (),
        near_duplicates: tuple[tuple[str, str, float], ...] = (),
        excluded: tuple[Exclusion, ...] = (),
        median_pair_correlation: float | None = None,
        weights_used: Mapping[str, float] | None = None,
        stable_between: tuple[float, float] | None = None,
        notes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(clusters)
        #: Share of the *measured* book per holding, after exclusions and
        #: renormalisation. Sums to 1 across holdings_used. Exposed because a
        #: renormalisation nobody can see is a renormalisation nobody can check.
        self.weights_used = dict(weights_used or {})
        #: The range of correlation cutoffs over which this exact grouping is
        #: unchanged. The single most useful defence against "you picked the
        #: threshold to get that answer": a grouping that survives a wide
        #: plateau is a property of the book, one that exists at exactly one
        #: cutoff is a property of the cutoff.
        self.stable_between = stable_between
        self.window = window
        self.holdings_used = holdings_used
        self.singletons = singletons
        self.near_duplicates = near_duplicates
        self.excluded = excluded
        self.median_pair_correlation = median_pair_correlation
        self.notes = notes

    @property
    def n_groups(self) -> int:
        """Distinct correlation groups: clusters plus holdings that joined none.

        This is the "you have 3 bets" number. It is a count of groups at one
        specific correlation threshold on one specific window, not a
        measurement of anything intrinsic — move the threshold and it moves.
        """
        return len(self) + len(self.singletons)


# --------------------------------------------------------------------------
# Panel handling
# --------------------------------------------------------------------------


def _as_returns(panel: Any, kind: str = "prices") -> pd.DataFrame:
    """Coerce a holdings panel to a simple-return DataFrame.

    Accepts a price DataFrame (the default, and the documented contract), a
    dict of ticker -> price Series, or an object exposing ``.returns`` or
    ``.prices``. We never *guess* whether a DataFrame holds prices or returns:
    guessing is how you silently produce a plausible wrong answer. Pass
    ``kind="returns"`` if you are handing over returns. A price panel
    containing a non-positive value raises rather than silently becoming
    nonsense, since that is the signature of returns passed as prices.
    """
    if hasattr(panel, "returns") and isinstance(panel.returns, pd.DataFrame):
        frame, kind = panel.returns, "returns"
    elif hasattr(panel, "prices") and isinstance(panel.prices, pd.DataFrame):
        frame, kind = panel.prices, "prices"
    elif isinstance(panel, pd.DataFrame):
        frame = panel
    elif isinstance(panel, Mapping):
        frame = pd.DataFrame(dict(panel))
    else:
        raise TypeError(
            "panel must be a DataFrame of prices, a mapping of ticker -> price "
            f"Series, or an object with .prices/.returns; got {type(panel)!r}"
        )

    if kind not in ("prices", "returns"):
        raise ValueError(f"kind must be 'prices' or 'returns', got {kind!r}")

    frame = frame.sort_index()
    frame.index = _to_dates(frame.index)
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.astype("float64")

    if kind == "returns":
        return frame

    bad = [c for c in frame.columns if (frame[c].dropna() <= 0).any()]
    if bad:
        raise ValueError(
            f"columns {bad} contain non-positive values, so this is not a price "
            "panel. If you are passing returns, say so with kind='returns' — "
            "this module will not guess."
        )
    return frame.pct_change()


def _to_dates(index: Any) -> pd.Index:
    """Normalise any date-ish index to plain ``datetime.date`` objects."""
    converted = pd.to_datetime(pd.Index(index))
    if getattr(converted, "tz", None) is not None:
        converted = converted.tz_convert(None)
    return pd.Index([d.date() for d in converted], name=getattr(index, "name", None))


def _normalise_weights(
    weights: Mapping[str, float], available: Sequence[str]
) -> tuple[dict[str, float], list[Exclusion], list[str]]:
    """Restrict weights to what the panel covers and renormalise to 1.

    Renormalisation is reported, not assumed: if a quarter of the book was
    dropped for want of history, every weight downstream is a share of the
    *remaining* three quarters and saying so is the difference between a fact
    and a wrong number.
    """
    excluded: list[Exclusion] = []
    kept: dict[str, float] = {}
    notes: list[str] = []
    available_set = set(available)

    for name, raw in weights.items():
        value = float(raw)
        if not math.isfinite(value):
            excluded.append(Exclusion(name, "non-finite weight", repr(raw)))
            continue
        if value == 0.0:
            excluded.append(Exclusion(name, "zero weight", "not held"))
            continue
        if name not in available_set:
            excluded.append(Exclusion(name, "no price history in panel", "ticker absent or unresolved"))
            continue
        kept[name] = value

    unweighted = [c for c in available if c not in weights]
    for name in unweighted:
        excluded.append(Exclusion(name, "present in panel but no weight supplied", ""))

    total = sum(kept.values())
    if not kept or not math.isfinite(total) or abs(total) < 1e-12:
        return {}, excluded, notes
    if any(v < 0 for v in kept.values()):
        notes.append(
            "Portfolio contains negative weights; correlation groups are "
            "reported on absolute co-movement and a short leg inside a group "
            "offsets rather than adds to it."
        )
    if abs(total - 1.0) > 1e-9:
        notes.append(f"Supplied weights summed to {total:.6g}; renormalised to 1.")
    return {k: v / total for k, v in kept.items()}, excluded, notes


def _common_window(returns: pd.DataFrame, min_days: int) -> tuple[pd.DataFrame, list[Exclusion]]:
    """Intersect calendars until every remaining column shares >= min_days.

    Greedy: at each step drop whichever column's removal buys the most shared
    days, tie-broken by fewest own observations then by name so the result is
    deterministic. This is the honest alternative to pairwise-complete
    correlations, which silently compute each cell on a different sample.
    Columns dropped here are returned as exclusions, never swallowed.
    """
    excluded: list[Exclusion] = []
    frame = returns.dropna(axis=1, how="all")
    for column in returns.columns:
        if column not in frame.columns:
            excluded.append(Exclusion(column, "no return observations", "column was entirely empty"))

    for column in list(frame.columns):
        observed = int(frame[column].notna().sum())
        if observed < min_days:
            excluded.append(
                Exclusion(
                    column,
                    "insufficient history",
                    f"{observed} trading days of returns, need {min_days}",
                )
            )
            frame = frame.drop(columns=[column])

    if frame.shape[1] == 1:
        only = frame.columns[0]
        return frame.loc[frame[only].notna()], excluded

    while frame.shape[1] >= 2:
        mask = frame.notna().all(axis=1)
        if int(mask.sum()) >= min_days:
            return frame.loc[mask], excluded
        best: str | None = None
        best_key: tuple[int, int, str] | None = None
        for column in frame.columns:
            gain = int(frame.drop(columns=[column]).notna().all(axis=1).sum())
            own = int(frame[column].notna().sum())
            key = (gain, -own, str(column))
            if best_key is None or key > best_key:
                best, best_key = str(column), key
        shared = int(mask.sum())
        excluded.append(
            Exclusion(
                str(best),
                "calendar does not overlap the rest of the book",
                f"keeping it would cap every correlation at {shared} shared days",
            )
        )
        frame = frame.drop(columns=[best])

    return frame.iloc[0:0], excluded


def _partition(linkage_matrix: np.ndarray, names: Sequence[str], threshold: float) -> frozenset:
    labels = fcluster(linkage_matrix, t=1.0 - threshold, criterion="distance")
    groups: dict[int, list[str]] = {}
    for name, label in zip(names, labels):
        groups.setdefault(int(label), []).append(name)
    return frozenset(frozenset(v) for v in groups.values())


def _stability_range(
    linkage_matrix: np.ndarray, names: Sequence[str], threshold: float, step: float = 0.01
) -> tuple[float, float]:
    """Widen outward from `threshold` while the partition is identical.

    Reported, not used to choose anything. A narrow plateau is a warning that
    the grouping is an artefact of where the cutoff landed.
    """
    target = _partition(linkage_matrix, names, threshold)
    low = high = threshold
    probe = threshold
    while probe - step >= -0.99:
        probe = round(probe - step, 4)
        if _partition(linkage_matrix, names, probe) != target:
            break
        low = probe
    probe = threshold
    while probe + step <= 0.99:
        probe = round(probe + step, 4)
        if _partition(linkage_matrix, names, probe) != target:
            break
        high = probe
    return (low, high)


def threshold_sweep(
    panel: Any,
    weights: Mapping[str, float],
    *,
    kind: str = "prices",
    min_days: int = MIN_TRADING_DAYS,
    thresholds: Sequence[float] = tuple(round(0.8 - 0.05 * i, 2) for i in range(11)),
) -> list[dict]:
    """Every grouping this book produces, at every cutoff. A calibration tool.

    ``correlation_clusters`` has to commit to one threshold, and that choice
    changes the headline: the same twelve-holding book can be nine groups or
    four. Rather than tune the default until it produces a pleasing number —
    which is the garden of forking paths aimed at a marketing line — this
    exposes the whole curve so the choice is made once, deliberately, on
    evidence, and can be disclosed.

    Returns one row per threshold with the group count and the grouping, so
    plateaus (a partition that survives a wide range of cutoffs) are visible.
    """
    rows = []
    for t in thresholds:
        result = correlation_clusters(panel, weights, kind=kind, min_days=min_days, correlation_threshold=t)
        rows.append(
            {
                "threshold": t,
                "n_groups": result.n_groups,
                "groups": [list(c.members) for c in result],
                "grouped_weight": float(sum(c.weight for c in result)),
                "stable_between": result.stable_between,
            }
        )
    return rows


def _fisher_ci(rho: float, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """95% interval for a correlation via the Fisher z transform.

    Assumes bivariate normality and independent observations. Daily returns
    are fat-tailed and mildly autocorrelated, so the true interval is somewhat
    wider than this. Quoted as indicative.
    """
    if n <= 3:
        return (float("nan"), float("nan"))
    rho = float(np.clip(rho, -0.999999, 0.999999))
    zed = np.arctanh(rho)
    half = z / math.sqrt(n - 3)
    return (float(np.tanh(zed - half)), float(np.tanh(zed + half)))


# --------------------------------------------------------------------------
# 1. Correlation clusters
# --------------------------------------------------------------------------


def correlation_clusters(
    panel: Any,
    weights: Mapping[str, float],
    *,
    kind: str = "prices",
    min_days: int = MIN_TRADING_DAYS,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> list[Cluster]:
    """Group holdings that moved together, and say what was left out.

    Agglomerative average-linkage clustering on the distance ``1 - rho``
    between daily simple returns, cut at ``1 - correlation_threshold``. Under
    average linkage that cut means the *average* correlation between any two
    merged groups clears the threshold, so an individual pair inside a cluster
    can sit below it — which is why every cluster reports its weakest pair.

    Returns a ``ClusterSet``: a real ``list[Cluster]``, ordered by how
    surprising the group is to a human rather than by statistical magnitude
    (see below), carrying ``.window``, ``.excluded``, ``.singletons``,
    ``.near_duplicates`` and ``.n_groups``.

    Where it misleads:

    * ``surprise`` is a presentation ordering, not a statistic. It is
      ``weight * (size - 1) * mean_rho``: it deliberately promotes a large,
      heavy group over a tight pair of two, because eight names carrying 60%
      of a book is the finding and two names at 0.99 is a footnote. The
      ``size - 1`` term is what does that, and it means a *more* correlated
      pair can rank below a *less* correlated group of eight — by design. Do
      not quote it as a measurement of anything.
      (An earlier version subtracted the book's median pair correlation to
      reward "unusually" correlated groups. On a book whose single big cluster
      supplies most of the pairs, that median *is* the cluster, so the term
      cancelled the finding and ranked a 2-name pair first. It was removed.)
    * The number of groups depends on ``correlation_threshold``. It is a
      choice, not a discovery, and on a real Indian large-cap book it is the
      whole headline: the same twelve holdings are nine groups at 0.60 and
      four at 0.40. ``.stable_between`` reports the range of cutoffs over
      which the returned grouping is unchanged — quote it, because a grouping
      that survives a wide plateau is a property of the book and one that
      holds at a single cutoff is a property of the cutoff. Use
      ``threshold_sweep`` to choose the default on evidence rather than
      tuning it until the headline reads well.
    * Everything is one window with one n, shown on every cluster. Holdings
      whose history was too short or whose trading calendar did not overlap
      are in ``.excluded`` with the reason — they are not in the count.
    """
    returns = _as_returns(panel, kind)
    weights_norm, weight_exclusions, notes = _normalise_weights(weights, list(returns.columns))
    if not weights_norm:
        return ClusterSet(
            excluded=tuple(weight_exclusions),
            notes=tuple(notes) + ("No holding had both a usable weight and price history.",),
        )

    returns = returns[list(weights_norm)]
    aligned, window_exclusions = _common_window(returns, min_days)
    excluded = tuple(weight_exclusions + window_exclusions)

    if aligned.shape[1] < 2 or aligned.shape[0] < min_days:
        return ClusterSet(
            excluded=excluded,
            notes=tuple(notes)
            + (
                f"Not enough shared history to compute any correlation: "
                f"{aligned.shape[1]} holdings share {aligned.shape[0]} trading days, "
                f"and this module reports nothing below {min_days}.",
            ),
        )

    dates = list(aligned.index)
    window = Window(dates[0], dates[-1], len(dates))

    # Re-normalise across what actually survived, and say so.
    surviving = {k: weights_norm[k] for k in aligned.columns}
    survived_share = sum(surviving.values())
    surviving = {k: v / survived_share for k, v in surviving.items()}
    if survived_share < 0.999999:
        notes.append(
            f"Cluster weights are shares of the {survived_share:.1%} of the book that had "
            f"usable overlapping history, not of the whole book."
        )

    corr = aligned.corr()
    values = np.array(corr.to_numpy(dtype=float), dtype=float, copy=True)
    np.fill_diagonal(values, 1.0)
    offdiag = values[np.triu_indices_from(values, k=1)]
    median_pair = float(np.median(offdiag))

    names = list(corr.columns)
    near_duplicates = tuple(
        (names[i], names[j], float(values[i, j]))
        for i, j in zip(*np.triu_indices_from(values, k=1))
        if values[i, j] >= NEAR_DUPLICATE_THRESHOLD
    )

    distance = np.clip(1.0 - values, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    labels = fcluster(linkage_matrix, t=1.0 - correlation_threshold, criterion="distance")
    stable_between = _stability_range(linkage_matrix, names, correlation_threshold)

    clusters: list[Cluster] = []
    singletons: list[str] = []
    for label in sorted(set(labels)):
        idx = [i for i, value in enumerate(labels) if value == label]
        members = tuple(names[i] for i in idx)
        if len(idx) == 1:
            singletons.append(members[0])
            continue
        block = values[np.ix_(idx, idx)]
        pairs = block[np.triu_indices_from(block, k=1)]
        mean_rho = float(pairs.mean())
        min_rho = float(pairs.min())
        weight = float(sum(surviving[m] for m in members))
        surprise = weight * (len(members) - 1) * mean_rho
        clusters.append(
            Cluster(
                members=members,
                weight=weight,
                mean_correlation=mean_rho,
                min_correlation=min_rho,
                min_correlation_ci=_fisher_ci(min_rho, window.n_days),
                window=window,
                surprise=float(surprise),
            )
        )

    clusters.sort(key=lambda c: (-c.surprise, -c.weight, c.members))
    return ClusterSet(
        clusters,
        window=window,
        holdings_used=tuple(names),
        singletons=tuple(sorted(singletons)),
        near_duplicates=near_duplicates,
        excluded=excluded,
        median_pair_correlation=median_pair,
        weights_used=surviving,
        stable_between=stable_between,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# 2. Factor exposure
# --------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_reference_returns_cached(
    db_path: str, mtime: float, size: int
) -> tuple[pd.DataFrame, tuple[tuple[str, str, str, str], ...]]:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            """
            SELECT e.entity_id,
                   e.canonical_name,
                   json_extract(e.metadata_json, '$.ticker'),
                   json_extract(e.metadata_json, '$.asset_class'),
                   o.observed_at,
                   o.value_json
              FROM entity_observations o
              JOIN entities e ON e.entity_id = o.entity_id
             WHERE o.observation_type = 'instrument_daily'
               AND e.entity_type = 'instrument'
            """
        ).fetchall()
    finally:
        conn.close()

    series: dict[str, dict[dt.date, float]] = {}
    meta: dict[str, tuple[str, str, str, str]] = {}
    for _eid, name, ticker, asset_class, observed_at, value_json in rows:
        if not ticker:
            continue
        payload = json.loads(value_json)
        log_return = payload.get("log_return")
        if log_return is None or not math.isfinite(float(log_return)):
            continue
        day = dt.datetime.fromtimestamp(float(observed_at), dt.UTC).date()
        series.setdefault(ticker, {})[day] = math.expm1(float(log_return))
        meta[ticker] = (ticker, name or ticker, asset_class or "unknown", _quote_currency(ticker))

    frame = pd.DataFrame({t: pd.Series(v) for t, v in series.items()}).sort_index()
    frame.index = _to_dates(frame.index)
    return frame, tuple(meta[t] for t in frame.columns)


def _quote_currency(ticker: str) -> str:
    """Currency a reference series is quoted in. Not cosmetic: regressing an
    INR-quoted holding on a USD-quoted factor silently folds USD/INR into the
    beta, and the reader has to be able to see that."""
    if ticker.endswith("=X"):
        base, _, _ = ticker.partition("=")
        return f"{base[:3]}/{base[3:6]}" if len(base) >= 6 else base
    return "USD"


def load_reference_returns(db_path: str | os.PathLike[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the reference instruments' daily simple returns, read-only.

    Opens the live pipeline DB with ``mode=ro`` and never anything else. Stored
    log returns are converted with ``expm1``; a stored return that spans a
    market holiday spans it here too, which is why factor coverage is checked
    against the portfolio's own window rather than assumed.

    Returns ``(returns, metadata)`` where metadata is indexed by ticker with
    ``name``, ``asset_class`` and ``quote_currency`` columns.
    """
    path = os.fspath(db_path)
    stat = os.stat(path)
    frame, meta = _load_reference_returns_cached(path, stat.st_mtime, stat.st_size)
    metadata = pd.DataFrame(
        [{"ticker": t, "name": n, "asset_class": a, "quote_currency": c} for t, n, a, c in meta]
    ).set_index("ticker")
    return frame.copy(), metadata


def _newey_west_lag(n: int) -> int:
    return max(1, int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least squares with intercept. Returns (betas_without_intercept, r2, adj_r2)."""
    n, k = x.shape
    design = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")
    return coef[1:], float(r2), float(adj)


def _forward_select(
    y: np.ndarray,
    candidates: pd.DataFrame,
    max_factors: int,
    alpha: float,
) -> tuple[list[str], list[str]]:
    """Forward stepwise selection with a Bonferroni gate. Returns (chosen, log).

    At each step every remaining candidate is fitted alongside those already
    chosen; the one with the highest adjusted R-squared wins, but only enters
    if its HAC t-statistic clears a Bonferroni threshold for the number of
    candidates *examined at that step*. That correction is approximate — it
    does not price in the selection of earlier steps — which is precisely why
    ``factor_exposure`` also reports a holdout it never saw.
    """
    import statsmodels.api as sm

    chosen: list[str] = []
    log: list[str] = []
    n = len(y)
    lag = _newey_west_lag(n)
    best_adj = -np.inf

    while len(chosen) < max_factors:
        remaining = [c for c in candidates.columns if c not in chosen]
        if not remaining:
            break
        threshold = alpha / len(remaining)
        scored = []
        for cand in remaining:
            cols = chosen + [cand]
            design = sm.add_constant(candidates[cols].to_numpy(dtype=float), has_constant="add")
            fit = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
            scored.append((float(fit.rsquared_adj), cand, float(fit.pvalues[-1])))
        scored.sort(key=lambda t: -t[0])
        adj, cand, pval = scored[0]
        if adj <= best_adj:
            log.append(
                f"stopped: best remaining candidate {cand} did not improve adjusted R2 ({adj:.4f} <= {best_adj:.4f})"
            )
            break
        if pval > threshold:
            log.append(
                f"stopped: best remaining candidate {cand} p={pval:.2g} failed the "
                f"Bonferroni gate {threshold:.2g} (0.05 / {len(remaining)} candidates)"
            )
            break
        chosen.append(cand)
        best_adj = adj
        log.append(f"step {len(chosen)}: added {cand} (adj R2 {adj:.4f}, p={pval:.2g} < {threshold:.2g})")
    return chosen, log


def factor_exposure(
    panel: Any,
    weights: Mapping[str, float],
    *,
    db_path: str | os.PathLike[str],
    kind: str = "prices",
    min_days: int = MIN_TRADING_DAYS,
    max_factors: int = DEFAULT_MAX_FACTORS,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    alpha: float = 0.05,
) -> dict:
    """Regress the portfolio against the reference instruments in the live DB.

    The output a broker app never shows: not what you own, but what you are
    *exposed to*. "83% of this book's daily variance tracked the India ETF and
    the dollar" is a fact about 573 days that have already happened.

    Method, stated because on a public page the method is part of the claim:

    * Portfolio series is ``sum(w_i * r_i)`` on the common window, weights held
      constant. A what-if on today's book, not realised P&L.
    * Candidates are the reference instruments covering >= 95% of that window
      with non-degenerate variance. Everything excluded is listed with a reason.
    * Selection is forward stepwise, capped at ``max_factors`` (default 3),
      each entry gated at Bonferroni ``alpha / n_remaining_candidates`` on
      Newey-West (HAC) standard errors.
    * ``variance_share`` per factor is ``beta_i * cov(f_i, r_p) / var(r_p)``;
      the shares sum to R-squared and ``unexplained_share`` is ``1 - R2``.
    * The gate is calibrated, not asserted. Under a circular-shift null that
      preserves the portfolio's entire autocorrelation structure and destroys
      only its alignment with the factors, this procedure selected zero
      factors in 30 of 30 runs against all 89 candidates. It errs toward
      saying nothing. ``test_selection_gate_holds_under_circular_shift_null``
      keeps it that way.
    * **The holdout is the honest part.** Selection is re-run from scratch on
      the first ``1 - holdout_fraction`` of the window and scored on the last
      part, which selection never touched. If ``holdout.r2`` collapses against
      ``adjusted_r2``, the full-window fit is the garden of forking paths and
      should not be shown.

    Where it misleads:

    * Correlated factors split a shared exposure arbitrarily between them.
      ``max_selected_factor_correlation`` is reported, with a warning past 0.8;
      past that, read the betas as one joint exposure, not two separate ones.
    * Betas are realised sensitivities over this window only.
    * A factor quoted in a different currency from the holdings absorbs the
      exchange rate. ``quote_currency`` is on every factor for that reason.
    * R-squared measures co-movement of daily returns, not that the factor
      caused anything.

    Returns a dict; ``sufficient`` is False with a ``reason`` whenever the data
    cannot support the statistic, and in that case no numbers are reported.
    """
    returns = _as_returns(panel, kind)
    weights_norm, weight_exclusions, notes = _normalise_weights(weights, list(returns.columns))
    excluded = list(weight_exclusions)

    if not weights_norm:
        return {
            "sufficient": False,
            "reason": "No holding had both a usable weight and price history.",
            "excluded": [e.__dict__ for e in excluded],
            "notes": notes,
        }

    aligned, window_exclusions = _common_window(returns[list(weights_norm)], min_days)
    excluded.extend(window_exclusions)

    if aligned.shape[1] < 1 or aligned.shape[0] < min_days:
        return {
            "sufficient": False,
            "reason": (
                f"{aligned.shape[1]} holdings share only {aligned.shape[0]} trading days; "
                f"this module reports no regression below {min_days}."
            ),
            "excluded": [e.__dict__ for e in excluded],
            "notes": notes,
        }

    surviving = {k: weights_norm[k] for k in aligned.columns}
    survived_share = sum(surviving.values())
    surviving = {k: v / survived_share for k, v in surviving.items()}
    if survived_share < 0.999999:
        notes.append(
            f"Regression covers the {survived_share:.1%} of the book with usable overlapping "
            "history; the rest is excluded by name below and is not represented in any number here."
        )

    weight_vector = pd.Series(surviving)
    portfolio = (aligned[weight_vector.index] * weight_vector).sum(axis=1)

    reference, meta = load_reference_returns(db_path)
    factors = reference.reindex(portfolio.index)

    candidates: list[str] = []
    for ticker in factors.columns:
        coverage = float(factors[ticker].notna().mean())
        if coverage < MIN_FACTOR_COVERAGE:
            excluded.append(Exclusion(ticker, "reference factor does not cover the window", f"{coverage:.0%} of days"))
            continue
        std = float(factors[ticker].std(skipna=True))
        if not math.isfinite(std) or std < 1e-9:
            excluded.append(Exclusion(ticker, "reference factor has no variation", f"std={std:.3g}"))
            continue
        candidates.append(ticker)

    usable = factors[candidates]
    mask = portfolio.notna() & usable.notna().all(axis=1)
    y_frame = portfolio[mask]
    x_frame = usable.loc[mask]
    days_lost = int(portfolio.notna().sum() - mask.sum())
    if days_lost:
        notes.append(
            f"{days_lost} of the portfolio's {int(portfolio.notna().sum())} trading days are not "
            "used in the regression because at least one reference factor did not trade that day "
            "(differing exchange holidays). Every factor below is measured on the same remaining days."
        )

    if len(y_frame) < min_days or not candidates:
        return {
            "sufficient": False,
            "reason": (
                f"Only {len(y_frame)} trading days overlap between the portfolio and "
                f"{len(candidates)} usable reference factors; below the {min_days}-day floor."
            ),
            "excluded": [e.__dict__ for e in excluded],
            "notes": notes,
        }

    dates = list(y_frame.index)
    window = Window(dates[0], dates[-1], len(dates))
    y = y_frame.to_numpy(dtype=float)

    chosen, log = _forward_select(y, x_frame, max_factors, alpha)

    result: dict[str, Any] = {
        "sufficient": True,
        "window": {"start": window.start.isoformat(), "end": window.end.isoformat(), "n_days": window.n_days},
        "holdings_used": list(aligned.columns),
        "book_share_covered": float(survived_share),
        "candidates_examined": len(candidates),
        "selection": (
            f"forward stepwise, max {max_factors} of {len(candidates)} candidates, "
            f"each entry gated at Bonferroni alpha={alpha}/n_remaining on Newey-West "
            f"(HAC, lag {_newey_west_lag(len(y))}) standard errors"
        ),
        "selection_log": log,
        "factors": [],
        "excluded": [e.__dict__ for e in excluded],
        "notes": notes,
        "warnings": [],
    }

    if not chosen:
        result["r2"] = 0.0
        result["adjusted_r2"] = 0.0
        result["unexplained_share"] = 1.0
        result["holdout"] = None
        result["warnings"].append(
            "No reference factor survived the selection gate: over this window nothing in the "
            "89-instrument reference set explained this portfolio's daily movement."
        )
        return result

    import statsmodels.api as sm

    lag = _newey_west_lag(len(y))
    design = sm.add_constant(x_frame[chosen].to_numpy(dtype=float), has_constant="add")
    fit = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    betas = fit.params[1:]
    var_p = float(np.var(y, ddof=1))

    for i, ticker in enumerate(chosen):
        f = x_frame[ticker].to_numpy(dtype=float)
        cov = float(np.cov(f, y, ddof=1)[0, 1])
        info = meta.loc[ticker]
        share = float(betas[i] * cov / var_p) if var_p > 0 else float("nan")
        result["factors"].append(
            {
                "ticker": ticker,
                "name": str(info["name"]),
                "asset_class": str(info["asset_class"]),
                "quote_currency": str(info["quote_currency"]),
                "beta": float(betas[i]),
                "hac_t_stat": float(fit.tvalues[i + 1]),
                "hac_p_value": float(fit.pvalues[i + 1]),
                "variance_share": share,
                "material": bool(abs(share) >= MIN_MATERIAL_VARIANCE_SHARE),
                "role": "exposure" if share >= 0 else "offset",
            }
        )

    result["r2"] = float(fit.rsquared)
    result["adjusted_r2"] = float(fit.rsquared_adj)
    result["unexplained_share"] = float(1.0 - fit.rsquared)
    result["alpha_daily"] = float(fit.params[0])
    result["alpha_hac_p_value"] = float(fit.pvalues[0])

    if len(chosen) > 1:
        sub = np.array(x_frame[chosen].corr().to_numpy(dtype=float), dtype=float, copy=True)
        max_corr = float(np.abs(sub[np.triu_indices_from(sub, k=1)]).max())
        result["max_selected_factor_correlation"] = max_corr
        if max_corr > 0.8:
            result["warnings"].append(
                f"Selected factors are themselves correlated at {max_corr:.2f}; their individual "
                "betas split one shared exposure between them and should be read jointly."
            )
    else:
        result["max_selected_factor_correlation"] = 0.0

    offsets = [f["ticker"] for f in result["factors"] if f["role"] == "offset"]
    if offsets:
        result["warnings"].append(
            f"{', '.join(offsets)} carries a negative share of variance. That is not a holding "
            "moving against the portfolio; it is a correction term subtracting the part of an "
            "already-selected factor that this portfolio does not share. Do not render it as "
            "an exposure to that instrument."
        )

    immaterial = [f["ticker"] for f in result["factors"] if not f["material"]]
    if immaterial:
        result["warnings"].append(
            f"{', '.join(immaterial)} cleared the significance gate but each accounts for under "
            f"{MIN_MATERIAL_VARIANCE_SHARE:.0%} of daily variance. Statistically detectable over "
            f"{window.n_days} days, too small to describe this portfolio; flagged material=False."
        )

    result["holdout"] = _holdout_check(y_frame, x_frame, max_factors, alpha, holdout_fraction, min_days)
    if result["holdout"] is None:
        result["warnings"].append(
            "Window too short to hold anything back, so the fit above has not been checked "
            "out of sample and the selection could be fitting noise."
        )
    elif result["holdout"]["r2"] < 0.5 * result["adjusted_r2"]:
        result["warnings"].append(
            f"Out-of-sample R2 ({result['holdout']['r2']:.2f}) is less than half the in-sample "
            f"adjusted R2 ({result['adjusted_r2']:.2f}): the full-window fit is not stable and "
            "should not be presented as a description of this portfolio."
        )
    return result


def _holdout_check(
    y_frame: pd.Series,
    x_frame: pd.DataFrame,
    max_factors: int,
    alpha: float,
    holdout_fraction: float,
    min_days: int,
) -> dict | None:
    """Re-run selection on the early part only, score on the untouched tail.

    This is the guard against the forking paths. Selection sees only the train
    slice; the test slice is scored with the train betas, so its R-squared can
    be negative, and if it is, the headline fit was noise.
    """
    n = len(y_frame)
    n_test = int(round(n * holdout_fraction))
    n_train = n - n_test
    if n_test < 40 or n_train < min_days:
        return None

    y_train = y_frame.iloc[:n_train].to_numpy(dtype=float)
    x_train = x_frame.iloc[:n_train]
    y_test = y_frame.iloc[n_train:].to_numpy(dtype=float)
    x_test = x_frame.iloc[n_train:]

    chosen, _ = _forward_select(y_train, x_train, max_factors, alpha)
    if not chosen:
        return {
            "train_window": _win(x_train),
            "test_window": _win(x_test),
            "selected_on_train": [],
            "r2": 0.0,
            "note": "Nothing cleared the selection gate on the training slice.",
        }

    design_train = np.column_stack([np.ones(n_train), x_train[chosen].to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
    design_test = np.column_stack([np.ones(len(y_test)), x_test[chosen].to_numpy(dtype=float)])
    pred = design_test @ coef
    ss_res = float(((y_test - pred) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "train_window": _win(x_train),
        "test_window": _win(x_test),
        "selected_on_train": chosen,
        "r2": float(r2),
        "note": (
            "Factors were re-selected from scratch on the training slice; this R2 is scored on "
            "days the selection never saw. A negative value means the model is worse than the "
            "test slice's own mean."
        ),
    }


def _win(frame: pd.DataFrame) -> str:
    idx = list(frame.index)
    return f"{idx[0].isoformat()}..{idx[-1].isoformat()} (n={len(idx)})"


# --------------------------------------------------------------------------
# Rendering (debug / CLI; the web layer owns real presentation)
# --------------------------------------------------------------------------


def format_overlap_report(clusters: list[Cluster], exposure: dict | None = None) -> str:
    """Plain-text rendering of both analyses. Facts only, no advice.

    This exists so the numbers can be eyeballed in a terminal and pasted into a
    review. It is not the product surface.
    """
    out: list[str] = []
    cs = clusters if isinstance(clusters, ClusterSet) else ClusterSet(clusters)

    if cs.window is not None:
        out.append(f"WINDOW  {cs.window}")
        out.append(f"HOLDINGS MEASURED  {len(cs.holdings_used)}")
        out.append(f"DISTINCT CO-MOVEMENT GROUPS  {cs.n_groups}")
        if cs.median_pair_correlation is not None:
            out.append(f"MEDIAN PAIRWISE CORRELATION  {cs.median_pair_correlation:.2f}")
        if cs.stable_between is not None:
            lo, hi = cs.stable_between
            out.append(f"GROUPING UNCHANGED FOR ANY CUTOFF  {lo:.2f} to {hi:.2f}")
        out.append("")

    if cs:
        out.append("GROUPS THAT MOVED TOGETHER")
        for i, c in enumerate(cs, 1):
            out.append(f"  {i}. {c.statement()}")
    else:
        out.append("GROUPS THAT MOVED TOGETHER: none at this threshold.")
    if cs.singletons:
        out.append(f"  Joined no group: {', '.join(cs.singletons)}")
    if cs.near_duplicates:
        out.append("")
        out.append("NEAR-IDENTICAL PAIRS")
        for a, b, r in cs.near_duplicates:
            out.append(f"  {a} and {b} moved at correlation {r:.3f} over {cs.window}.")
    for note in cs.notes:
        out.append(f"  note: {note}")
    if cs.excluded:
        out.append("")
        out.append("EXCLUDED FROM THE CORRELATION ANALYSIS")
        for e in cs.excluded:
            out.append(f"  {e}")

    if exposure is None:
        return "\n".join(out)

    out.append("")
    out.append("WHAT THE PORTFOLIO TRACKED")
    if not exposure.get("sufficient"):
        out.append(f"  Not computed: {exposure.get('reason')}")
        return "\n".join(out)

    w = exposure["window"]
    out.append(f"  Window {w['start']}..{w['end']} (n={w['n_days']} trading days)")
    out.append(f"  Selection: {exposure['selection']}")
    for line in exposure["selection_log"]:
        out.append(f"    - {line}")
    for f in exposure["factors"]:
        out.append(
            f"  {f['ticker']:<10} {f['name']:<24} beta {f['beta']:+.3f}  "
            f"{f['variance_share']:6.1%} of daily variance  "
            f"[{f['asset_class']}, quoted {f['quote_currency']}, HAC p={f['hac_p_value']:.2g}]"
            + ("" if f["material"] else "  <- immaterial, not a finding")
            + ("" if f["role"] == "exposure" else "  <- offset, not an exposure")
        )
    out.append(f"  R2 {exposure['r2']:.3f}   adjusted R2 {exposure['adjusted_r2']:.3f}")
    out.append(f"  Unexplained by any selected factor: {exposure['unexplained_share']:.1%}")
    h = exposure.get("holdout")
    if h:
        out.append(
            f"  Holdout: selected on {h['train_window']}, scored on {h['test_window']} "
            f"-> out-of-sample R2 {h['r2']:.3f} (train picked {', '.join(h['selected_on_train']) or 'nothing'})"
        )
    for warning in exposure.get("warnings", []):
        out.append(f"  warning: {warning}")
    return "\n".join(out)
