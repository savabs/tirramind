"""TirraMind — the publishable scoreboard for automatic ghost-chain discovery.

WHAT THIS MODULE IS FOR
    An automatic chain finder is a p-hacking machine unless the multiple-testing
    accounting is airtight. With 12 node types, 18 link types and a lag grid you
    can generate tens of thousands of candidates, so thousands will look
    significant by chance. This module is the part that publishes the result, and
    it is built so that the denominator cannot be dropped:

      * ``Scoreboard`` raises unless ``enumerated == untestable + tested``;
      * ``render_scoreboard`` prints that arithmetic above the fold, before any
        survivor;
      * the graveyard (what was tested and died) is a required section, not an
        appendix, and it carries ``nominal_survivors`` — how many chains would
        have been published by someone who skipped the correction.

WHAT IT COMPUTES
    Nothing statistical. Every number on the page is computed upstream by
    ``agent.verify`` (Benjamini-Hochberg, block bootstrap, power) and
    ``agent.mechanism`` (routes, independent time-varying sources). This module
    validates that those numbers are mutually consistent and renders them as
    markdown. The only arithmetic it does itself is the funnel reconciliation,
    the median/quantile summaries, and the minimum detectable effect, which uses
    the SAME normal approximation as ``agent.verify.stats.power_estimate`` so the
    two cannot disagree:

        mde(n, sigma, power=0.8) = sigma * (z_{1-alpha/2} + z_{0.8}) / sqrt(n)

WHERE IT MISLEADS
    * A rendered page is only as honest as the family handed to it. If the caller
      ran hypotheses it never put in ``tested``, this module cannot know, and the
      m it prints will be too small. The reconciliation catches candidates that
      were *enumerated and then lost*; it cannot catch candidates that were never
      enumerated. That is why ``provenance`` carries the grid definition: a
      reader can recount the space by hand.
    * Testability is decided on ``n_events`` and ``n_clusters`` ONLY — never on
      p, effect or sign. That is what makes moving a candidate to UNTESTABLE
      after running it defensible rather than a selection. ``Scoreboard``
      enforces the n/cluster floors on everything in ``tested``, but it has no
      way to prove the caller did not peek; the floors are printed so the rule
      is checkable.
    * ``independent_sources`` is an UPPER bound (see
      ``agent.mechanism.routes.independence``): two ingest pipelines fed by one
      publisher count as two. A chain with 2 independent sources may have 1
      witness.
    * ``p_adjusted`` is not a p-value of the individual chain. It is the smallest
      FDR at which the whole procedure would reject it, and it is not comparable
      across families of different size. Two scoreboards with different
      ``tested`` counts cannot have their adjusted p's compared.
    * ``power`` is a stipulated-effect calculation, not an observed one. A 6%
      power figure means "if the effect were the size we stipulated, we would see
      it 6 times in 100". It does not say the effect is absent.
    * The effect CI is a circular block bootstrap over event returns. On a chain
      whose events sit on 3 distinct dates the CI is decorative, which is exactly
      why ``n_clusters`` is printed beside every effect and why the cluster floor
      exists.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_MIN_CLUSTERS",
    "DEFAULT_MIN_EVENTS",
    "ChainResult",
    "FilterStep",
    "Scoreboard",
    "ScoreboardError",
    "SeriesCheck",
    "UntestableReason",
    "mde_at_power",
    "render_scoreboard",
    "series_check",
    "summarise",
]

# Floors below which a chain is declared UNTESTABLE rather than tested. These are
# the published CFTC study's own thresholds: that study's headline moved by a
# factor of five purely from counting the calendar instead of the events, so a
# cluster floor is not optional. 30 events / 20 clusters is the repo default and
# is printed on the page so a reader can disagree with it explicitly.
DEFAULT_MIN_EVENTS = 30
DEFAULT_MIN_CLUSTERS = 20

# Two-sided normal quantiles, hardcoded so this module imports without scipy and
# matches `agent.verify.stats.power_estimate`, which uses the same approximation.
_Z_ALPHA_2 = 1.959963984540054  # Phi^-1(0.975)
_Z_POWER_80 = 0.8416212335729143  # Phi^-1(0.80)

_TOL = 1e-9


class ScoreboardError(ValueError):
    """A scoreboard whose numbers do not reconcile.

    Raised rather than rendered. A page that cannot account for its own
    candidates is the single failure this module exists to prevent, so it fails
    loudly at construction instead of publishing a smaller, flattering m.
    """


def _z_quantile(p: float) -> float:
    """Inverse standard normal CDF, Acklam's rational approximation.

    Accurate to ~1.15e-9 relative error over (0, 1), which is far below any
    precision this page quotes. Present only so ``mde_at_power`` accepts a power
    other than 0.80 without a scipy dependency.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"_z_quantile needs p in (0, 1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def mde_at_power(n: int, sigma: float, *, power: float = 0.80, alpha: float = 0.05) -> float:
    """Smallest effect this sample could have detected, in the units of ``sigma``.

    WHAT IT COMPUTES
        ``sigma * (z_{1-alpha/2} + z_{power}) / sqrt(n)`` — the inverse of the
        normal-approximation power curve in ``agent.verify.stats.power_estimate``,
        dropping that function's wrong-tail term (which is below 1e-3 at any
        power worth quoting). Feeding this value back into ``power_estimate``
        returns ``power`` to about 1e-3; the test asserts it.

    WHY IT IS ON THE PAGE
        It is the only number that makes a null result readable. "Nothing
        survived" plus "we could not have seen anything smaller than 1.8% per
        5 days" is a bounded, falsifiable statement. "Nothing survived" alone is
        not.

    WHERE IT MISLEADS
        * It uses ``n`` as given. If those n events sit on 16 distinct dates, the
          real detectable effect is larger than this by roughly the square root
          of the clustering factor, and this number is optimistic. Always read it
          next to ``n_clusters``.
        * ``sigma`` is the observed dispersion of the event returns, so it is
          itself estimated from the same small sample.
        * It is two-sided and symmetric. A directional hypothesis would have a
          slightly smaller MDE, so quoting this one is the conservative choice.

    Raises:
        ValueError: n < 1, sigma < 0, or power/alpha outside (0, 1).
    """
    if n < 1:
        raise ValueError(f"mde_at_power needs n >= 1, got {n}")
    if sigma < 0 or not math.isfinite(sigma):
        raise ValueError(f"mde_at_power needs a finite sigma >= 0, got {sigma}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"mde_at_power needs power in (0, 1), got {power}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"mde_at_power needs alpha in (0, 1), got {alpha}")
    z_a = _Z_ALPHA_2 if abs(alpha - 0.05) < _TOL else _z_quantile(1.0 - alpha / 2.0)
    z_b = _Z_POWER_80 if abs(power - 0.80) < _TOL else _z_quantile(power)
    return sigma * (z_a + z_b) / math.sqrt(n)


@dataclass(frozen=True)
class SeriesCheck:
    """Does a series of computed numbers actually vary? (LESSONS F-18.)

    A frozen model was once ranked first in this repo for having a good random
    initialisation, and the untrained HGT's "learned attention" came back as the
    uniform fractions 0.0 / 0.333 / 0.5 / 0.667 / 1.0. Both were caught by
    looking at the spread of the series, so no number reaches the page without
    one of these beside it.

    ``distinct`` counts unique values at full float precision. ``degenerate`` is
    True when a series of 2 or more takes a single value — which is a defect
    claim about the pipeline, not about the market.

    Where it misleads: a varying series is not a correct one. This catches a
    constant, not a bias.
    """

    label: str
    n: int
    distinct: int
    minimum: float | None
    median: float | None
    maximum: float | None

    @property
    def degenerate(self) -> bool:
        """True when 2+ values were computed and all of them are identical."""
        return self.n >= 2 and self.distinct <= 1

    def line(self) -> str:
        """One-line rendering, with a shout when the series does not vary."""
        if self.n == 0:
            return f"{self.label}: no values computed"
        span = f"min {_fmt(self.minimum)} / median {_fmt(self.median)} / max {_fmt(self.maximum)}"
        flag = "  <-- DEGENERATE: does not vary" if self.degenerate else ""
        return f"{self.label}: n={self.n}, {self.distinct} distinct, {span}{flag}"


def series_check(label: str, values: Sequence[float]) -> SeriesCheck:
    """Summarise a computed series so a constant cannot pass for a result.

    Non-finite values are DROPPED and the drop is visible as a gap between ``n``
    here and the length you passed in — this function does not silently pretend
    a NaN was a number, but it also does not raise, because a NaN p-value is a
    real (and reportable) outcome for an ungradeable chain.

    Where it misleads: ``distinct`` is exact float equality. Two p-values that
    differ in the 15th decimal count as two, so a near-degenerate series can
    still report a high distinct count. Read ``minimum``/``maximum`` too.
    """
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not finite:
        return SeriesCheck(label=label, n=0, distinct=0, minimum=None, median=None, maximum=None)
    ordered = sorted(finite)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return SeriesCheck(
        label=label,
        n=len(ordered),
        distinct=len(set(ordered)),
        minimum=ordered[0],
        median=median,
        maximum=ordered[-1],
    )


@dataclass(frozen=True)
class FilterStep:
    """One narrowing of the candidate space, with its own arithmetic.

    ``examined - dropped == kept`` is enforced, so no stage of the funnel can
    lose candidates into a rounding. ``reason`` is the rule applied, in words a
    reader can disagree with.

    Where it misleads: a funnel of steps that each reconcile can still be the
    wrong funnel. These counts prove nothing was lost; they do not prove the
    space was the right one to enumerate.
    """

    name: str
    examined: int
    dropped: int
    reason: str

    def __post_init__(self) -> None:
        if self.examined < 0 or self.dropped < 0:
            raise ScoreboardError(f"FilterStep {self.name!r} has a negative count")
        if self.dropped > self.examined:
            raise ScoreboardError(f"FilterStep {self.name!r} dropped {self.dropped} of {self.examined} examined")

    @property
    def kept(self) -> int:
        """Candidates that survived this step."""
        return self.examined - self.dropped


@dataclass(frozen=True)
class UntestableReason:
    """A bucket of candidates that were never given a p-value, and why.

    Every candidate that leaves the funnel must land in exactly one of these, so
    ``sum(count)`` plus ``len(tested)`` has to equal ``enumerated``. The reason
    strings are the product: "38,904 untestable" is a shrug, "38,904 untestable,
    of which 30,192 had fewer than 30 gradeable events" is a specification.
    """

    reason: str
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ScoreboardError(f"UntestableReason {self.reason!r} has count {self.count}")


@dataclass(frozen=True)
class ChainResult:
    """One chain that was actually tested — survivor or kill, same shape.

    Survivors and kills share this type on purpose. A renderer that needed a
    different object for a survivor would make it possible to build a page with
    survivors and no graveyard.

    Fields:
        chain_id: stable identifier, so a kill can be looked up in a later run.
        name: the chain as a named path, including the lag and the threshold.
        route: the mechanism route's node/link labels, outermost first. Empty
            when no route was computed — which is itself printed, never blanked.
        hops: links traversed by ``route``. 0 means "no mechanism route", NOT
            "direct".
        independent_sources: distinct sources on TIME-VARYING hops only, from
            ``agent.mechanism.routes.independence``. An upper bound.
        lag_steps / lag_label: the horizon, in series steps of the TARGET. On a
            weekly target, "10 steps" is 10 weeks and ``lag_label`` says so.
        n_events / n_clusters: gradeable events, and the distinct event
            timestamps they fall on. The second is the one that matters.
        sigma: dispersion of the event returns, used for the MDE.
        power: from ``agent.verify.stats.power_estimate`` at the study's
            stipulated effect.
        p_uncorrected: the study's cluster-resampled p where available, else the
            i.i.d. one. Which of the two is recorded in ``notes``.
        p_adjusted / rejected: from ``agent.verify.stats.benjamini_hochberg`` over
            the WHOLE tested family.
        effect: ``event_mean - baseline``. This is the quantity of interest.
        event_mean / baseline: the two means the effect is the difference of.
        ci_low / ci_high: the circular block bootstrap CI of the EVENT MEAN, not
            of the effect. `agent.verify.study` bootstraps the event returns
            alone, so this interval does not carry the baseline's own
            uncertainty and must never be read as a CI on ``effect``.
        route_is_scaffold_only: True when the highest-mass route between the
            pair is pure static geography. The displayed route is then the
            highest-mass route carrying at least one time-varying hop, and the
            page says so — a scaffold route is not a mechanism.
        falsifier: what observation would kill this chain. Required, non-empty.
        notes: anything that qualifies the row, e.g. undatable links.

    Where it misleads: ``effect`` is a mean log return over the event set, not a
    tradeable return. It carries no cost, no slippage and no capacity, and this
    module never converts it into one.
    """

    chain_id: str
    name: str
    route: tuple[str, ...]
    hops: int
    independent_sources: int
    evidence_sources: tuple[str, ...]
    lag_steps: int
    lag_label: str
    market: str
    mechanism: str
    n_events: int
    n_clusters: int
    sigma: float
    power: float
    p_uncorrected: float
    p_adjusted: float
    rejected: bool
    effect: float
    event_mean: float
    baseline: float
    ci_low: float
    ci_high: float
    falsifier: str
    route_is_scaffold_only: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.chain_id or not self.name:
            raise ScoreboardError("ChainResult needs a chain_id and a name")
        if not self.falsifier.strip():
            raise ScoreboardError(
                f"chain {self.chain_id!r} has no falsifier. An unfalsifiable chain is not publishable by this module."
            )
        for label, p in (("p_uncorrected", self.p_uncorrected), ("p_adjusted", self.p_adjusted)):
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                raise ScoreboardError(f"chain {self.chain_id!r} has {label}={p}, not a p-value")
        if self.p_adjusted < self.p_uncorrected - 1e-12:
            raise ScoreboardError(
                f"chain {self.chain_id!r} has an adjusted p ({self.p_adjusted}) below its "
                f"uncorrected p ({self.p_uncorrected}); Benjamini-Hochberg never shrinks a p."
            )
        if not math.isfinite(self.power) or not 0.0 <= self.power <= 1.0:
            raise ScoreboardError(f"chain {self.chain_id!r} has power={self.power}")
        if self.n_events < 0 or self.n_clusters < 0:
            raise ScoreboardError(f"chain {self.chain_id!r} has a negative sample count")
        if self.n_clusters > self.n_events:
            raise ScoreboardError(
                f"chain {self.chain_id!r} claims {self.n_clusters} distinct event dates from {self.n_events} events"
            )
        if self.hops < 0 or self.independent_sources < 0:
            raise ScoreboardError(f"chain {self.chain_id!r} has a negative hop/source count")

    @property
    def clustering_factor(self) -> float | None:
        """Events per distinct event date. 1.0 means one event per date.

        In the published CFTC study four weeks had five correlated contracts
        firing together, and re-running with a cluster bootstrap moved the
        adjusted p from 0.102 to 0.294. A factor of 81 here means 81 correlated
        observations share one date, and the nominal n is fiction.
        """
        if self.n_clusters <= 0:
            return None
        return self.n_events / self.n_clusters

    @property
    def heavily_clustered(self) -> bool:
        """More than 5 events per distinct date — the CFTC study's own failure ratio."""
        cf = self.clustering_factor
        return cf is not None and cf > 5.0

    @property
    def nominal(self) -> bool:
        """Would have looked significant at 0.05 WITHOUT the correction."""
        return self.p_uncorrected <= 0.05

    @property
    def mde_80(self) -> float | None:
        """Effect this chain could have detected 80% of the time, or None."""
        if self.n_events < 1 or not math.isfinite(self.sigma) or self.sigma <= 0:
            return None
        return mde_at_power(self.n_events, self.sigma)

    @property
    def underpowered(self) -> bool:
        """Power below 0.80 — i.e. a non-rejection here is not evidence of absence."""
        return self.power < 0.80


@dataclass(frozen=True)
class Scoreboard:
    """The whole run: what was enumerated, what could not be tested, what died.

    THE INVARIANT
        ``enumerated == untestable_total + len(tested)``. Construction fails
        otherwise. This is the only defence against the specific lie this product
        cannot survive — reporting survivors without the denominator that
        produced them.

    THE SECOND INVARIANT
        every chain in ``tested`` clears ``min_events`` and ``min_clusters``.
        Testability is a data rule, so a chain below the floors belongs in
        ``untestable`` whether or not its p looked good. If the caller filtered
        on outcome, this will not catch it — but the floors are printed, so the
        rule is auditable.

    Where it misleads: ``enumerated`` is the caller's count of the grid it built.
    A grid that was quietly pruned before enumeration would reconcile perfectly
    and still be a smaller family than the analyst actually searched. That is why
    ``grid`` is a required field and is printed verbatim.
    """

    title: str
    generated_at: str
    db_path: str
    as_of_label: str
    alpha: float
    min_events: int
    min_clusters: int
    enumerated: int
    filters: tuple[FilterStep, ...]
    untestable: tuple[UntestableReason, ...]
    tested: tuple[ChainResult, ...]
    grid: Mapping[str, str]
    caveats: tuple[str, ...]
    provenance: Mapping[str, str] = field(default_factory=dict)
    checks: tuple[SeriesCheck, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ScoreboardError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.enumerated < 0:
            raise ScoreboardError(f"enumerated must be >= 0, got {self.enumerated}")
        total = self.untestable_total + len(self.tested)
        if total != self.enumerated:
            raise ScoreboardError(
                f"the funnel does not reconcile: {self.enumerated} chains enumerated but "
                f"{self.untestable_total} untestable + {len(self.tested)} tested = {total}. "
                f"{abs(self.enumerated - total)} chains are unaccounted for, and a scoreboard "
                "that cannot say where every candidate went is exactly the artefact this "
                "module refuses to render."
            )
        for c in self.tested:
            if c.n_events < self.min_events or c.n_clusters < self.min_clusters:
                raise ScoreboardError(
                    f"chain {c.chain_id!r} is in `tested` with n={c.n_events} events on "
                    f"{c.n_clusters} clusters, below the declared floors "
                    f"({self.min_events}/{self.min_clusters}). It belongs in `untestable`; "
                    "testing it and then reporting the floors is a lie about the family."
                )
            if c.rejected and c.p_adjusted > self.alpha * (1.0 + 1e-9):
                raise ScoreboardError(
                    f"chain {c.chain_id!r} is marked rejected with an adjusted p of "
                    f"{c.p_adjusted} above alpha={self.alpha}"
                )
        ids = [c.chain_id for c in self.tested]
        if len(set(ids)) != len(ids):
            raise ScoreboardError("duplicate chain_id in `tested`; the family would double-count")
        p_check = series_check("p (uncorrected)", [c.p_uncorrected for c in self.tested])
        if p_check.degenerate and self.survivors:
            raise ScoreboardError(
                f"every one of {p_check.n} tested chains returned the same p-value "
                f"({p_check.minimum}) and {len(self.survivors)} are reported as survivors. "
                "A constant test statistic with discoveries is a broken pipeline, not a "
                "finding (LESSONS F-18)."
            )

    @property
    def untestable_total(self) -> int:
        """Candidates that never received a p-value."""
        return sum(u.count for u in self.untestable)

    @property
    def survivors(self) -> tuple[ChainResult, ...]:
        """Tested chains rejected by Benjamini-Hochberg, strongest first."""
        return tuple(sorted((c for c in self.tested if c.rejected), key=lambda c: c.p_adjusted))

    @property
    def kills(self) -> tuple[ChainResult, ...]:
        """Tested chains that did NOT survive, nearest miss first."""
        return tuple(sorted((c for c in self.tested if not c.rejected), key=lambda c: c.p_uncorrected))

    @property
    def nominal_survivors(self) -> int:
        """How many chains would have been published without the correction.

        The gap between this and ``len(survivors)`` is the entire argument for
        the correction, measured on this run rather than asserted.
        """
        return sum(1 for c in self.tested if c.nominal)

    @property
    def median_power(self) -> float | None:
        """Median power across the tested family, or None if nothing was tested."""
        c = series_check("power", [x.power for x in self.tested])
        return c.median

    @property
    def median_mde(self) -> float | None:
        """Median smallest-detectable effect across the tested family."""
        vals = [c.mde_80 for c in self.tested if c.mde_80 is not None]
        return series_check("mde", vals).median

    @property
    def n_adequately_powered(self) -> int:
        """Tested chains with power >= 0.80 — where a null means something."""
        return sum(1 for c in self.tested if not c.underpowered)

    @property
    def distinct_survivor_mechanisms(self) -> int:
        """How many genuinely different mechanisms the survivors represent.

        Two survivors differing only in lag, or in which point of one yield curve
        fired, are one mechanism measured twice. Reporting the row count as a
        discovery count is the second-most-tempting lie available here, after
        dropping the denominator.
        """
        return len({c.mechanism for c in self.survivors})

    @property
    def distinct_tested_mechanisms(self) -> int:
        """Distinct mechanisms in the tested family, for the same reason."""
        return len({c.mechanism for c in self.tested})

    @property
    def markets(self) -> tuple[str, ...]:
        """Distinct target markets represented in the tested family."""
        return tuple(sorted({c.market for c in self.tested}))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _fmt(x: float | None, digits: int = 4) -> str:
    """Render a number, or ``n/a`` for a missing one. Never returns an empty cell."""
    if x is None:
        return "n/a"
    if isinstance(x, float) and not math.isfinite(x):
        return "not computed"
    return f"{x:.{digits}g}"


def _fmt_p(x: float) -> str:
    """Render a p-value, refusing to print a bare 0.

    A bootstrap p of exactly 0.0 means "no resample reached the observed
    statistic", i.e. p < 1/n_resamples. Printing "0" invites a reader to treat
    it as zero probability, which is the one thing it is not.
    """
    if not math.isfinite(x):
        return "not computed"
    if x == 0.0:
        return "0 (no bootstrap resample reached the observed value; read as < 1/n_resamples)"
    return f"{x:.4g}"


def _pct(x: float | None, digits: int = 1) -> str:
    """Render a fraction as a percentage, or ``n/a``."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def _bp(x: float | None) -> str:
    """Render a log return as a percentage with an explicit sign."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x * 100:+.3f}%"


def _md(text: str) -> str:
    """Escape the pipe, which is the only character that can break a table row."""
    return text.replace("|", "\\|")


def _expected_by_chance(m: int, alpha: float = 0.05) -> str:
    """How many of m tests clear ``alpha`` under a complete null, in words.

    Rendered with a decimal below ten so a family of 8 reads "0.4 of them", not
    "roughly 0", which would silently turn the argument for the correction into an
    argument against it.
    """
    expected = m * alpha
    return f"{expected:,.1f}" if expected < 10 else f"about {expected:,.0f}"


def _thousands(n: int) -> str:
    return f"{n:,}"


def _headline_block(board: Scoreboard) -> list[str]:
    """The denominator, above the fold, in a fixed-width block.

    Deliberately not a markdown table: this block is meant to survive being
    screenshotted, pasted into a chat, or read in a terminal, and it is the one
    part of the page a reader is guaranteed to see.
    """
    width = max(
        len(_thousands(board.enumerated)),
        len(_thousands(board.untestable_total)),
        len(_thousands(len(board.tested))),
        len(_thousands(len(board.survivors))),
        6,
    )

    def row(label: str, value: str, note: str = "") -> str:
        return f"    {label:<22}{value:>{width}}{('   ' + note) if note else ''}"

    lines = ["```"]
    lines.append(row("CHAINS ENUMERATED", _thousands(board.enumerated), "<- the denominator"))
    lines.append(
        row(
            "UNTESTABLE",
            _thousands(board.untestable_total),
            f"(no gradeable pair at all, or n < {board.min_events} events / "
            f"< {board.min_clusters} distinct dates — itemised below)",
        )
    )
    lines.append(row("TESTED", _thousands(len(board.tested)), "<- the BH family size, m"))
    lines.append(row(f"SURVIVED BH @ {board.alpha:g}", _thousands(len(board.survivors))))
    lines.append(row("MEDIAN POWER", _pct(board.median_power)))
    lines.append("```")
    return lines


def _funnel_section(board: Scoreboard) -> list[str]:
    out = ["## How the space narrowed", ""]
    if board.filters:
        out += [
            "| # | filter | examined | dropped | kept | rule |",
            "|---|---|---:|---:|---:|---|",
        ]
        for i, f in enumerate(board.filters, start=1):
            out.append(
                f"| {i} | {_md(f.name)} | {_thousands(f.examined)} | {_thousands(f.dropped)} "
                f"| {_thousands(f.kept)} | {_md(f.reason)} |"
            )
        out.append("")
        out.append(
            "Each row reconciles: `examined - dropped == kept`. No candidate leaves this "
            "funnel without being counted somewhere below."
        )
    else:
        out.append("No funnel was recorded for this run.")
    out += ["", "### Why chains were untestable", ""]
    if board.untestable:
        out += ["| candidates | reason |", "|---:|---|"]
        for u in sorted(board.untestable, key=lambda u: -u.count):
            out.append(f"| {_thousands(u.count)} | {_md(u.reason)} |")
        out.append(f"| **{_thousands(board.untestable_total)}** | **total untestable** |")
    else:
        out.append("Every enumerated chain was testable.")
    out += [
        "",
        f"Untestable means *we did not compute a p-value*, decided on `n_events` and "
        f"`n_clusters` alone — never on the effect, its sign or its p. Those {_thousands(board.untestable_total)} "
        "chains are outside the Benjamini-Hochberg family because they were never tested, "
        "not because they disappointed.",
        "",
    ]
    return out


def _chain_header(c: ChainResult) -> list[str]:
    route = " -> ".join(c.route) if c.route else "(no mechanism route computed)"
    return [
        f"#### {_md(c.name)}",
        "",
        f"- **path** `{_md(route)}`",
        f"- **hops** {c.hops}" + ("  (no route computed; this is not a claim of directness)" if c.hops == 0 else ""),
        *(
            [
                "- **note on the path** the highest-mass route between this pair is pure static "
                "geography, which routes a signal without witnessing anything. The path above is "
                "the highest-mass route that carries at least one time-varying hop."
            ]
            if c.route_is_scaffold_only
            else []
        ),
        f"- **independent time-varying sources** {c.independent_sources}"
        + (f"  ({', '.join(sorted(c.evidence_sources))})" if c.evidence_sources else "")
        + (
            "  — NO route between this pair carries a time-varying hop, so nothing in the "
            "graph independently witnesses this chain. The only evidence is the event stream "
            "that defined it, and a static link is not a mechanism."
            if c.independent_sources == 0
            else "  — an upper bound; two pipelines fed by one publisher count as two"
        ),
        f"- **lag** {c.lag_steps} {_md(c.lag_label)}",
        f"- **market** {_md(c.market)}",
    ]


def _chain_stats(c: ChainResult, alpha: float) -> list[str]:
    cf = c.clustering_factor
    cluster_note = (
        f"  — {cf:.1f} events per date; HEAVILY CLUSTERED, so the nominal n overstates the "
        "independent information by roughly that factor"
        if c.heavily_clustered
        else (f"  — {cf:.1f} events per date" if cf is not None else "")
    )
    return [
        f"- **n** {_thousands(c.n_events)} gradeable events on {_thousands(c.n_clusters)} distinct dates{cluster_note}",
        f"- **power** {_pct(c.power)}"
        + ("  — UNDERPOWERED: below 80%, so a null here would not have meant absence" if c.underpowered else ""),
        f"- **smallest effect detectable at 80% power** {_bp(c.mde_80)} (ignoring clustering, so optimistic)",
        f"- **p** {_fmt_p(c.p_uncorrected)} uncorrected, {_fmt_p(c.p_adjusted)} BH-adjusted "
        f"(alpha = {alpha:g}, family m = see denominator)",
        f"- **effect** {_bp(c.effect)} = an event mean of {_bp(c.event_mean)} minus a "
        f"{_bp(c.baseline)} population baseline",
        f"- **95% CI of the event mean** [{_bp(c.ci_low)}, {_bp(c.ci_high)}] "
        "(circular block bootstrap). This interval is on the event mean, NOT on the effect: "
        "the baseline's own uncertainty is not in it.",
        f"- **what would falsify it** {_md(c.falsifier)}",
    ]


def _survivor_section(board: Scoreboard) -> list[str]:
    survivors = board.survivors
    if not survivors:
        return _null_result_section(board)
    n_mech = board.distinct_survivor_mechanisms
    out = [
        f"## Survivors — {len(survivors)} rows, {n_mech} distinct "
        f"mechanism{'' if n_mech == 1 else 's'}, of {_thousands(len(board.tested))} tested",
        "",
        f"These cleared Benjamini-Hochberg at alpha = {board.alpha:g} over the whole tested "
        f"family of {_thousands(len(board.tested))}. Read each one's `n`, cluster count and "
        "power before its effect.",
        "",
    ]
    if len(survivors) > n_mech:
        out += [
            f"**{len(survivors)} surviving rows are {n_mech} mechanisms.** The rest are the same "
            "event stream against the same target at another lag, or another point on the same "
            "curve. Count mechanisms, not rows: "
            + "; ".join(
                f"`{_md(k)}` x{v}"
                for k, v in sorted(
                    {
                        m: sum(1 for c in survivors if c.mechanism == m) for m in {c.mechanism for c in survivors}
                    }.items(),
                    key=lambda kv: -kv[1],
                )
            ),
            "",
        ]
    for c in survivors:
        out += _chain_header(c)
        out += _chain_stats(c, board.alpha)
        if c.notes:
            out.append("- **notes** " + "; ".join(_md(n) for n in c.notes))
        out.append("")
    weak = [c for c in survivors if c.underpowered]
    if weak:
        out += [
            f"**{len(weak)} of {len(survivors)} survivors are underpowered** "
            f"(power < 80%). At low power, a survivor is more often a chance excursion "
            "than a small real effect: the effects that reach significance are the ones "
            "that happened to be large in this sample. Treat these as candidates for "
            "out-of-sample testing, not as findings.",
            "",
        ]
    saturated = [c for c in survivors if c.p_uncorrected == 0.0]
    if saturated:
        out += [
            f"**{len(saturated)} of {len(survivors)} survivors have a bootstrap p of exactly 0.** "
            "No resample reached the observed statistic, so the p is bounded only by the number "
            "of resamples. Such a chain is rejected at ANY alpha, which means the correction "
            "cannot rank it against a chain whose p is 1/n_resamples, and lowering alpha will "
            "never remove it. Treat the ordering of these rows as undetermined, and read their "
            "cluster counts instead.",
            "",
        ]
    clustered = [c for c in survivors if c.heavily_clustered]
    if clustered:
        out += [
            f"**{len(clustered)} of {len(survivors)} survivors are heavily clustered** (more "
            "than 5 gradeable events per distinct event date). Correlated observations sharing "
            "a date are not independent draws: in this repo's own CFTC study, switching to a "
            "cluster bootstrap over dates moved an adjusted p from 0.102 to 0.294. The p above "
            "is already cluster-resampled where the study could compute it, but the nominal n "
            "still overstates the information.",
            "",
        ]
    thin = [c for c in survivors if c.n_clusters < 30]
    if thin:
        out += [
            f"**{len(thin)} of {len(survivors)} survivors rest on fewer than 30 distinct "
            "event dates.** A bootstrap over that few clusters has a coarse support; the CI "
            "above is wider in reality than it prints.",
            "",
        ]
    return out


def _null_result_section(board: Scoreboard) -> list[str]:
    """The zero-survivor page. A finding, written as one.

    This is the branch that has to read well, because it is the likely one: every
    headline this project has produced died when measured properly. A null with a
    power analysis beside it is a bounded claim and is publishable; a null on its
    own is just an absence.
    """
    m = len(board.tested)
    out = ["## Result: no chain survived correction", ""]
    if m == 0:
        out += [
            f"Of {_thousands(board.enumerated)} chains enumerated, **none was testable**: "
            f"every one fell below the floors of {board.min_events} gradeable events and "
            f"{board.min_clusters} independent clusters. No p-value was computed, so there is "
            "nothing to correct and nothing to report as a discovery.",
            "",
            "That is a statement about this graph's density, not about the markets. The "
            "reasons are itemised above; each is a specific, addressable data gap.",
            "",
        ]
        return out
    out += [
        f"We enumerated {_thousands(board.enumerated)} chains, tested {_thousands(m)} of them "
        f"(covering {board.distinct_tested_mechanisms} distinct event-stream/target "
        f"mechanisms), and **{_thousands(len(board.survivors))} survived** "
        f"Benjamini-Hochberg at alpha = {board.alpha:g}.",
        "",
        f"{_thousands(board.nominal_survivors)} of the {_thousands(m)} tested chains had an "
        f"uncorrected p at or below 0.05. Publishing those would have been the entire "
        f"finding — and with a family of {_thousands(m)}, "
        f"{_expected_by_chance(m)} are expected to clear 0.05 by chance alone. That is the "
        "whole reason the correction is applied, and the reason this page leads with m.",
        "",
        "### What we could have detected",
        "",
    ]
    mm = board.median_mde
    out += [
        f"- median power across the tested family: **{_pct(board.median_power)}**",
        f"- tested chains with power at or above 80%: **{board.n_adequately_powered} of {_thousands(m)}**",
        f"- median smallest effect detectable at 80% power: **{_bp(mm)}** per holding period",
        "",
        "Read that last number as the boundary of the claim. An effect smaller than it "
        "would have been missed by this design most of the time, so this null rules out "
        f"effects of roughly {_bp(mm)} and larger, and says nothing about anything smaller. "
        "It is also optimistic: it uses `n_events`, and the events cluster on far fewer "
        "distinct dates.",
        "",
    ]
    if board.n_adequately_powered == 0:
        out += [
            "**No tested chain reached 80% power.** A family in which nothing is adequately "
            "powered cannot produce a credible discovery *or* a credible refutation. The "
            "honest reading is that this graph does not yet carry enough independent dated "
            "observations to settle these questions, and the fix is more distinct event "
            "dates, not more candidates.",
            "",
        ]
    return out


def _graveyard_section(board: Scoreboard, max_rows: int = 40) -> list[str]:
    kills = board.kills
    out = [
        f"## The graveyard — {_thousands(len(kills))} tested and killed",
        "",
        "Anyone can produce survivors. Almost nobody publishes the list of things they "
        "tested that did not work, which is the list that tells you whether to believe the "
        "survivors.",
        "",
    ]
    if not kills:
        out += ["Nothing was tested and killed in this run.", ""]
        return out
    nominal_kills = [c for c in kills if c.nominal]
    out += [
        (
            f"**{_thousands(len(nominal_kills))} of these {_thousands(len(kills))} killed chains "
            "had an uncorrected p at or below 0.05.** Those are the ones that would have looked "
            "like discoveries to a run that did not correct for the size of the family."
            if nominal_kills
            else f"**None of these {_thousands(len(kills))} killed chains reached even an "
            "uncorrected 0.05.** They did not fail the correction; they failed outright."
        ),
        "",
        "| chain | lag | n | clusters | ev/date | power | p | BH p | effect | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in kills[:max_rows]:
        verdict = "nominal only" if c.nominal else "not supported"
        out.append(
            f"| {_md(c.name)} | {c.lag_steps} | {_thousands(c.n_events)} | "
            f"{_thousands(c.n_clusters)} | {_fmt(c.clustering_factor, 3)} | {_pct(c.power)} | "
            f"{_fmt(c.p_uncorrected)} | {_fmt(c.p_adjusted)} | {_bp(c.effect)} | {verdict} |"
        )
    if len(kills) > max_rows:
        out.append(
            f"| ... | | | | | | | | | **{_thousands(len(kills) - max_rows)} further killed "
            "chains not shown, ordered by uncorrected p** |"
        )
    out.append("")
    return out


def _checks_section(board: Scoreboard) -> list[str]:
    out = ["## Degeneracy checks (LESSONS F-18)", ""]
    checks = list(board.checks)
    if not checks:
        checks = [
            series_check("p (uncorrected)", [c.p_uncorrected for c in board.tested]),
            series_check("power", [c.power for c in board.tested]),
            series_check("effect", [c.effect for c in board.tested]),
        ]
    out.append(
        "Before any number here was called good, its full series was printed and checked "
        "for variation. A frozen model was once ranked first in this repo for having a "
        "lucky random initialisation, and the previous discovery engine's \"learned "
        'attention" was a set of uniform 1/k fractions.'
    )
    out.append("")
    out.append("```")
    for c in checks:
        out.append(c.line())
    out.append("```")
    out.append("")
    degenerate = [c for c in checks if c.degenerate]
    if degenerate:
        out += [
            "**"
            + f"{len(degenerate)} of these series do not vary"
            + ".** That is a defect claim about this pipeline, not a result about markets. "
            "Do not act on any figure derived from a constant series.",
            "",
        ]
    return out


def _limits_section(board: Scoreboard) -> list[str]:
    out = ["## Where these claims stop applying", ""]
    markets = board.markets
    if len(markets) == 1:
        out.append(
            f"- **One market.** Every tested chain targets {_md(markets[0])}. Nothing here "
            "generalises to another asset class, and the family size would change if it did."
        )
    elif markets:
        out.append(
            f"- **{len(markets)} target markets** ({', '.join(_md(m) for m in markets)}). "
            "Results are not pooled across them and a chain's effect is only about its own "
            "target."
        )
    out += [
        "- **One window.** "
        + (
            "No as-of bound was applied: the run sees the graph and every series as they "
            "stand today, so it is one span ending at the most recent observation. "
            if board.as_of_label.strip().lower().startswith("none")
            else f"The run covers a single historical span, bounded at {_md(board.as_of_label)}. "
        )
        + "A rolling re-run is the only thing that would show whether any of this is stable; "
        "this repo's own history is that a single-window headline became a 45% beat rate "
        "when measured rolling.",
        "- **No cost model.** Effects are mean log returns over event sets. No spread, no "
        "slippage, no capacity, no borrow. They are not returns you could have taken.",
        "- **Correction within this family only.** BH here controls the false-discovery rate "
        "across the chains in *this* run. It does not account for previous runs on the same "
        "data, and it cannot: earlier looks at the same graph are a multiplicity this page "
        "does not price.",
        "- **The graph is the limit.** A chain that does not exist as a link in the entity "
        "graph was never a candidate. Absence here is absence from the graph first, and from "
        "the world only maybe.",
        "- **Upper-bound independence.** Source counts come from ingest-pipeline labels. Two "
        "pipelines reading one publisher count as two witnesses.",
        "- **Correlated candidates.** Many enumerated chains are near-duplicates of each other "
        "(sixteen points on one yield curve; the same threshold at adjacent lags). "
        "Benjamini-Hochberg remains valid under positive dependence, but the family is smaller "
        "in effective terms than m suggests, so the correction is doing less work than the "
        "count implies.",
    ]
    for extra in board.caveats:
        out.append(f"- {_md(extra)}")
    out.append("")
    return out


def _provenance_section(board: Scoreboard) -> list[str]:
    out = ["## Provenance", "", "```"]
    out.append(f"generated_at   {board.generated_at}")
    out.append(f"database       {board.db_path}  (opened mode=ro)")
    out.append(f"as_of          {board.as_of_label}")
    out.append(f"alpha          {board.alpha:g}")
    out.append(f"testability    n_events >= {board.min_events} and n_clusters >= {board.min_clusters}")
    for k, v in board.grid.items():
        out.append(f"grid.{k:<9} {v}")
    for k, v in board.provenance.items():
        out.append(f"{k:<14} {v}")
    out.append("```")
    out.append("")
    out.append(
        "The grid is printed in full so the denominator can be recomputed by hand. "
        "A reader who multiplies it out and does not get the enumerated count should "
        "distrust this page."
    )
    out.append("")
    return out


def render_scoreboard(board: Scoreboard) -> str:
    """Render a run as publishable markdown, denominator first.

    WHAT IT COMPUTES
        Ordering and formatting only. Every statistic is taken from ``board``; the
        only derived figures are the medians, the nominal-significance count and
        the minimum detectable effect.

    THE ORDER IS THE ARGUMENT
        1. the denominator block — enumerated / untestable / tested / survived /
           median power;
        2. how the space narrowed, with each filter's counts;
        3. survivors, or (if none) the null result written as a bounded claim
           with its power analysis;
        4. the graveyard, including how many chains would have been published
           without the correction;
        5. degeneracy checks;
        6. where the claims stop applying;
        7. provenance, including the grid, so the denominator is recomputable.

        Sections 1, 2, 4, 6 and 7 are emitted unconditionally. There is no
        argument and no flag that produces a page of survivors with no
        denominator and no graveyard.

    WHERE IT MISLEADS
        * It renders what it is given. A caller that under-reports ``enumerated``
          gets a page that reconciles and is still wrong; the reconciliation is a
          consistency check, not an audit of the enumeration.
        * The graveyard table truncates at 40 rows by uncorrected p. The count is
          exact and stated; the list is not complete.
        * Markdown tables round. The full precision is in the objects, not here.

    Returns:
        A markdown string. No file is written, nothing is uploaded, and no state
        is mutated — publishing is the caller's decision.
    """
    if not isinstance(board, Scoreboard):
        raise TypeError(f"render_scoreboard expects a Scoreboard, got {type(board).__name__}")

    survived = len(board.survivors)
    m = len(board.tested)
    if m == 0:
        verdict = f"{_thousands(board.enumerated)} chains enumerated, none testable"
    elif survived == 0:
        verdict = f"{_thousands(board.enumerated)} enumerated, {_thousands(m)} tested, 0 survived correction"
    else:
        verdict = f"{_thousands(board.enumerated)} enumerated, {_thousands(m)} tested, {survived} survived correction"

    lines = [f"# {board.title}", "", f"**{verdict}.**", ""]
    lines += _headline_block(board)
    lines += [
        "",
        "Every number below is conditioned on that block. A chain count without its "
        "denominator is not a result, and this page is generated so that the two cannot "
        "be separated.",
        "",
    ]
    lines += _funnel_section(board)
    lines += _survivor_section(board)
    lines += _graveyard_section(board)
    lines += _checks_section(board)
    lines += _limits_section(board)
    lines += _provenance_section(board)
    return "\n".join(lines).rstrip() + "\n"


def summarise(board: Scoreboard) -> str:
    """One-screen terminal summary of a run. Same numbers, no markdown.

    Where it misleads: it omits the graveyard and the limits section, so it is a
    progress report and is not publishable on its own.
    """
    out = [
        f"{board.title}",
        f"  enumerated   {_thousands(board.enumerated)}",
        f"  untestable   {_thousands(board.untestable_total)}",
        f"  tested       {_thousands(len(board.tested))}",
        f"  survived     {len(board.survivors)} at BH alpha={board.alpha:g}",
        f"  nominal      {board.nominal_survivors} would have passed an uncorrected 0.05",
        f"  median power {_pct(board.median_power)}",
        f"  median MDE   {_bp(board.median_mde)} at 80% power",
    ]
    for u in sorted(board.untestable, key=lambda u: -u.count):
        out.append(f"    untestable: {_thousands(u.count):>9}  {u.reason}")
    return "\n".join(out)
