"""
TirraMind — verification report: the statistical verdict joined to the mechanism.

WHAT THIS BUILDS

The sellable object. A customer states a hypothesis ("when X happens, Y moves
within N days") and gets back one page carrying two independently computed
halves:

  * the STATISTICAL verdict — did the effect survive multiple-testing
    correction, at what power, on how many genuinely independent observations
    (from `agent/verify/study.py`, whose reference implementation is
    `scripts/cftc_event_study.py`);
  * the STRUCTURAL mechanism — which routes through the entity graph connect
    cause to effect, how much of the graph's mass flows along them, and how many
    genuinely independent TIME-VARYING sources witness them (from
    `agent/mechanism/`).

An LLM can write a plausible reason. It cannot compute one. Everything below is
computed or is printed as "not computed".

THE THREE RULES THIS MODULE EXISTS TO ENFORCE

1. NO SILENT OMISSION. Every line of the report is always present. A line whose
   inputs are missing renders as "not computed" together with the reason. A
   report with no graph therefore has the same nine lines as one with a graph —
   four of them saying they were not computed. A missing line and a negative
   finding must never look alike, because a reader skims for absence.

2. NO PLAUSIBLE DEFAULTS. This module never substitutes alpha=0.05, never
   assumes a unit for an effect size, never multiplies a return by 100 to make
   it read as a percentage, and never invents a distinct-period count from an
   event count. If `study_result` does not carry a number, the report says so.

3. THE INDEPENDENCE FIGURE CARRIES ITS SEARCH WIDTH. Measured on the live graph,
   Russia -> WTI Crude Oil has ONE independent time-varying source (gdelt) among
   its top k=10 routes and THREE at k=400. Both are true. Quoting either without
   the k that produced it is a lie by omission, so `build_report` runs the route
   search at every width in `route_widths` and the EVIDENCE line names all of
   them.

WHERE THIS MISLEADS

  * It is a renderer, not a referee. It re-derives nothing: if `study_result`
    carries a wrong p-value, this module formats a wrong p-value faithfully. The
    one arithmetic check it does perform is the `significant` vs
    (`p_adj` <= `alpha`) consistency check, and a disagreement is reported as
    CONTRADICTORY rather than resolved in either direction.
  * The mechanism half and the statistical half are computed from different
    data. The graph explains how Russia COULD move WTI; the event study measures
    whether a CFTC positioning anomaly DID precede a return. A strong mechanism
    beside a null verdict is not a contradiction and this report does not
    present it as one.
  * PageRank mass measures routing, not evidence. The STRENGTH line is a
    structural fact and is deliberately printed below the VERDICT, never as a
    substitute for it. The single highest-mass Russia -> WTI route is a direct
    `produced_in` edge — static geography, zero evidential weight — which is why
    the CAVEAT line exists and is derived, not optional.
  * "not detectable" is not "no effect". At ~10% power those are very different
    statements and only the first is supported; the POWER line is mandatory for
    exactly this reason and renders "not computed" rather than being dropped
    when the study did not estimate it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# The mechanism layer is imported lazily inside `build_report`: importing
# `agent.mechanism` transitively pulls in torch (~5s) and a statistics-only
# report must not pay that cost. See agent/mechanism/__init__.py.

__all__ = [
    "LINE_ORDER",
    "MECHANISM_LABELS",
    "ReportLine",
    "VerificationReport",
    "build_report",
    "render_markdown",
]

#: The nine lines of the product, in order. Every report has all nine.
LINE_ORDER: tuple[str, ...] = (
    "VERDICT",
    "POWER",
    "SAMPLE",
    "STRENGTH",
    "ROUTES",
    "EVIDENCE",
    "SCAFFOLD",
    "CAVEAT",
    "UPGRADE PATH",
)

#: The subset of `LINE_ORDER` that needs a graph. Without one these render
#: "not computed", which is why the statistical verdict alone is shippable.
MECHANISM_LABELS: frozenset[str] = frozenset({"STRENGTH", "ROUTES", "EVIDENCE", "SCAFFOLD"})

#: Default route-search widths. 10 is the mechanism layer's own default; 400 is
#: the width at which longer prediction-market routes enter on the live graph
#: and the measured source count changes from 1 to 3. Both are reported.
DEFAULT_ROUTE_WIDTHS: tuple[int, ...] = (10, 400)

_NOT_COMPUTED = "not computed"

# Field-name aliases. `scripts/cftc_event_study.py` row dicts are consumable
# directly; `agent/verify/study.py` is being written concurrently, so each
# quantity is looked up under every name it plausibly ships as. A name is never
# guessed at: an unresolved quantity becomes "not computed".
_ALIASES: dict[str, tuple[str, ...]] = {
    "hypothesis": ("hypothesis", "label", "name", "description"),
    "field": ("field", "feature", "variable"),
    "threshold": ("z_threshold", "threshold", "z"),
    "horizon": ("horizon_d", "horizon", "horizon_days"),
    "p_value": ("p_value", "p", "p_uncorrected", "p_raw"),
    # `p_value_clustered` is the study's own cluster-resampled p. On the published
    # COT cell it moves 0.102 -> 0.294; a report that quoted only the i.i.d. figure
    # would overstate its own precision by roughly a factor of three.
    "p_clustered": ("p_value_clustered", "p_clustered", "p_cluster"),
    "p_adj": ("p_adj", "p_adj_bh", "p_adjusted", "p_bh", "p_value_adjusted"),
    "alpha": ("alpha", "fdr_alpha", "significance_level"),
    "significant": ("significant", "significant_bh", "reject", "detected"),
    "n_tests": ("n_tests", "n_hypotheses", "n_comparisons", "m"),
    "correction": ("correction", "correction_method", "method"),
    "power": ("power", "power_estimate", "estimated_power"),
    "n_for_80_power": ("n_for_80_power", "n_required_80", "n_for_power_80"),
    "n_events": ("n_events", "n_event", "n"),
    "n_effective": ("effective_sample_size", "n_effective", "n_eff"),
    "max_per_period": ("max_events_per_cluster", "max_events_per_period"),
    # `verdict` on agent.verify.study.StudyResult is explicitly UNCORRECTED
    # ("supported_uncorrected"). It is read for context and is never accepted as
    # a significance decision — see `_verdict_line`.
    "study_verdict": ("verdict", "study_verdict"),
    "reconciled": ("reconciled",),
    "n_periods": (
        "n_distinct_weeks",
        "distinct_weeks",
        "n_distinct_periods",
        "n_clusters",
        "n_event_clusters",
        "n_independent_periods",
    ),
    "period_name": ("period_name", "period_label", "cluster_label"),
    "effect": ("effect", "edge", "effect_size"),
    "mean_event": ("mean_event_ret", "mean_event_return", "mean_event", "event_mean"),
    "mean_baseline": ("mean_baseline_ret", "baseline_mean_return", "mean_baseline", "baseline_mean"),
    "effect_units": ("effect_units", "units", "effect_unit"),
    "ci": ("ci", "confidence_interval"),
    "ci_lo": ("ci_lo", "ci_low", "ci_lower"),
    "ci_hi": ("ci_hi", "ci_high", "ci_upper"),
}


# ── Report objects ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReportLine:
    """One labelled line of the report, with where its number came from.

    `computed` is the load-bearing field. False means the inputs were absent,
    None or non-finite, and `value` is then the string "not computed (<why>)" —
    never a substituted default. Read `computed` before `value`; a caller that
    string-matches `value` will eventually match a reason as if it were a
    finding.

    `provenance` names the `study_result` attribute or mechanism function the
    number was read from, so a disputed figure is traceable without re-running
    anything. `caveats` are the conditions that apply to THIS line only;
    report-wide caveats live on `VerificationReport.caveats`.
    """

    label: str
    value: str
    computed: bool
    provenance: str
    caveats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.label not in LINE_ORDER:
            raise ValueError(f"unknown report line label {self.label!r}; expected one of {LINE_ORDER}")


@dataclass(frozen=True)
class VerificationReport:
    """The verdict and the mechanism as one object. Always nine lines.

    `lines` is in `LINE_ORDER` and complete: iterate it to render, and read
    `uncomputed` to find out what the report could not establish. `has_mechanism`
    is False when no graph was supplied, in which case the four
    `MECHANISM_LABELS` lines are present and uncomputed — the statistical half
    ships on its own.

    Misleads: `mechanism` and `study` hold the raw computed dicts for audit. They
    are the inputs this report formatted, not independent confirmation of it.
    """

    hypothesis: str
    lines: tuple[ReportLine, ...]
    has_mechanism: bool
    caveats: tuple[str, ...] = ()
    study: Mapping[str, Any] = field(default_factory=dict)
    mechanism: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = tuple(ln.label for ln in self.lines)
        if labels != LINE_ORDER:
            missing = [lbl for lbl in LINE_ORDER if lbl not in labels]
            raise ValueError(
                f"a VerificationReport must carry all {len(LINE_ORDER)} lines in order; "
                f"got {labels} (missing {missing}). A silently omitted line is the "
                f"failure mode this class exists to prevent."
            )

    def line(self, label: str) -> ReportLine:
        """The line with this label. Raises `KeyError` on an unknown label."""
        for ln in self.lines:
            if ln.label == label:
                return ln
        raise KeyError(f"no report line labelled {label!r}")

    @property
    def uncomputed(self) -> tuple[str, ...]:
        """Labels of every line whose inputs were unavailable, in order."""
        return tuple(ln.label for ln in self.lines if not ln.computed)


# ── study_result field access ─────────────────────────────────────────────


def _raw(study_result: Any, key: str) -> tuple[Any, str | None]:
    """Resolve one logical quantity, returning (value, attribute_name_used).

    Looks up every alias for `key` in order, over a Mapping's keys or an
    object's attributes. Returns (None, None) when no alias is present; the
    caller must distinguish that from an alias that is present and holds None,
    which is why the attribute name is returned rather than a bare value.
    """
    names = _ALIASES.get(key, (key,))
    if isinstance(study_result, Mapping):
        for name in names:
            if name in study_result:
                return study_result[name], name
        return None, None
    for name in names:
        if hasattr(study_result, name):
            return getattr(study_result, name), name
    return None, None


def _num(study_result: Any, key: str) -> tuple[float | None, str]:
    """A finite float for `key`, or (None, reason). Never coerces silently.

    A bool is rejected: `True` is not a p-value, and Python would happily read
    it as 1.0. NaN and infinity are rejected as "not computed" rather than
    formatted, because a NaN that reaches a customer-facing line reads as a
    number to a formatter and as nothing to a reader.
    """
    val, name = _raw(study_result, key)
    if name is None:
        return None, f"no {key} field on study_result (looked for {list(_ALIASES.get(key, (key,)))})"
    if val is None:
        return None, f"study_result.{name} is None"
    if isinstance(val, bool):
        return None, f"study_result.{name} is a bool ({val!r}), not a number"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None, f"study_result.{name} is {type(val).__name__} ({val!r}), not a number"
    if not math.isfinite(f):
        return None, f"study_result.{name} is {f!r} (non-finite)"
    return f, f"study_result.{name}"


def _int(study_result: Any, key: str) -> tuple[int | None, str]:
    """An integer count for `key`, or (None, reason). Rejects non-integral floats."""
    f, prov = _num(study_result, key)
    if f is None:
        return None, prov
    if f != int(f):
        return None, f"{prov} is {f!r}, not an integer count"
    return int(f), prov


def _flag(study_result: Any, key: str) -> tuple[bool | None, str]:
    """A tri-state boolean for `key`: True, False, or None with a reason.

    Only real booleans and 0/1 are accepted. A string "True" is refused, because
    every non-empty string is truthy and that is how a typo becomes a positive
    verdict.
    """
    val, name = _raw(study_result, key)
    if name is None:
        return None, f"no {key} field on study_result (looked for {list(_ALIASES.get(key, (key,)))})"
    if val is None:
        return None, f"study_result.{name} is None"
    if isinstance(val, bool):
        return val, f"study_result.{name}"
    if isinstance(val, int | float) and not isinstance(val, bool) and float(val) in (0.0, 1.0):
        return bool(val), f"study_result.{name}"
    return None, f"study_result.{name} is {type(val).__name__} ({val!r}), not a boolean"


def _text(study_result: Any, key: str) -> tuple[str | None, str]:
    """A non-empty string for `key`, or (None, reason)."""
    val, name = _raw(study_result, key)
    if name is None:
        return None, f"no {key} field on study_result"
    if val is None:
        return None, f"study_result.{name} is None"
    s = str(val).strip()
    if not s:
        return None, f"study_result.{name} is empty"
    return s, f"study_result.{name}"


def _fmt(x: float, digits: int = 3) -> str:
    """Format a float without inventing precision, and without eating a digit.

    Trailing zeros are stripped only from the FRACTIONAL part. The first version
    of this function called `.rstrip("0")` on the whole string, which turned
    `_fmt(20.0, 0)` into "2" and `_fmt(10.0, 0)` into "1" — a 20-day horizon
    rendered as 2 days and 10% power as 1%. Every number in this report passes
    through here, so the strip must never touch a digit left of the point.
    """
    if x != 0 and (abs(x) < 10 ** (-digits) or abs(x) >= 1e6):
        return f"{x:.2e}"
    s = f"{x:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _hypothesis_text(study_result: Any) -> str:
    """The hypothesis as stated, or one reconstructed from field/threshold/horizon.

    `agent.verify.study.StudyResult.hypothesis` is a `Hypothesis` OBJECT, not a
    string. Its `.label` (the customer's own words, when set) is preferred, then
    its `.describe()`. Falling through to `str()` would put a dataclass repr on
    the front page of the product, so an object with neither is refused rather
    than stringified.

    Misleads: `describe()` is a machine description of a test cell
    ("cftc/futures_positioning/mm_net_pct_oi |z| >= 2 -> ... @ 20 steps"), not the
    customer's sentence. When `label` is set, that is used verbatim.
    """
    raw, _name = _raw(study_result, "hypothesis")
    if raw is not None and not isinstance(raw, str):
        lbl = getattr(raw, "label", None)
        if isinstance(lbl, str) and lbl.strip():
            return lbl.strip()
        describe = getattr(raw, "describe", None)
        if callable(describe):
            try:
                out = describe()
            except Exception:  # noqa: BLE001 — a broken describe() must not sink the report
                out = None
            if isinstance(out, str) and out.strip():
                return out.strip()
        raw = None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    fld, _ = _text(study_result, "field")
    thr, _ = _num(study_result, "threshold")
    hor, _ = _num(study_result, "horizon")
    bits = []
    if fld:
        bits.append(fld)
    if thr is not None:
        bits.append(f"at |z| >= {_fmt(thr, 1)}")
    if hor is not None:
        bits.append(f"over {_fmt(hor, 0)} days")
    if bits:
        return " ".join(bits)
    return "hypothesis not stated by study_result"


# ── statistical lines ─────────────────────────────────────────────────────


def _effect_clause(study_result: Any) -> str:
    """ " — event mean X vs baseline Y (edge Z)" when computed, else "".

    Units are NOT assumed. `mean_event_ret` in the reference implementation is a
    log return expressed as a fraction; the published paper prints it times 100.
    This function prints whatever it was handed and appends a unit only if
    `study_result` names one, because silently rescaling by 100 would turn
    0.02584 into "2.584" with no record of the transform.
    """
    units, _ = _text(study_result, "effect_units")
    suffix = f" {units}" if units else ""
    ev, _ = _num(study_result, "mean_event")
    base, _ = _num(study_result, "mean_baseline")
    eff, _ = _num(study_result, "effect")
    parts = []
    if ev is not None:
        parts.append(f"event mean {_fmt(ev, 5)}{suffix}")
    if base is not None:
        parts.append(f"baseline {_fmt(base, 5)}{suffix}")
    if eff is not None:
        parts.append(f"edge {eff:+.5g}{suffix}")
    if not parts:
        return ""
    return " — " + ", ".join(parts)


def _verdict_line(study_result: Any, correction: Mapping[str, Any] | None = None) -> ReportLine:
    """Detectable / not detectable, with the adjusted p and the correction used.

    A single `StudyResult` cannot know its own adjusted p: Benjamini-Hochberg is a
    property of the FAMILY of hypotheses the study was one of. `correction` is
    where a caller holding an `agent.verify.stats.BHResult` supplies that
    family-level fact — `{"p_adjusted", "alpha", "n_tested", "method"}`. Without
    it there is no corrected verdict, and this function says so rather than
    promoting the study's own uncorrected one.

    Two things it refuses outright:

      * a disagreement between an explicit `significant` flag and
        `p_adj <= alpha`. A study that ships both and disagrees with itself has a
        bug, and picking the more flattering one is how F-18 happened: a
        favourable number ended the investigation.
      * a study whose `reconciled` is False. `StudyResult`'s own docstring says a
        False `reconciled` means rows were lost and its numbers are not to be
        used. Formatting them anyway would launder a bookkeeping failure into a
        verdict.
    """
    reconciled, rec_prov = _flag(study_result, "reconciled")
    if reconciled is False:
        return ReportLine(
            label="VERDICT",
            value=(
                f"{_NOT_COMPUTED} — the study reports reconciled=False ({rec_prov}): it "
                f"could not account for every observation that entered, so its numbers "
                f"are not to be used. No verdict is issued from unreconciled bookkeeping."
            ),
            computed=False,
            provenance=rec_prov,
            caveats=("fix the study's bookkeeping before any figure here is quoted.",),
        )

    corr_map: Mapping[str, Any] = correction or {}
    p_adj, p_adj_prov = _num(study_result, "p_adj")
    supplied = corr_map.get("p_adjusted", corr_map.get("p_adj"))
    if supplied is not None:
        pc, _ = _num({"p_adj": supplied}, "p_adj")
        if pc is not None:
            if p_adj is not None and abs(pc - p_adj) > 1e-12:
                return ReportLine(
                    label="VERDICT",
                    value=(
                        f"CONTRADICTORY INPUT — the correction argument says adjusted "
                        f"p = {_fmt(pc)} and {p_adj_prov} says {_fmt(p_adj)}. No verdict "
                        f"is issued from two disagreeing adjusted p-values."
                    ),
                    computed=False,
                    provenance=f"correction['p_adjusted'] vs {p_adj_prov}",
                    caveats=("the caller and the study disagree; establish which family this cell belongs to.",),
                )
            p_adj, p_adj_prov = pc, "correction['p_adjusted']"

    alpha, alpha_prov = _num(study_result, "alpha")
    if corr_map.get("alpha") is not None:
        ac, _ = _num({"alpha": corr_map["alpha"]}, "alpha")
        if ac is not None:
            alpha, alpha_prov = ac, "correction['alpha']"

    sig, sig_prov = _flag(study_result, "significant")
    p_raw, p_raw_prov = _num(study_result, "p_value")
    p_clu, p_clu_prov = _num(study_result, "p_clustered")
    corr, _ = _text(study_result, "correction")
    if corr_map.get("method"):
        corr = str(corr_map["method"])
    n_tests, _ = _int(study_result, "n_tests")
    n_supplied = corr_map.get("n_tested", corr_map.get("n_tests"))
    if n_supplied is not None:
        nc, _ = _int({"n_tests": n_supplied}, "n_tests")
        if nc is not None:
            n_tests = nc

    caveats: list[str] = []
    from_threshold = None if (p_adj is None or alpha is None) else bool(p_adj <= alpha)

    if sig is not None and from_threshold is not None and sig != from_threshold:
        return ReportLine(
            label="VERDICT",
            value=(
                f"CONTRADICTORY INPUT — study_result reports significant={sig!r} but "
                f"adjusted p = {_fmt(p_adj)} against alpha = {_fmt(alpha)} implies "
                f"{from_threshold!r}. No verdict is issued from a self-inconsistent study."
            ),
            computed=False,
            provenance=f"{sig_prov} vs {p_adj_prov} and {alpha_prov}",
            caveats=("the two disagree; fix the study before quoting either.",),
        )

    decided = sig if sig is not None else from_threshold
    if decided is None:
        why = []
        if sig is None:
            why.append(sig_prov)
        if p_adj is None:
            why.append(p_adj_prov)
        if alpha is None:
            why.append(alpha_prov)
        sv, sv_prov = _text(study_result, "study_verdict")
        context = ""
        extra: list[str] = [
            "a verdict needs either an explicit significance flag or both an "
            "adjusted p-value and an alpha; none was supplied. Pass a `correction` "
            "mapping built from agent.verify.stats.benjamini_hochberg over the "
            "family this cell belongs to."
        ]
        if sv:
            context = f". The study's own UNCORRECTED verdict is {sv!r}"
            extra.append(
                f"{sv_prov} is uncorrected. 51 cells tested at alpha=0.05 expect ~2.5 "
                f"false positives, so an uncorrected 'supported' is not a detection and "
                f"is not promoted to one here."
            )
        return ReportLine(
            label="VERDICT",
            value=f"{_NOT_COMPUTED} — no significance decision available ({'; '.join(why)}){context}",
            computed=False,
            provenance=sv_prov if sv else "none",
            caveats=tuple(extra),
        )

    head = "DETECTED" if decided else "not detectable"
    detail: list[str] = []
    if p_adj is not None:
        corr_name = corr or "multiplicity-adjusted"
        detail.append(f"{corr_name} p = {_fmt(p_adj)}")
    if alpha is not None:
        detail.append(f"alpha = {_fmt(alpha)}")
    if n_tests is not None:
        detail.append(f"{n_tests} hypotheses tested")
    if p_raw is not None:
        detail.append(f"uncorrected p = {_fmt(p_raw)}")
    if p_clu is not None:
        detail.append(f"cluster-resampled p = {_fmt(p_clu)}")
    value = head + (f" ({', '.join(detail)})" if detail else "")
    value += _effect_clause(study_result)

    if p_adj is None:
        caveats.append(
            "no adjusted p-value was supplied; the verdict rests on the study's own "
            "significance flag and cannot be re-derived here."
        )
    if p_raw is not None and p_adj is not None and p_raw <= (alpha if alpha is not None else 0.05) < p_adj:
        caveats.append(
            f"uncorrected p = {_fmt(p_raw)} would have looked significant; the correction is what changed the answer."
        )
    if p_clu is not None and p_raw is not None and p_clu > p_raw:
        caveats.append(
            f"resampling CLUSTERS rather than events moves p from {_fmt(p_raw)} to "
            f"{_fmt(p_clu)} ({p_clu_prov}). Events on the same date are not "
            f"independent draws, so the clustered figure is the defensible one."
        )
    if not decided:
        caveats.append(
            "'not detectable' is a measurement of this test, not a statement that "
            "no effect exists — read the POWER line before treating it as either."
        )
    if reconciled is None:
        caveats.append(
            "the study did not report a `reconciled` flag, so nothing here confirms "
            "that every observation it read is accounted for."
        )

    provs = [p for p in (sig_prov, p_adj_prov, alpha_prov) if p.startswith(("study_result.", "correction["))]
    return ReportLine(
        label="VERDICT",
        value=value,
        computed=True,
        provenance=" + ".join(provs) if provs else "none",
        caveats=tuple(caveats),
    )


def _power_line(study_result: Any) -> ReportLine:
    """Estimated power, and what a null result at that power is worth.

    This line is mandatory and renders "not computed" rather than vanishing: a
    null result with no power figure beside it is the single easiest way to
    overstate a negative finding, and an absent line is invisible.
    """
    power, prov = _num(study_result, "power")
    if power is None:
        return ReportLine(
            label="POWER",
            value=f"{_NOT_COMPUTED} — {prov}",
            computed=False,
            provenance="none",
            caveats=(
                "without a power estimate, a null result cannot be distinguished "
                "from an underpowered test. Do not read 'not detectable' as 'no effect'.",
            ),
        )
    pct = power * 100.0 if power <= 1.0 else power
    caveats: list[str] = []
    n80, n80_prov = _int(study_result, "n_for_80_power")
    extra = ""
    if n80 is not None:
        extra = f"; {n80} observations would be needed for 80%"
    if pct < 50.0:
        caveats.append(
            f"at ~{_fmt(pct, 0)}% power a null result is close to preordained; the "
            f"sample, not the effect, is the binding constraint."
        )
    return ReportLine(
        label="POWER",
        value=f'~{_fmt(pct, 0)}% — "no detectable effect" is not "no effect"{extra}',
        computed=True,
        provenance=prov if n80 is None else f"{prov} + {n80_prov}",
        caveats=tuple(caveats),
    )


def _sample_line(study_result: Any) -> ReportLine:
    """Event count, distinct-period count, and the ratio between them.

    The ratio is the clustering measure: 123 events on 67 distinct weeks is 0.545
    and means an i.i.d. bootstrap over events overstates its own precision. The
    denominator is never inferred — a study that does not count distinct periods
    gets a line that says the ratio was not computed, with the event count still
    shown.
    """
    n_ev, ev_prov = _int(study_result, "n_events")
    n_per, per_prov = _int(study_result, "n_periods")
    period_name, _ = _text(study_result, "period_name")
    unit = period_name or "distinct period"
    plural = unit if unit.endswith("s") else unit + "s"

    if n_ev is None and n_per is None:
        return ReportLine(
            label="SAMPLE",
            value=f"{_NOT_COMPUTED} — {ev_prov}; {per_prov}",
            computed=False,
            provenance="none",
            caveats=("sample size is unknown, so nothing here bounds the uncertainty.",),
        )
    if n_ev is None:
        return ReportLine(
            label="SAMPLE",
            value=f"{n_per} {plural}; event count {_NOT_COMPUTED} ({ev_prov})",
            computed=False,
            provenance=per_prov,
            caveats=("the clustering ratio needs both counts and was not computed.",),
        )
    if n_per is None:
        return ReportLine(
            label="SAMPLE",
            value=(f"{n_ev} events; {plural} {_NOT_COMPUTED} ({per_prov}), so the clustering ratio is unknown"),
            computed=False,
            provenance=ev_prov,
            caveats=(
                "events that fall on the same period are not independent draws. "
                "Without the period count, the effective sample size is unbounded below.",
            ),
        )

    ratio = (n_per / n_ev) if n_ev else None
    caveats: list[str] = []
    if n_per > n_ev:
        caveats.append(
            f"{n_per} {plural} exceeds {n_ev} events, which cannot happen if periods "
            f"are counted over these events — check the study's denominators."
        )
    elif ratio is not None and ratio < 0.8:
        caveats.append(
            f"ratio {_fmt(ratio, 2)}: these events cluster. An i.i.d. resample over "
            f"events assumes {n_ev} independent draws and there are at most {n_per}."
        )
    if n_ev < 30:
        caveats.append(f"{n_ev} events is a small-sample regime; bootstrap intervals are wide and fragile.")
    ratio_txt = f" (ratio {_fmt(ratio, 2)})" if ratio is not None else ""
    n_eff, eff_prov = _int(study_result, "n_effective")
    eff_txt = ""
    if n_eff is not None:
        eff_txt = f", effective n = {n_eff}"
        if n_ev and n_eff < n_ev:
            caveats.append(
                f"the study's own effective sample size is {n_eff}, not {n_ev}; any "
                f"interval computed on {n_ev} draws is narrower than the data supports."
            )
    mx, mx_prov = _int(study_result, "max_per_period")
    if mx is not None and mx > 1:
        caveats.append(f"up to {mx} events fall on a single {unit} ({mx_prov}).")
    return ReportLine(
        label="SAMPLE",
        value=f"{n_ev} events across {n_per} {plural}{ratio_txt}{eff_txt}",
        computed=True,
        provenance=f"{ev_prov} + {per_prov}" + (f" + {eff_prov}" if n_eff is not None else ""),
        caveats=tuple(caveats),
    )


# ── mechanism lines ───────────────────────────────────────────────────────


def _no_graph(label: str, reason: str) -> ReportLine:
    """A mechanism line that could not be computed, stated rather than dropped."""
    return ReportLine(
        label=label,
        value=f"{_NOT_COMPUTED} — {reason}",
        computed=False,
        provenance="none",
        caveats=("the statistical verdict above stands on its own; this line adds mechanism, not confidence.",),
    )


def _strength_line(conn: Mapping[str, Any] | None, err: str | None) -> ReportLine:
    """Peer-group PageRank rank, refusing to quote a rank for an unreachable pair.

    `connectivity()` returns a `type_rank` even when the mass is numerical zero,
    and its own notes forbid quoting it: a zero-mass "#1 of 93" is a tie behind
    everything reachable, not a finding. This function honours that.
    """
    if conn is None:
        return _no_graph("STRENGTH", err or "no graph supplied")
    dst_type = conn.get("dst_type") or "entity"
    src_name = conn.get("src_name") or conn.get("src_id")
    if not conn.get("connected"):
        return ReportLine(
            label="STRENGTH",
            value=(
                f"unreachable — PageRank mass {conn.get('mass', 0.0):.3e} from {src_name} "
                f"is numerical zero, so the {dst_type} rank is degenerate and is not quoted"
            ),
            computed=True,
            provenance="agent.mechanism.connectivity.connectivity",
            caveats=("a computed 'no route' is a result: nothing in this graph routes cause to effect.",),
        )
    ratio = conn.get("ratio_to_type_median")
    ratio_txt = f", {_fmt(ratio, 1)}x median" if isinstance(ratio, int | float) else ", ratio to median undefined"
    ties = int(conn.get("type_ties") or 0)
    value = (
        f"{dst_type} #{conn.get('type_rank')} of {conn.get('type_count')} from {src_name}{ratio_txt} "
        f"(mass {conn.get('mass', float('nan')):.3e}, global rank "
        f"{conn.get('global_rank')} of {conn.get('n_entities')})"
    )
    caveats = list(conn.get("notes") or ())
    if ties:
        caveats.append(f"{ties} peers of the same type share this exact mass.")
    return ReportLine(
        label="STRENGTH",
        value=value,
        computed=True,
        provenance="agent.mechanism.connectivity.connectivity",
        caveats=tuple(caveats),
    )


def _route_families(routes: Sequence[Any]) -> list[dict[str, Any]]:
    """Group routes by their link_type sequence, merging intermediate nodes.

    Five routes that differ only in which oil-producing country they pass
    through are one shape of explanation, not five. Grouping by `link_types` and
    collapsing the intermediates into a set is what turns ten rows into
    "Russia --event_involves--> {US, Iran, Canada} --produced_in--> WTI".

    Misleads: the merge hides that the members have different masses. Total mass
    per family is returned so the ordering is still by measured mass, but a
    family is a rendering convenience and never a count of evidence.
    """
    fams: dict[tuple[str, ...], dict[str, Any]] = {}
    for r in routes:
        key = tuple(r.link_types)
        fam = fams.setdefault(
            key,
            {
                "link_types": key,
                "kinds": tuple(r.kinds),
                "mids": [set() for _ in range(max(len(r.node_names) - 2, 0))],
                "src": r.node_names[0],
                "dst": r.node_names[-1],
                "mass": 0.0,
                "n": 0,
                "hub_dominated": 0,
            },
        )
        for i, nm in enumerate(r.node_names[1:-1]):
            fam["mids"][i].add(nm)
        fam["mass"] += float(r.mass)
        fam["n"] += 1
        fam["hub_dominated"] += 1 if r.hub_dominated else 0
    return sorted(fams.values(), key=lambda f: -f["mass"])


def _render_family(fam: Mapping[str, Any], max_names: int = 4) -> str:
    """One family as `A --lt--> {B, C} --lt--> D`, marked if all-scaffold."""
    out = [fam["src"]]
    for i, lt in enumerate(fam["link_types"]):
        out.append(f"--{lt}-->")
        if i < len(fam["mids"]):
            names = sorted(fam["mids"][i])
            shown = names[:max_names]
            more = len(names) - len(shown)
            body = ", ".join(shown) + (f", +{more} more" if more else "")
            out.append(body if len(names) == 1 else "{" + body + "}")
    out.append(fam["dst"])
    line = " ".join(out)
    if fam["kinds"] and all(k == "scaffold" for k in fam["kinds"]):
        line += "  [all-scaffold]"
    return line


def _routes_line(routes: Sequence[Any] | None, err: str | None, max_families: int = 3) -> ReportLine:
    """The route shapes that connect cause to effect, ordered by measured mass."""
    if routes is None:
        return _no_graph("ROUTES", err or "no graph supplied")
    if len(routes) == 0:
        return ReportLine(
            label="ROUTES",
            value="no route found within the search bound — these entities are not connected in this graph",
            computed=True,
            provenance="agent.mechanism.routes.top_routes",
            caveats=("a computed absence of any path, not a failed search; see the search block.",),
        )
    fams = _route_families(routes)
    shown = fams[:max_families]
    text = " | ".join(_render_family(f) for f in shown)
    if len(fams) > len(shown):
        text += f" | +{len(fams) - len(shown)} further route shapes"
    caveats = [
        f"{len(routes)} routes collapse to {len(fams)} distinct link_type sequences; "
        f"members of one sequence are one shape of explanation, not several confirmations."
    ]
    return ReportLine(
        label="ROUTES",
        value=text,
        computed=True,
        provenance="agent.mechanism.routes.top_routes",
        caveats=tuple(caveats),
    )


def _evidence_line(per_width: Sequence[Mapping[str, Any]], err: str | None) -> ReportLine:
    """Independent time-varying sources, at every search width that was run.

    The width is part of the number. Measured: Russia -> WTI Crude Oil returns 1
    independent source at k=10 and 3 at k=400 — the extra two enter on longer
    routes through prediction-market topics. Reporting "1 independent source"
    without "at k=10" implies an exhaustive search that did not happen, and
    reporting "3" without "at k=400" implies breadth that the default search
    does not have.
    """
    if not per_width:
        return _no_graph("EVIDENCE", err or "no graph supplied")
    bits: list[str] = []
    caveats: list[str] = []
    for w in per_width:
        ind = w["independence"]
        k = w["k"]
        n = ind["independent_sources"]
        if ind["status"] == "no_route":
            bits.append(f"no route at k={k}")
            continue
        names = ind["evidence_sources"]
        label = f"{n} independent time-varying source" + ("" if n == 1 else "s")
        names_txt = f" ({', '.join(names)})" if names else ""
        bits.append(f"{label}{names_txt} at k={k}")
        if not ind["complete"]:
            caveats.append(
                f"k={k}: the search was not exhaustive (more_routes_exist="
                f"{ind['search']['more_routes_exist']}, budget_exhausted="
                f"{ind['search']['budget_exhausted']}); this is a count over the "
                f"top-{k} routes only."
            )
        if ind["naive_source_count"] > n:
            n_inflated = ind["naive_source_count"] - n
            caveats.append(
                f"k={k}: naive counting of every source label on every hop would "
                f"claim {ind['naive_source_count']}; {n_inflated} of those "
                f"{'is' if n_inflated == 1 else 'are'} static scaffold and "
                f"{'witnesses' if n_inflated == 1 else 'witness'} nothing."
            )
        hub_free = ind["hub_free_independent_sources"]
        if n and hub_free < n:
            caveats.append(
                f"k={k}: only {hub_free} of {n} "
                f"{'source survives' if n == 1 else 'sources survive'} dropping "
                f"hub-dominated routes — the rest are supported only by routes "
                f"through a node that touches everything."
            )
        for warn in ind["warnings"]:
            caveats.append(f"k={k}: {warn}")
    counts = {w["independence"]["independent_sources"] for w in per_width}
    if len(counts) > 1:
        caveats.insert(
            0,
            "the source count CHANGES with search width: "
            + ", ".join(f"{w['independence']['independent_sources']} at k={w['k']}" for w in per_width)
            + ". Neither figure is wrong and neither is quotable without its k.",
        )
    return ReportLine(
        label="EVIDENCE",
        value="; ".join(bits),
        computed=True,
        provenance="agent.mechanism.routes.independence over agent.mechanism.routes.top_routes",
        caveats=tuple(caveats),
    )


def _scaffold_line(per_width: Sequence[Mapping[str, Any]], err: str | None) -> ReportLine:
    """The static-geography relations the routes travelled, named as non-evidence.

    `produced_in` and `exchange_country` are facts about the world that were true
    in 1990 and are true today. They ROUTE a signal without witnessing anything,
    so a report that let them raise its confidence would be actively harmful.
    They are named here so a reader can see what the mass was flowing through.
    """
    if not per_width:
        return _no_graph("SCAFFOLD", err or "no graph supplied")
    link_types: set[str] = set()
    sources: set[str] = set()
    unclassified: set[str] = set()
    hops = 0
    for w in per_width:
        ind = w["independence"]
        sources.update(ind["scaffold_sources_non_evidence"])
        unclassified.update(ind["unclassified_link_types"])
        hops += ind["scaffold_hops"]
        for r in w["routes"]:
            for hop in r.hops_detail:
                if hop.kind == "scaffold":
                    link_types.add(hop.link_type)
    caveats: list[str] = []
    if unclassified:
        caveats.append(
            f"link_type(s) {sorted(unclassified)} are classified as neither evidence "
            f"nor scaffold and are counted as neither. Classify them in "
            f"agent/models/gnn/graph_builder.py before quoting this report."
        )
    if not link_types:
        return ReportLine(
            label="SCAFFOLD",
            value="none — no static-geography hop appears on the returned routes",
            computed=True,
            provenance="agent.mechanism.routes.Route.hops_detail",
            caveats=tuple(caveats),
        )
    srcs = f" (sources: {', '.join(sorted(sources))})" if sources else ""
    return ReportLine(
        label="SCAFFOLD",
        value=(
            f"{', '.join(sorted(link_types))} — static geography, not evidence; {hops} scaffold hops traversed{srcs}"
        ),
        computed=True,
        provenance="agent.mechanism.routes.Route.hops_detail",
        caveats=tuple(caveats),
    )


def _caveat_line(per_width: Sequence[Mapping[str, Any]], study_caveats: Sequence[str]) -> ReportLine:
    """The findings that would mislead if read without them, derived not canned.

    Three structural checks run here: is the highest-mass route pure scaffold
    (measured: for Russia -> WTI it is, a direct `produced_in` edge); what share
    of the returned routes are hub-dominated (measured: 9 of 10); and does any
    evidenced route avoid hubs at all. Each caveat is emitted only when its
    condition holds, so an empty caveat list is itself a computed statement.
    """
    items: list[str] = []
    for w in per_width:
        routes = w["routes"]
        ind = w["independence"]
        k = w["k"]
        tag = f"k={k}: " if len(per_width) > 1 else ""
        if routes:
            top = max(routes, key=lambda r: r.mass)
            if top.kinds and all(kind == "scaffold" for kind in top.kinds):
                items.append(
                    f"{tag}top route by mass is pure scaffold "
                    f"({' + '.join(top.link_types)}) and carries zero evidential weight"
                )
            hub = ind["hub_dominated_routes"]
            if hub:
                items.append(f"{tag}{hub}/{len(routes)} routes hub-dominated")
            if ind["routes_without_evidence"]:
                items.append(f"{tag}{ind['routes_without_evidence']}/{len(routes)} routes carry no event hop at all")
    items.extend(study_caveats)
    if not items:
        return ReportLine(
            label="CAVEAT",
            value=(
                "none triggered — the highest-mass route carries an event hop, no route "
                "is hub-dominated, and every route carries evidence"
            ),
            computed=True,
            provenance="derived from agent.mechanism.routes (top-mass kind, hub_dominated_routes, routes_without_evidence)",
        )
    return ReportLine(
        label="CAVEAT",
        value="; ".join(items),
        computed=True,
        provenance="derived from agent.mechanism.routes (top-mass kind, hub_dominated_routes, routes_without_evidence)",
    )


# ── upgrade path ──────────────────────────────────────────────────────────


def _qualifying_sources(graph: Any, counted: set[str], counted_link_types: set[str]) -> dict[str, Any]:
    """Evidence-class sources present in the graph that are not already counted.

    Walks `graph.edges` and classifies each `link_type` with
    `agent.mechanism.routes.link_kind`, so an unclassified relation is never
    offered as an upgrade. A candidate whose every link_type is one this finding
    already uses is listed separately as co-dependent: a second pipeline writing
    the SAME relation (a repair pass over an earlier ingest) is one witness, not
    two, and offering it as independent evidence would be the inflation this
    product exists to refuse.

    Misleads: presence in the graph is not relevance. `form144` has 12,180 links
    and none of them may touch this pair. This names what COULD qualify, not
    what would.
    """
    from agent.mechanism.routes import link_kind

    by_source: dict[str, set[str]] = {}
    unclassified: set[str] = set()
    for e in getattr(graph, "edges", ()):  # MechanismEdge
        kind = link_kind(e.link_type)
        if kind == "unclassified":
            unclassified.add(e.link_type)
            continue
        if kind != "evidence":
            continue
        by_source.setdefault(e.source, set()).add(e.link_type)

    qualifying: list[str] = []
    codependent: dict[str, list[str]] = {}
    for src, lts in sorted(by_source.items()):
        if src in counted:
            continue
        fresh = lts - counted_link_types
        if fresh:
            qualifying.append(src)
        else:
            codependent[src] = sorted(lts)
    return {
        "qualifying": qualifying,
        "codependent": codependent,
        "unclassified_link_types": sorted(unclassified),
        "n_evidence_sources_in_graph": len(by_source),
    }


def _upgrade_line(
    study_result: Any,
    per_width: Sequence[Mapping[str, Any]],
    graph: Any | None,
    mech_err: str | None,
) -> ReportLine:
    """What evidence would change the verdict — derived from what is missing.

    Checks four binding constraints in the order they bind, and names the
    missing evidence CLASS for each one that holds: statistical power, event
    clustering, source independence, and route staticness. When source
    independence binds and a graph is present, the qualifying sources are named
    from the graph rather than described in the abstract.

    Misleads: this is a list of what would make the test better, not a promise
    that it would make the answer positive. A well-powered re-run of a true null
    returns the same verdict with more authority, which is the outcome this
    product is for.
    """
    needs: list[str] = []
    checked: list[str] = []

    power, _ = _num(study_result, "power")
    checked.append("power")
    if power is not None:
        pct = power * 100.0 if power <= 1.0 else power
        if pct < 80.0:
            n80, _ = _int(study_result, "n_for_80_power")
            want = f" ({n80} needed for 80%)" if n80 is not None else ""
            needs.append(f"MORE OBSERVATIONS: power is ~{_fmt(pct, 0)}%, below 80%{want}")
    else:
        needs.append("A POWER ESTIMATE: none was computed, so the strength of this null is unknown")

    n_ev, _ = _int(study_result, "n_events")
    n_per, _ = _int(study_result, "n_periods")
    checked.append("event clustering")
    if n_ev and n_per and n_ev > 0 and (n_per / n_ev) < 0.8:
        needs.append(
            f"INDEPENDENT PERIODS, NOT MORE EVENTS: {n_ev} events fall on {n_per} "
            f"periods, so additional events from the same periods add no information"
        )

    checked.append("source independence")
    if per_width:
        widest = max(per_width, key=lambda w: w["k"])
        ind = widest["independence"]
        n = ind["independent_sources"]
        n_hub_free = ind["hub_free_independent_sources"]
        counted = set(ind["evidence_sources"])
        counted_lts = set(ind["evidence_link_types"])
        # A source that appears only on hub-dominated routes is weak evidence by
        # construction — everything touches the United States. So independence
        # binds when EITHER the raw count or the hub-free count is <= 1, and
        # checking only the raw count is how a 3 at k=400 would hide a 1.
        if n <= 1 or n_hub_free <= 1:
            if n <= 1:
                head = (
                    f"the finding rests on {next(iter(counted))} alone, "
                    if n == 1 and counted
                    else "no time-varying source supports these routes, "
                )
            else:
                head = (
                    f"{n} sources appear ({', '.join(sorted(counted))}) but only "
                    f"{n_hub_free} survives dropping hub-dominated routes, "
                )
            claim = (
                "AN INDEPENDENT TIME-VARYING SOURCE: "
                + head
                + f"so a failure of it takes the whole finding (measured at k={widest['k']})"
            )
            if graph is not None:
                avail = _qualifying_sources(graph, counted, counted_lts)
                if avail["qualifying"]:
                    claim += (
                        f". Sources in this graph that would qualify: "
                        f"{', '.join(avail['qualifying'])} "
                        f"({len(avail['qualifying'])} of "
                        f"{avail['n_evidence_sources_in_graph']} evidence-class sources present)"
                    )
                else:
                    claim += ". No other evidence-class source is present in this graph"
                if avail["codependent"]:
                    claim += (
                        f". Excluded as co-dependent (they write a relation already "
                        f"counted here): {', '.join(sorted(avail['codependent']))}"
                    )
            needs.append(claim)
        checked.append("route staticness")
        routes = widest["routes"]
        if routes:
            top = max(routes, key=lambda r: r.mass)
            if top.kinds and all(kind == "scaffold" for kind in top.kinds):
                needs.append(
                    "A DATED ROUTE THAT OUTRANKS GEOGRAPHY: the highest-mass route is "
                    "static scaffold, so an effective_from backfill on the event links "
                    "would change which route the mass flows along"
                )
    else:
        needs.append(
            f"THE MECHANISM HALF: {mech_err or 'no graph was supplied'}, so no statement "
            f"about evidential independence is possible"
        )

    if not needs:
        return ReportLine(
            label="UPGRADE PATH",
            value=(
                f"none identified — checked {', '.join(checked)}: power is adequate, "
                f"events are not clustered, and more than one independent time-varying "
                f"source supports the routes"
            ),
            computed=True,
            provenance="derived from study_result power/sample fields and agent.mechanism.routes.independence",
        )
    return ReportLine(
        label="UPGRADE PATH",
        value="; ".join(needs),
        computed=True,
        provenance="derived from study_result power/sample fields and agent.mechanism.routes.independence",
    )


# ── public API ────────────────────────────────────────────────────────────


def build_report(
    study_result: Any,
    *,
    graph: Any = None,
    src_id: str | None = None,
    dst_id: str | None = None,
    correction: Mapping[str, Any] | None = None,
    route_widths: Sequence[int] = DEFAULT_ROUTE_WIDTHS,
    max_hops: int = 4,
) -> VerificationReport:
    """Join a statistical study result to a structural mechanism, as one report.

    Computes the five statistical lines from `study_result` and, when `graph`,
    `src_id` and `dst_id` are all supplied, the four mechanism lines by running
    `connectivity()` once and `top_routes()` + `independence()` once per width in
    `route_widths`. Every one of the nine `LINE_ORDER` lines is present in the
    result whether or not its inputs were: a line with no inputs says
    "not computed" and why.

    Args:
        study_result: a mapping or object carrying the study's numbers. Field
            names are resolved through `_ALIASES`, so a row dict from
            `scripts/cftc_event_study.py` works unchanged. Nothing is required;
            each absent quantity costs exactly the line that needed it.
        graph: any object satisfying the `agent.mechanism` graph contract, or
            None for a statistics-only report.
        src_id: cause entity id in `graph`.
        dst_id: effect entity id in `graph`.
        correction: the family-level multiple-testing facts a lone `StudyResult`
            cannot carry — `{"p_adjusted", "alpha", "n_tested", "method"}`, as
            produced by `agent.verify.stats.benjamini_hochberg` for the family
            this cell belongs to. Without it there is no corrected verdict and
            the VERDICT line says so; the study's own uncorrected verdict is
            reported as context and is never promoted to a detection.
        route_widths: the `k` values to run the route search at. Reporting more
            than one is the point — see `_evidence_line`. Duplicates are
            collapsed and the order is ascending.
        max_hops: passed to `top_routes`.

    Returns:
        `VerificationReport` with all nine lines.

    Raises:
        ValueError: if `graph` is supplied without both `src_id` and `dst_id`,
            or if `route_widths` is empty or contains a non-positive k. A graph
            that is silently ignored for want of an id would produce a report
            reading "no graph supplied" while holding one, which is a lie.

        Nothing else. A mechanism computation that fails (unknown entity,
        isolated entity, empty graph) is caught, and its exception text becomes
        the reason on the four mechanism lines. A graph problem must not destroy
        a valid statistical verdict.

    Where it misleads:
        The two halves are computed from different data over different windows.
        This function does not check that `graph`'s `as_of` matches the study's
        sample period, and a mechanism measured today beside a study run over
        2023-2026 is a real mismatch that no line here will catch. It also
        re-derives none of the study's arithmetic apart from the
        significant/alpha consistency check.
    """
    widths = sorted({int(k) for k in route_widths})
    if not widths:
        raise ValueError("route_widths is empty; the independence figure is meaningless without a search width")
    if any(k <= 0 for k in widths):
        raise ValueError(f"route_widths must all be positive, got {sorted(route_widths)}")
    if graph is not None and (src_id is None or dst_id is None):
        raise ValueError(
            "a graph was supplied without src_id and/or dst_id. Ignoring it would "
            "produce a report saying 'no graph supplied' while holding one."
        )

    conn: dict[str, Any] | None = None
    per_width: list[dict[str, Any]] = []
    mech_err: str | None = None if graph is not None else "no graph supplied"

    if graph is not None:
        from agent.mechanism.connectivity import connectivity
        from agent.mechanism.routes import independence, top_routes

        try:
            conn = connectivity(graph, src_id, dst_id)
        except Exception as exc:  # noqa: BLE001 — the reason is reported, not swallowed
            conn = None
            mech_err = f"connectivity() failed: {type(exc).__name__}: {exc}"
        for k in widths:
            try:
                routes = top_routes(graph, src_id, dst_id, k=k, max_hops=max_hops)
                per_width.append({"k": k, "routes": routes, "independence": independence(routes)})
            except Exception as exc:  # noqa: BLE001 — same
                mech_err = f"route search at k={k} failed: {type(exc).__name__}: {exc}"
                per_width = []
                break

    study_caveats: list[str] = []
    verdict = _verdict_line(study_result, correction)
    power = _power_line(study_result)
    sample = _sample_line(study_result)
    for ln in (verdict, power, sample):
        study_caveats.extend(ln.caveats)

    lines = (
        verdict,
        power,
        sample,
        _strength_line(conn, mech_err),
        _routes_line(per_width[0]["routes"] if per_width else None, mech_err),
        _evidence_line(per_width, mech_err),
        _scaffold_line(per_width, mech_err),
        _caveat_line(per_width, study_caveats),
        _upgrade_line(study_result, per_width, graph, mech_err),
    )

    report_caveats: list[str] = []
    if per_width and conn is not None:
        report_caveats.append(
            "the statistical verdict and the mechanism are computed from different "
            "data; a strong mechanism beside a null verdict is not a contradiction."
        )
    if mech_err and mech_err != "no graph supplied":
        report_caveats.append(mech_err)

    return VerificationReport(
        hypothesis=_hypothesis_text(study_result),
        lines=lines,
        has_mechanism=bool(per_width),
        caveats=tuple(report_caveats),
        study=dict(study_result) if isinstance(study_result, Mapping) else {},
        mechanism={
            "connectivity": conn,
            "widths": [{"k": w["k"], "independence": w["independence"]} for w in per_width],
            "error": mech_err,
        },
    )


def _md_escape(text: str) -> str:
    """Escape the two characters that break a markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(report: VerificationReport) -> str:
    """Render a `VerificationReport` as markdown: nine rows, then caveats.

    The table always has nine rows. A row whose inputs were missing says
    "not computed" in the same column a finding would occupy, so scanning the
    column shows what is known and what is not — an omitted row would not.

    Caveats are printed AFTER the table, on purpose, because that is what a
    reader of a verdict skips and most needs. Provenance is last, so a disputed
    figure can be traced to the attribute it came from without re-running
    anything.

    Misleads: markdown is presentation. This function cannot make a
    "not computed" line into a finding, and it does not try to soften a negative
    verdict — a null result read as a result is the product.
    """
    if not isinstance(report, VerificationReport):
        raise TypeError(f"render_markdown expects a VerificationReport, got {type(report).__name__}")

    out: list[str] = ["# Verification report", "", f"**Hypothesis:** {report.hypothesis}", ""]
    if not report.has_mechanism:
        out += [
            "> Mechanism half not computed. The statistical verdict below stands on "
            "its own; the four mechanism lines state why they are empty.",
            "",
        ]
    out += ["| | |", "| --- | --- |"]
    for ln in report.lines:
        marker = "" if ln.computed else " "
        out.append(f"| **{ln.label}**{marker} | {_md_escape(ln.value)} |")
    out.append("")

    uncomputed = report.uncomputed
    if uncomputed:
        out += [f"**Not computed:** {', '.join(uncomputed)}.", ""]

    caveat_items: list[tuple[str, str]] = []
    for ln in report.lines:
        for c in ln.caveats:
            caveat_items.append((ln.label, c))
    for c in report.caveats:
        caveat_items.append(("REPORT", c))
    if caveat_items:
        out += ["## Caveats", ""]
        out += [f"- **{lbl}** — {txt}" for lbl, txt in caveat_items]
        out.append("")

    out += ["## Provenance", ""]
    out += [f"- **{ln.label}** — {ln.provenance}" for ln in report.lines]
    out.append("")
    return "\n".join(out)
