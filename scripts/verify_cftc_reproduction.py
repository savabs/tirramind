"""Acceptance test for the generalised verification subsystem: did it change the answer?

WHAT THIS IS FOR

    ``scripts/cftc_event_study.py`` is a correct event study welded to CFTC. It
    produced ``docs/publications/cot_null_result.md``. ``agent/verify/`` is that
    script with CFTC cut out of it. This module expresses the CFTC study as 54
    ``Hypothesis`` objects, runs them through the generalised path, and compares the
    result against BOTH of the things it is supposed to agree with:

        A.  the REFERENCE IMPLEMENTATION, executed as a subprocess right now against
            the same database, its stdout table parsed. Not re-implemented here --
            re-implementing the thing you are checking against is how four tests in
            this repo came to assert the bug they were meant to catch.
        B.  the PUBLISHED TABLE, parsed out of the markdown of the paper.

    Comparison A is the real acceptance test: same data, same instant, two code
    paths. A divergence there is a bug in the generalisation.

    Comparison B is a DIFFERENT question, and conflating the two is the trap this
    module exists to avoid. The paper was cut at commit 2597e8b on 2026-08-29. The
    live database has been collected into since. B can therefore fail for an
    innocent reason (more data) and for a guilty one (changed arithmetic), and only
    A distinguishes them. So B is reported as a data-drift ledger -- per cell, how
    many events and baseline points were added -- and never as a pass/fail on the
    generalisation.

WHAT A DIVERGENCE MEANS

    Nothing here adjusts anything to force agreement. ``divergences()`` returns the
    list; ``main()`` prints it; ``tests/test_verify_reproduction.py`` fails on it.
    A generalisation that silently changes a published result is worse than no
    generalisation, and the exit code is 1 when comparison A does not hold.

WHERE THIS MODULE MISLEADS -- read before quoting it

    *   **It checks agreement, not correctness.** If the reference and the
        generalisation share a mistake, this module prints a clean bill of health.
        It bounds the RISK OF THE REFACTOR, which is all an acceptance test can do.
    *   **Bootstrap p-values do NOT agree bit-for-bit, and that is a measured fact,
        not a tolerance.** Both sides seed ``default_rng(7)`` and draw B=2000, but
        ``rng.choice`` resamples POSITIONS, so the p-value depends on the order of
        the event array. The reference builds that array in ``entity_links`` row
        order; the generalised path builds it in (event entity_id, target
        entity_id) order. Same estimator, different Monte Carlo realisation.
        So the comparison is split in two and only one half can fail the run:

            STRUCTURAL -- n_events, n_pop, mean event and baseline return, both hit
            rates, and the BH reject/accept decision. These are functions of the
            data alone. They must match EXACTLY. A divergence here is a bug.

            MONTE CARLO -- p_value and p_adj_bh. Checked against a band measured by
            re-running the generalised path's own bootstrap over random permutations
            of its event array. A reference p inside that band is the same estimator
            disagreeing with itself; a reference p OUTSIDE it is a changed estimator
            and fails the run.

        This is not a softer test. It is a test of the right thing: widening the
        p tolerance to 0.06 to make the run pass would also have accepted a genuinely
        changed estimator, which is the failure this repo has eighteen entries of.
    *   **A p-value that moves with row order is a product defect even though it is
        not a refactor defect.** ``main()`` says so in its output. At B=2000 the
        smallest non-zero two-sided p is 0.001, and BH at rank 1 of m=54 multiplies
        that granularity to 0.054 -- coarser than alpha. The headline cell prints
        p_BH = 0.018 from the reference and 0.036 from the generalised path off a
        one-draw difference. Neither number should be quoted to three decimals.
    *   **The paper's section-6 clustered p-values are NOT compared.** The paper used
        B=10,000 over as-of weeks; ``agent.verify.study`` uses B=2000 over distinct
        event timestamps. Two different estimators of two different quantities. The
        cluster counts are printed for the record and excluded from every verdict.
    *   **The "51 tested" figure is a property of the data, not of the code.** Three
        cells were untestable in the paper because they had n_events < 3 at the time.
        Cells cross that floor as data arrives. A run that tests 54 is not a bug.
    *   **"116 distinct weeks" is a retracted number.** Appendix A of the paper
        withdraws it: it belonged to the 251 PRE-FILTER events. The 123 labelled
        events of the headline cell fall on 67 distinct as-of weeks. This module
        checks against 67 and states the retraction rather than reproducing the
        error. See ``PUBLISHED_HEADLINE``.

Read-only: the generalised path opens the database ``mode=ro`` and the reference is
run as a separate process that does the same. Nothing here writes.

Usage
-----
    .venv/bin/python scripts/verify_cftc_reproduction.py
    .venv/bin/python scripts/verify_cftc_reproduction.py --skip-reference   # B only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.verify.stats import benjamini_hochberg  # noqa: E402
from agent.verify.study import Hypothesis, verify  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DB_PATH = ".tirra_pipeline/pipeline.db"
REFERENCE_SCRIPT = "scripts/cftc_event_study.py"
PUBLICATION = "docs/publications/cot_null_result.md"

# --- the CFTC study, restated as the generalised path's parameters ------------
# Every one of these is a constant lifted from scripts/cftc_event_study.py. They are
# spelled out rather than imported because importing them would make this file agree
# with the reference by construction, which is the opposite of a reproduction.
FIELDS = (
    "mm_net",
    "open_interest",
    "swap_net",
    "pm_net",
    "conc_top4_long",
    "conc_top4_short",
    "mm_net_pct_oi",
    "mm_weekly_flow",
    "oi_change",
)
Z_THRESHOLDS = (2.0, 3.0)
HORIZONS = (1, 5, 20)
PUB_LAG_S = 3 * 86400.0  # Tuesday as-of -> Friday release
ENTRY_TOLERANCE_S = 7 * 86400.0
MIN_HISTORY = 20
MIN_TARGET_POINTS = 40
ALPHA = 0.05

# --- the published headline, transcribed from docs/publications/cot_null_result.md ---
# `distinct_weeks_of_best` is 67, NOT the 116 that Appendix A of the paper retracts:
# 116 was the pre-filter count. Transcribing 116 here would re-publish a known error.
PUBLISHED_HEADLINE = {
    "n_tested": 51,
    "n_cells": 54,
    "n_surviving_bh": 0,
    "best_field": "mm_net_pct_oi",
    "best_z": 2.0,
    "best_horizon": 20,
    "best_p": 0.002,
    "best_p_adj": 0.102,
    "best_n_events": 123,
    "distinct_weeks_of_best": 67,
    "retracted_distinct_weeks": 116,
    "date": "2026-08-29",
    "commit": "2597e8b",
}

Key = tuple[str, float, int]

# Column meanings, shared by all three sources so a comparison cannot line up the
# wrong pair of numbers.
COMPARED_COUNTS = ("n_events", "n_pop")
COMPARED_RATES = ("mean_event_ret", "mean_baseline_ret", "hit_rate_event", "hit_rate_baseline")
COMPARED_PVALS = ("p_value", "p_adj_bh")

# Functions of the data alone. Two correct implementations must agree exactly.
STRUCTURAL_QUANTITIES = (*COMPARED_COUNTS, *COMPARED_RATES, "significant_bh")
# Functions of the data AND the resampling order. Checked against a measured band.
MONTE_CARLO_QUANTITIES = COMPARED_PVALS

DEFAULT_BAND_PERMS = 32


@dataclass(frozen=True)
class Divergence:
    """One quantity on which two sources disagree, with both values and the gap."""

    source_a: str
    source_b: str
    key: Key
    quantity: str
    value_a: Any
    value_b: Any
    tolerance: float | None

    def __str__(self) -> str:
        field, zt, h = self.key
        gap = ""
        if isinstance(self.value_a, (int, float)) and isinstance(self.value_b, (int, float)):
            with np.errstate(invalid="ignore"):
                gap = f"  (delta {float(self.value_b) - float(self.value_a):+g}"
                gap += f", tol {self.tolerance:g})" if self.tolerance is not None else ")"
        return (
            f"{field}/|z|>={zt:g}/h={h}  {self.quantity}: "
            f"{self.source_a}={self.value_a!r}  {self.source_b}={self.value_b!r}{gap}"
        )


def cftc_hypotheses() -> list[Hypothesis]:
    """The 54 cells of the published study as ``Hypothesis`` objects, in a fixed order.

    ``direction="abs"`` because the published specification is two-sided on |z|. The
    paper says so explicitly and says so is a weakness: pooling "extremely long" and
    "extremely short" averages a signed effect toward zero by construction.

    ``link_types=("cftc_tracks",)`` pins the routing to the 19 contract->instrument
    links the reference iterates. Without it the generalised path would take ANY link
    to an instrument entity, which is a wider study than the one being reproduced.
    """
    out: list[Hypothesis] = []
    for field in FIELDS:
        for zt in Z_THRESHOLDS:
            for h in HORIZONS:
                out.append(
                    Hypothesis(
                        event_source="cftc",
                        event_obs_type="futures_positioning",
                        event_field=field,
                        z_threshold=zt,
                        direction="abs",
                        target_entity_type="instrument",
                        target_obs_type="instrument_daily",
                        horizon_days=h,
                        publication_lag_s=PUB_LAG_S,
                        target_value_field="close",
                        link_types=("cftc_tracks",),
                        event_aggregate="last",
                        target_aggregate="last",
                        min_history=MIN_HISTORY,
                        min_target_points=MIN_TARGET_POINTS,
                        entry_tolerance_s=ENTRY_TOLERANCE_S,
                        label=f"{field} |z|>={zt:g} h={h}d",
                    )
                )
    return out


def run_generalised(db_path: str = DB_PATH, *, as_of: float | None = None) -> dict[str, Any]:
    """Run all 54 hypotheses through ``agent.verify`` and apply one BH family.

    The BH family is all 54 cells that produced a finite p-value -- not a subset, and
    not padded with 1.0 for the untestable ones. Padding would enlarge m and flatter
    every real p; dropping tested cells would shrink m and is laundering.
    """
    rows: list[dict[str, Any]] = []
    results = []
    for h in cftc_hypotheses():
        r = verify(db_path, h, as_of=as_of)
        results.append(r)
        rows.append(
            {
                "key": (h.event_field, h.z_threshold, h.horizon_days),
                "field": h.event_field,
                "z_threshold": h.z_threshold,
                "horizon_d": h.horizon_days,
                "n_events": r.n_events,
                "n_pop": r.n_baseline,
                "mean_event_ret": r.mean_event_return,
                "mean_baseline_ret": r.baseline_mean_return,
                "edge": r.edge,
                "hit_rate_event": r.hit_rate_event,
                "hit_rate_baseline": r.hit_rate_baseline,
                "p_value": r.p_value,
                "p_value_clustered": r.p_value_clustered,
                "n_clusters": r.n_event_clusters,
                "power": r.power,
                "verdict": r.verdict,
                "reconciled": r.reconciled,
                "n_pairs_used": r.n_pairs_used,
                "n_pairs_entered": r.n_pairs_entered,
                # kept so a p-value disagreement can be audited against a permutation
                # band without re-running the whole study
                "event_returns": r.event_returns,
            }
        )

    testable = [r for r in rows if np.isfinite(r["p_value"])]
    bh = benjamini_hochberg([r["p_value"] for r in testable], alpha=ALPHA)
    for r, padj, rej in zip(testable, bh.p_adjusted, bh.rejected, strict=True):
        r["p_adj_bh"] = float(padj)
        r["significant_bh"] = bool(rej)
    for r in rows:
        r.setdefault("p_adj_bh", float("nan"))
        r.setdefault("significant_bh", False)

    return {
        "rows": {r["key"]: r for r in rows},
        "order": [r["key"] for r in rows],
        "n_cells": len(rows),
        "n_tested": bh.n_tested,
        "n_surviving_bh": bh.n_rejected,
        "results": results,
        "all_reconciled": all(r["reconciled"] for r in rows),
    }


_REF_ROW = re.compile(
    r"^(?P<field>[a-z_0-9]+)\s+"
    r"(?P<zt>\d+\.\d)\s+"
    r"(?P<h>\d+)\s+"
    r"(?P<n_ev>\d+)\s+"
    r"(?P<n_pop>\d+)\s+"
    r"(?P<mean_ev>-?\d+\.\d+)\s+"
    r"(?P<mean_base>-?\d+\.\d+)\s+"
    r"(?P<edge>-?\d+\.\d+)\s+"
    r"(?P<hit_ev>-?\d+\.\d+)%\s*"
    r"(?P<hit_base>-?\d+\.\d+)%\s*"
    r"(?P<p>-?[\d.]+|nan)\s+"
    r"(?P<p_bh>-?[\d.]+|nan)\s+"
    r"(?P<sig>True|False)\s*$"
)


def parse_reference_stdout(text: str) -> dict[str, Any]:
    """Parse the reference script's results table out of its stdout.

    The reference prints fixed-width columns with no separators, so the parse is
    anchored on the column ORDER in its f-string, and a line that does not match the
    full row shape is not silently skipped -- ``n_cells`` is compared against the
    reference's own "N / M tested" footer, and a mismatch raises. A parser that
    quietly dropped half the table would manufacture agreement.
    """
    rows: dict[Key, dict[str, Any]] = {}
    order: list[Key] = []
    for line in text.splitlines():
        m = _REF_ROW.match(line.strip())
        if not m:
            continue
        g = m.groupdict()
        key: Key = (g["field"], float(g["zt"]), int(g["h"]))
        rows[key] = {
            "key": key,
            "field": g["field"],
            "z_threshold": float(g["zt"]),
            "horizon_d": int(g["h"]),
            "n_events": int(g["n_ev"]),
            "n_pop": int(g["n_pop"]),
            # the reference prints these already multiplied by 100
            "mean_event_ret": float(g["mean_ev"]) / 100.0,
            "mean_baseline_ret": float(g["mean_base"]) / 100.0,
            "edge": float(g["edge"]) / 100.0,
            "hit_rate_event": float(g["hit_ev"]) / 100.0,
            "hit_rate_baseline": float(g["hit_base"]) / 100.0,
            "p_value": float(g["p"]),
            "p_adj_bh": float(g["p_bh"]),
            "significant_bh": g["sig"] == "True",
        }
        order.append(key)

    footer = re.search(r"Rows surviving Benjamini-Hochberg correction at alpha=0\.05: (\d+) / (\d+) tested", text)
    if footer is None:
        raise ValueError("reference stdout has no BH footer; the parse cannot be trusted")
    n_surviving, n_tested = int(footer.group(1)), int(footer.group(2))
    if len(rows) != n_tested:
        raise ValueError(
            f"parsed {len(rows)} reference rows but its own footer says {n_tested} were tested; "
            "the table parse is losing lines and its agreement would be meaningless"
        )
    return {
        "rows": rows,
        "order": order,
        "n_cells": len(rows),
        "n_tested": n_tested,
        "n_surviving_bh": n_surviving,
    }


def run_reference(db_path: str = DB_PATH, *, cwd: Path = REPO) -> dict[str, Any]:
    """Execute the reference implementation as a subprocess and parse its stdout.

    Never imported, never modified, never re-implemented. It is run with the repo's
    own interpreter so its numpy/statsmodels are the ones the paper was cut with.

    ``db_path`` is accepted for symmetry but the reference hardcodes its own path; a
    mismatch raises rather than silently comparing two different databases.
    """
    if db_path != DB_PATH:
        raise ValueError(
            f"the reference implementation hardcodes DB_PATH={DB_PATH!r} and must not be edited; "
            f"cannot run it against {db_path!r}"
        )
    exe = cwd / ".venv/bin/python"
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [str(exe if exe.exists() else sys.executable), REFERENCE_SCRIPT],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"reference script exited {proc.returncode}:\n{proc.stderr[-4000:]}")
    out = parse_reference_stdout(proc.stdout)
    out["stdout"] = proc.stdout
    return out


_PUB_ROW = re.compile(
    r"^\|\s*(?P<field>[a-z_0-9]+)\s*\|\s*(?P<zt>\d+\.\d)\s*\|\s*(?P<h>\d+)\s*\|"
    r"\s*(?P<n_ev>\d+)\s*\|\s*(?P<n_pop>\d+)\s*\|"
    r"\s*(?P<mean_ev>-?[\d.]+)\s*\|\s*(?P<mean_base>-?[\d.]+)\s*\|\s*(?P<edge>-?[\d.]+)\s*\|"
    r"\s*(?P<hit_ev>-?[\d.]+)%\s*\|\s*(?P<hit_base>-?[\d.]+)%\s*\|"
    r"\s*(?P<p>[\d.]+|—)\s*\|\s*(?P<p_bh>[\d.]+|—)\s*\|\s*(?P<sig>[^|]*?)\s*\|\s*$"
)


def parse_publication(path: Path | None = None) -> dict[str, Any]:
    """Parse the section-5 results table of the paper.

    The paper is the record of what was claimed in public; it is read, not trusted
    to memory. ``n_cells`` is checked against the 54 the paper says it reports, so a
    regex that silently stops matching cannot turn "we compared 54 cells" into "we
    compared 6 and they were fine".
    """
    md = (path or (REPO / PUBLICATION)).read_text()
    # Section 5 only: section 6's table has different columns and would mis-parse.
    start = md.index("## 5. Results")
    end = md.index("## 6.", start)
    rows: dict[Key, dict[str, Any]] = {}
    order: list[Key] = []
    for line in md[start:end].splitlines():
        m = _PUB_ROW.match(line)
        if not m:
            continue
        g = m.groupdict()
        key: Key = (g["field"], float(g["zt"]), int(g["h"]))
        testable = g["p"] != "—"
        rows[key] = {
            "key": key,
            "field": g["field"],
            "z_threshold": float(g["zt"]),
            "horizon_d": int(g["h"]),
            "n_events": int(g["n_ev"]),
            "n_pop": int(g["n_pop"]),
            "mean_event_ret": float(g["mean_ev"]) / 100.0,
            "mean_baseline_ret": float(g["mean_base"]) / 100.0,
            "edge": float(g["edge"]) / 100.0,
            "hit_rate_event": float(g["hit_ev"]) / 100.0,
            "hit_rate_baseline": float(g["hit_base"]) / 100.0,
            "p_value": float(g["p"]) if testable else float("nan"),
            "p_adj_bh": float(g["p_bh"]) if testable else float("nan"),
            "significant_bh": g["sig"].strip().lower() == "yes",
            "testable": testable,
        }
        order.append(key)
    if len(rows) != PUBLISHED_HEADLINE["n_cells"]:
        raise ValueError(
            f"parsed {len(rows)} cells from {PUBLICATION} but the paper reports "
            f"{PUBLISHED_HEADLINE['n_cells']}; the table parse is incomplete"
        )
    return {
        "rows": rows,
        "order": order,
        "n_cells": len(rows),
        "n_tested": sum(1 for r in rows.values() if r["testable"]),
        "n_surviving_bh": sum(1 for r in rows.values() if r["significant_bh"]),
    }


def _eq(a: Any, b: Any, tol: float | None) -> bool:
    """Equality that treats NaN == NaN as agreement: both sides said "not computable"."""
    if isinstance(a, bool) or isinstance(b, bool) or tol is None:
        return a == b
    fa, fb = float(a), float(b)
    if not np.isfinite(fa) or not np.isfinite(fb):
        return np.isnan(fa) and np.isnan(fb)
    return abs(fa - fb) <= tol


def divergences(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    name_a: str,
    name_b: str,
    quantities: tuple[str, ...] = (*STRUCTURAL_QUANTITIES, *MONTE_CARLO_QUANTITIES),
    count_tol: int = 0,
    rate_tol: float = 5e-5,
    p_tol: float = 5e-4,
) -> list[Divergence]:
    """Every cell-level disagreement between two sources, and the missing cells too.

    Default tolerances are the PRINT PRECISION of the sources being compared, not a
    fudge factor: the reference prints rates to 3 decimal places of a percent (so
    5e-5 as a fraction is half its last digit) and p-values to 3 decimals (5e-4).
    Counts must match exactly. Loosening these to make a run pass would be the whole
    failure this module exists to prevent -- which is why a Monte Carlo disagreement
    is routed to ``monte_carlo_audit`` instead of being absorbed into ``p_tol``.

    ``quantities`` selects which columns to compare; pass ``STRUCTURAL_QUANTITIES``
    for the half that must match exactly.
    """
    tol_of: dict[str, float | None] = {
        **{q: float(count_tol) for q in COMPARED_COUNTS},
        **{q: rate_tol for q in COMPARED_RATES},
        **{q: p_tol for q in COMPARED_PVALS},
        "significant_bh": None,
    }
    out: list[Divergence] = []
    ra, rb = a["rows"], b["rows"]
    for key in sorted(set(ra) | set(rb)):
        if key not in ra or key not in rb:
            present, absent = (name_a, name_b) if key in ra else (name_b, name_a)
            out.append(Divergence(name_a, name_b, key, "cell present", present, f"absent from {absent}", None))
            continue
        for quantity in quantities:
            va, vb = ra[key].get(quantity), rb[key].get(quantity)
            if not _eq(va, vb, tol_of[quantity]):
                out.append(Divergence(name_a, name_b, key, quantity, va, vb, tol_of[quantity]))
    return out


BAND_SIGMA = 4.0


@dataclass(frozen=True)
class BandCheck:
    """Whether the reference's p-value is a plausible draw from the generalised bootstrap's own spread.

    The criterion is ``|p_ref - band_mean| <= BAND_SIGMA * band_sd``, NOT "inside
    [min, max]". A min/max band widens monotonically with ``n_perm`` -- at 16
    permutations one cell fell outside it and at 200 the same cell fell inside --
    so a min/max verdict is a statement about how long the audit ran. Mean and sd
    converge instead, which is the property a pass/fail needs.

    ``BAND_SIGMA = 4`` is a stated choice, not a fitted one: two-sided, it excludes
    about 6e-5 of a normal per cell, so across ~50 banded cells the chance of
    flagging a cell that is merely noisy is under 0.5%. Raising it to make a run
    green would be the failure this module exists to catch, so ``sigma`` is printed
    for every cell and the maximum is printed for the run -- the margin is visible
    rather than asserted.
    """

    key: Key
    n_events: int
    p_reference: float
    p_generalised: float
    band_low: float
    band_high: float
    band_mean: float
    band_sd: float
    n_perm: int

    @property
    def sigma(self) -> float:
        """How many band standard deviations the reference p sits from the band mean."""
        if self.band_sd <= 0:
            return 0.0 if self.p_reference == self.band_mean else float("inf")
        return abs(self.p_reference - self.band_mean) / self.band_sd

    @property
    def inside(self) -> bool:
        return self.sigma <= BAND_SIGMA

    def __str__(self) -> str:
        f, z, h = self.key
        return (
            f"{f}/|z|>={z:g}/h={h}  n_ev={self.n_events:<5} p_ref={self.p_reference:.3f} "
            f"p_new={self.p_generalised:.3f}  band mean={self.band_mean:.4f} sd={self.band_sd:.4f} "
            f"[{self.band_low:.3f},{self.band_high:.3f}]  {self.sigma:.2f} sigma  "
            f"{'OK (Monte Carlo)' if self.inside else '*** OUTSIDE (estimator changed) ***'}"
        )


def permutation_band(
    event_returns: np.ndarray,
    baseline_mean: float,
    *,
    n_perm: int = DEFAULT_BAND_PERMS,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """(low, high, mean, sd) of the generalised bootstrap p over permutations of its input.

    The i.i.d. bootstrap resamples POSITIONS, so p is a function of the event array's
    order as well as its contents. Permuting the array and re-running the SAME
    function measures how much of a disagreement the order alone can explain. This
    calls ``agent.verify.study._bootstrap_p_iid`` rather than reimplementing it: a
    band computed from a copy of the code would not bound the code being tested.

    Misleads: this is a spread, not a confidence interval. It says "the estimator
    disagrees with itself by this much"; it says nothing about whether the estimator
    is right. Widen ``n_perm`` and the band widens -- an extreme value is always
    reachable -- so a band check is evidence about a magnitude, not a proof.
    """
    from agent.verify.study import _bootstrap_p_iid  # noqa: PLC0415 - private on purpose

    ev = np.asarray(event_returns, dtype=float)
    rng = np.random.default_rng(seed)
    ps = np.array([_bootstrap_p_iid(ev[rng.permutation(len(ev))], baseline_mean) for _ in range(n_perm)])
    return float(ps.min()), float(ps.max()), float(ps.mean()), float(ps.std())


def monte_carlo_audit(
    ref: dict[str, Any],
    new: dict[str, Any],
    *,
    n_perm: int = DEFAULT_BAND_PERMS,
    p_tol: float = 5e-4,
) -> list[BandCheck]:
    """For every cell whose p-value disagrees, is the disagreement explainable by order?

    Only cells that actually disagree are banded, so a run where the p-values happen
    to match costs nothing. A cell whose reference p falls outside the band is a
    changed estimator and must fail the run; ``main()`` treats it that way.
    """
    checks: list[BandCheck] = []
    for key in new["order"]:
        if key not in ref["rows"]:
            continue
        pr, pn = ref["rows"][key]["p_value"], new["rows"][key]["p_value"]
        if _eq(pr, pn, p_tol):
            continue
        row = new["rows"][key]
        ev = np.asarray(row["event_returns"], dtype=float)
        if len(ev) < 3:
            continue
        lo, hi, mean, sd = permutation_band(ev, row["mean_baseline_ret"], n_perm=n_perm)
        checks.append(
            BandCheck(
                key=key,
                n_events=row["n_events"],
                p_reference=pr,
                p_generalised=pn,
                band_low=lo,
                band_high=hi,
                band_mean=mean,
                band_sd=sd,
                n_perm=n_perm,
            )
        )
    return checks


def drift_ledger(new: dict[str, Any], pub: dict[str, Any]) -> list[dict[str, Any]]:
    """Per cell, how much data arrived since the paper was cut.

    This is the innocent explanation for a B-comparison failure, stated as numbers
    so it can be checked instead of asserted. If ``d_pop`` is 0 everywhere and the
    cells still disagree, the explanation is NOT data drift and something is wrong.
    """
    out = []
    for key in pub["order"]:
        if key not in new["rows"]:
            continue
        p, n = pub["rows"][key], new["rows"][key]
        out.append(
            {
                "key": key,
                "pub_n_events": p["n_events"],
                "new_n_events": n["n_events"],
                "d_events": n["n_events"] - p["n_events"],
                "pub_n_pop": p["n_pop"],
                "new_n_pop": n["n_pop"],
                "d_pop": n["n_pop"] - p["n_pop"],
                "pub_p": p["p_value"],
                "new_p": n["p_value"],
            }
        )
    return out


def _fmt_table(rows: list[dict[str, Any]], order: list[Key]) -> str:
    hdr = (
        f"{'field':<17}{'|z|':<5}{'h':<4}{'n_ev':<6}{'n_pop':<7}{'clust':<7}"
        f"{'mean_ev%':<10}{'mean_base%':<12}{'p':<8}{'p_bh':<8}{'p_clust':<9}{'verdict':<22}"
    )
    lines = [hdr, "-" * len(hdr)]
    by_p = sorted(order, key=lambda k: rows[k]["p_value"] if np.isfinite(rows[k]["p_value"]) else 1.0)
    for k in by_p:
        r = rows[k]
        lines.append(
            f"{r['field']:<17}{r['z_threshold']:<5.1f}{r['horizon_d']:<4}{r['n_events']:<6}{r['n_pop']:<7}"
            f"{r.get('n_clusters', 0):<7}{r['mean_event_ret'] * 100:<10.3f}{r['mean_baseline_ret'] * 100:<12.3f}"
            f"{r['p_value']:<8.3f}{r['p_adj_bh']:<8.3f}{r.get('p_value_clustered', float('nan')):<9.4f}"
            f"{r.get('verdict', ''):<22}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--skip-reference", action="store_true", help="skip comparison A (subprocess, ~6s)")
    ap.add_argument(
        "--band-perms",
        type=int,
        default=DEFAULT_BAND_PERMS,
        help="permutations per diverging p-value (0 disables the audit, which then cannot clear a divergence)",
    )
    args = ap.parse_args(argv)

    print("=" * 108)
    print("CFTC REPRODUCTION -- does agent/verify/ give the same answer as scripts/cftc_event_study.py?")
    print("=" * 108)

    new = run_generalised(args.db)
    print(
        f"\ngeneralised path: {new['n_cells']} cells, {new['n_tested']} testable, "
        f"{new['n_surviving_bh']} surviving BH at alpha={ALPHA}"
    )
    first = new["rows"][new["order"][0]]
    print(f"pairs: {first['n_pairs_used']} used / {first['n_pairs_entered']} entered")
    print(f"every cell reconciled (counts add up): {new['all_reconciled']}")
    print("\n--- GENERALISED PATH (agent/verify/study.verify), sorted by uncorrected p ---")
    print(_fmt_table(new["rows"], new["order"]))

    exit_code = 0

    # ---- comparison A: the acceptance test --------------------------------------
    if args.skip_reference:
        print("\n[A] SKIPPED by --skip-reference. Nothing below establishes the generalisation is sound.")
    else:
        ref = run_reference(args.db)
        print("\n" + "=" * 108)
        print("[A] GENERALISED vs REFERENCE -- same database, same minute, two code paths")
        print("=" * 108)
        print(f"reference: {ref['n_cells']} cells, {ref['n_tested']} testable, {ref['n_surviving_bh']} surviving BH")
        print(f"generalised: {new['n_cells']} cells, {new['n_tested']} testable, {new['n_surviving_bh']} surviving BH")
        # --- A1: the half that must match exactly ----------------------------
        struct = divergences(ref, new, name_a="reference", name_b="generalised", quantities=STRUCTURAL_QUANTITIES)
        print(f"\n[A1] STRUCTURAL (functions of the data alone: {', '.join(STRUCTURAL_QUANTITIES)})")
        if struct:
            exit_code = 1
            print(f"  *** {len(struct)} DIVERGENCE(S). The generalisation changed the arithmetic. ***")
            for d in struct:
                print(f"    {d}")
        else:
            print(f"  All {new['n_cells']} cells agree exactly. The data path is unchanged.")
        for label, k in (("n_cells", "n_cells"), ("n_tested", "n_tested"), ("BH survivors", "n_surviving_bh")):
            if ref[k] != new[k]:
                exit_code = 1
                print(f"  *** {label}: reference={ref[k]} generalised={new[k]} ***")

        # --- A2: the half that is a Monte Carlo realisation -------------------
        mc = divergences(ref, new, name_a="reference", name_b="generalised", quantities=MONTE_CARLO_QUANTITIES)
        n_p = sum(1 for d in mc if d.quantity == "p_value")
        n_padj = sum(1 for d in mc if d.quantity == "p_adj_bh")
        print(
            f"\n[A2] MONTE CARLO: {n_p}/{new['n_cells']} cells disagree on the uncorrected p, "
            f"{n_padj} on the BH-adjusted p"
        )
        print("  Cause: rng.choice resamples POSITIONS, so the bootstrap p depends on the order of the")
        print("  event array. The reference orders by entity_links row; the generalised path by")
        print("  (event entity_id, target entity_id). Same estimator, different realisation.")
        if args.band_perms <= 0:
            if mc:
                exit_code = 1
                print("  *** --band-perms=0: the audit that would clear these was not run, so they stand. ***")
        elif mc:
            checks = monte_carlo_audit(ref, new, n_perm=args.band_perms)
            outside = [c for c in checks if not c.inside]
            print(
                f"  Banded {len(checks)} cells over {args.band_perms} permutations of the generalised"
                f" path's own event array:"
            )
            for c in sorted(checks, key=lambda c: -c.sigma)[:10]:
                print(f"    {c}")
            if len(checks) > 10:
                print(f"    ... {len(checks) - 10} more, all inside" if not outside else "    ...")
            sds = [c.band_sd for c in checks]
            print(
                f"  median band sd {float(np.median(sds)):.4f}; "
                f"max |p_new - p_ref| {max(abs(c.p_generalised - c.p_reference) for c in checks):.3f}; "
                f"worst {max(c.sigma for c in checks):.2f} sigma of {BAND_SIGMA:g} allowed"
            )
            if outside:
                exit_code = 1
                print(f"  *** {len(outside)} cell(s) OUTSIDE the band. That is a changed estimator, not noise. ***")
            else:
                print(f"  All {len(checks)} reference p-values fall inside the band: the disagreement is")
                print("  reordering, not a different test.")
        print("\n  PRODUCT DEFECT, separate from the refactor: a p-value that moves with join order is not")
        print("  reproducible for a customer. At B=2000 the smallest non-zero two-sided p is 0.001, and BH")
        print(
            f"  at rank 1 of m={new['n_tested']} multiplies that granularity to "
            f"{0.001 * new['n_tested']:.3f} -- coarser than alpha={ALPHA}."
        )
        print("  Fix in agent/verify/study.py, not here: sort the event array canonically before")
        print("  resampling, or raise B until the last printed digit is stable. Do not quote p_BH to 3 dp.")

    # ---- comparison B: the paper, which is a different question ------------------
    pub = parse_publication()
    print("\n" + "=" * 108)
    print(
        f"[B] TODAY vs THE PAPER ({PUBLICATION}, {PUBLISHED_HEADLINE['date']}, commit {PUBLISHED_HEADLINE['commit']})"
    )
    print("=" * 108)
    print("This is NOT a test of the generalisation. The database has been collected into since the")
    print("paper was cut, so BOTH code paths now differ from it. [A] above is what isolates the refactor.")
    print(f"\n{'':<22}{'paper':<12}{'today':<12}")
    for label, k in (("cells reported", "n_cells"), ("cells testable", "n_tested"), ("BH survivors", "n_surviving_bh")):
        print(f"{label:<22}{pub[k]:<12}{new[k]:<12}")

    best_key: Key = (
        str(PUBLISHED_HEADLINE["best_field"]),
        float(PUBLISHED_HEADLINE["best_z"]),
        int(PUBLISHED_HEADLINE["best_horizon"]),
    )
    bn = new["rows"][best_key]
    print(f"\nheadline cell {best_key[0]} |z|>={best_key[1]:g} h={best_key[2]}d:")
    print(f"{'':<22}{'paper':<12}{'today':<12}")
    print(f"{'n_events':<22}{PUBLISHED_HEADLINE['best_n_events']:<12}{bn['n_events']:<12}")
    print(f"{'distinct clusters':<22}{PUBLISHED_HEADLINE['distinct_weeks_of_best']:<12}{bn['n_clusters']:<12}")
    print(f"{'uncorrected p':<22}{PUBLISHED_HEADLINE['best_p']:<12.3f}{bn['p_value']:<12.3f}")
    print(f"{'BH-adjusted p':<22}{PUBLISHED_HEADLINE['best_p_adj']:<12.3f}{bn['p_adj_bh']:<12.3f}")
    print(
        f"\nNOTE: '{PUBLISHED_HEADLINE['retracted_distinct_weeks']} distinct weeks' is RETRACTED by Appendix A of"
        f" the paper\n      (it was the pre-filter count). The published figure for this cell is"
        f" {PUBLISHED_HEADLINE['distinct_weeks_of_best']} distinct as-of weeks."
    )

    ledger = drift_ledger(new, pub)
    added_pop = [r for r in ledger if r["d_pop"] != 0]
    added_ev = [r for r in ledger if r["d_events"] != 0]
    print(
        f"\ndata drift since the paper: {len(added_pop)}/{len(ledger)} cells gained baseline points, "
        f"{len(added_ev)}/{len(ledger)} gained events"
    )
    if added_pop:
        d = sorted(r["d_pop"] for r in added_pop)
        print(f"  baseline delta range: {d[0]:+d} .. {d[-1]:+d}")
    if added_ev:
        d = sorted(r["d_events"] for r in added_ev)
        print(f"  event delta range:    {d[0]:+d} .. {d[-1]:+d}")
    div_b = divergences(pub, new, name_a="paper", name_b="today")
    print(f"\ncell-level differences vs the paper: {len(div_b)} (expected non-zero: the data grew)")
    if not added_pop and not added_ev and div_b:
        exit_code = 1
        print("  *** NO data arrived yet the cells differ. Drift does not explain this. ***")

    # ---- what the current data actually says, which is not what the paper says ----
    survivors = [new["rows"][k] for k in new["order"] if new["rows"][k]["significant_bh"]]
    print("\n" + "-" * 108)
    print(
        f"SUBSTANTIVE: the published answer was 0 of 51 surviving BH. On today's data it is "
        f"{len(survivors)} of {new['n_tested']}."
    )
    print("-" * 108)
    if survivors:
        print("The paper's null is NOT the current answer, and nobody has re-run it since 2026-08-29.")
        print(f"{'field':<17}{'|z|':<5}{'h':<4}{'n_ev':<6}{'clust':<7}{'p':<8}{'p_bh':<8}{'p_clust':<9}{'power':<8}")
        for r in survivors:
            print(
                f"{r['field']:<17}{r['z_threshold']:<5.1f}{r['horizon_d']:<4}{r['n_events']:<6}"
                f"{r['n_clusters']:<7}{r['p_value']:<8.3f}{r['p_adj_bh']:<8.3f}"
                f"{r['p_value_clustered']:<9.4f}{r['power']:<8.1%}"
            )
        thin = [r for r in survivors if r["n_events"] < 30]
        if thin:
            print(f"\nRead these before treating any of them as a discovery. {len(thin)} of {len(survivors)} have")
            print("n_events < 30, and the paper's own Section 5 warns about exactly that:")
            for r in thin:
                print(
                    f"  {r['field']} |z|>={r['z_threshold']:g} h={r['horizon_d']}d has n_events="
                    f"{r['n_events']} on {r['n_clusters']} distinct timestamps, at "
                    f"{r['power']:.1%} power."
                )
            tiny = [r for r in thin if r["n_events"] <= 5]
            for r in tiny:
                print(
                    f"  *** {r['field']} |z|>={r['z_threshold']:g} h={r['horizon_d']}d has n_events="
                    f"{r['n_events']}: an i.i.d. bootstrap over {r['n_events']} points can take at most "
                    f"{r['n_events'] ** r['n_events']} distinct values."
                )
                print("      Its p is a resampling artefact of a handful of observations, not evidence.")
        print("\nThis module does not adjudicate that. It reports that the answer moved and that the cells")
        print("which moved it are thin, so the re-run is a decision someone has to make deliberately.")

    print("\n" + "=" * 108)
    if exit_code == 0:
        print("VERDICT: the generalisation reproduces the reference on every quantity that is a function")
        print("of the data. Its p-values differ by a measured Monte Carlo reordering and no more.")
    else:
        print("VERDICT: DIVERGENCE -- see [A1]/[A2] above. Do not ship this generalisation.")
    print("=" * 108)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
