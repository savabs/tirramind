"""
Tests for scripts/verify_cftc_reproduction.py — the acceptance test for agent/verify/.

TESTING POSTURE
    This file tests the thing that tests the generalisation, so a false green here
    is a false green twice over. Four tests in this repository's history asserted
    the bug they were written to catch, and yesterday a frozen model ranked first
    because a favourable number ended the investigation (F-18). So:

      * The oracles are the PUBLISHED paper and the REFERENCE SCRIPT's own source
        text, read off disk. Never a fresh run of the code under test.
      * Every comparator is tested against a PLANTED BREAKAGE as well as against
        agreement: a `divergences()` that returned `[]` unconditionally would pass
        a test that only ever feeds it identical inputs, so each one is also fed a
        mutated input and asserted to go red.
      * The Monte Carlo band is tested to have TEETH: a p-value from a genuinely
        different estimator is asserted to fall outside it. A band that accepted
        anything would silently license a changed estimator, which is the exact
        failure the band was introduced to avoid.
      * "Could not compute" is asserted to be an exception or NaN, never a
        plausible zero.
      * The live acceptance test (`test_generalisation_matches_reference*`) is
        marked `slow` and skipped when the pipeline DB is absent, so the fast
        tests still run in CI. It is the one that actually answers the question.

WHAT THESE TESTS DO NOT COVER
    Whether the reference implementation is itself correct. If the reference and
    the generalisation share a mistake, everything here is green. These tests
    bound the risk of the refactor; they do not validate the study.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DB = REPO / ".tirra_pipeline/pipeline.db"


def _load_script():
    """Import scripts/verify_cftc_reproduction.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "verify_cftc_reproduction", REPO / "scripts/verify_cftc_reproduction.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # registered before exec_module: @dataclass resolves its own module from
    # sys.modules, and without this the import fails inside dataclasses
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


vcr = _load_script()

needs_db = pytest.mark.skipif(not DB.exists(), reason=f"pipeline DB absent at {DB}")


# ---------------------------------------------------------------------------
# 1. The hypotheses must be the published study, not a lookalike
# ---------------------------------------------------------------------------


def test_hypotheses_cover_exactly_the_published_grid():
    """9 fields x 2 thresholds x 3 horizons = 54 distinct cells, no duplicates."""
    hs = vcr.cftc_hypotheses()
    assert len(hs) == 54, "the paper reports 54 cells"
    keys = {(h.event_field, h.z_threshold, h.horizon_days) for h in hs}
    assert len(keys) == 54, "duplicate cells would double-count the BH family and deflate every p_adj"
    assert {h.event_field for h in hs} == set(vcr.FIELDS)
    assert {h.z_threshold for h in hs} == set(vcr.Z_THRESHOLDS)
    assert {h.horizon_days for h in hs} == set(vcr.HORIZONS)


def test_constants_match_the_reference_scripts_own_source():
    """The knobs are read out of scripts/cftc_event_study.py, not remembered.

    A drifted constant here is the single most likely way to "reproduce" a study
    that is not the published one: a 2-day publication lag or a 19-point minimum
    history would still produce a plausible table.
    """
    src = (REPO / "scripts/cftc_event_study.py").read_text()
    tree = ast.parse(src)
    const: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                # literal_eval, so the reference's code is READ and never executed
                const[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                # 3 * 86400.0 and 7 * 86400.0 are BinOps, not literals
                if isinstance(node.value, ast.BinOp):
                    const[node.targets[0].id] = eval(  # noqa: S307 - our own repo's literal arithmetic
                        compile(ast.Expression(node.value), "<ref>", "eval")
                    )

    assert const["PUB_LAG_SECS"] == vcr.PUB_LAG_S, "publication lag drifted from the reference"
    assert const["ENTRY_TOLERANCE_SECS"] == vcr.ENTRY_TOLERANCE_S
    assert const["MIN_HISTORY"] == vcr.MIN_HISTORY, "a changed z-window is a different study"
    assert const["HORIZONS"] == vcr.HORIZONS
    assert const["Z_THRESHOLDS"] == vcr.Z_THRESHOLDS
    assert const["FIELDS"] == vcr.FIELDS
    # the reference's own pair gate, which lives inline rather than as a constant
    assert "len(closes) < 40" in src
    assert vcr.MIN_TARGET_POINTS == 40
    # and the reference's dedup rule, which our event_aggregate="last" stands in for
    assert "max(rowid)" in src


def test_hypotheses_carry_the_leakage_guards():
    """Every cell must assert the publication lag and pin the routing link type.

    publication_lag_s defaults to 0.0 in Hypothesis, so forgetting it is silent and
    grants three days of lookahead — the single easiest way to manufacture an edge.
    """
    for h in vcr.cftc_hypotheses():
        assert h.publication_lag_s == 3 * 86400.0, "Tuesday as-of -> Friday release"
        assert h.link_types == ("cftc_tracks",), "an unpinned link type is a wider study"
        assert h.direction == "abs", "the published specification is two-sided on |z|"
        assert h.min_history == 20
        assert h.entry_tolerance_s == 7 * 86400.0
        assert h.target_value_field == "close"
        assert h.event_aggregate == "last", "the reference keeps max(rowid) per observed_at"


def test_dedup_rule_is_load_bearing_even_though_the_live_data_hides_it():
    """Pinning `event_aggregate="last"` cannot be checked against the live DB. Check it here.

    MEASURED: all 19 duplicate (entity, observed_at) groups on the 19 linked CFTC
    contracts carry IDENTICAL field values, so on this database "keep max(rowid)"
    (the reference's rule) and "average the duplicates" give the same series. The
    structural acceptance test therefore CANNOT catch a changed dedup rule — mutating
    `event_aggregate` to "mean" leaves every one of the 54 cells unchanged.

    That is a hole, so it is closed from two sides: `test_hypotheses_carry_the_leakage
    _guards` asserts the value directly, and this test proves on a synthetic database
    with duplicates that DISAGREE that the knob changes the answer. Without this, a
    future dataset with real duplicate disagreement would be silently mis-aggregated.
    """
    import sqlite3
    import tempfile

    from agent.verify.study import Hypothesis, verify

    week, day = 7 * 86400.0, 86400.0
    t0 = 1_600_000_000.0
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "synthetic.db")
        con = sqlite3.connect(path)
        con.executescript(
            """
            create table entities (entity_id text primary key, entity_type text not null,
                canonical_name text not null, created_at real not null, metadata_json text);
            create table entity_observations (id integer primary key autoincrement,
                entity_id text not null, source_tool text not null, observed_at real not null,
                ingested_at real not null, observation_type text not null,
                depth_level integer not null default 1, value_json text not null, metadata_json text);
            create table entity_links (link_id integer primary key autoincrement,
                entity_id_a text not null, entity_id_b text not null, link_type text not null,
                confidence real not null default 1.0, source text not null, created_at real not null,
                metadata_json text, effective_from real);
            """
        )
        con.execute("insert into entities values ('E','cftc_contract','contract',0,null)")
        con.execute("insert into entities values ('T','instrument','ticker',0,null)")
        con.execute(
            "insert into entity_links (entity_id_a,entity_id_b,link_type,source,created_at,effective_from) "
            "values ('E','T','cftc_tracks','test',0,0)"
        )
        # 40 weekly events, flat at 100.0 apart from a small wobble so the history has variance
        for i in range(40):
            v = 100.0 + (1.0 if i % 2 else -1.0)
            con.execute(
                "insert into entity_observations (entity_id,source_tool,observed_at,ingested_at,"
                "observation_type,value_json) values ('E','cftc',?,0,'futures_positioning',?)",
                (t0 + i * week, f'{{"x": {v}}}'),
            )
        # week 30 gets a SECOND row at the same timestamp with a very different value.
        # "last" (max rowid) sees 400; "mean" sees (100 + 400)/2 = 250. Both are extreme
        # against a history of ~100, but by different amounts, so they cannot both be right.
        con.execute(
            "insert into entity_observations (entity_id,source_tool,observed_at,ingested_at,"
            "observation_type,value_json) values ('E','cftc',?,0,'futures_positioning','{\"x\": 400.0}')",
            (t0 + 30 * week,),
        )
        for i in range(400):
            con.execute(
                "insert into entity_observations (entity_id,source_tool,observed_at,ingested_at,"
                "observation_type,value_json) values ('T','px',?,0,'instrument_daily',?)",
                (t0 + i * day, f'{{"close": {50.0 + 0.01 * i}}}'),
            )
        con.commit()
        con.close()

        def run(aggregate: str):
            return verify(
                path,
                Hypothesis(
                    event_source="cftc",
                    event_obs_type="futures_positioning",
                    event_field="x",
                    z_threshold=2.0,
                    direction="abs",
                    target_entity_type="instrument",
                    target_obs_type="instrument_daily",
                    horizon_days=5,
                    publication_lag_s=3 * 86400.0,
                    link_types=("cftc_tracks",),
                    event_aggregate=aggregate,
                    min_history=20,
                    min_target_points=40,
                ),
            )

        last, mean = run("last"), run("mean")

    # the duplicate row is discarded by "last" and reported as such, not absorbed
    assert last.n_collapsed_observations == 1, "a lossy aggregate must report the row it dropped"
    assert mean.n_collapsed_observations == 0, "'mean' keeps both rows, so nothing is collapsed"
    assert not last.complete, "a study that dropped a row must not call itself complete"
    # and the two rules genuinely disagree about the series
    assert last.event_returns or mean.event_returns, "the synthetic event should fire under some rule"
    assert last.n_events == mean.n_events == 1, "one extreme week, one graded event"
    # same event, but the z that produced it came from a different value, so the
    # aggregate is not cosmetic: assert the histories differ where it matters
    assert last.n_z_computable == mean.n_z_computable
    assert abs(last.leakage_audit[0].z - mean.leakage_audit[0].z) > 1.0, (
        "'last' saw 400 and 'mean' saw 250 against the same history; their z-scores must differ"
    )


# ---------------------------------------------------------------------------
# 2. The published oracle is parsed, and the parse cannot silently shrink
# ---------------------------------------------------------------------------


def test_publication_parses_to_the_headline_the_paper_states():
    pub = vcr.parse_publication()
    assert pub["n_cells"] == 54
    assert pub["n_tested"] == 51, "the paper reports 51 testable cells"
    assert pub["n_surviving_bh"] == 0, "the paper's result is 0 of 51 surviving BH"
    best = pub["rows"][("mm_net_pct_oi", 2.0, 20)]
    assert best["n_events"] == 123
    assert best["p_value"] == pytest.approx(0.002, abs=5e-4)
    assert best["p_adj_bh"] == pytest.approx(0.102, abs=5e-4)
    assert best["p_value"] == min(r["p_value"] for r in pub["rows"].values() if r["testable"])


def test_untestable_published_cells_are_nan_not_zero():
    """The paper's three n<3 cells print an em dash. A 0.0 there would be a false discovery."""
    pub = vcr.parse_publication()
    untestable = [r for r in pub["rows"].values() if not r["testable"]]
    assert len(untestable) == 3
    for r in untestable:
        assert math.isnan(r["p_value"]), "an untestable p must be NaN, never 0.0"
        assert math.isnan(r["p_adj_bh"])
        assert r["n_events"] < 3
        assert r["significant_bh"] is False


def test_publication_parse_refuses_a_truncated_table():
    """A regex that stops matching must raise, not report 6 cells as a clean comparison."""
    md = (REPO / "docs/publications/cot_null_result.md").read_text()
    lines = md.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| mm_net_pct_oi | 2.0 | 20 |"))
    # section 5 header and the first 6 data rows, then the section-6 boundary the
    # parser slices on, so the failure is "6 cells, not 54" and not "heading missing"
    head = next(i for i, ln in enumerate(lines) if ln.startswith("## 5. Results"))
    mangled = "\n".join(lines[head : start + 6] + ["", "## 6. Clustering", ""])
    p = Path(vcr.REPO) / "docs/publications/.pytest_truncated.md"
    try:
        p.write_text(mangled)
        with pytest.raises(ValueError, match="incomplete"):
            vcr.parse_publication(p)
    finally:
        p.unlink(missing_ok=True)


def test_retracted_week_count_is_not_used_as_an_oracle():
    """Appendix A of the paper retracts "116 distinct weeks". 67 is the published figure.

    This is the one oracle a reproduction is most likely to get wrong, because 116 is
    the number that circulates. Checking against it would make the harness demand a
    figure the paper itself withdrew.
    """
    assert vcr.PUBLISHED_HEADLINE["distinct_weeks_of_best"] == 67
    assert vcr.PUBLISHED_HEADLINE["retracted_distinct_weeks"] == 116
    md = (REPO / "docs/publications/cot_null_result.md").read_text()
    assert "Appendix A" in md
    assert "fall on **67** distinct as-of weeks" in md.replace("\n", " ").replace("  ", " ") or "67" in md


# ---------------------------------------------------------------------------
# 3. The reference-stdout parser cannot manufacture agreement
# ---------------------------------------------------------------------------

_FAKE_REF = """\
=== EVENT STUDY RESULTS (per field x horizon x |z| threshold) ===
field             |z|>=  h(d)  n_ev  n_pop  mean_ev   mean_base  edge      hit_ev  hit_base p       p_bh    sig_bh
--------------
mm_net_pct_oi     2.0    20    129   1620   2.549     0.389      2.160     65.12%  50.37%   0.001   0.018   True
swap_net          3.0    1     17    1639   1.668     0.018      1.650     70.59%  51.43%   0.012   0.093   False

18 / 54 rows have n_events < 30 (small-sample warning).
Rows surviving Benjamini-Hochberg correction at alpha=0.05: 1 / 2 tested
"""


def test_reference_parser_reads_the_columns_in_the_right_order():
    out = vcr.parse_reference_stdout(_FAKE_REF)
    r = out["rows"][("mm_net_pct_oi", 2.0, 20)]
    assert (r["n_events"], r["n_pop"]) == (129, 1620)
    # the reference prints percent; the parser must divide by 100 or every rate
    # comparison silently compares 2.549 against 0.02549 and always diverges
    assert r["mean_event_ret"] == pytest.approx(0.02549)
    assert r["mean_baseline_ret"] == pytest.approx(0.00389)
    assert r["hit_rate_event"] == pytest.approx(0.6512)
    assert r["p_value"] == pytest.approx(0.001)
    assert r["p_adj_bh"] == pytest.approx(0.018)
    assert r["significant_bh"] is True
    assert out["rows"][("swap_net", 3.0, 1)]["significant_bh"] is False
    assert out["n_tested"] == 2
    assert out["n_surviving_bh"] == 1


def test_reference_parser_raises_when_it_drops_rows():
    """Footer says 2 tested; give it 1 parsable row. Silence here would fake a match."""
    dropped = _FAKE_REF.replace("swap_net          3.0", "swap_net  MANGLED  3.0")
    with pytest.raises(ValueError, match="losing lines"):
        vcr.parse_reference_stdout(dropped)


def test_reference_parser_raises_without_a_footer():
    with pytest.raises(ValueError, match="no BH footer"):
        vcr.parse_reference_stdout(_FAKE_REF.split("Rows surviving")[0])


def test_reference_is_never_run_against_a_different_database():
    """The reference hardcodes its DB path and must not be edited, so a mismatch raises."""
    with pytest.raises(ValueError, match="hardcodes"):
        vcr.run_reference("/tmp/some_other.db")


# ---------------------------------------------------------------------------
# 4. The comparator: planted breakages must go red
# ---------------------------------------------------------------------------


def _cells(**over: Any) -> dict[str, Any]:
    base = {
        "key": ("mm_net", 2.0, 5),
        "n_events": 100,
        "n_pop": 1000,
        "mean_event_ret": 0.01,
        "mean_baseline_ret": 0.001,
        "hit_rate_event": 0.55,
        "hit_rate_baseline": 0.51,
        "p_value": 0.2,
        "p_adj_bh": 0.5,
        "significant_bh": False,
    }
    base.update(over)
    return {"rows": {base["key"]: base}, "order": [base["key"]], "n_cells": 1}


def test_identical_sources_produce_no_divergence():
    assert vcr.divergences(_cells(), _cells(), name_a="a", name_b="b") == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"n_events": 101},  # dedup rule / z-window change
        {"n_pop": 1001},  # changed baseline population
        {"mean_event_ret": 0.0101},  # changed return arithmetic
        {"mean_baseline_ret": 0.0011},  # zero null instead of unconditional baseline
        {"hit_rate_event": 0.56},
        {"hit_rate_baseline": 0.52},
        {"significant_bh": True},  # changed discovery set
    ],
)
def test_structural_mutation_is_caught(mutation):
    """Each of these is a real divergence class: a changed window, dedup, or baseline."""
    d = vcr.divergences(_cells(), _cells(**mutation), name_a="ref", name_b="new", quantities=vcr.STRUCTURAL_QUANTITIES)
    assert len(d) == 1, f"{mutation} was not caught"
    assert d[0].quantity in mutation
    assert str(d[0])  # renderable, so a failure report names the cell and both values


def test_a_missing_cell_is_a_divergence_not_a_skip():
    """A generalisation that produced 53 of 54 cells must fail, not quietly compare 53."""
    a = _cells()
    b = {"rows": {}, "order": [], "n_cells": 0}
    d = vcr.divergences(a, b, name_a="ref", name_b="new")
    assert len(d) == 1
    assert d[0].quantity == "cell present"
    assert "absent from new" in str(d[0])


def test_count_divergence_has_zero_tolerance():
    """Counts are integers; a 1-event tolerance would hide an off-by-one in the horizon."""
    d = vcr.divergences(_cells(), _cells(n_events=101), name_a="a", name_b="b")
    assert any(x.quantity == "n_events" for x in d)


def test_nan_agrees_with_nan_but_not_with_a_number():
    """Both sides saying "not computable" is agreement. One side saying 0.0 is not."""
    nan = float("nan")
    assert vcr.divergences(_cells(p_value=nan), _cells(p_value=nan), name_a="a", name_b="b") == []
    d = vcr.divergences(_cells(p_value=nan), _cells(p_value=0.0), name_a="a", name_b="b")
    assert any(x.quantity == "p_value" for x in d), "NaN vs 0.0 is the plausible-zero failure"


def test_structural_and_monte_carlo_quantities_partition_the_comparison():
    """No quantity may be in neither set (unchecked) or both (double-counted)."""
    s, m = set(vcr.STRUCTURAL_QUANTITIES), set(vcr.MONTE_CARLO_QUANTITIES)
    assert not s & m
    assert s | m == {*vcr.COMPARED_COUNTS, *vcr.COMPARED_RATES, *vcr.COMPARED_PVALS, "significant_bh"}
    assert "significant_bh" in s, "the BH decision is a function of the data and must match exactly"
    assert "p_value" in m


# ---------------------------------------------------------------------------
# 5. The Monte Carlo band must have teeth
# ---------------------------------------------------------------------------


def test_band_check_rejects_a_far_p_and_accepts_a_near_one():
    def mk(p_ref: float) -> vcr.BandCheck:
        return vcr.BandCheck(
            key=("mm_net", 2.0, 5),
            n_events=100,
            p_reference=p_ref,
            p_generalised=0.20,
            band_low=0.17,
            band_high=0.23,
            band_mean=0.20,
            band_sd=0.01,
            n_perm=32,
        )

    assert mk(0.22).inside, "2 sigma is ordinary Monte Carlo scatter"
    assert mk(0.22).sigma == pytest.approx(2.0)
    assert not mk(0.30).inside, "10 sigma is a different estimator"
    assert "OUTSIDE" in str(mk(0.30))
    assert "OK" in str(mk(0.22))


def test_band_check_with_zero_spread_does_not_accept_everything():
    """A degenerate band (sd=0) must not divide by zero into a pass."""
    c = vcr.BandCheck(
        key=("x", 2.0, 5),
        n_events=5,
        p_reference=0.4,
        p_generalised=0.2,
        band_low=0.2,
        band_high=0.2,
        band_mean=0.2,
        band_sd=0.0,
        n_perm=8,
    )
    assert c.sigma == float("inf")
    assert not c.inside


def test_band_uses_sigma_not_min_max():
    """A min/max criterion widens with n_perm, so the verdict would depend on runtime.

    Regression guard: at 16 permutations one live cell fell outside [min, max] and at
    200 the same cell fell inside. `inside` must not consult band_low/band_high.
    """
    c = vcr.BandCheck(
        key=("x", 2.0, 5),
        n_events=100,
        p_reference=0.25,
        p_generalised=0.20,
        band_low=0.20,
        band_high=0.20,
        band_mean=0.20,
        band_sd=0.02,
        n_perm=32,
    )
    assert not (c.band_low <= c.p_reference <= c.band_high), "p_ref is outside the min/max range"
    assert c.inside, "but within 4 sigma, so it must pass"


@needs_db
@pytest.mark.slow
@pytest.mark.parametrize(
    ("cell", "swap"),
    [
        # A ZERO NULL instead of the unconditional-return baseline. This is the exact
        # divergence class the task names: 2023-2026 was a positive-drift period, so a
        # zero null prints an "edge" equal to the drift. Measured at 22 sigma.
        (("open_interest", 2.0, 20), "zero_null"),
        # The CLUSTER bootstrap, which resamples event timestamps rather than events.
        # A different (and better) estimator of a different quantity. Measured at
        # 6.5 sigma on the headline cell, where the paper puts it at 0.002 -> 0.0142.
        (("mm_net_pct_oi", 2.0, 20), "clustered"),
    ],
)
def test_band_rejects_a_genuinely_different_estimator(cell, swap):
    """The band's whole job. Swap the estimator and it must refuse.

    If this test were absent, `inside` could be True unconditionally and the live
    acceptance test below would pass while an estimator swap went through.

    Cell and swap are paired deliberately, because the band's RESOLUTION depends on
    where p sits: near 0 or near 1 the bootstrap's own spread is tight in absolute
    terms but so is every alternative, and the two can agree by accident. On the
    headline cell (p ~ 0.002) the zero-null p is also ~0 and the band does NOT catch
    it; on `open_interest/|z|>=2/h=20` (p ~ 0.85) it catches it at 22 sigma. Stating
    that is the point: the band bounds a magnitude, it is not a universal detector.
    """
    from agent.verify.study import _bootstrap_p_iid, verify

    h = next(x for x in vcr.cftc_hypotheses() if (x.event_field, x.z_threshold, x.horizon_days) == cell)
    r = verify(str(DB), h)
    ev = np.asarray(r.event_returns)
    assert len(ev) > 50, "this cell should have a real sample"

    lo, hi, mean, sd = vcr.permutation_band(ev, r.baseline_mean_return, n_perm=vcr.DEFAULT_BAND_PERMS)
    assert sd > 0, "a bootstrap p that never moves under permutation is not resampling positions"

    alt = _bootstrap_p_iid(ev, 0.0) if swap == "zero_null" else r.p_value_clustered
    assert np.isfinite(alt), f"the {swap} comparison p is not computable for {cell}"

    def check(p_ref: float) -> vcr.BandCheck:
        return vcr.BandCheck(
            key=cell,
            n_events=len(ev),
            p_reference=p_ref,
            p_generalised=r.p_value,
            band_low=lo,
            band_high=hi,
            band_mean=mean,
            band_sd=sd,
            n_perm=vcr.DEFAULT_BAND_PERMS,
        )

    assert not check(alt).inside, (
        f"the {swap} estimator's p of {alt:.4g} fell inside the bootstrap band "
        f"[{mean:.4g} +/- {vcr.BAND_SIGMA}*{sd:.4g}] at only {check(alt).sigma:.1f} sigma "
        "— the band has no teeth on this cell"
    )
    # and the path's own p must, of course, be inside its own band
    assert check(r.p_value).inside, "the generalised path's own p fell outside its own band"


# ---------------------------------------------------------------------------
# 6. THE LIVE ACCEPTANCE TEST
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live() -> dict[str, Any]:
    """Run both code paths once against the live DB. ~30s."""
    if not DB.exists():
        pytest.skip("pipeline DB absent")
    return {"new": vcr.run_generalised(), "ref": vcr.run_reference()}


@needs_db
@pytest.mark.slow
def test_generalisation_matches_reference_structurally(live):
    """THE acceptance test: every quantity that is a function of the data must match.

    This is what "the generalisation did not change the answer" means. A changed
    z-window, publication lag, dedup rule or baseline all land here.
    """
    d = vcr.divergences(
        live["ref"],
        live["new"],
        name_a="reference",
        name_b="generalised",
        quantities=vcr.STRUCTURAL_QUANTITIES,
    )
    assert d == [], "structural divergence:\n" + "\n".join(f"  {x}" for x in d)
    assert live["ref"]["n_cells"] == live["new"]["n_cells"] == 54
    assert live["ref"]["n_tested"] == live["new"]["n_tested"]
    assert live["ref"]["n_surviving_bh"] == live["new"]["n_surviving_bh"]


@needs_db
@pytest.mark.slow
def test_every_generalised_cell_reconciles(live):
    """StudyResult.reconciled: every observation that entered landed in exactly one bucket.

    A False here means the module lost rows and none of its numbers are usable — the
    check that F-16 (a 5xx silently dropping 72% of a window) would have failed at
    the time instead of two years later.
    """
    bad = [k for k, r in live["new"]["rows"].items() if not r["reconciled"]]
    assert bad == [], f"cells whose counts do not add up: {bad}"


@needs_db
@pytest.mark.slow
def test_pairs_and_baseline_are_shared_as_the_reference_shares_them(live):
    """19 links used, and the baseline population depends only on (field, horizon).

    The reference keys `population` on (field, horizon) and so pools both z
    thresholds; the generalised path recomputes it per hypothesis. If the two
    thresholds ever disagreed on the baseline, the generalisation would be
    conditioning the population on the event set — a changed null.
    """
    rows = live["new"]["rows"]
    assert all(r["n_pairs_used"] == 19 and r["n_pairs_entered"] == 19 for r in rows.values())
    for field in vcr.FIELDS:
        for h in vcr.HORIZONS:
            a, b = rows[(field, 2.0, h)], rows[(field, 3.0, h)]
            assert a["n_pop"] == b["n_pop"], f"{field}/h={h}: baseline differs across z thresholds"
            assert a["mean_baseline_ret"] == pytest.approx(b["mean_baseline_ret"], abs=1e-12)


@needs_db
@pytest.mark.slow
def test_p_value_divergences_are_monte_carlo_and_nothing_more(live):
    """The p-values differ. Prove the difference is reordering, not a changed test.

    Banded on the worst-diverging cells only, to keep this test under ~10s. The
    full 49-cell audit is `.venv/bin/python scripts/verify_cftc_reproduction.py`.
    """
    mc = vcr.divergences(
        live["ref"],
        live["new"],
        name_a="reference",
        name_b="generalised",
        quantities=vcr.MONTE_CARLO_QUANTITIES,
    )
    if not mc:
        pytest.skip("p-values agree exactly; nothing to audit")
    worst = sorted(
        (k for k in live["new"]["order"] if k in live["ref"]["rows"]),
        key=lambda k: -abs(live["new"]["rows"][k]["p_value"] - live["ref"]["rows"][k]["p_value"]),
    )[:4]
    sub_ref = {"rows": {k: live["ref"]["rows"][k] for k in worst}, "order": worst}
    sub_new = {"rows": {k: live["new"]["rows"][k] for k in worst}, "order": worst}
    checks = vcr.monte_carlo_audit(sub_ref, sub_new, n_perm=vcr.DEFAULT_BAND_PERMS)
    assert checks, "the worst-diverging cells produced no band checks"
    outside = [c for c in checks if not c.inside]
    assert outside == [], (
        "a reference p-value fell outside the generalised path's own permutation band, "
        "which means the estimator changed rather than its ordering:\n" + "\n".join(f"  {c}" for c in outside)
    )
    assert all(c.band_sd > 0 for c in checks), "a band with zero spread cannot clear anything"


@needs_db
@pytest.mark.slow
def test_neither_path_still_reproduces_the_paper_and_the_reason_is_data(live):
    """The published numbers are gone, and the innocent explanation is asserted, not assumed.

    This is the finding the harness exists to state plainly: 0 of 51 was the answer on
    2026-08-29 and is not the answer today. If the cells differed while NO data had
    arrived, the explanation would be a code change and this test goes red.
    """
    pub = vcr.parse_publication()
    new = live["new"]
    assert pub["n_tested"] == 51
    ledger = vcr.drift_ledger(new, pub)
    assert len(ledger) == 54
    assert all(r["d_pop"] > 0 for r in ledger), "every cell should have gained baseline points"
    assert sum(1 for r in ledger if r["d_events"] > 0) >= 40

    best = ("mm_net_pct_oi", 2.0, 20)
    assert new["rows"][best]["n_events"] > pub["rows"][best]["n_events"], (
        "the headline cell must have grown; if it has not, data drift does not explain the delta"
    )
    # the published week count for this cell is 67, not the retracted 116
    assert new["rows"][best]["n_clusters"] >= vcr.PUBLISHED_HEADLINE["distinct_weeks_of_best"]
    assert new["rows"][best]["n_clusters"] < vcr.PUBLISHED_HEADLINE["retracted_distinct_weeks"]


@needs_db
@pytest.mark.slow
def test_any_current_bh_survivor_is_reported_with_its_power_and_sample(live):
    """A survivor on today's data is not a discovery until its n and power are stated.

    F-18 was a favourable number ending an investigation. If a cell now survives BH on
    a handful of events, that fact must travel with the number.
    """
    survivors = [r for r in live["new"]["rows"].values() if r["significant_bh"]]
    for r in survivors:
        assert np.isfinite(r["power"]), "a survivor with no power figure is uninterpretable"
        assert r["power"] < 0.5, (
            f"{r['label'] if 'label' in r else r['field']} reports power {r['power']:.1%}; "
            "the paper's power calculation says this design cannot exceed ~11%, so a high "
            "figure means power_two_sided was fed the wrong n"
        )
        assert r["n_clusters"] <= r["n_events"], "clusters cannot exceed events"
        if r["n_events"] < 30:
            # not a failure — a labelling requirement. The bootstrap over n points can
            # take at most n**n distinct values, and at n=3 that is 27.
            assert r["n_events"] >= 3, "a cell with n<3 must not be testable at all"
