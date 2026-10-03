"""
Tests for agent.verify.stats — the statistical primitives of the verdict.

TESTING POSTURE
    Four tests in this repository's history asserted the bug they were written
    to catch (LESSONS), and yesterday a frozen model ranked first because a
    favourable number ended the investigation (F-18). So the tests here are
    written to fail when the code is broken, which is a different thing from
    passing when the code is right:

      * Oracles are computed by hand or taken from the PUBLISHED study
        (docs/publications/cot_null_result.md), never from a fresh run of the
        code under test.
      * The no-lookahead property of causal_zscore is tested by asserting the
        causal value AND asserting it differs from the lookahead value, so a
        mutation that includes the current point turns the test red.
      * "Could not compute" is asserted to be None / an exception, never 0.0,
        because a plausible zero is the failure mode of this codebase.
      * Where a result should vary across inputs, variation itself is asserted
        (a constant answer is the shape a dead branch takes).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agent.quant.scoring import block_bootstrap_ci as quant_block_bootstrap_ci
from agent.verify.stats import (
    MIN_HISTORY,
    benjamini_hochberg,
    block_bootstrap_ci,
    causal_zscore,
    effective_sample_size,
    power_estimate,
    sample_size_for_power,
)

# --------------------------------------------------------------------------- #
# Published fixtures — docs/publications/cot_null_result.md, Section 5.
# The 51 testable cells, ascending, exactly as printed in the `p` column.
# --------------------------------------------------------------------------- #
COT_P_VALUES = (
    0.002,
    0.010,
    0.016,
    0.019,
    0.024,
    0.029,
    0.033,
    0.040,
    0.043,
    0.075,
    0.079,
    0.094,
    0.099,
    0.171,
    0.179,
    0.187,
    0.204,
    0.214,
    0.217,
    0.234,
    0.254,
    0.312,
    0.318,
    0.332,
    0.341,
    0.344,
    0.349,
    0.374,
    0.376,
    0.452,
    0.472,
    0.480,
    0.509,
    0.561,
    0.573,
    0.584,
    0.597,
    0.606,
    0.610,
    0.631,
    0.678,
    0.685,
    0.699,
    0.707,
    0.760,
    0.812,
    0.854,
    0.868,
    0.871,
    0.888,
    0.899,
)
# The `p_BH` column of the same table, same order.
COT_P_ADJ = (
    0.102,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.244,
    0.244,
    0.366,
    0.366,
    0.388,
    0.388,
    0.582,
    0.582,
    0.582,
    0.582,
    0.582,
    0.582,
    0.597,
    0.617,
    0.659,
    0.659,
    0.659,
    0.659,
    0.659,
    0.659,
    0.661,
    0.661,
    0.765,
    0.765,
    0.765,
    0.787,
    0.798,
    0.798,
    0.798,
    0.798,
    0.798,
    0.798,
    0.805,
    0.819,
    0.819,
    0.819,
    0.819,
    0.861,
    0.899,
    0.899,
    0.899,
    0.899,
    0.899,
    0.899,
)
# Section 7.1: a reference t of 2.14 measured on 1,451 weekly cross-sections.
COT_REF_T = 2.14
COT_REF_N = 1451


def cot_effect_over_sigma(reference_t: float = COT_REF_T) -> float:
    """effect/sigma implied by a reference t-stat at n=1451 (published Section 7.1)."""
    return reference_t / math.sqrt(COT_REF_N)


# =========================================================================== #
# 1. benjamini_hochberg
# =========================================================================== #
class TestBenjaminiHochbergByHand:
    def test_five_evenly_spaced_pvalues_all_adjust_to_alpha(self):
        """Hand-checkable: p_i = i/100 with m=5 gives m/i * p_i = 0.05 for every i.

        0.01*5/1 = 0.05, 0.02*5/2 = 0.05, 0.03*5/3 = 0.05, 0.04*5/4 = 0.05,
        0.05*5/5 = 0.05. Every adjusted value is exactly alpha, so every
        hypothesis is rejected at alpha=0.05 (the rule is <=, not <).
        """
        res = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05], alpha=0.05)
        assert res.p_adjusted == pytest.approx((0.05,) * 5, abs=1e-12)
        assert res.rejected == (True,) * 5
        assert res.n_tested == 5
        assert res.n_rejected == 5
        assert res.largest_rejected_p == pytest.approx(0.05)

    def test_three_pvalues_hand_computed(self):
        """m=3: 0.005*3/1=0.015, 0.03*3/2=0.045, 0.5*3/3=0.5."""
        res = benjamini_hochberg([0.005, 0.03, 0.5], alpha=0.05)
        assert res.p_adjusted == pytest.approx((0.015, 0.045, 0.5), abs=1e-12)
        assert res.rejected == (True, True, False)
        assert res.n_rejected == 2
        assert res.largest_rejected_p == pytest.approx(0.03)

    def test_monotonicity_is_enforced_downwards(self):
        """m=4, p=(0.01, 0.045, 0.05, 0.9).

        Raw scaling gives 0.04, 0.09, 0.0667, 0.9 — NOT monotone (rank 3 scales
        below rank 2). BH takes the running minimum from the top, so rank 2
        must come down to 0.0667, not stay at 0.09.
        """
        res = benjamini_hochberg([0.01, 0.045, 0.05, 0.9])
        assert res.p_adjusted == pytest.approx((0.04, 1 / 15, 1 / 15, 0.9), abs=1e-12)
        assert res.p_adjusted[1] == res.p_adjusted[2]
        assert res.p_adjusted[1] < 0.09

    def test_step_up_rejects_a_gap_below_a_significant_rank(self):
        """BH is step-up: everything below the largest passing rank is rejected.

        m=3, alpha=0.05, p=(0.001, 0.04, 0.045). Rank 2 fails (0.04 > 0.0333)
        but rank 3 passes (0.045 <= 0.05), so ALL THREE are rejected. A
        per-rank ("step-down") rule would wrongly reject only rank 1.
        """
        res = benjamini_hochberg([0.001, 0.04, 0.045], alpha=0.05)
        assert res.rejected == (True, True, True)
        assert res.n_rejected == 3

    def test_results_come_back_in_input_order(self):
        shuffled = [0.5, 0.005, 0.03]
        res = benjamini_hochberg(shuffled)
        assert res.p_adjusted == pytest.approx((0.5, 0.015, 0.045), abs=1e-12)
        assert res.rejected == (False, True, True)

    def test_ties_are_given_the_same_adjusted_value(self):
        res = benjamini_hochberg([0.02, 0.02, 0.02, 0.9])
        assert res.p_adjusted[0] == res.p_adjusted[1] == res.p_adjusted[2]

    def test_adjusted_values_are_capped_at_one(self):
        res = benjamini_hochberg([0.9, 0.95, 1.0])
        assert max(res.p_adjusted) <= 1.0
        assert res.n_rejected == 0
        assert res.largest_rejected_p is None


class TestBenjaminiHochbergAgainstPublishedCFTC:
    def test_rank_one_of_fiftyone_reproduces_the_headline_figure(self):
        """The acceptance check named in the brief: p=0.002 at rank 1 of 51 -> 0.102."""
        res = benjamini_hochberg(COT_P_VALUES, alpha=0.05)
        assert res.n_tested == 51
        assert res.p_adjusted[0] == pytest.approx(0.102, abs=5e-4)
        assert COT_P_VALUES[0] == 0.002

    def test_whole_published_padj_column_reproduces(self):
        res = benjamini_hochberg(COT_P_VALUES, alpha=0.05)
        got = [round(v, 3) for v in res.p_adjusted]
        assert got == list(COT_P_ADJ)

    def test_zero_of_fiftyone_survive(self):
        """The published null. If this ever goes green, the correction broke."""
        res = benjamini_hochberg(COT_P_VALUES, alpha=0.05)
        assert res.n_rejected == 0
        assert not any(res.rejected)
        assert res.largest_rejected_p is None
        assert min(res.p_adjusted) == pytest.approx(0.102, abs=5e-4)

    def test_uncorrected_would_have_found_nine_false_discoveries(self):
        """Why the correction is load-bearing, stated as a number.

        Nine of the 51 cells have an uncorrected p below 0.05. Shipping those as
        confirmations is exactly the harm the product exists to prevent.
        """
        assert sum(1 for p in COT_P_VALUES if p <= 0.05) == 9
        assert benjamini_hochberg(COT_P_VALUES).n_rejected == 0

    def test_dropping_the_boring_tests_manufactures_significance(self):
        """The family-definition trap, made concrete.

        Correct the 9 best cells only, as if the other 42 had never been run,
        and 1 becomes 'significant'. Same data, same arithmetic, different
        claim — which is why the docstring insists m is every test performed.
        """
        honest = benjamini_hochberg(COT_P_VALUES, alpha=0.05)
        cherry = benjamini_hochberg(COT_P_VALUES[:9], alpha=0.05)
        assert honest.n_rejected == 0
        assert cherry.n_rejected >= 1
        assert cherry.p_adjusted[0] < honest.p_adjusted[0]

    def test_matches_statsmodels_on_the_published_family(self):
        multitest = pytest.importorskip("statsmodels.stats.multitest")
        reject, p_adj, _, _ = multitest.multipletests(list(COT_P_VALUES), alpha=0.05, method="fdr_bh")
        res = benjamini_hochberg(COT_P_VALUES, alpha=0.05)
        assert res.p_adjusted == pytest.approx(tuple(p_adj), abs=1e-12)
        assert res.rejected == tuple(bool(r) for r in reject)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_matches_statsmodels_on_random_families(self, seed):
        """The reference implementation uses statsmodels; parity must hold off-fixture."""
        multitest = pytest.importorskip("statsmodels.stats.multitest")
        rng = np.random.default_rng(seed)
        m = int(rng.integers(1, 80))
        pvals = rng.random(m) ** 2  # skewed toward small p so rejections occur
        reject, p_adj, _, _ = multitest.multipletests(pvals, alpha=0.05, method="fdr_bh")
        res = benjamini_hochberg(pvals, alpha=0.05)
        assert res.p_adjusted == pytest.approx(tuple(p_adj), abs=1e-12)
        assert res.rejected == tuple(bool(r) for r in reject)


class TestBenjaminiHochbergDegenerateInput:
    def test_empty_family_is_not_an_error_but_says_nothing_was_tested(self):
        res = benjamini_hochberg([])
        assert res.n_tested == 0
        assert res.n_rejected == 0
        assert res.p_adjusted == ()
        assert res.rejected == ()
        assert res.largest_rejected_p is None

    def test_single_pvalue_is_unchanged(self):
        res = benjamini_hochberg([0.03])
        assert res.p_adjusted == pytest.approx((0.03,))
        assert res.rejected == (True,)
        assert res.n_tested == 1

    def test_all_identical_pvalues(self):
        res = benjamini_hochberg([0.2] * 10, alpha=0.05)
        assert res.p_adjusted == pytest.approx((0.2,) * 10)
        assert res.n_rejected == 0

    def test_zero_pvalue_is_rejected_not_confused_with_missing(self):
        res = benjamini_hochberg([0.0, 0.5])
        assert res.p_adjusted[0] == 0.0
        assert res.rejected[0] is True
        assert res.largest_rejected_p == 0.0

    def test_nan_raises_rather_than_being_treated_as_one(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            benjamini_hochberg([0.01, float("nan"), 0.3])

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            benjamini_hochberg([0.01, float("inf")])

    @pytest.mark.parametrize("bad", [[-0.1, 0.2], [0.2, 1.5]])
    def test_out_of_range_pvalue_raises(self, bad):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            benjamini_hochberg(bad)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.2, float("nan")])
    def test_bad_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            benjamini_hochberg([0.01, 0.2], alpha=alpha)

    def test_two_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            benjamini_hochberg([[0.01, 0.2], [0.3, 0.4]])

    def test_non_numeric_input_raises(self):
        with pytest.raises(ValueError):
            benjamini_hochberg(["a", "b"])


# =========================================================================== #
# 2. block_bootstrap_ci
# =========================================================================== #
def drifting_sample(n: int = 120, mean: float = 0.02, sd: float = 0.05, seed: int = 11):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, sd, n)


class TestBlockBootstrapIsAWrapper:
    def test_point_and_ci_are_bit_identical_to_the_wrapped_function(self):
        """It must WRAP agent.quant.scoring, not re-derive an almost-equal CI."""
        x = drifting_sample()
        ref_point, ref_lo, ref_hi = quant_block_bootstrap_ci(
            x, np.mean, confidence=0.95, n_bootstrap=500, block_length=None, seed=42
        )
        got = block_bootstrap_ci(x, np.mean, n_bootstrap=500, seed=42)
        assert got.point == ref_point
        assert got.lo == ref_lo
        assert got.hi == ref_hi

    def test_point_estimate_is_the_statistic_of_the_sample_itself(self):
        x = drifting_sample()
        got = block_bootstrap_ci(x, np.mean, n_bootstrap=200)
        assert got.point == pytest.approx(float(np.mean(x)), abs=1e-15)

    def test_reproducible_across_calls_and_seed_sensitive(self):
        x = drifting_sample()
        a = block_bootstrap_ci(x, np.mean, n_bootstrap=400, seed=1)
        b = block_bootstrap_ci(x, np.mean, n_bootstrap=400, seed=1)
        c = block_bootstrap_ci(x, np.mean, n_bootstrap=400, seed=2)
        assert a == b
        assert (a.lo, a.hi) != (c.lo, c.hi)

    def test_interval_brackets_the_point_and_has_width(self):
        got = block_bootstrap_ci(drifting_sample(), np.mean, n_bootstrap=500)
        assert got.lo < got.point < got.hi
        assert got.hi - got.lo > 0

    def test_block_length_changes_the_interval(self):
        """A block_len that never reaches the resampler is a silent dead knob."""
        x = drifting_sample()
        widths = {
            bl: (lambda r: r.hi - r.lo)(block_bootstrap_ci(x, np.mean, block_len=bl, n_bootstrap=400, seed=5))
            for bl in (1, 5, 20)
        }
        assert len(set(widths.values())) == 3, widths

    def test_wider_confidence_gives_a_wider_interval(self):
        x = drifting_sample()
        narrow = block_bootstrap_ci(x, np.mean, confidence=0.80, n_bootstrap=600, seed=3)
        wide = block_bootstrap_ci(x, np.mean, confidence=0.99, n_bootstrap=600, seed=3)
        assert (wide.hi - wide.lo) > (narrow.hi - narrow.lo)

    def test_p_is_computed_from_the_same_draws_as_the_interval(self):
        """The p and the CI must come from ONE resample, not two.

        The oracle re-runs the wrapped resampler with its own recording metric
        and derives the p-value by hand from those exact draws. If the module
        ever computes its p from a different (even statistically equivalent)
        distribution, this exact equality breaks.
        """
        x = drifting_sample(n=90, mean=0.01, sd=0.05, seed=31)
        null = 0.004
        n_boot = 1500
        recorded: list[float] = []

        def recording(arr):
            value = float(np.mean(arr))
            recorded.append(value)
            return value

        point, lo, hi = quant_block_bootstrap_ci(
            x, recording, confidence=0.95, n_bootstrap=n_boot, block_length=None, seed=13
        )
        boot = np.asarray(recorded[1:], dtype=float)
        assert boot.size == n_boot
        p_low = float(np.mean(boot <= null))
        p_high = float(np.mean(boot >= null))
        expected = max(min(1.0, 2.0 * min(p_low, p_high)), 1.0 / n_boot)
        assert 0.05 < expected < 0.95  # mid-range, so a re-draw would show up

        got = block_bootstrap_ci(x, np.mean, n_bootstrap=n_boot, seed=13, null_value=null)
        assert (got.point, got.lo, got.hi) == (point, lo, hi)
        assert got.p == pytest.approx(expected, abs=1e-12)

    def test_a_changed_resampler_contract_raises_instead_of_a_wrong_p(self, monkeypatch):
        """The wrapper reads the bootstrap distribution off the metric calls.

        That is brittle by design, so the brittleness must be loud: if
        agent.quant.scoring ever stops calling the metric once for the point
        estimate and then n_bootstrap times, this wrapper must refuse rather
        than compute a p-value from a misaligned distribution.
        """
        import agent.verify.stats as stats_mod

        def fake_resampler(returns, metric_fn, *, confidence, n_bootstrap, block_length, seed):
            rng = np.random.default_rng(seed)
            vals = [
                metric_fn(rng.choice(returns, size=returns.size, replace=True)) for _ in range(n_bootstrap)
            ]  # one call short: no point-estimate call
            return float(np.mean(returns)), min(vals), max(vals)

        monkeypatch.setattr(stats_mod, "_quant_block_bootstrap_ci", fake_resampler)
        with pytest.raises(RuntimeError, match="no longer calls the metric"):
            stats_mod.block_bootstrap_ci(drifting_sample(n=40), np.mean, n_bootstrap=50)

    def test_works_for_a_statistic_other_than_the_mean(self):
        x = drifting_sample()
        got = block_bootstrap_ci(x, np.median, n_bootstrap=400, null_value=0.0)
        assert got.point == pytest.approx(float(np.median(x)))
        assert got.lo < got.point < got.hi


class TestBlockBootstrapPValue:
    def test_a_clearly_positive_sample_is_significant_against_a_zero_null(self):
        got = block_bootstrap_ci(drifting_sample(mean=0.05, sd=0.02), np.mean, n_bootstrap=2000)
        assert got.p <= 0.001

    def test_two_sided_a_clearly_negative_sample_is_also_significant(self):
        got = block_bootstrap_ci(drifting_sample(mean=-0.05, sd=0.02), np.mean, n_bootstrap=2000)
        assert got.point < 0
        assert got.p <= 0.001

    def test_a_sample_centred_on_the_null_is_not_significant(self):
        got = block_bootstrap_ci(drifting_sample(mean=0.0, sd=0.05), np.mean, n_bootstrap=2000)
        assert got.p > 0.2

    def test_the_null_value_changes_the_verdict(self):
        """The published study's central choice: baseline, not zero.

        A drifting series is 'significant' against zero and unremarkable against
        its own unconditional mean. Same sample, opposite conclusions.
        """
        x = drifting_sample(mean=0.03, sd=0.02)
        vs_zero = block_bootstrap_ci(x, np.mean, n_bootstrap=2000, null_value=0.0)
        vs_baseline = block_bootstrap_ci(x, np.mean, n_bootstrap=2000, null_value=float(np.mean(x)))
        assert vs_zero.p <= 0.001
        assert vs_baseline.p > 0.5

    def test_p_is_floored_at_the_bootstrap_resolution_never_zero(self):
        got = block_bootstrap_ci(drifting_sample(mean=1.0, sd=0.01), np.mean, n_bootstrap=500)
        assert got.p == pytest.approx(1.0 / 500)
        assert got.p > 0.0

    def test_raising_n_bootstrap_lowers_the_floor(self):
        x = drifting_sample(mean=1.0, sd=0.01)
        coarse = block_bootstrap_ci(x, np.mean, n_bootstrap=500)
        fine = block_bootstrap_ci(x, np.mean, n_bootstrap=4000)
        assert coarse.p == pytest.approx(1 / 500)
        assert fine.p == pytest.approx(1 / 4000)

    def test_p_is_bounded(self):
        x = drifting_sample(mean=0.0, sd=0.05)
        got = block_bootstrap_ci(x, np.mean, n_bootstrap=1000, null_value=float(np.mean(x)))
        assert 0.0 < got.p <= 1.0

    def test_iid_mode_reproduces_the_reference_recipe(self):
        """p_resample='iid' must match scripts/cftc_event_study.py exactly.

        The reference computes its p-value as
            rng = np.random.default_rng(7)
            boot = [mean(rng.choice(ev, size=len(ev), replace=True)) for _ in range(2000)]
            p = min(1, 2 * min((boot <= base).mean(), (boot >= base).mean()))
        recomputed inline here as the oracle (that script is never imported or
        modified). The only permitted difference is this module's 1/B floor.
        """
        x = drifting_sample(mean=0.01, sd=0.05, n=80)
        baseline = 0.004
        n_boot = 2000
        rng = np.random.default_rng(7)
        boot = np.array([float(np.mean(rng.choice(x, size=len(x), replace=True))) for _ in range(n_boot)])
        p_low = float((boot <= baseline).mean())
        p_high = float((boot >= baseline).mean())
        expected = min(1.0, 2.0 * min(p_low, p_high))

        got = block_bootstrap_ci(x, np.mean, n_bootstrap=n_boot, null_value=baseline, p_resample="iid", p_seed=7)
        assert got.p == pytest.approx(max(expected, 1.0 / n_boot), abs=1e-12)

    def test_block_and_iid_disagree_on_a_serially_correlated_sample(self):
        """If they agreed, the block structure would not be reaching the p-value.

        AR(1) with phi=0.9: an i.i.d. resample treats 300 observations as 300
        independent draws and produces a standard error small enough to call
        almost anything significant. Blocks of 30 concede the correlation. Both
        p-values here are above the 1/B floor, so the gap is real resolution,
        not saturation.
        """
        rng = np.random.default_rng(4)
        noise = rng.normal(0, 0.02, 300)
        ar = np.zeros(300)
        for i in range(1, 300):
            ar[i] = 0.9 * ar[i - 1] + noise[i]
        ar = ar + 0.002
        kw = dict(block_len=30, n_bootstrap=2000, seed=9, null_value=0.014)
        blocked = block_bootstrap_ci(ar, np.mean, **kw)
        iid = block_bootstrap_ci(ar, np.mean, **kw, p_resample="iid", p_seed=9)
        assert iid.p > 1.0 / 2000  # not floored: a genuine measured value
        assert iid.p < 0.01
        assert blocked.p > 0.3
        assert blocked.p > 20 * iid.p  # blocks concede it, iid ignores it


class TestBlockBootstrapRefusals:
    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_samples_below_min_n_are_refused_not_bootstrapped(self, n):
        """n=1 would give a zero-width interval and p at the floor: a fake verdict."""
        with pytest.raises(ValueError, match="min_n|empty|n="):
            block_bootstrap_ci([0.03] * n if n else [], np.mean, n_bootstrap=100)

    def test_min_n_can_be_raised_but_not_lowered_below_three(self):
        x = drifting_sample(n=10)
        with pytest.raises(ValueError, match="min_n must be at least 3"):
            block_bootstrap_ci(x, np.mean, min_n=1, n_bootstrap=100)
        with pytest.raises(ValueError, match="min_n=30"):
            block_bootstrap_ci(x, np.mean, min_n=30, n_bootstrap=100)

    def test_all_identical_sample_raises_instead_of_reporting_certainty(self):
        with pytest.raises(ValueError, match="degenerate"):
            block_bootstrap_ci([0.04] * 50, np.mean, n_bootstrap=200)

    def test_nan_in_sample_raises(self):
        x = list(drifting_sample(n=40))
        x[7] = float("nan")
        with pytest.raises(ValueError, match="NaN or infinite"):
            block_bootstrap_ci(x, np.mean, n_bootstrap=100)

    def test_inf_in_sample_raises(self):
        x = list(drifting_sample(n=40))
        x[3] = float("inf")
        with pytest.raises(ValueError, match="NaN or infinite"):
            block_bootstrap_ci(x, np.mean, n_bootstrap=100)

    def test_statistic_returning_nan_raises_rather_than_a_nan_p(self):
        with pytest.raises(RuntimeError):
            block_bootstrap_ci(drifting_sample(n=40), lambda a: float("nan"), n_bootstrap=100)

    def test_non_finite_null_value_raises(self):
        with pytest.raises(ValueError, match="null_value"):
            block_bootstrap_ci(drifting_sample(n=40), np.mean, null_value=float("nan"))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"confidence": 0.0},
            {"confidence": 1.0},
            {"n_bootstrap": 1},
            {"block_len": 0},
            {"p_resample": "bootstrap"},
        ],
    )
    def test_bad_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            block_bootstrap_ci(drifting_sample(n=40), np.mean, **kwargs)

    def test_two_dimensional_sample_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            block_bootstrap_ci(np.zeros((5, 5)), np.mean)


# =========================================================================== #
# 3. causal_zscore — the F-04 guard
# =========================================================================== #
def lookahead_zscore(values) -> float:
    """The BUG this function must not have: the current point in its own stats."""
    x = np.asarray(values, dtype=float)
    return float((x[-1] - x.mean()) / (x.std() + 1e-12))


def reference_zscore(values):
    """Oracle: zscore_causal() from scripts/cftc_event_study.py, retyped.

    That script is the reference implementation and is never imported (it is a
    script, not a module) nor modified. The formula is reproduced here so a
    divergence in agent.verify.stats turns this test red.
    """
    if len(values) < MIN_HISTORY:
        return None
    x = np.array(values, dtype=float)
    hist = x[:-1]
    if float(np.std(hist)) < 1e-12:
        return None
    return float((x[-1] - np.mean(hist)) / (np.std(hist) + 1e-12))


class TestCausalZScoreValue:
    def test_hand_computed_value(self):
        """history = [1..19, 100].

        mean(1..19) = 10. var = sum((i-10)^2)/19 = 570/19 = 30, so std = sqrt(30)
        = 5.4772255750. z = (100 - 10) / 5.4772255750 = 16.4316767252.
        """
        history = list(range(1, 20)) + [100]
        assert len(history) == 20
        assert causal_zscore(history) == pytest.approx(16.4316767252, abs=1e-9)

    def test_zero_when_the_point_sits_on_its_historical_mean(self):
        """0.0 here is a real measurement, and is why 'could not compute' is None."""
        history = list(range(1, 20)) + [10.0]
        z = causal_zscore(history)
        assert z == pytest.approx(0.0, abs=1e-9)
        assert z is not None

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_matches_the_reference_implementation_exactly(self, seed):
        rng = np.random.default_rng(seed)
        series = rng.normal(0, 1, int(rng.integers(20, 200)))
        assert causal_zscore(series) == reference_zscore(list(series))

    def test_sign_follows_the_direction_of_the_excursion(self):
        up = list(range(1, 20)) + [100.0]
        down = list(range(1, 20)) + [-100.0]
        assert causal_zscore(up) > 0
        assert causal_zscore(down) < 0


class TestCausalZScoreHasNoLookahead:
    def test_differs_from_the_lookahead_formula(self):
        """The mutation guard. A spike shrinks its own z once it joins the stats.

        history = [1..19, 100], hand-derived both ways:
          causal     mean = 10,   var = 570/19  = 30,     std = 5.47722558,
                     z = 90 / 5.47722558 = 16.4316767252
          lookahead  mean = 14.5, var = 8265/20 = 413.25, std = 20.32856112,
                     z = 85.5 / 20.32856112 = 4.2059071759
        A factor of 3.9. If someone "fixes" this function to use the full
        window, the first assertion is what fails.
        """
        history = list(range(1, 20)) + [100.0]
        causal = causal_zscore(history)
        contaminated = lookahead_zscore(history)
        assert causal == pytest.approx(16.4316767252, abs=1e-9)
        assert contaminated == pytest.approx(4.2059071759, abs=1e-9)
        assert causal > contaminated * 3.5

    def test_the_scored_point_is_absent_from_its_own_mean_and_std(self):
        """Algebraic identity: replace the last point and the history stats hold."""
        base = list(np.linspace(0.0, 1.0, 40))
        z_small = causal_zscore(base + [2.0])
        z_big = causal_zscore(base + [200.0])
        hist = np.asarray(base)
        expected_ratio = (200.0 - hist.mean()) / (2.0 - hist.mean())
        assert z_big / z_small == pytest.approx(expected_ratio, rel=1e-9)

    def test_a_prefix_zscore_is_unaffected_by_later_data(self):
        """Future-blindness, stated as the property that matters (LESSONS F-04)."""
        rng = np.random.default_rng(17)
        series = list(rng.normal(0, 1, 60))
        prefix_zs = [causal_zscore(series[:k]) for k in range(MIN_HISTORY, 41)]
        extended = series + [1e6] * 20
        again = [causal_zscore(extended[:k]) for k in range(MIN_HISTORY, 41)]
        assert prefix_zs == again
        assert len({round(z, 9) for z in prefix_zs if z is not None}) > 5  # it varies

    def test_only_the_last_element_is_the_point_under_test(self):
        rng = np.random.default_rng(23)
        series = list(rng.normal(0, 1, 30))
        z_full = causal_zscore(series)
        z_one_shorter = causal_zscore(series[:-1])
        assert z_full != z_one_shorter


class TestCausalZScoreUndefinedCases:
    def test_empty_history_returns_none(self):
        assert causal_zscore([]) is None

    def test_single_point_returns_none_not_zero(self):
        assert causal_zscore([42.0]) is None

    @pytest.mark.parametrize("n", [2, 5, 19])
    def test_below_min_history_returns_none(self, n):
        assert causal_zscore(list(np.linspace(0, 1, n))) is None

    def test_exactly_min_history_returns_a_number(self):
        z = causal_zscore(list(np.linspace(0, 1, MIN_HISTORY)))
        assert isinstance(z, float)

    def test_min_history_boundary_is_at_twenty_matching_the_reference(self):
        series = list(np.linspace(0, 1, 40))
        assert causal_zscore(series[:19]) is None
        assert causal_zscore(series[:20]) is not None

    def test_all_identical_history_returns_none_not_a_huge_z(self):
        """Zero-variance history: (x - mean)/1e-12 would be ~1e12, a fake signal."""
        assert causal_zscore([5.0] * 30) is None
        assert causal_zscore([5.0] * 29 + [9.0]) is None

    def test_a_flat_history_with_a_jump_is_refused_rather_than_scaled_by_1e_minus_12(self):
        z = causal_zscore([0.0] * 25 + [1.0])
        assert z is None

    def test_min_history_can_be_lowered_explicitly(self):
        series = list(np.linspace(0, 1, 5))
        assert causal_zscore(series) is None
        assert causal_zscore(series, min_history=5) is not None

    def test_min_history_below_two_raises(self):
        with pytest.raises(ValueError, match="min_history"):
            causal_zscore(list(range(30)), min_history=1)

    def test_nan_raises_rather_than_returning_nan(self):
        series = list(np.linspace(0, 1, 30))
        series[4] = float("nan")
        with pytest.raises(ValueError, match="NaN or infinite"):
            causal_zscore(series)

    def test_nan_as_the_current_point_raises(self):
        series = list(np.linspace(0, 1, 29)) + [float("nan")]
        with pytest.raises(ValueError, match="NaN or infinite"):
            causal_zscore(series)

    def test_inf_raises(self):
        series = list(np.linspace(0, 1, 29)) + [float("inf")]
        with pytest.raises(ValueError, match="NaN or infinite"):
            causal_zscore(series)

    def test_two_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            causal_zscore(np.zeros((30, 2)))

    def test_never_returns_nan(self):
        rng = np.random.default_rng(5)
        for _ in range(50):
            series = rng.normal(0, rng.random() + 1e-6, 25)
            z = causal_zscore(series)
            assert z is None or math.isfinite(z)


# =========================================================================== #
# 4. power_estimate / sample_size_for_power
# =========================================================================== #
class TestPowerAgainstPublishedTable:
    @pytest.mark.parametrize(
        "reference_t,power_123,power_160",
        [
            (1.50, 0.072, 0.079),
            (2.00, 0.090, 0.102),
            (2.14, 0.096, 0.110),
            (2.50, 0.113, 0.132),
            (3.00, 0.141, 0.169),
        ],
    )
    def test_sensitivity_table_reproduces(self, reference_t, power_123, power_160):
        """docs/publications/cot_null_result.md Section 7.1, every row."""
        d = cot_effect_over_sigma(reference_t)
        assert power_estimate(123, d, 1.0) == pytest.approx(power_123, abs=5e-4)
        assert power_estimate(160, d, 1.0) == pytest.approx(power_160, abs=5e-4)

    def test_the_headline_ten_percent_power_claim(self):
        """'9.3% to 11.0%' for n=116 and n=160 at the 2.14 benchmark."""
        d = cot_effect_over_sigma()
        assert power_estimate(116, d, 1.0) == pytest.approx(0.093, abs=5e-4)
        assert power_estimate(160, d, 1.0) == pytest.approx(0.110, abs=5e-4)

    def test_expected_t_at_n_116_is_the_published_0_61(self):
        d = cot_effect_over_sigma()
        assert d * math.sqrt(116) == pytest.approx(0.61, abs=5e-3)

    @pytest.mark.parametrize(
        "reference_t,published_n",
        [(1.50, 5062), (2.00, 2847), (2.14, 2487), (2.50, 1822), (3.00, 1265)],
    )
    def test_sample_size_column_reproduces_within_a_rounding_unit(self, reference_t, published_n):
        """The published column truncates; this module ceils, so it can be +1.

        Verified as the correct direction below: the published n lands a hair
        UNDER 80% power for three of the five rows, ours never does.
        """
        d = cot_effect_over_sigma(reference_t)
        n = sample_size_for_power(d, 1.0, power=0.80, alpha=0.05)
        assert abs(n - published_n) <= 1
        assert power_estimate(n, d, 1.0) >= 0.80
        assert power_estimate(n - 1, d, 1.0) < 0.80

    def test_the_study_would_need_thousands_of_observations(self):
        """The published conclusion: 1,300-5,000 weekly observations."""
        needed = [sample_size_for_power(cot_effect_over_sigma(t), 1.0) for t in (1.5, 2.0, 2.14, 2.5, 3.0)]
        assert min(needed) > 1_200
        assert max(needed) < 5_200
        assert len(set(needed)) == 5  # it varies with the assumption


class TestPowerProperties:
    def test_zero_effect_gives_exactly_alpha(self):
        """Power against nothing is the false-positive rate, not zero."""
        assert power_estimate(500, 0.0, 1.0, alpha=0.05) == pytest.approx(0.05, abs=1e-12)
        assert power_estimate(500, 0.0, 1.0, alpha=0.01) == pytest.approx(0.01, abs=1e-12)

    def test_monotone_increasing_in_sample_size(self):
        d = cot_effect_over_sigma()
        powers = [power_estimate(n, d, 1.0) for n in (10, 50, 123, 500, 2000, 10000)]
        assert powers == sorted(powers)
        assert len(set(powers)) == 6
        assert powers[-1] > 0.99

    def test_monotone_increasing_in_effect_size(self):
        powers = [power_estimate(100, e, 1.0) for e in (0.0, 0.05, 0.1, 0.2, 0.5)]
        assert powers == sorted(powers)

    def test_monotone_decreasing_in_sigma(self):
        powers = [power_estimate(100, 0.1, s) for s in (0.1, 0.5, 1.0, 5.0)]
        assert powers == sorted(powers, reverse=True)

    def test_sign_of_the_effect_does_not_matter(self):
        assert power_estimate(100, 0.1, 1.0) == power_estimate(100, -0.1, 1.0)

    def test_never_exceeds_one_or_falls_below_alpha(self):
        assert power_estimate(10**7, 1.0, 1.0) <= 1.0
        assert power_estimate(1, 1e-12, 1.0, alpha=0.05) >= 0.05

    def test_stricter_alpha_costs_power(self):
        d = cot_effect_over_sigma()
        assert power_estimate(123, d, 1.0, alpha=0.01) < power_estimate(123, d, 1.0, alpha=0.05)

    def test_clustered_n_overstates_power(self):
        """The link to effective_sample_size: 123 events, 67 weeks."""
        d = cot_effect_over_sigma()
        assert power_estimate(123, d, 1.0) > power_estimate(67, d, 1.0)

    def test_round_trip_with_sample_size_for_power(self):
        for target in (0.50, 0.80, 0.90, 0.99):
            n = sample_size_for_power(0.1, 1.0, power=target)
            assert power_estimate(n, 0.1, 1.0) >= target


class TestPowerRefusals:
    @pytest.mark.parametrize("n", [0, -1, -100])
    def test_non_positive_n_raises(self, n):
        with pytest.raises(ValueError, match="n_events"):
            power_estimate(n, 0.1, 1.0)

    @pytest.mark.parametrize("n", [1.5, "12", None, True])
    def test_non_integer_n_raises(self, n):
        with pytest.raises(ValueError, match="n_events"):
            power_estimate(n, 0.1, 1.0)

    @pytest.mark.parametrize("sigma", [0.0, -1.0])
    def test_non_positive_sigma_raises(self, sigma):
        with pytest.raises(ValueError, match="sigma"):
            power_estimate(100, 0.1, sigma)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"effect_size": float("nan")},
            {"effect_size": float("inf")},
            {"sigma": float("nan")},
            {"alpha": float("nan")},
        ],
    )
    def test_non_finite_arguments_raise(self, kwargs):
        args = {"n_events": 100, "effect_size": 0.1, "sigma": 1.0}
        args.update(kwargs)
        with pytest.raises(ValueError):
            power_estimate(**args)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.2, 1.7])
    def test_bad_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            power_estimate(100, 0.1, 1.0, alpha=alpha)

    def test_zero_effect_size_has_no_finite_sample_size(self):
        with pytest.raises(ValueError, match="effect_size is 0"):
            sample_size_for_power(0.0, 1.0)

    @pytest.mark.parametrize("power", [0.0, 1.0, -0.1, 1.2])
    def test_bad_target_power_raises(self, power):
        with pytest.raises(ValueError, match="power"):
            sample_size_for_power(0.1, 1.0, power=power)

    def test_target_power_at_or_below_alpha_raises(self):
        with pytest.raises(ValueError, match="not above alpha"):
            sample_size_for_power(0.1, 1.0, power=0.05, alpha=0.05)

    def test_sample_size_rejects_bad_sigma_and_nan(self):
        with pytest.raises(ValueError, match="sigma"):
            sample_size_for_power(0.1, 0.0)
        with pytest.raises(ValueError, match="effect_size"):
            sample_size_for_power(float("nan"), 1.0)


# =========================================================================== #
# 5. effective_sample_size
# =========================================================================== #
WEEK = 7 * 86400.0


def build_clustered_weeks(week_counts: list[int], week_stride: int = 1) -> list[float]:
    """Timestamps placed mid-week so bucket phase cannot split a group by accident."""
    out: list[float] = []
    for i, count in enumerate(week_counts):
        base = (i * week_stride) * WEEK + WEEK / 2
        out.extend(base + j * 60.0 for j in range(count))
    return out


class TestEffectiveSampleSizePublishedShape:
    def test_reproduces_the_123_events_on_67_weeks_shape(self):
        """Section 6: the headline cell, 123 events, 67 distinct weeks, max 5.

        Reconstructed as 4 weeks of 5 + 7 weeks of 4 + ... summing to 123 over
        67 weeks. The published numbers are the counts; this asserts the
        function recovers them from raw timestamps.
        """
        week_counts = [5, 5, 5, 5] + [4] * 7 + [2] * 19 + [1] * 37
        assert sum(week_counts) == 123
        assert len(week_counts) == 67
        res = effective_sample_size(build_clustered_weeks(week_counts), window=WEEK)
        assert res["n_observations"] == 123
        assert res["n_clusters"] == 67
        assert res["max_cluster_size"] == 5
        assert res["distinct_ratio"] == pytest.approx(67 / 123)
        assert res["pseudo_replication"] == pytest.approx(123 / 67)
        assert res["singleton_clusters"] == 37

    def test_the_ratio_is_the_factor_the_published_study_paid(self):
        """123/67 = 1.84 events per week; the p-value cost a factor of five."""
        res = effective_sample_size(build_clustered_weeks([5, 5, 5, 5] + [4] * 7 + [2] * 19 + [1] * 37), window=WEEK)
        assert 1.8 < res["pseudo_replication"] < 1.9
        assert res["distinct_ratio"] < 1.0

    def test_fully_independent_events_give_ratio_one(self):
        res = effective_sample_size(build_clustered_weeks([1] * 30), window=WEEK)
        assert res["n_clusters"] == 30
        assert res["distinct_ratio"] == 1.0
        assert res["pseudo_replication"] == 1.0
        assert res["max_cluster_size"] == 1
        assert res["singleton_clusters"] == 30

    def test_it_does_not_invent_an_effective_n(self):
        """Deliberate absence: an effective n needs an intra-cluster correlation."""
        res = effective_sample_size(build_clustered_weeks([3, 3, 3]), window=WEEK)
        assert "effective_n" not in res
        assert set(res) == {
            "n_observations",
            "n_clusters",
            "distinct_ratio",
            "pseudo_replication",
            "max_cluster_size",
            "mean_cluster_size",
            "singleton_clusters",
            "cluster_sizes",
            "window",
            "origin",
        }


class TestEffectiveSampleSizeBehaviour:
    def test_all_events_on_one_timestamp(self):
        res = effective_sample_size([1_700_000_000.0] * 7, window=WEEK)
        assert res["n_observations"] == 7
        assert res["n_clusters"] == 1
        assert res["distinct_ratio"] == pytest.approx(1 / 7)
        assert res["pseudo_replication"] == 7.0
        assert res["max_cluster_size"] == 7

    def test_single_event(self):
        res = effective_sample_size([1_700_000_000.0], window=WEEK)
        assert res["n_observations"] == 1
        assert res["n_clusters"] == 1
        assert res["distinct_ratio"] == 1.0

    def test_order_does_not_matter(self):
        ts = build_clustered_weeks([2, 1, 3])
        rng = np.random.default_rng(1)
        shuffled = list(rng.permutation(ts))
        a = effective_sample_size(ts, window=WEEK)
        b = effective_sample_size(shuffled, window=WEEK)
        assert a["cluster_sizes"] == b["cluster_sizes"]
        assert a["n_clusters"] == b["n_clusters"]

    def test_window_size_changes_the_answer(self):
        ts = build_clustered_weeks([1] * 10)
        daily = effective_sample_size(ts, window=86400.0)
        weekly = effective_sample_size(ts, window=WEEK)
        monthly = effective_sample_size(ts, window=30 * 86400.0)
        assert daily["n_clusters"] == 10
        assert weekly["n_clusters"] == 10
        assert monthly["n_clusters"] < 10
        assert len({daily["n_clusters"], monthly["n_clusters"]}) == 2

    def test_the_grid_phase_is_visible_and_can_change_the_count(self):
        """The documented gotcha: buckets are a fixed grid, not clustering.

        Two events 2 hours apart either side of a daily boundary count as 2
        clusters; shifting the origin by 3 hours merges them into 1.
        """
        ts = [86400.0 - 3600.0, 86400.0 + 3600.0]
        split = effective_sample_size(ts, window=86400.0, origin=0.0)
        merged = effective_sample_size(ts, window=86400.0, origin=3 * 3600.0)
        assert split["n_clusters"] == 2
        assert merged["n_clusters"] == 1
        assert split["origin"] == 0.0
        assert merged["origin"] == pytest.approx(10800.0)

    def test_window_and_origin_are_echoed_back(self):
        res = effective_sample_size([0.0, 10.0], window=5.0, origin=1.0)
        assert res["window"] == 5.0
        assert res["origin"] == 1.0

    def test_cluster_sizes_are_descending_and_sum_to_n(self):
        res = effective_sample_size(build_clustered_weeks([1, 4, 2, 3]), window=WEEK)
        sizes = res["cluster_sizes"]
        assert list(sizes) == sorted(sizes, reverse=True)
        assert sum(sizes) == res["n_observations"]
        assert len(sizes) == res["n_clusters"]

    def test_negative_timestamps_work(self):
        res = effective_sample_size([-WEEK * 3, -WEEK * 3 + 60, -WEEK], window=WEEK)
        assert res["n_observations"] == 3
        assert res["n_clusters"] == 2


class TestEffectiveSampleSizeRefusals:
    def test_empty_raises_rather_than_returning_a_ratio(self):
        with pytest.raises(ValueError, match="empty"):
            effective_sample_size([], window=WEEK)

    def test_nan_timestamp_raises(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            effective_sample_size([0.0, float("nan")], window=WEEK)

    def test_inf_timestamp_raises(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            effective_sample_size([0.0, float("inf")], window=WEEK)

    @pytest.mark.parametrize("window", [0.0, -1.0, float("nan"), float("inf")])
    def test_bad_window_raises(self, window):
        with pytest.raises(ValueError, match="window"):
            effective_sample_size([0.0, 1.0], window=window)

    def test_non_finite_origin_raises(self):
        with pytest.raises(ValueError, match="origin"):
            effective_sample_size([0.0, 1.0], window=1.0, origin=float("nan"))

    def test_overflowing_bucket_index_raises(self):
        with pytest.raises(ValueError, match="overflow"):
            effective_sample_size([1e30, 2e30], window=1e-6)

    def test_two_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            effective_sample_size(np.zeros((3, 3)), window=WEEK)


# =========================================================================== #
# 6. The primitives together: the published verdict, end to end
# =========================================================================== #
class TestTheVerdictTheseFunctionsProduce:
    def test_the_cot_null_result_is_reconstructible_from_these_primitives(self):
        """Three numbers, three functions, one honest verdict.

        1. BH over the family of 51: nothing survives, best adjusted p 0.102.
        2. Power at the stipulated benchmark: ~10%, so absence is not evidence.
        3. Pseudo-replication: 123 events are 67 weeks, so even the n was
           optimistic.

        A product that reported only step 1 would be selling 'no effect'. All
        three together are what makes the report showable to a risk committee.
        """
        bh = benjamini_hochberg(COT_P_VALUES, alpha=0.05)
        assert bh.n_tested == 51
        assert bh.n_rejected == 0
        assert min(bh.p_adjusted) == pytest.approx(0.102, abs=5e-4)

        power = power_estimate(123, cot_effect_over_sigma(), 1.0, alpha=0.05)
        assert 0.09 < power < 0.11

        ess = effective_sample_size(build_clustered_weeks([5, 5, 5, 5] + [4] * 7 + [2] * 19 + [1] * 37), window=WEEK)
        assert ess["n_observations"] == 123
        assert ess["n_clusters"] == 67

        honest_power = power_estimate(ess["n_clusters"], cot_effect_over_sigma(), 1.0)
        assert honest_power < power
        assert sample_size_for_power(cot_effect_over_sigma(), 1.0) > 2_000

    def test_a_real_effect_is_still_detectable_end_to_end(self):
        """The suite must not merely prove everything is null.

        A genuine +5% mean against a 0.4% baseline, bootstrapped, corrected over
        a family of 51 mostly-null cells, survives BH. If this fails, the
        pipeline is rejecting truth, not protecting against falsehood.
        """
        real = block_bootstrap_ci(
            drifting_sample(n=150, mean=0.05, sd=0.03, seed=99),
            np.mean,
            n_bootstrap=2000,
            null_value=0.004,
        )
        assert real.point > 0.04
        family = [real.p] + list(COT_P_VALUES[1:])
        bh = benjamini_hochberg(family, alpha=0.05)
        assert bh.n_tested == 51
        assert bh.rejected[0] is True
        assert bh.n_rejected >= 1
