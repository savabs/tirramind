"""Tests for Batch 3 convergence extractors: labor_disruptions, gov_contracts, academic_preprints.

Covers:
  - Valid data → correct signal_ids, categories, directions, confidence, ttl
  - Empty/None/malformed data → [] (no crash)
  - Overview vs single-series mode (labor_disruptions)
  - US vs UK region (gov_contracts)
  - Papers vs trials mode (academic_preprints)
  - Direction logic threshold edge cases
  - Defense share boundary cases
"""

from __future__ import annotations

import pytest

from agent.convergence.extractors import extract_evidence, registered_tools


# ════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════

def _sigs(tool: str, data: dict) -> dict:
    """Return {signal_id: Evidence} for a tool extraction."""
    return {e.signal_id: e for e in extract_evidence(tool, data)}


# ════════════════════════════════════════════════════════════════
#  Registration
# ════════════════════════════════════════════════════════════════

class TestRegistration:
    def test_49_total_extractors(self):
        assert len(registered_tools()) == 49

    @pytest.mark.parametrize("name", ["labor_disruptions", "gov_contracts", "academic_preprints"])
    def test_new_extractors_registered(self, name: str):
        assert name in registered_tools()


# ════════════════════════════════════════════════════════════════
#  labor_disruptions
# ════════════════════════════════════════════════════════════════

class TestLaborDisruptionsOverview:
    """Overview mode: signals dict has nested workers/idle_days sub-dicts."""

    @pytest.fixture()
    def overview_data(self):
        return {
            "signals": {
                "workers": {
                    "latest_value": 250,
                    "period_average": 120.5,
                    "trend": "ESCALATING",
                    "trend_ratio": 2.1,
                    "alert": "WARNING — >100K workers in stoppages",
                },
                "idle_days": {
                    "latest_value": 5000,
                    "period_average": 2000.0,
                    "trend": "RISING",
                    "trend_ratio": 1.3,
                    "alert": "NOTICE — ongoing idle time",
                },
                "intensity_ratio": 20.0,
                "consecutive_active_months": 5,
                "combined_alert": "WARNING — >100K workers in stoppages",
            },
        }

    def test_produces_4_signals(self, overview_data):
        sigs = _sigs("labor_disruptions", overview_data)
        assert len(sigs) == 4

    def test_workers_signal(self, overview_data):
        e = _sigs("labor_disruptions", overview_data)["strike.us.workers_involved"]
        assert e.value == 250.0
        assert e.direction == 1  # ESCALATING
        assert e.confidence == 0.75
        assert e.category == "behavioral_intent"
        assert e.ttl == 2_592_000

    def test_idle_days_signal(self, overview_data):
        e = _sigs("labor_disruptions", overview_data)["strike.us.idle_days"]
        assert e.value == 5000.0
        assert e.direction == 1  # RISING
        assert e.confidence == 0.70
        assert e.category == "macro_momentum"

    def test_intensity_signal(self, overview_data):
        e = _sigs("labor_disruptions", overview_data)["strike.us.intensity"]
        assert e.value == 20.0
        assert e.direction == 1  # > 1.5
        assert e.confidence == 0.65

    def test_consecutive_months_signal(self, overview_data):
        e = _sigs("labor_disruptions", overview_data)["strike.us.consecutive_months"]
        assert e.value == 5.0
        assert e.direction == 1  # >= 3
        assert e.confidence == 0.70

    def test_tags_present(self, overview_data):
        e = _sigs("labor_disruptions", overview_data)["strike.us.workers_involved"]
        assert "labor" in e.tags
        assert "strike" in e.tags


class TestLaborDisruptionsSingleSeries:
    """Single-series mode: data has 'label' key, flat signals dict."""

    def test_workers_single(self):
        data = {
            "label": "workers",
            "signals": {
                "latest_value": 600,
                "trend": "DECLINING",
                "alert": "CRITICAL — >500K workers in stoppages",
            },
        }
        sigs = _sigs("labor_disruptions", data)
        assert len(sigs) == 1
        e = sigs["strike.us.workers_involved"]
        assert e.value == 600.0
        assert e.direction == -1  # DECLINING

    def test_idle_days_single(self):
        data = {
            "label": "idle_days",
            "signals": {
                "latest_value": 15000,
                "trend": "STABLE",
            },
        }
        sigs = _sigs("labor_disruptions", data)
        assert len(sigs) == 1
        e = sigs["strike.us.idle_days"]
        assert e.value == 15000.0
        assert e.direction == 0  # STABLE

    def test_single_no_overview_fields(self):
        """Single-series mode should NOT produce intensity/consecutive signals."""
        data = {
            "label": "workers",
            "signals": {
                "latest_value": 100,
                "trend": "RISING",
            },
        }
        sigs = _sigs("labor_disruptions", data)
        assert "strike.us.intensity" not in sigs
        assert "strike.us.consecutive_months" not in sigs


class TestLaborDisruptionsDirectionEdgeCases:
    """Edge cases for trend → direction mapping."""

    @pytest.mark.parametrize(
        "trend,expected_dir",
        [
            ("ESCALATING", 1),
            ("RISING", 1),
            ("NEW_ACTIVITY", 1),
            ("DECLINING", -1),
            ("STABLE", 0),
            ("QUIET", 0),
            ("INSUFFICIENT_DATA", 0),
            (None, 0),
            ("", 0),
            ("UNKNOWN_VALUE", 0),
        ],
    )
    def test_trend_to_direction(self, trend, expected_dir):
        data = {
            "label": "workers",
            "signals": {"latest_value": 100, "trend": trend},
        }
        sigs = _sigs("labor_disruptions", data)
        assert sigs["strike.us.workers_involved"].direction == expected_dir

    def test_intensity_low_threshold(self):
        """intensity_ratio < 0.5 → direction -1."""
        data = {
            "signals": {
                "workers": {"latest_value": 100, "trend": "STABLE"},
                "idle_days": {"latest_value": 50, "trend": "STABLE"},
                "intensity_ratio": 0.3,
                "consecutive_active_months": 0,
            },
        }
        e = _sigs("labor_disruptions", data)["strike.us.intensity"]
        assert e.direction == -1

    def test_intensity_mid_range(self):
        """0.5 <= intensity_ratio <= 1.5 → direction 0."""
        data = {
            "signals": {
                "workers": {"latest_value": 100, "trend": "STABLE"},
                "idle_days": {"latest_value": 50, "trend": "STABLE"},
                "intensity_ratio": 1.0,
                "consecutive_active_months": 0,
            },
        }
        e = _sigs("labor_disruptions", data)["strike.us.intensity"]
        assert e.direction == 0

    def test_intensity_exact_boundary_low(self):
        """intensity_ratio == 0.5 → direction 0 (not -1; boundary is strict <)."""
        data = {
            "signals": {
                "workers": {"latest_value": 100, "trend": "STABLE"},
                "idle_days": {"latest_value": 50, "trend": "STABLE"},
                "intensity_ratio": 0.5,
                "consecutive_active_months": 0,
            },
        }
        e = _sigs("labor_disruptions", data)["strike.us.intensity"]
        assert e.direction == 0

    def test_intensity_exact_boundary_high(self):
        """intensity_ratio == 1.5 → direction 0 (not 1; boundary is strict >)."""
        data = {
            "signals": {
                "workers": {"latest_value": 100, "trend": "STABLE"},
                "idle_days": {"latest_value": 50, "trend": "STABLE"},
                "intensity_ratio": 1.5,
                "consecutive_active_months": 0,
            },
        }
        e = _sigs("labor_disruptions", data)["strike.us.intensity"]
        assert e.direction == 0

    def test_consecutive_months_below_threshold(self):
        """1 or 2 consecutive months → direction 0."""
        data = {
            "signals": {
                "workers": {"latest_value": 100, "trend": "STABLE"},
                "idle_days": {"latest_value": 50, "trend": "STABLE"},
                "intensity_ratio": 1.0,
                "consecutive_active_months": 2,
            },
        }
        e = _sigs("labor_disruptions", data)["strike.us.consecutive_months"]
        assert e.direction == 0

    def test_consecutive_months_exact_threshold(self):
        """3 consecutive months → direction 1."""
        data = {
            "signals": {
                "workers": {"latest_value": 100, "trend": "STABLE"},
                "idle_days": {"latest_value": 50, "trend": "STABLE"},
                "intensity_ratio": 1.0,
                "consecutive_active_months": 3,
            },
        }
        e = _sigs("labor_disruptions", data)["strike.us.consecutive_months"]
        assert e.direction == 1


class TestLaborDisruptionsDefensive:
    """Defensive handling: bad/missing data returns []."""

    def test_none_data(self):
        assert extract_evidence("labor_disruptions", None) == []

    def test_empty_dict(self):
        assert extract_evidence("labor_disruptions", {}) == []

    def test_string_data(self):
        assert extract_evidence("labor_disruptions", "bad") == []

    def test_list_data(self):
        assert extract_evidence("labor_disruptions", [1, 2, 3]) == []

    def test_no_signals_key(self):
        assert extract_evidence("labor_disruptions", {"foo": "bar"}) == []

    def test_signals_not_dict(self):
        assert extract_evidence("labor_disruptions", {"signals": "whoops"}) == []

    def test_signals_none(self):
        assert extract_evidence("labor_disruptions", {"signals": None}) == []

    def test_overview_workers_missing_latest(self):
        data = {
            "signals": {
                "workers": {"trend": "RISING"},
                "idle_days": {"latest_value": 100, "trend": "STABLE"},
                "intensity_ratio": 1.0,
                "consecutive_active_months": 1,
            },
        }
        sigs = _sigs("labor_disruptions", data)
        assert "strike.us.workers_involved" not in sigs
        assert "strike.us.idle_days" in sigs

    def test_overview_workers_none_value(self):
        data = {
            "signals": {
                "workers": {"latest_value": None, "trend": "RISING"},
                "idle_days": {"latest_value": 100, "trend": "STABLE"},
                "intensity_ratio": 1.0,
                "consecutive_active_months": 1,
            },
        }
        sigs = _sigs("labor_disruptions", data)
        assert "strike.us.workers_involved" not in sigs

    def test_overview_no_intensity(self):
        """Overview without intensity_ratio → no intensity signal."""
        data = {
            "signals": {
                "workers": {"latest_value": 100, "trend": "STABLE"},
                "idle_days": {"latest_value": 50, "trend": "STABLE"},
                "consecutive_active_months": 1,
            },
        }
        sigs = _sigs("labor_disruptions", data)
        assert "strike.us.intensity" not in sigs

    def test_consecutive_zero(self):
        """consecutive_active_months == 0 → no signal produced."""
        data = {
            "signals": {
                "workers": {"latest_value": 100, "trend": "STABLE"},
                "idle_days": {"latest_value": 50, "trend": "STABLE"},
                "intensity_ratio": 1.0,
                "consecutive_active_months": 0,
            },
        }
        sigs = _sigs("labor_disruptions", data)
        assert "strike.us.consecutive_months" not in sigs

    def test_single_unknown_label(self):
        """Unknown label with latest_value → no signal (not workers/idle_days)."""
        data = {
            "label": "something_else",
            "signals": {"latest_value": 999, "trend": "RISING"},
        }
        assert extract_evidence("labor_disruptions", data) == []


# ════════════════════════════════════════════════════════════════
#  gov_contracts
# ════════════════════════════════════════════════════════════════

class TestGovContractsUS:
    """US awards from USASpending API."""

    @pytest.fixture()
    def us_data(self):
        return {
            "awards": [
                {
                    "award_id": "A1",
                    "recipient": "Lockheed Martin",
                    "amount_usd": 5_000_000,
                    "agency": "Department of Defense",
                    "description": "Fighter jet maintenance",
                },
                {
                    "award_id": "A2",
                    "recipient": "ACME Corp",
                    "amount_usd": 2_000_000,
                    "agency": "Department of Education",
                    "description": "School supplies",
                },
                {
                    "award_id": "A3",
                    "recipient": "Boeing",
                    "amount_usd": 3_000_000,
                    "agency": "Department of Defense",
                    "description": "Military transport",
                },
            ],
            "total": 100,
            "count": 3,
        }

    def test_produces_3_signals(self, us_data):
        sigs = _sigs("gov_contracts", us_data)
        assert len(sigs) == 3

    def test_award_count(self, us_data):
        e = _sigs("gov_contracts", us_data)["gov_contract.us.award_count"]
        assert e.value == 3.0
        assert e.direction == 1
        assert e.confidence == 0.65
        assert e.category == "regulatory_action"
        assert e.ttl == 21_600

    def test_total_value(self, us_data):
        e = _sigs("gov_contracts", us_data)["gov_contract.us.total_value"]
        assert e.value == 10_000_000.0
        assert e.direction == 1
        assert e.confidence == 0.70

    def test_defense_share(self, us_data):
        e = _sigs("gov_contracts", us_data)["gov_contract.us.defense_share"]
        # 2 out of 3 are defense → 0.6667
        assert abs(e.value - 0.6667) < 0.001
        assert e.direction == 1  # > 0.3
        assert e.confidence == 0.75
        assert e.category == "geopolitical"

    def test_tags(self, us_data):
        e = _sigs("gov_contracts", us_data)["gov_contract.us.defense_share"]
        assert "gov" in e.tags
        assert "us" in e.tags


class TestGovContractsUK:
    """UK awards from Contracts Finder."""

    @pytest.fixture()
    def uk_data(self):
        return {
            "awards": [
                {
                    "award_id": "UK-001",
                    "recipient": "BAE Systems",
                    "amount": 1_000_000,
                    "agency": "Ministry of Defence",
                    "description": "Naval equipment",
                },
                {
                    "award_id": "UK-002",
                    "recipient": "NHS Trust",
                    "amount": 500_000,
                    "agency": "Department of Health",
                    "description": "Hospital supplies",
                },
            ],
            "total": 50,
            "count": 2,
            "region": "uk",
        }

    def test_uk_region_in_signal_id(self, uk_data):
        sigs = _sigs("gov_contracts", uk_data)
        assert "gov_contract.uk.award_count" in sigs
        assert "gov_contract.uk.total_value" in sigs
        assert "gov_contract.uk.defense_share" in sigs

    def test_uk_total_value_uses_amount_field(self, uk_data):
        e = _sigs("gov_contracts", uk_data)["gov_contract.uk.total_value"]
        assert e.value == 1_500_000.0

    def test_uk_defense_share(self, uk_data):
        e = _sigs("gov_contracts", uk_data)["gov_contract.uk.defense_share"]
        # 1 out of 2 = 0.5
        assert e.value == 0.5
        assert e.direction == 1  # > 0.3

    def test_uk_tags(self, uk_data):
        e = _sigs("gov_contracts", uk_data)["gov_contract.uk.award_count"]
        assert "uk" in e.tags


class TestGovContractsDefenseEdgeCases:
    """Defense keyword detection and threshold edge cases."""

    def test_defense_share_zero(self):
        """All civilian → defense_share 0, direction 0."""
        data = {
            "awards": [
                {"award_id": "1", "agency": "Dept of Education", "amount_usd": 100, "description": "books"},
                {"award_id": "2", "agency": "Dept of HHS", "amount_usd": 200, "description": "health"},
            ],
            "count": 2,
        }
        e = _sigs("gov_contracts", data)["gov_contract.us.defense_share"]
        assert e.value == 0.0
        assert e.direction == 0

    def test_defense_share_exact_threshold(self):
        """Exactly 0.3 → direction 0 (boundary is strict >)."""
        data = {
            "awards": [
                {"award_id": f"{i}", "agency": "Civilian", "amount_usd": 100, "description": ""}
                for i in range(7)
            ] + [
                {"award_id": f"d{i}", "agency": "DoD", "amount_usd": 100, "description": ""}
                for i in range(3)
            ],
            "count": 10,
        }
        e = _sigs("gov_contracts", data)["gov_contract.us.defense_share"]
        assert e.value == 0.3
        assert e.direction == 0

    def test_defense_keyword_in_description(self):
        """Defense keyword in description (not agency) should count."""
        data = {
            "awards": [
                {"award_id": "1", "agency": "General Services", "amount_usd": 100, "description": "Military logistics support"},
            ],
            "count": 1,
        }
        e = _sigs("gov_contracts", data)["gov_contract.us.defense_share"]
        assert e.value == 1.0
        assert e.direction == 1

    def test_mod_keyword_for_uk(self):
        """'MoD' in agency → defense."""
        data = {
            "awards": [
                {"award_id": "1", "agency": "MoD Procurement", "amount": 100, "description": ""},
            ],
            "count": 1,
            "region": "uk",
        }
        e = _sigs("gov_contracts", data)["gov_contract.uk.defense_share"]
        assert e.value == 1.0

    def test_defence_british_spelling(self):
        """British 'defence' is in keyword set."""
        data = {
            "awards": [
                {"award_id": "1", "agency": "Ministry of Defence", "amount": 100, "description": ""},
            ],
            "count": 1,
            "region": "uk",
        }
        e = _sigs("gov_contracts", data)["gov_contract.uk.defense_share"]
        assert e.value == 1.0


class TestGovContractsDefensive:
    """Defensive handling: bad/missing data returns []."""

    def test_none_data(self):
        assert extract_evidence("gov_contracts", None) == []

    def test_empty_dict(self):
        assert extract_evidence("gov_contracts", {}) == []

    def test_string_data(self):
        assert extract_evidence("gov_contracts", "bad") == []

    def test_no_awards_key(self):
        assert extract_evidence("gov_contracts", {"total": 100}) == []

    def test_awards_empty_list(self):
        assert extract_evidence("gov_contracts", {"awards": []}) == []

    def test_awards_not_list(self):
        assert extract_evidence("gov_contracts", {"awards": "oops"}) == []

    def test_awards_none(self):
        assert extract_evidence("gov_contracts", {"awards": None}) == []

    def test_all_zero_amounts(self):
        """Awards with zero/None amounts → total_value signal skipped."""
        data = {
            "awards": [
                {"award_id": "1", "agency": "X", "amount_usd": 0, "description": ""},
                {"award_id": "2", "agency": "Y", "amount_usd": None, "description": ""},
            ],
            "count": 2,
        }
        sigs = _sigs("gov_contracts", data)
        assert "gov_contract.us.total_value" not in sigs
        assert "gov_contract.us.award_count" in sigs  # count is still valid

    def test_awards_with_non_dict_entries(self):
        """Non-dict items in awards list → skipped gracefully."""
        data = {
            "awards": [
                None,
                "bad",
                42,
                {"award_id": "1", "agency": "X", "amount_usd": 1000, "description": ""},
            ],
            "count": 4,
        }
        sigs = _sigs("gov_contracts", data)
        assert sigs["gov_contract.us.total_value"].value == 1000.0

    def test_missing_agency_and_description(self):
        """Awards missing agency/description → defense detection still works (0 defense)."""
        data = {
            "awards": [
                {"award_id": "1", "amount_usd": 100},
            ],
            "count": 1,
        }
        sigs = _sigs("gov_contracts", data)
        assert sigs["gov_contract.us.defense_share"].value == 0.0

    def test_count_fallback_to_len(self):
        """If data['count'] is missing, use len(awards)."""
        data = {
            "awards": [
                {"award_id": "1", "agency": "X", "amount_usd": 100, "description": ""},
                {"award_id": "2", "agency": "Y", "amount_usd": 200, "description": ""},
            ],
        }
        sigs = _sigs("gov_contracts", data)
        assert sigs["gov_contract.us.award_count"].value == 2.0


# ════════════════════════════════════════════════════════════════
#  academic_preprints
# ════════════════════════════════════════════════════════════════

class TestAcademicPreprintsTrials:
    """Clinical trials mode (data['trials'] present)."""

    @pytest.fixture()
    def trials_data(self):
        return {
            "trials": [
                {"nct_id": "NCT001", "status": "Recruiting", "sponsor": "Pfizer", "sponsor_class": "INDUSTRY"},
                {"nct_id": "NCT002", "status": "Active, not recruiting", "sponsor": "NIH", "sponsor_class": "NIH"},
                {"nct_id": "NCT003", "status": "Completed", "sponsor": "Moderna", "sponsor_class": "INDUSTRY"},
                {"nct_id": "NCT004", "status": "Completed", "sponsor": "Harvard", "sponsor_class": "OTHER"},
                {"nct_id": "NCT005", "status": "Not yet recruiting", "sponsor": "AstraZeneca", "sponsor_class": "INDUSTRY"},
                {"nct_id": "NCT006", "status": "Withdrawn", "sponsor": "Unknown", "sponsor_class": "OTHER"},
            ],
            "total": 6,
            "count": 6,
            "source": "clinicaltrials",
        }

    def test_produces_3_signals(self, trials_data):
        sigs = _sigs("academic_preprints", trials_data)
        assert len(sigs) == 3
        assert "trials.active_count" in sigs
        assert "trials.completed_count" in sigs
        assert "trials.industry_ratio" in sigs

    def test_active_count(self, trials_data):
        e = _sigs("academic_preprints", trials_data)["trials.active_count"]
        # Recruiting + Active not recruiting + Not yet recruiting = 3
        assert e.value == 3.0
        assert e.direction == 1
        assert e.confidence == 0.60
        assert e.category == "biological"
        assert e.ttl == 86_400

    def test_completed_count(self, trials_data):
        e = _sigs("academic_preprints", trials_data)["trials.completed_count"]
        assert e.value == 2.0
        assert e.direction == 1
        assert e.confidence == 0.75
        assert e.category == "regulatory_action"

    def test_industry_ratio(self, trials_data):
        e = _sigs("academic_preprints", trials_data)["trials.industry_ratio"]
        # 3 INDUSTRY / 6 total with sponsor_class = 0.5
        assert e.value == 0.5
        assert e.direction == 0  # NOT > 0.5 (equal is not greater)
        assert e.confidence == 0.60
        assert e.category == "behavioral_intent"

    def test_industry_ratio_above_half(self):
        """More than half are INDUSTRY → direction 1."""
        data = {
            "trials": [
                {"nct_id": "1", "status": "Recruiting", "sponsor_class": "INDUSTRY"},
                {"nct_id": "2", "status": "Recruiting", "sponsor_class": "INDUSTRY"},
                {"nct_id": "3", "status": "Recruiting", "sponsor_class": "OTHER"},
            ],
            "count": 3,
        }
        e = _sigs("academic_preprints", data)["trials.industry_ratio"]
        assert abs(e.value - 0.6667) < 0.001
        assert e.direction == 1

    def test_enrolling_by_invitation_counted(self):
        """'Enrolling by invitation' is counted as active."""
        data = {
            "trials": [
                {"nct_id": "1", "status": "Enrolling by invitation", "sponsor_class": "INDUSTRY"},
            ],
            "count": 1,
        }
        sigs = _sigs("academic_preprints", data)
        assert sigs["trials.active_count"].value == 1.0

    def test_no_active_no_completed(self):
        """All terminated/withdrawn → no active or completed signals."""
        data = {
            "trials": [
                {"nct_id": "1", "status": "Terminated", "sponsor_class": "INDUSTRY"},
                {"nct_id": "2", "status": "Withdrawn", "sponsor_class": "OTHER"},
            ],
            "count": 2,
        }
        sigs = _sigs("academic_preprints", data)
        assert "trials.active_count" not in sigs
        assert "trials.completed_count" not in sigs
        # industry_ratio still produced
        assert "trials.industry_ratio" in sigs

    def test_no_arxiv_signal_in_trials_mode(self, trials_data):
        """Trials mode should not produce arxiv signals."""
        sigs = _sigs("academic_preprints", trials_data)
        assert "arxiv.volume" not in sigs


class TestAcademicPreprintsPapers:
    """arXiv papers mode (data['papers'] present)."""

    @pytest.fixture()
    def papers_data(self):
        return {
            "papers": [
                {"arxiv_id": "2401.001", "title": "Paper 1", "categories": ["cs.AI"]},
                {"arxiv_id": "2401.002", "title": "Paper 2", "categories": ["q-fin"]},
                {"arxiv_id": "2401.003", "title": "Paper 3", "categories": ["cs.LG"]},
            ],
            "total": 100,
            "count": 3,
            "source": "arxiv",
        }

    def test_produces_1_signal(self, papers_data):
        sigs = _sigs("academic_preprints", papers_data)
        assert len(sigs) == 1
        assert "arxiv.volume" in sigs

    def test_volume_signal(self, papers_data):
        e = _sigs("academic_preprints", papers_data)["arxiv.volume"]
        assert e.value == 3.0  # uses data["count"]
        assert e.direction == 1
        assert e.confidence == 0.50
        assert e.category == "behavioral_intent"
        assert e.ttl == 86_400

    def test_volume_fallback_to_total(self):
        """If count is missing, use total."""
        data = {
            "papers": [{"arxiv_id": "1"}],
            "total": 50,
        }
        e = _sigs("academic_preprints", data)["arxiv.volume"]
        assert e.value == 50.0

    def test_volume_fallback_to_len(self):
        """If both count and total missing, use len(papers)."""
        data = {
            "papers": [{"arxiv_id": "1"}, {"arxiv_id": "2"}],
        }
        e = _sigs("academic_preprints", data)["arxiv.volume"]
        assert e.value == 2.0

    def test_no_trial_signals_in_papers_mode(self, papers_data):
        """Papers mode should not produce trial signals."""
        sigs = _sigs("academic_preprints", papers_data)
        assert "trials.active_count" not in sigs
        assert "trials.completed_count" not in sigs


class TestAcademicPreprintsDefensive:
    """Defensive handling: bad/missing data returns []."""

    def test_none_data(self):
        assert extract_evidence("academic_preprints", None) == []

    def test_empty_dict(self):
        assert extract_evidence("academic_preprints", {}) == []

    def test_string_data(self):
        assert extract_evidence("academic_preprints", "bad") == []

    def test_list_data(self):
        assert extract_evidence("academic_preprints", [1, 2]) == []

    def test_neither_trials_nor_papers(self):
        """Dict without trials or papers keys → []."""
        assert extract_evidence("academic_preprints", {"something": "else"}) == []

    def test_trials_not_list(self):
        assert extract_evidence("academic_preprints", {"trials": "bad"}) == []

    def test_papers_not_list(self):
        assert extract_evidence("academic_preprints", {"papers": "bad"}) == []

    def test_trials_empty_list(self):
        """Empty trials list → no signals."""
        assert extract_evidence("academic_preprints", {"trials": []}) == []

    def test_papers_empty_list(self):
        """Empty papers list → no signal (volume 0 skipped)."""
        assert extract_evidence("academic_preprints", {"papers": [], "count": 0}) == []

    def test_trials_with_non_dict_entries(self):
        """Non-dict entries in trials → skipped."""
        data = {
            "trials": [
                None,
                42,
                "bad",
                {"nct_id": "1", "status": "Recruiting", "sponsor_class": "INDUSTRY"},
            ],
            "count": 4,
        }
        sigs = _sigs("academic_preprints", data)
        assert sigs["trials.active_count"].value == 1.0

    def test_trial_missing_status(self):
        """Trials with missing status → counted as neither active nor completed."""
        data = {
            "trials": [
                {"nct_id": "1", "sponsor_class": "INDUSTRY"},
            ],
            "count": 1,
        }
        sigs = _sigs("academic_preprints", data)
        assert "trials.active_count" not in sigs
        assert "trials.completed_count" not in sigs
        # industry_ratio still works
        assert sigs["trials.industry_ratio"].value == 1.0

    def test_trial_missing_sponsor_class(self):
        """Trials with None sponsor_class → not counted in ratio denominator."""
        data = {
            "trials": [
                {"nct_id": "1", "status": "Recruiting"},
                {"nct_id": "2", "status": "Recruiting", "sponsor_class": None},
            ],
            "count": 2,
        }
        sigs = _sigs("academic_preprints", data)
        assert sigs["trials.active_count"].value == 2.0
        assert "trials.industry_ratio" not in sigs  # 0 valid sponsors

    def test_trials_all_none_sponsor_class(self):
        """All studies with sponsor_class=None → no industry_ratio."""
        data = {
            "trials": [
                {"nct_id": "1", "status": "Recruiting", "sponsor_class": None},
            ],
        }
        sigs = _sigs("academic_preprints", data)
        assert "trials.industry_ratio" not in sigs

    def test_trials_priority_over_papers(self):
        """If both trials and papers present, only trials mode is used."""
        data = {
            "trials": [
                {"nct_id": "1", "status": "Recruiting", "sponsor_class": "INDUSTRY"},
            ],
            "papers": [
                {"arxiv_id": "1"},
            ],
            "count": 1,
        }
        sigs = _sigs("academic_preprints", data)
        assert "trials.active_count" in sigs
        assert "arxiv.volume" not in sigs


# ════════════════════════════════════════════════════════════════
#  Cross-cutting
# ════════════════════════════════════════════════════════════════

class TestCrossCutting:
    """Properties that should hold for all 3 new extractors."""

    @pytest.mark.parametrize("tool", ["labor_disruptions", "gov_contracts", "academic_preprints"])
    def test_none_returns_empty(self, tool):
        assert extract_evidence(tool, None) == []

    @pytest.mark.parametrize("tool", ["labor_disruptions", "gov_contracts", "academic_preprints"])
    def test_empty_dict_returns_empty(self, tool):
        assert extract_evidence(tool, {}) == []

    @pytest.mark.parametrize("tool", ["labor_disruptions", "gov_contracts", "academic_preprints"])
    def test_int_returns_empty(self, tool):
        assert extract_evidence(tool, 42) == []

    def test_all_signals_have_source(self):
        """Every Evidence from any extractor has source == tool_name."""
        data = {
            "signals": {
                "workers": {"latest_value": 100, "trend": "STABLE"},
                "idle_days": {"latest_value": 50, "trend": "STABLE"},
                "intensity_ratio": 1.0,
                "consecutive_active_months": 1,
            },
        }
        for e in extract_evidence("labor_disruptions", data):
            assert e.source == "labor_disruptions"

    def test_all_signals_have_timestamp(self):
        data = {
            "awards": [{"award_id": "1", "agency": "X", "amount_usd": 100, "description": ""}],
            "count": 1,
        }
        for e in extract_evidence("gov_contracts", data):
            assert isinstance(e.timestamp, (int, float))
            assert e.timestamp > 0
