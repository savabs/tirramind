"""
Phase 38 — Downstream Pipeline Integration Tests

Validates:
- DAG node source names (table_name) match convergence extractor registry
- Convergence detection finds evidence when pipeline_data has correct sources
- MacroStateFeatureBuilder produces values when pipeline_data has macro_data
- Full pipeline smoke test: seeded data → convergence → features
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.pipeline.dag import DAG
from agent.pipeline.dags.daily_collection import build_daily_collection_dag
from agent.convergence.extractors import registered_tools


# ═══════════════════════════════════════════════════════════════
#  38.1 — Source Name Alignment
# ═══════════════════════════════════════════════════════════════


class TestSourceNameAlignment:
    """Every tool-backed DAG node must have table_name matching the
    convergence extractor registry, so pipeline_data rows are findable."""

    @pytest.fixture
    def dag(self):
        return build_daily_collection_dag()

    def test_all_tool_nodes_have_table_name(self, dag):
        """Every node whose operator is a string (tool name) must set table_name."""
        for node in dag.nodes.values():
            if isinstance(node.operator, str):
                assert node.table_name is not None, (
                    f"Node {node.id!r} has operator={node.operator!r} "
                    f"but table_name is None. Source name mismatch will break "
                    f"convergence detection."
                )

    def test_table_names_match_extractor_registry(self, dag):
        """table_name for each tool node should be a registered extractor name."""
        extractor_names = set(registered_tools())
        for node in dag.nodes.values():
            if isinstance(node.operator, str) and node.table_name:
                assert node.table_name in extractor_names, (
                    f"Node {node.id!r} has table_name={node.table_name!r} "
                    f"which is not in the extractor registry. "
                    f"Available: {sorted(extractor_names)}"
                )

    def test_table_name_matches_operator(self, dag):
        """table_name should equal the operator (tool name) for simple nodes."""
        for node in dag.nodes.values():
            if isinstance(node.operator, str):
                assert node.table_name == node.operator, (
                    f"Node {node.id!r}: table_name={node.table_name!r} != "
                    f"operator={node.operator!r}. These should match."
                )

    def test_specific_source_names(self, dag):
        """Spot-check critical source name mappings."""
        checks = {
            "fetch_cftc": "cftc",
            "fetch_finra_scan": "finra_short_volume",
            "fetch_power_demand": "power_grid",
            "fetch_power_fuel": "power_grid",
            "fetch_gdelt": "gdelt",
            "fetch_polymarket": "polymarket",
            "fetch_macro": "macro_data",
        }
        for node_id, expected_source in checks.items():
            node = dag.nodes[node_id]
            assert node.table_name == expected_source, (
                f"Node {node_id!r}: expected table_name={expected_source!r}, "
                f"got {node.table_name!r}"
            )


# ═══════════════════════════════════════════════════════════════
#  38.2 — Macro Data Node
# ═══════════════════════════════════════════════════════════════


class TestMacroDataNode:
    """The fetch_macro node must exist and be correctly configured."""

    @pytest.fixture
    def dag(self):
        return build_daily_collection_dag()

    def test_fetch_macro_exists(self, dag):
        assert "fetch_macro" in dag.nodes

    def test_fetch_macro_operator(self, dag):
        assert dag.nodes["fetch_macro"].operator == "macro_data"

    def test_fetch_macro_table_name(self, dag):
        assert dag.nodes["fetch_macro"].table_name == "macro_data"

    def test_fetch_macro_has_series_id(self, dag):
        params = dag.nodes["fetch_macro"].params
        assert "series_id" in params
        series = params["series_id"]
        for sid in ["DFF", "GS10", "GS2", "WALCL"]:
            assert sid in series, f"Missing FRED series {sid} in fetch_macro params"


# ═══════════════════════════════════════════════════════════════
#  38.3 — Convergence Evidence from Correct Source Names
# ═══════════════════════════════════════════════════════════════


class TestConvergenceEvidenceLoading:
    """Verify convergence detection finds evidence when pipeline_data
    has rows stored with correct source names (matching tool names)."""

    def _make_store_with_data(self, tmp_path):
        """Create a PipelineStore seeded with mock tool data."""
        from agent.pipeline.store import PipelineStore

        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path)
        now = time.time()

        # Seed CFTC data (matches _extract_cftc extractor expectations)
        store.store_data(
            source="cftc",  # correct source name
            params={"mode": "latest"},
            data={
                "mode": "latest",
                "contracts": [
                    {
                        "commodity": "CRUDE OIL",
                        "exchange": "NYMEX",
                        "report_date": "2026-04-18",
                        "noncommercial_long": 300000,
                        "noncommercial_short": 250000,
                        "open_interest": 2000000,
                        "net_noncommercial": 50000,
                        "net_pct_oi": 2.5,
                    }
                ],
            },
        )

        # Seed GDELT data (matches _extract_gdelt extractor expectations)
        store.store_data(
            source="gdelt",
            params={"mode": "events"},
            data={
                "mode": "events",
                "event_count": 100,
                "events": [
                    {
                        "event_code": "190",
                        "goldstein_scale": -8.0,
                        "num_mentions": 50,
                        "avg_tone": -3.5,
                        "actor1_country": "USA",
                        "actor2_country": "CHN",
                        "event_date": "20260418",
                    }
                ],
            },
        )

        # Seed Polymarket data (matches _extract_polymarket extractor expectations)
        store.store_data(
            source="polymarket",
            params={"category": "all"},
            data={
                "markets": [
                    {
                        "question": "Will Fed cut rates in May 2026?",
                        "category": "economics",
                        "probability": 0.72,
                        "volume": 5_000_000,
                        "liquidity": 100_000,
                    }
                ],
                "category_counts": {"economics": 1},
            },
        )

        return store

    def test_evidence_nonzero_with_correct_sources(self, tmp_path):
        """When pipeline_data has correct source names, extractors find data."""
        from agent.pipeline.dags.convergence_detection import (
            _load_evidence_from_store,
        )

        store = self._make_store_with_data(tmp_path)
        evidence = _load_evidence_from_store(store, lookback_days=7)
        store.close()
        assert len(evidence) > 0, "Expected non-zero evidence from seeded data"

    def test_evidence_zero_with_wrong_sources(self, tmp_path):
        """When pipeline_data has fetch_* source names, extractors find nothing."""
        from agent.pipeline.store import PipelineStore
        from agent.pipeline.dags.convergence_detection import (
            _load_evidence_from_store,
        )

        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path)
        # Store with wrong source name (the pre-fix behavior)
        store.store_data(
            source="fetch_cftc",  # WRONG — extractor looks for "cftc"
            params={"mode": "latest"},
            data={"contracts": [{"commodity": "CRUDE OIL"}]},
        )
        evidence = _load_evidence_from_store(store, lookback_days=7)
        store.close()
        assert len(evidence) == 0, "Evidence should be 0 with wrong source names"

    def test_evidence_from_at_least_one_tool(self, tmp_path):
        """Evidence should come from at least one tool with correct source names.
        Not all extractors will fire from minimal mock data (field name
        requirements vary per extractor), but the source-name alignment
        must allow at least one to succeed."""
        from agent.pipeline.dags.convergence_detection import (
            _load_evidence_from_store,
        )

        store = self._make_store_with_data(tmp_path)
        evidence = _load_evidence_from_store(store, lookback_days=7)
        store.close()
        sources = {e.source for e in evidence}
        assert len(sources) >= 1, f"Expected evidence from >=1 tools, got: {sources}"


# ═══════════════════════════════════════════════════════════════
#  38.3b — MacroStateFeatureBuilder with Correct Source Name
# ═══════════════════════════════════════════════════════════════


class TestMacroFeatureBuilderIntegration:
    """MacroStateFeatureBuilder should produce non-None values when
    pipeline_data has rows with source='macro_data'."""

    def _seed_macro_data(self, tmp_path):
        from agent.pipeline.store import PipelineStore

        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path)

        # Seed macro_data with FRED series (matching _extract_series expectations)
        store.store_data(
            source="macro_data",
            params={"series_id": "DFF,GS10,GS2,WALCL"},
            data={
                "DFF": [
                    {"date": "2026-03-01", "value": "4.33"},
                    {"date": "2026-03-15", "value": "4.33"},
                    {"date": "2026-04-01", "value": "4.33"},
                    {"date": "2026-04-15", "value": "4.25"},
                ],
                "GS10": [
                    {"date": "2026-04-15", "value": "4.50"},
                ],
                "GS2": [
                    {"date": "2026-04-15", "value": "4.10"},
                ],
                "WALCL": [
                    {"date": "2026-01-22", "value": "7800000"},
                    {"date": "2026-01-29", "value": "7810000"},
                    {"date": "2026-02-05", "value": "7820000"},
                    {"date": "2026-02-12", "value": "7830000"},
                    {"date": "2026-02-19", "value": "7840000"},
                    {"date": "2026-02-26", "value": "7850000"},
                    {"date": "2026-03-05", "value": "7860000"},
                    {"date": "2026-03-12", "value": "7870000"},
                    {"date": "2026-03-19", "value": "7880000"},
                    {"date": "2026-03-26", "value": "7890000"},
                    {"date": "2026-04-02", "value": "7900000"},
                    {"date": "2026-04-09", "value": "7910000"},
                ],
            },
        )
        return store

    def test_macro_builder_produces_values(self, tmp_path):
        """MacroStateFeatureBuilder produces non-None values from seeded data."""
        from agent.features.builders import MacroStateFeatureBuilder

        store = self._seed_macro_data(tmp_path)
        builder = MacroStateFeatureBuilder()
        features = builder.build(store, time.time())
        store.close()

        assert len(features) == 3
        named = {f.feature_name: f for f in features}

        # Rate momentum: 4.25 - 4.33 = -0.08 → -8 bps
        rm = named["macro.rate_momentum.30d"]
        assert rm.value is not None, "rate_momentum should have a value"

        # Yield curve slope: 4.50 - 4.10 = 0.40 → 40 bps
        yc = named["macro.yield_curve_slope.spot"]
        assert yc.value is not None, "yield_curve_slope should have a value"
        assert abs(yc.value - 40.0) < 0.1

        # Liquidity pressure: z-score of weekly diffs
        lp = named["macro.liquidity_pressure.30d"]
        assert lp.value is not None, "liquidity_pressure should have a value"

    def test_macro_builder_missing_when_no_data(self, tmp_path):
        """MacroStateFeatureBuilder returns 3 missing features with no data."""
        from agent.pipeline.store import PipelineStore
        from agent.features.builders import MacroStateFeatureBuilder

        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path)
        builder = MacroStateFeatureBuilder()
        features = builder.build(store, time.time())
        store.close()

        # Empty store now returns 3 None-valued features for consistent
        # GNN dimensionality (value=None, missing_reason set).
        assert len(features) == 3
        assert all(f.value is None for f in features)


# ═══════════════════════════════════════════════════════════════
#  38.3c — Full Pipeline Smoke Test
# ═══════════════════════════════════════════════════════════════


class TestPipelineSmokeTest:
    """Seed pipeline_data → run convergence → run features → non-empty output."""

    def test_convergence_then_features(self, tmp_path):
        """Full downstream pipeline produces non-empty results."""
        from agent.pipeline.store import PipelineStore
        from agent.pipeline.dags.convergence_detection import (
            run_convergence_detection,
        )
        from agent.pipeline.dags.feature_generation import run_feature_generation
        from agent.features.builders import (
            ConvergenceFeatureBuilder,
            MacroStateFeatureBuilder,
        )

        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path)
        now = time.time()

        # Seed CFTC data with multiple contracts for positioning signals
        for i in range(5):
            store.store_data(
                source="cftc",
                params={"mode": "latest"},
                data={
                    "mode": "latest",
                    "contracts": [
                        {
                            "commodity": "CRUDE OIL",
                            "exchange": "NYMEX",
                            "report_date": f"2026-04-{13 + i:02d}",
                            "noncommercial_long": 300000 + i * 10000,
                            "noncommercial_short": 250000,
                            "open_interest": 2000000,
                            "net_noncommercial": 50000 + i * 10000,
                            "net_pct_oi": 2.5 + i * 0.5,
                        }
                    ],
                },
            )

        # Seed GDELT data for geopolitical signals
        for i in range(5):
            store.store_data(
                source="gdelt",
                params={"mode": "events"},
                data={
                    "mode": "events",
                    "event_count": 50 + i * 20,
                    "events": [
                        {
                            "event_code": "190",
                            "goldstein_scale": -8.0 + i,
                            "num_mentions": 50,
                            "avg_tone": -3.5,
                            "actor1_country": "USA",
                            "actor2_country": "CHN",
                            "event_date": f"2026041{3 + i}",
                        }
                    ],
                },
            )

        # Seed macro data
        store.store_data(
            source="macro_data",
            params={"series_id": "DFF,GS10,GS2,WALCL"},
            data={
                "DFF": [
                    {"date": "2026-03-15", "value": "4.33"},
                    {"date": "2026-04-15", "value": "4.25"},
                ],
                "GS10": [{"date": "2026-04-15", "value": "4.50"}],
                "GS2": [{"date": "2026-04-15", "value": "4.10"}],
                "WALCL": [
                    {
                        "date": f"2026-0{2 + i // 4}-{(i % 4) * 7 + 1:02d}",
                        "value": str(7800000 + i * 10000),
                    }
                    for i in range(12)
                ],
            },
        )
        store.close()

        # Run convergence detection
        conv_result = run_convergence_detection(
            params={"db_path": db_path, "lookback_days": 30},
            upstream={},
        )

        # Even if convergence detects 0 events (requires signal pairs),
        # the evidence loading itself should succeed.
        # The key test is that it doesn't crash and processes real data.
        assert isinstance(conv_result, dict)
        assert "detected" in conv_result
        assert "emitted" in conv_result

        # Run feature generation (without GNN — only convergence + macro)
        feat_result = run_feature_generation(
            params={
                "db_path": db_path,
                "builders": [
                    ConvergenceFeatureBuilder(),
                    MacroStateFeatureBuilder(),
                ],
            },
            upstream={},
        )

        assert feat_result["produced"] == 6  # 3 convergence + 3 macro
        # Macro features should have values
        macro_summary = next(
            b
            for b in feat_result["builders"]
            if b["builder"] == "MacroStateFeatureBuilder"
        )
        assert macro_summary["features_produced"] == 3
        # At least yield curve slope should be non-missing
        assert (
            macro_summary["missing"] < 3
        ), "All macro features missing — data not found"


# ═══════════════════════════════════════════════════════════════
#  38.4 — DAG Structure (updated counts)
# ═══════════════════════════════════════════════════════════════


class TestDagStructureUpdated:
    """Updated DAG structure tests reflecting Phase 38 additions."""

    @pytest.fixture
    def dag(self):
        return build_daily_collection_dag()

    def test_node_count(self, dag):
        # 50 string-operator nodes + 2 callable nodes
        # (fetch_instruments + fetch_cert_domains) = 52
        assert len(dag.nodes) == 52

    def test_expected_node_ids(self, dag):
        expected = {
            "fetch_ais_vessel",
            "fetch_capital_flows",
            "fetch_central_bank_balance",
            "fetch_cftc",
            "fetch_comtrade",
            "fetch_defi_flows",
            "fetch_finra_scan",
            "fetch_form144",
            "fetch_gdelt",
            "fetch_global_pmi",
            "fetch_gov_contracts",
            "fetch_insider_filings",
            "fetch_instruments",
            "fetch_lobbying",
            "fetch_macro",
            "fetch_patent_filings",
            "fetch_political_risk",
            "fetch_polymarket",
            "fetch_power_demand",
            "fetch_power_fuel",
            "fetch_regulatory_gazette",
            "fetch_sanctions_monitor",
            "fetch_sovereign_debt_eu",
            "fetch_sovereign_debt_us",
            "fetch_supply_chain",
            "fetch_whale_alert",
            "fetch_wikipedia_pageviews",
            # Phase 45 — cert/dns
            "fetch_cert_domains",
            "fetch_dns_domains",
            # Phase 45.3 — remaining 23 tools
            "fetch_academic_preprints",
            "fetch_bankruptcy_court",
            "fetch_building_permits",
            "fetch_consumer_sentiment",
            "fetch_creditor_filings",
            "fetch_disease_surveillance",
            "fetch_drug_regulatory",
            "fetch_earthquake_proximity",
            "fetch_electricity_monitor",
            "fetch_energy_supply",
            "fetch_foia_requests",
            "fetch_food_security",
            "fetch_interconnection_queue",
            "fetch_internet_infrastructure",
            "fetch_internet_outages",
            "fetch_job_postings",
            "fetch_labor_disruptions",
            "fetch_migration_flows",
            "fetch_polymarket_whales",
            "fetch_satellite_activity",
            "fetch_transport_throughput",
            "fetch_treasury_receipts",
            "fetch_weather_alerts",
        }
        assert set(dag.nodes.keys()) == expected

    def test_all_nodes_independent(self, dag):
        for node in dag.nodes.values():
            assert node.depends_on == [], f"Node {node.id} has deps"

    def test_single_parallel_layer(self, dag):
        layers = dag.topo_sort()
        assert len(layers) == 1
        assert len(layers[0]) == 52

    def test_all_roots(self, dag):
        assert len(dag.roots()) == 52

    def test_tool_nodes_have_string_operators(self, dag):
        """Tool-backed nodes have string operators; callable nodes are allowed."""
        tool_nodes = [n for n in dag.nodes.values() if isinstance(n.operator, str)]
        assert (
            len(tool_nodes) == 50
        )  # all except fetch_instruments + fetch_cert_domains
        callable_nodes = [n for n in dag.nodes.values() if callable(n.operator)]
        assert len(callable_nodes) == 2  # fetch_instruments + fetch_cert_domains
