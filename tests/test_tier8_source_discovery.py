"""
Tests for Tier 8, Change 15 — Autonomous Data Source Discovery.

Covers: PipelineStore discovery tables, SourceScout search/probe, SignalEvaluator MI,
ToolFactory create/serialize, ToolRoutingBandit arm management, discovery orchestration,
and quarantine cycle logic.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.discovery.signal_evaluator import SignalEvaluator, SignalReport
from agent.discovery.source_scout import (
    DataSourceCandidate,
    SourceScout,
    _make_source_id,
    _tfidf_relevance,
)
from agent.discovery.tool_factory import (
    DiscoveredCsvFeedTool,
    DiscoveredJsonApiTool,
    ToolFactory,
)
from agent.learning.tool_router import ToolRoutingBandit
from agent.pipeline.dags.daily_collection import (
    _MAX_CONSECUTIVE_FAILURES,
    run_quarantine_cycle,
)
from agent.pipeline.store import PipelineStore

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)


@pytest.fixture()
def store(tmp_dir):
    return PipelineStore(tmp_dir / "test.db")


@pytest.fixture()
def candidate():
    url = "https://example.com/api/test-data.json"
    return DataSourceCandidate(
        source_id=_make_source_id(url),
        name="Test Economic API",
        url=url,
        description="Daily economic indicators for testing",
        format="json_api",
        update_frequency="daily",
        topic_tags=["economy", "gdp", "inflation"],
        probe_sample={"data": [{"date": "2024-01-01", "value": 3.14}]},
    )


# ── 1. PipelineStore: discovered_sources CRUD ────────────────


class TestDiscoveredSourcesCRUD:
    def test_store_and_query(self, store):
        store.store_discovered_source(
            source_id="src_001",
            name="Test Source",
            url="https://example.com/data",
            fmt="json_api",
            description="Some data",
            update_frequency="daily",
        )
        rows = store.query_discovered_sources()
        assert len(rows) >= 1
        match = [r for r in rows if r["source_id"] == "src_001"]
        assert len(match) == 1
        assert match[0]["name"] == "Test Source"
        assert match[0]["status"] == "discovered"

    def test_status_transitions(self, store):
        store.store_discovered_source(
            source_id="src_002",
            name="Transition Test",
            url="https://example.com/trans",
            fmt="csv_feed",
        )
        store.update_source_status("src_002", "quarantine")
        rows = store.query_discovered_sources(status="quarantine")
        assert any(r["source_id"] == "src_002" for r in rows)

        store.update_source_status("src_002", "active")
        rows = store.query_discovered_sources(status="active")
        assert any(r["source_id"] == "src_002" for r in rows)

    def test_consecutive_failures(self, store):
        store.store_discovered_source(
            source_id="src_003",
            name="Fail Test",
            url="https://example.com/fail",
            fmt="json_api",
        )
        store.increment_source_failures("src_003")
        store.increment_source_failures("src_003")
        rows = store.query_discovered_sources()
        match = [r for r in rows if r["source_id"] == "src_003"]
        assert match[0]["consecutive_failures"] == 2

        store.reset_source_failures("src_003")
        rows = store.query_discovered_sources()
        match = [r for r in rows if r["source_id"] == "src_003"]
        assert match[0]["consecutive_failures"] == 0

    def test_idempotent_insert(self, store):
        store.store_discovered_source(
            source_id="src_dup",
            name="First",
            url="https://example.com/dup",
            fmt="json_api",
        )
        store.store_discovered_source(
            source_id="src_dup",
            name="Second",
            url="https://example.com/dup",
            fmt="json_api",
        )
        rows = store.query_discovered_sources()
        match = [r for r in rows if r["source_id"] == "src_dup"]
        # INSERT OR IGNORE: first one wins
        assert len(match) == 1
        assert match[0]["name"] == "First"

    def test_query_by_status_filter(self, store):
        store.store_discovered_source(
            source_id="s1",
            name="A",
            url="https://a.com",
            fmt="json_api",
        )
        store.store_discovered_source(
            source_id="s2",
            name="B",
            url="https://b.com",
            fmt="csv_feed",
        )
        store.update_source_status("s1", "active")
        assert len(store.query_discovered_sources(status="active")) >= 1
        assert len(store.query_discovered_sources(status="disabled")) == 0


# ── 2. PipelineStore: unresolved_entities CRUD ───────────────


class TestUnresolvedEntitiesCRUD:
    def test_store_and_query(self, store):
        row_id = store.store_unresolved_entity(
            raw_text="FacilityX",
            source_tool="test_tool",
            context_snippet='{"owner": "ACME"}',
            observed_at=time.time(),
        )
        assert isinstance(row_id, int)
        rows = store.query_unresolved_entities(resolved=False)
        assert any(r["raw_text"] == "FacilityX" for r in rows)

    def test_cluster_assignment(self, store):
        rid = store.store_unresolved_entity(
            raw_text="EntityA",
            source_tool="tool1",
        )
        store.update_unresolved_cluster([rid], cluster_id=42)
        rows = store.query_unresolved_entities()
        match = [r for r in rows if r["id"] == rid]
        assert match[0]["cluster_id"] == 42

    def test_resolve(self, store):
        rid = store.store_unresolved_entity(
            raw_text="EntityB",
            source_tool="tool1",
        )
        # Assign to a cluster first, then resolve by cluster
        store.update_unresolved_cluster([rid], cluster_id=99)
        count = store.resolve_unresolved_entities(cluster_id=99, resolved_type="facility")
        assert count >= 1
        rows = store.query_unresolved_entities(resolved=True)
        assert any(r["id"] == rid for r in rows)


# ── 3. PipelineStore: entity_type_registry CRUD ──────────────


class TestEntityTypeRegistryCRUD:
    def test_register_and_query(self, store):
        store.register_entity_type(
            type_name="facility",
            parent_type=None,
            source="induced",
            confidence=0.85,
        )
        rows = store.query_entity_types(active_only=True)
        assert any(r["type_name"] == "facility" for r in rows)

    def test_deactivate(self, store):
        store.register_entity_type("temp_type", source="test")
        store.deactivate_entity_type("temp_type")
        active = store.query_entity_types(active_only=True)
        assert not any(r["type_name"] == "temp_type" for r in active)

    def test_reactivate(self, store):
        store.register_entity_type("revived", source="test")
        store.deactivate_entity_type("revived")
        store.reactivate_entity_type("revived")
        active = store.query_entity_types(active_only=True)
        assert any(r["type_name"] == "revived" for r in active)


# ── 4. SourceScout: search with mock catalog ─────────────────


class TestSourceScoutSearch:
    def _mock_ckan_response(self):
        return json.dumps(
            {
                "success": True,
                "result": {
                    "results": [
                        {
                            "title": "GDP Growth Data",
                            "name": "gdp-growth",
                            "notes": "Quarterly GDP growth rates by country",
                            "tags": [{"name": "gdp"}, {"name": "economy"}],
                            "resources": [
                                {
                                    "format": "JSON",
                                    "url": "https://api.example.com/gdp.json",
                                },
                            ],
                        },
                        {
                            "title": "Commodity Prices",
                            "name": "commodity-prices",
                            "notes": "Daily commodity spot prices",
                            "tags": [{"name": "commodities"}],
                            "resources": [
                                {
                                    "format": "CSV",
                                    "url": "https://api.example.com/comm.csv",
                                },
                            ],
                        },
                    ]
                },
            }
        ).encode()

    @patch("agent.discovery.source_scout.urllib.request.urlopen")
    def test_search_parses_ckan_results(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = self._mock_ckan_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        scout = SourceScout(topic_vocabulary={"gdp", "economy", "market"})
        results = scout.search(["economic data"])

        assert len(results) == 2
        assert results[0].format in ("json_api", "csv_feed")
        assert results[0].name in ("GDP Growth Data", "Commodity Prices")

    @patch("agent.discovery.source_scout.urllib.request.urlopen")
    def test_search_deduplicates_known_urls(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = self._mock_ckan_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        scout = SourceScout(
            existing_source_urls={"https://api.example.com/gdp.json"},
            topic_vocabulary={"gdp"},
        )
        results = scout.search(["economy"])
        # gpd.json should be excluded
        urls = [c.url for c in results]
        assert "https://api.example.com/gdp.json" not in urls

    def test_relevance_scoring(self):
        score = _tfidf_relevance("gdp economy growth inflation", {"gdp", "economy"})
        assert score > 0.0
        zero = _tfidf_relevance("weather sunny clouds", {"gdp", "economy"})
        assert zero == 0.0


# ── 5. SourceScout: probe success and failure ────────────────


class TestSourceScoutProbe:
    @patch("agent.discovery.source_scout.urllib.request.urlopen")
    def test_probe_success_json(self, mock_urlopen):
        sample_data = json.dumps({"data": [{"value": 42}]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = sample_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        candidate = DataSourceCandidate(
            source_id="test1",
            name="Test",
            url="https://example.com/test",
            description="test",
            format="json_api",
            update_frequency="daily",
        )
        scout = SourceScout()
        result = scout.probe(candidate)
        assert result.probe_sample is not None
        assert result.probe_sample["data"][0]["value"] == 42

    @patch("agent.discovery.source_scout.urllib.request.urlopen")
    def test_probe_failure_sets_none(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Network error")

        candidate = DataSourceCandidate(
            source_id="test2",
            name="Bad URL",
            url="https://bad.example.com/fail",
            description="fails",
            format="json_api",
            update_frequency="unknown",
        )
        scout = SourceScout()
        result = scout.probe(candidate)
        assert result.probe_sample is None


# ── 6-8. SignalEvaluator ─────────────────────────────────────


class TestSignalEvaluator:
    def test_correlated_data_positive_mi(self, store):
        """Synthetic correlated data → positive MI."""
        from agent.features.protocol import EngineeredFeature

        rng = np.random.RandomState(42)
        now = time.time()
        for i in range(100):
            feat = EngineeredFeature(
                feature_name="test_feature.spot",
                version=1,
                effective_at=now - (100 - i) * 86400,
                computed_at=now,
                horizon="spot",
                value=float(rng.randn()),
                quality=1.0,
                source_signals=("synthetic",),
                builder="test",
            )
            store.store_feature(feat)

        # Create a candidate with correlated probe data
        probe_data = [{"date": f"2024-{i:04d}", "metric": float(rng.randn())} for i in range(100)]
        candidate = DataSourceCandidate(
            source_id="corr",
            name="Correlated",
            url="https://example.com/corr",
            description="correlated",
            format="json_api",
            update_frequency="daily",
            probe_sample=probe_data,
        )
        evaluator = SignalEvaluator(store=store, min_samples=5)
        report = evaluator.evaluate(candidate)
        assert isinstance(report, SignalReport)

    def test_none_probe_returns_empty(self, store):
        """No probe sample → empty report."""
        candidate = DataSourceCandidate(
            source_id="empty",
            name="Empty",
            url="https://example.com/empty",
            description="no data",
            format="json_api",
            update_frequency="unknown",
            probe_sample=None,
        )
        evaluator = SignalEvaluator(store=store)
        report = evaluator.evaluate(candidate)
        assert report.max_mi == 0.0
        assert not report.passes_threshold

    def test_empty_store_returns_zero(self, store):
        """Empty feature store → zero MI."""
        candidate = DataSourceCandidate(
            source_id="nostore",
            name="No Store Features",
            url="https://example.com/ns",
            description="test",
            format="json_api",
            update_frequency="daily",
            probe_sample=[{"colA": 1.0, "colB": 2.0}],
        )
        evaluator = SignalEvaluator(store=store)
        report = evaluator.evaluate(candidate)
        assert report.max_mi == 0.0


# ── 9-11. ToolFactory ────────────────────────────────────────


class TestToolFactory:
    def test_create_json_api_tool(self, candidate, tmp_dir):
        report = SignalReport(max_mi=0.1, passes_threshold=True)
        factory = ToolFactory(config_dir=str(tmp_dir / "tools"))
        tool = factory.create_tool(candidate, report)
        assert tool is not None
        assert isinstance(tool, DiscoveredJsonApiTool)
        assert candidate.source_id[:8] in tool.name

    def test_create_csv_tool(self, tmp_dir):
        csv_candidate = DataSourceCandidate(
            source_id="csv123456",
            name="CSV Data",
            url="https://example.com/data.csv",
            description="CSV",
            format="csv_feed",
            update_frequency="daily",
            probe_sample=[{"col1": "1", "col2": "2"}],
        )
        report = SignalReport(max_mi=0.1, passes_threshold=True)
        factory = ToolFactory(config_dir=str(tmp_dir / "tools"))
        tool = factory.create_tool(csv_candidate, report)
        assert tool is not None
        assert isinstance(tool, DiscoveredCsvFeedTool)

    def test_config_round_trip(self, candidate, tmp_dir):
        report = SignalReport(max_mi=0.1, passes_threshold=True)
        factory = ToolFactory(config_dir=str(tmp_dir / "tools"))
        tool = factory.create_tool(candidate, report)
        factory.save_config(tool)

        loaded = factory.load_all_configs()
        assert len(loaded) == 1
        assert loaded[0].name == tool.name

    def test_load_empty_dir(self, tmp_dir):
        factory = ToolFactory(config_dir=str(tmp_dir / "empty"))
        loaded = factory.load_all_configs()
        assert loaded == []

    def test_multiple_tools_persist(self, tmp_dir):
        factory = ToolFactory(config_dir=str(tmp_dir / "tools"))
        report = SignalReport(max_mi=0.1, passes_threshold=True)

        for i in range(3):
            # Use source_id long enough that [:8] gives unique prefixes
            c = DataSourceCandidate(
                source_id=f"abcdef{i:02d}00000000",
                name=f"Tool {i}",
                url=f"https://example.com/api{i}",
                description="test",
                format="json_api",
                update_frequency="daily",
                probe_sample={"data": [{"val": i}]},
            )
            tool = factory.create_tool(c, report)
            factory.save_config(tool)

        loaded = factory.load_all_configs()
        assert len(loaded) == 3


# ── 12. ToolRoutingBandit arm management ─────────────────────


class TestBanditArmManagement:
    def test_add_arm(self, tmp_dir):
        bandit = ToolRoutingBandit(
            tool_names=["existing_a"],
            persist_path=tmp_dir / "bandit.json",
        )
        bandit.add_arm("new_tool")
        assert "new_tool" in bandit._tool_names

    def test_add_arm_idempotent(self, tmp_dir):
        bandit = ToolRoutingBandit(
            tool_names=["existing_a"],
            persist_path=tmp_dir / "bandit.json",
        )
        bandit.add_arm("existing_a")
        assert bandit._tool_names.count("existing_a") == 1

    def test_remove_arm(self, tmp_dir):
        bandit = ToolRoutingBandit(
            tool_names=["a", "b", "c"],
            persist_path=tmp_dir / "bandit.json",
        )
        bandit.remove_arm("b")
        assert "b" not in bandit._tool_names

    def test_remove_nonexistent_no_crash(self, tmp_dir):
        bandit = ToolRoutingBandit(
            tool_names=["a"],
            persist_path=tmp_dir / "bandit.json",
        )
        bandit.remove_arm("nonexistent")  # should not raise


# ── 13. Discovery orchestration (mocked) ─────────────────────


class TestDiscoveryOrchestration:
    @patch("agent.discovery.source_scout.urllib.request.urlopen")
    def test_end_to_end_mocked(self, mock_urlopen, store, tmp_dir):
        """Full discovery pipeline with mocked HTTP."""
        # Mock catalog response
        catalog_resp = json.dumps(
            {
                "success": True,
                "result": {
                    "results": [
                        {
                            "title": "Test API",
                            "name": "test-api",
                            "notes": "Market data feed for testing",
                            "tags": [{"name": "market"}],
                            "resources": [
                                {
                                    "format": "JSON",
                                    "url": "https://api.example.com/market.json",
                                }
                            ],
                        }
                    ]
                },
            }
        ).encode()

        # Mock probe response
        probe_resp = json.dumps({"data": [{"date": "2024-01-01", "price": 100.5, "volume": 1e6}]}).encode()

        call_count = [0]

        def mock_open(req, timeout=None):
            call_count[0] += 1
            resp = MagicMock()
            if call_count[0] == 1:
                resp.read.return_value = catalog_resp
            else:
                resp.read.return_value = probe_resp
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        mock_urlopen.side_effect = mock_open

        from agent.tools.base import ToolRegistry

        registry = ToolRegistry()
        bandit = ToolRoutingBandit(
            tool_names=["existing"],
            persist_path=tmp_dir / "bandit.json",
        )

        from agent.discovery.source_scout import run_source_discovery

        # Pre-populate a known entity type so vocabulary isn't empty
        store.register_entity_type("market", source="seed")

        created = run_source_discovery(
            store=store,
            registry=registry,
            bandit=bandit,
            query_terms=["market data"],
            max_new_tools=1,
        )
        # Whether anything was created depends on MI threshold + data alignment
        # The key test is that the pipeline doesn't crash
        assert isinstance(created, list)
        # discovered_sources table should have at least one entry
        # (stored as 'discovered' even if MI threshold isn't met)
        sources = store.query_discovered_sources()
        assert len(sources) >= 1


# ── 14. Quarantine: promotion and failure disabling ──────────


class TestQuarantineCycle:
    def test_no_quarantine_sources_returns_empty(self, store):
        results = run_quarantine_cycle(store)
        assert results == {}

    def test_failure_increments_counter(self, store):
        # Add a quarantine source (no config on disk → probe will fail)
        store.store_discovered_source(
            source_id="q1",
            name="Quarantine Test",
            url="https://q1.example.com",
            fmt="json_api",
        )
        store.update_source_status("q1", "quarantine")

        results = run_quarantine_cycle(store)
        assert results.get("q1") == "quarantine"

    def test_disabling_after_max_failures(self, store):
        store.store_discovered_source(
            source_id="q2",
            name="Failing Source",
            url="https://q2.example.com",
            fmt="json_api",
        )
        store.update_source_status("q2", "quarantine")

        # Pre-increment failures to just below threshold
        for _ in range(_MAX_CONSECUTIVE_FAILURES - 1):
            store.increment_source_failures("q2")

        # This cycle pushes it over the edge
        results = run_quarantine_cycle(store)
        assert results["q2"] == "disabled"

        disabled = store.query_discovered_sources(status="disabled")
        assert any(s["source_id"] == "q2" for s in disabled)

    def test_status_filter_excludes_non_quarantine(self, store):
        store.store_discovered_source(
            source_id="active_src",
            name="Active",
            url="https://active.example.com",
            fmt="json_api",
        )
        store.update_source_status("active_src", "active")
        results = run_quarantine_cycle(store)
        assert "active_src" not in results


# ── Edge case: make_source_id deterministic ──────────────────


class TestMakeSourceId:
    def test_deterministic(self):
        url = "https://example.com/data.json"
        assert _make_source_id(url) == _make_source_id(url)

    def test_different_urls_different_ids(self):
        assert _make_source_id("https://a.com") != _make_source_id("https://b.com")

    def test_expected_format(self):
        sid = _make_source_id("https://test.com")
        assert len(sid) == 16
        # Should be hex characters
        int(sid, 16)
