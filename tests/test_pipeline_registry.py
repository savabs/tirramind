"""
Tests for DAGRegistry + daily_collection DAG (Step 7.7).

Covers: registry CRUD, validation, load_defaults, daily_collection DAG
structure (nodes, schedule, params, topo sort, independence).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.pipeline.dag import DAG
from agent.pipeline.dags.daily_collection import build_daily_collection_dag
from agent.pipeline.registry import DAGRegistry

# ── Helpers ───────────────────────────────────────────────────


def _make_dag(name: str = "test", schedule: str | None = None) -> DAG:
    dag = DAG(name=name, schedule=schedule, description=f"Test: {name}")
    dag.add("n1", operator=lambda p, u: "ok", params={})
    return dag


# ── DAGRegistry CRUD ──────────────────────────────────────────


class TestRegistryCRUD:
    """Basic register/get/list/remove operations."""

    def test_register_and_get(self):
        r = DAGRegistry()
        dag = _make_dag("alpha")
        r.register(dag)
        assert r.get("alpha") is dag

    def test_get_missing_returns_none(self):
        r = DAGRegistry()
        assert r.get("nope") is None

    def test_list_all_empty(self):
        r = DAGRegistry()
        assert r.list_all() == []

    def test_list_all_multiple(self):
        r = DAGRegistry()
        for name in ["charlie", "alpha", "bravo"]:
            r.register(_make_dag(name))
        names = [d.name for d in r.list_all()]
        assert names == ["alpha", "bravo", "charlie"]  # Sorted

    def test_list_names(self):
        r = DAGRegistry()
        r.register(_make_dag("b"))
        r.register(_make_dag("a"))
        assert r.list_names() == ["a", "b"]

    def test_len(self):
        r = DAGRegistry()
        assert len(r) == 0
        r.register(_make_dag("x"))
        assert len(r) == 1
        r.register(_make_dag("y"))
        assert len(r) == 2

    def test_remove_existing(self):
        r = DAGRegistry()
        r.register(_make_dag("d"))
        assert r.remove("d") is True
        assert r.get("d") is None
        assert len(r) == 0

    def test_remove_missing(self):
        r = DAGRegistry()
        assert r.remove("ghost") is False

    def test_overwrite_existing(self):
        r = DAGRegistry()
        r.register(_make_dag("d"))
        dag2 = _make_dag("d")
        r.register(dag2)
        assert r.get("d") is dag2
        assert len(r) == 1


# ── Registry Validation ──────────────────────────────────────


class TestRegistryValidation:
    """Validation behavior on register."""

    def test_reject_invalid_dag(self):
        r = DAGRegistry()
        bad = DAG(name="empty")  # No nodes
        with pytest.raises(ValueError, match="Invalid DAG"):
            r.register(bad)

    def test_reject_dag_with_cycle(self):
        r = DAGRegistry()
        dag = DAG(name="cyclic")
        dag.add("a", operator="x", depends_on=["b"])
        dag.add("b", operator="x", depends_on=["a"])
        with pytest.raises(ValueError, match="Invalid DAG"):
            r.register(dag)

    def test_accept_valid_dag(self):
        r = DAGRegistry()
        r.register(_make_dag())
        assert len(r) == 1

    def test_invalid_dag_not_stored(self):
        r = DAGRegistry()
        try:
            r.register(DAG(name="bad"))
        except ValueError:
            pass
        assert len(r) == 0


# ── load_defaults ─────────────────────────────────────────────


class TestLoadDefaults:
    """Tests for load_defaults() integration with daily_collection."""

    def test_load_defaults_registers_daily_collection(self):
        r = DAGRegistry()
        mock_registry = MagicMock()
        r.load_defaults(mock_registry)
        assert r.get("daily_collection") is not None

    def test_load_defaults_count(self):
        r = DAGRegistry()
        r.load_defaults(MagicMock())
        assert len(r) >= 1  # At least daily_collection

    def test_load_defaults_idempotent(self):
        r = DAGRegistry()
        r.load_defaults(MagicMock())
        count1 = len(r)
        r.load_defaults(MagicMock())  # Overwrites, doesn't duplicate
        assert len(r) == count1

    def test_daily_collection_is_valid_after_load(self):
        r = DAGRegistry()
        r.load_defaults(MagicMock())
        dag = r.get("daily_collection")
        assert dag.validate() == []


# ── daily_collection DAG Structure ────────────────────────────


class TestDailyCollectionStructure:
    """Tests for the daily_collection DAG definition."""

    @pytest.fixture
    def dag(self):
        return build_daily_collection_dag()

    def test_name(self, dag):
        assert dag.name == "daily_collection"

    def test_schedule_weekday_6pm_utc(self, dag):
        assert dag.schedule == "0 18 * * 1-5"

    def test_has_description(self, dag):
        assert dag.description  # Non-empty

    def test_validates_clean(self, dag):
        assert dag.validate() == []

    def test_node_count(self, dag):
        # 52 -> 56 on 2026-08-26: fetch_us_yield_curve, fetch_options_chains,
        # fetch_dividends (market-data-engineer, previously-unwired instrument
        # fetchers) + ingest_evidence_from_gdelt (feeds the Entity Graph from
        # this cycle's real GDELT events instead of static demo docs).
        assert len(dag.nodes) == 56

    def test_expected_node_ids(self, dag):
        expected = {
            "fetch_cftc",
            "fetch_finra_scan",
            "fetch_power_demand",
            "fetch_power_fuel",
            "fetch_gdelt",
            "fetch_polymarket",
            "fetch_macro",
            "fetch_instruments",
            "fetch_whale_alert",
            # Phase 42 — entity diversity expansion
            "fetch_insider_filings",
            "fetch_central_bank_balance",
            "fetch_sovereign_debt_us",
            "fetch_sovereign_debt_eu",
            "fetch_global_pmi",
            "fetch_capital_flows",
            "fetch_defi_flows",
            "fetch_wikipedia_pageviews",
            "fetch_lobbying",
            # Phase 43 — high-volume entity wiring
            "fetch_ais_vessel",
            "fetch_gov_contracts",
            "fetch_sanctions_monitor",
            "fetch_patent_filings",
            # Phase 44 — batch 2 entity wiring
            "fetch_regulatory_gazette",
            "fetch_form144",
            "fetch_supply_chain",
            "fetch_political_risk",
            "fetch_comtrade",
            # Phase 45 — cert/dns domain wiring
            "fetch_cert_domains",
            "fetch_dns_domains",
            # Phase 45.3 — remaining 23 unwired tools
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
            # 2026-08-26 additions (see test_node_count)
            "fetch_us_yield_curve",
            "fetch_options_chains",
            "fetch_dividends",
            "ingest_evidence_from_gdelt",
        }
        assert set(dag.nodes.keys()) == expected

    def test_all_nodes_independent(self, dag):
        """No dependencies between nodes — single parallel layer.

        Exception: ingest_evidence_from_gdelt depends on fetch_gdelt by
        design — it turns that cycle's fetched events into Entity Graph
        documents, so it must run after fetch_gdelt, not alongside it.
        """
        for node in dag.nodes.values():
            if node.id == "ingest_evidence_from_gdelt":
                continue
            assert node.depends_on == [], f"Node {node.id} has deps: {node.depends_on}"

    def test_single_parallel_layer(self, dag):
        # 2026-08-26: no longer a single layer — ingest_evidence_from_gdelt
        # deliberately depends on fetch_gdelt (see test_all_nodes_independent's
        # documented exception), giving 55 roots + 1 dependent layer.
        layers = dag.topo_sort()
        assert len(layers) == 2
        assert len(layers[0]) == 55
        assert len(layers[1]) == 1

    def test_all_roots(self, dag):
        # 55, not 56: ingest_evidence_from_gdelt is not a root (see above).
        assert len(dag.roots()) == 55

    def test_whale_alert_node_config(self, dag):
        n = dag.nodes["fetch_whale_alert"]
        assert n.operator == "whale_alert"
        assert n.table_name == "whale_alert"
        assert n.params["mode"] == "confirmed"
        assert n.params["min_btc"] == 10.0
        assert n.params["limit"] == 100
        assert n.timeout == 60
        assert n.retries == 2

    # ── Phase 42 — per-node config assertions ───────────────

    def test_insider_filings_node_config(self, dag):
        n = dag.nodes["fetch_insider_filings"]
        assert n.operator == "insider_filings"
        assert n.table_name == "insider_filings"
        assert n.params["days_back"] == 14
        assert n.params["min_cluster_size"] == 3
        assert n.timeout == 300
        assert n.retries == 2

    def test_central_bank_balance_node_config(self, dag):
        n = dag.nodes["fetch_central_bank_balance"]
        assert n.operator == "central_bank_balance"
        assert n.params["mode"] == "balance_sheets"
        assert n.params["period"] == "1y"
        assert n.timeout == 120

    def test_sovereign_debt_us_node_config(self, dag):
        n = dag.nodes["fetch_sovereign_debt_us"]
        assert n.operator == "sovereign_debt"
        assert n.params["mode"] == "us_yields"
        assert n.timeout == 120

    def test_sovereign_debt_eu_node_config(self, dag):
        n = dag.nodes["fetch_sovereign_debt_eu"]
        assert n.operator == "sovereign_debt"
        assert n.params["mode"] == "eu_yields"
        assert n.timeout == 120

    def test_global_pmi_node_config(self, dag):
        n = dag.nodes["fetch_global_pmi"]
        assert n.operator == "global_pmi"
        assert n.params["mode"] == "cli"
        assert n.timeout == 120

    def test_capital_flows_node_config(self, dag):
        n = dag.nodes["fetch_capital_flows"]
        assert n.operator == "capital_flows"
        assert n.params["mode"] == "holdings"
        assert n.timeout == 120

    def test_defi_flows_node_config(self, dag):
        n = dag.nodes["fetch_defi_flows"]
        assert n.operator == "defi_flows"
        assert n.params["mode"] == "tvl"
        assert n.params["limit"] == 20
        assert n.timeout == 120

    def test_wikipedia_pageviews_node_config(self, dag):
        n = dag.nodes["fetch_wikipedia_pageviews"]
        assert n.operator == "wikipedia_pageviews"
        assert n.params["mode"] == "spike"
        assert n.params["days_back"] == 30
        assert n.params["z_threshold"] == 2.0
        assert n.params["limit"] == 50
        assert n.timeout == 120

    def test_lobbying_node_config(self, dag):
        n = dag.nodes["fetch_lobbying"]
        assert n.operator == "lobbying"
        assert n.params["mode"] == "search"
        assert isinstance(n.params["year"], int)
        assert n.params["year"] >= 2024
        assert n.timeout == 120

    # ── Phase 43 — per-node config assertions ───────────────

    def test_ais_vessel_node_config(self, dag):
        n = dag.nodes["fetch_ais_vessel"]
        assert n.operator == "ais_vessel_tracking"
        assert n.table_name == "ais_vessel_tracking"
        assert n.params["mode"] == "area_daily_snapshot"
        assert n.params["area_name"] == "full_baltic"
        assert n.params["ship_type"] == "tanker"
        assert n.timeout == 180
        assert n.retries == 2

    def test_gov_contracts_node_config(self, dag):
        n = dag.nodes["fetch_gov_contracts"]
        assert n.operator == "gov_contracts"
        assert n.table_name == "gov_contracts"
        assert n.params["mode"] == "recent"
        assert n.params["limit"] == 100
        assert n.timeout == 120
        assert n.retries == 2

    def test_sanctions_monitor_node_config(self, dag):
        n = dag.nodes["fetch_sanctions_monitor"]
        assert n.operator == "sanctions_monitor"
        assert n.table_name == "sanctions_monitor"
        assert n.params["mode"] == "recent"
        assert n.params["days_back"] == 90
        assert n.params["limit"] == 100
        assert n.timeout == 120
        assert n.retries == 2

    def test_patent_filings_node_config(self, dag):
        n = dag.nodes["fetch_patent_filings"]
        assert n.operator == "patent_filings"
        assert n.table_name == "patent_filings"
        assert n.params["mode"] == "search"
        assert n.params["cpc_class"] == "G06N"
        assert n.params["limit"] == 50
        assert n.timeout == 120
        assert n.retries == 2


class TestDailyCollectionNodes:
    """Tests for individual node configurations."""

    @pytest.fixture
    def dag(self):
        return build_daily_collection_dag()

    def test_cftc_node(self, dag):
        n = dag.nodes["fetch_cftc"]
        assert n.operator == "cftc"
        assert n.params == {"mode": "latest"}
        assert n.retries >= 2
        assert n.timeout >= 60

    def test_finra_scan_node(self, dag):
        n = dag.nodes["fetch_finra_scan"]
        assert n.operator == "finra_short_volume"
        assert n.params["mode"] == "short_volume"
        assert "ticker" not in n.params  # scan mode: no ticker
        assert n.retries >= 2

    def test_power_demand_node(self, dag):
        n = dag.nodes["fetch_power_demand"]
        assert n.operator == "power_grid"
        assert n.params == {"mode": "demand"}

    def test_power_fuel_node(self, dag):
        n = dag.nodes["fetch_power_fuel"]
        assert n.operator == "power_grid"
        assert n.params == {"mode": "fuel_mix"}

    def test_gdelt_node(self, dag):
        n = dag.nodes["fetch_gdelt"]
        assert n.operator == "gdelt"
        assert n.params["mode"] == "events"
        assert n.params["hours_back"] == 24
        assert n.params["limit"] >= 100

    def test_polymarket_node(self, dag):
        n = dag.nodes["fetch_polymarket"]
        assert n.operator == "polymarket"
        assert n.params["category"] == "all"
        assert n.params["limit"] >= 50

    def test_tool_nodes_use_string_operators(self, dag):
        """Tool-backed nodes reference tools by string name.
        Callable operators (e.g., fetch_instruments) are also valid."""
        tool_nodes = [n for n in dag.nodes.values() if isinstance(n.operator, str)]
        assert len(tool_nodes) >= 7  # All tool nodes
        for n in tool_nodes:
            assert isinstance(n.operator, str), f"{n.id} operator is not a str"

    def test_all_nodes_store_results(self, dag):
        """Exception: ingest_evidence_from_gdelt's output is a summary dict
        (doc/sentence counts), not raw source data — nothing meaningful to
        persist to pipeline_data."""
        for node in dag.nodes.values():
            if node.id == "ingest_evidence_from_gdelt":
                continue
            assert node.store_result is True, f"{node.id} should store results"

    def test_all_timeouts_positive(self, dag):
        for node in dag.nodes.values():
            assert node.timeout > 0, f"{node.id} has non-positive timeout"

    def test_all_retries_at_least_one(self, dag):
        for node in dag.nodes.values():
            assert node.retries >= 1, f"{node.id} has zero retries"


# ── Phase 44 per-node config tests ────────────────────────────


class TestPhase44Nodes:
    """Node configuration tests for Phase 44 batch-2 wiring."""

    @pytest.fixture
    def dag(self):
        return build_daily_collection_dag()

    def test_fetch_regulatory_gazette_config(self, dag):
        n = dag.nodes["fetch_regulatory_gazette"]
        assert n.operator == "regulatory_gazette"
        assert n.params["days_back"] == 7
        assert n.params["limit"] >= 25
        assert n.timeout > 0
        assert n.retries >= 1

    def test_fetch_form144_config(self, dag):
        n = dag.nodes["fetch_form144"]
        assert n.operator == "form144"
        assert n.params["days_back"] == 14
        assert n.timeout > 0
        assert n.retries >= 1

    def test_fetch_supply_chain_config(self, dag):
        n = dag.nodes["fetch_supply_chain"]
        assert n.operator == "supply_chain_prices"
        assert n.params["mode"] == "producer_prices"
        assert n.timeout > 0
        assert n.retries >= 1

    def test_fetch_political_risk_config(self, dag):
        n = dag.nodes["fetch_political_risk"]
        assert n.operator == "political_risk"
        assert n.params["mode"] == "candidates"
        assert n.timeout > 0
        assert n.retries >= 1

    def test_fetch_comtrade_config(self, dag):
        n = dag.nodes["fetch_comtrade"]
        assert n.operator == "comtrade"
        assert n.params["mode"] == "partners"
        assert n.params["reporter"] == "USA"
        assert n.timeout > 0
        assert n.retries >= 1


# ── Integration: Registry + Scheduler ─────────────────────────


class TestRegistrySchedulerIntegration:
    """Verify DAGRegistry satisfies PipelineScheduler's DAGProvider protocol."""

    def test_registry_has_get_and_list_all(self):
        r = DAGRegistry()
        assert hasattr(r, "get")
        assert hasattr(r, "list_all")
        assert callable(r.get)
        assert callable(r.list_all)

    def test_registry_works_with_scheduler(self):
        from agent.pipeline.executor import DAGExecutor
        from agent.pipeline.scheduler import PipelineScheduler

        r = DAGRegistry()
        r.load_defaults(MagicMock())
        executor = MagicMock(spec=DAGExecutor)

        # This should not raise — registry satisfies DAGProvider
        scheduler = PipelineScheduler(executor=executor, registry=r)
        assert len(scheduler.list_dags()) >= 1

    def test_scheduler_registers_daily_collection_cron(self):
        from agent.pipeline.executor import DAGExecutor
        from agent.pipeline.scheduler import PipelineScheduler

        r = DAGRegistry()
        r.load_defaults(MagicMock())
        executor = MagicMock(spec=DAGExecutor)

        scheduler = PipelineScheduler(executor=executor, registry=r)
        scheduler.start(blocking=False)
        try:
            jobs = scheduler._scheduler.get_jobs()
            job_ids = {j.id for j in jobs}
            assert "daily_collection" in job_ids
        finally:
            scheduler.stop()


# ── Phase 45: cert_transparency + dns_monitor wiring ──────────────────────────


class TestPhase45Nodes:
    """Node configuration tests for Phase 45 cert/dns domain wiring."""

    @pytest.fixture
    def dag(self):
        return build_daily_collection_dag()

    def test_fetch_cert_domains_present(self, dag):
        assert "fetch_cert_domains" in dag.nodes

    def test_fetch_cert_domains_operator_is_callable(self, dag):
        n = dag.nodes["fetch_cert_domains"]
        assert callable(n.operator)

    def test_fetch_cert_domains_params(self, dag):
        n = dag.nodes["fetch_cert_domains"]
        assert "db_path" in n.params
        assert "domains" in n.params
        assert "days_back" in n.params
        assert isinstance(n.params["domains"], list)
        assert len(n.params["domains"]) == 20
        assert n.params["days_back"] == 30

    def test_fetch_cert_domains_timeout(self, dag):
        n = dag.nodes["fetch_cert_domains"]
        assert n.timeout >= 120

    def test_fetch_dns_domains_present(self, dag):
        assert "fetch_dns_domains" in dag.nodes

    def test_fetch_dns_domains_config(self, dag):
        n = dag.nodes["fetch_dns_domains"]
        assert n.operator == "dns_monitor"
        assert n.params["mode"] == "bulk_resolve"
        assert isinstance(n.params["domains"], list)
        assert len(n.params["domains"]) == 20
        assert n.timeout > 0
        assert n.retries >= 1

    def test_financial_domains_no_duplicates(self, dag):
        from agent.pipeline.dags.daily_collection import FINANCIAL_DOMAINS

        assert len(FINANCIAL_DOMAINS) == len(set(FINANCIAL_DOMAINS))

    def test_financial_domains_all_valid_format(self, dag):
        from agent.pipeline.dags.daily_collection import FINANCIAL_DOMAINS

        for domain in FINANCIAL_DOMAINS:
            assert "." in domain, f"Invalid domain: {domain}"
            assert " " not in domain, f"Domain has space: {domain}"
            assert domain == domain.lower(), f"Domain not lowercase: {domain}"

    def test_both_nodes_share_same_domain_list(self, dag):
        cert_node = dag.nodes["fetch_cert_domains"]
        dns_node = dag.nodes["fetch_dns_domains"]
        assert cert_node.params["domains"] == dns_node.params["domains"]


class TestPhase453Nodes:
    """Phase 45.3 — verify all 23 newly-wired tools are present and configured."""

    @pytest.fixture
    def dag(self):
        return build_daily_collection_dag()

    _PHASE_453_NODES = [
        ("fetch_academic_preprints", "academic_preprints", "trending"),
        ("fetch_bankruptcy_court", "bankruptcy_court", "us_bankruptcy"),
        ("fetch_building_permits", "building_permits", "permits"),
        ("fetch_consumer_sentiment", "consumer_sentiment", "us_sentiment"),
        ("fetch_creditor_filings", "creditor_filings", "stress_scan"),
        ("fetch_disease_surveillance", "disease_surveillance", "wastewater"),
        ("fetch_drug_regulatory", "drug_regulatory", "approvals"),
        ("fetch_earthquake_proximity", "earthquake_proximity", "recent"),
        ("fetch_electricity_monitor", "electricity_monitor", "demand"),
        ("fetch_energy_supply", "energy_supply", "petroleum_stocks"),
        ("fetch_foia_requests", "foia_requests", "agency_activity"),
        ("fetch_food_security", "food_security", "production"),
        ("fetch_interconnection_queue", "interconnection_queue", "queue"),
        ("fetch_internet_infrastructure", "internet_infrastructure", "outages"),
        ("fetch_internet_outages", "internet_outages", "outage_detection"),
        ("fetch_job_postings", "job_postings", "jolts"),
        ("fetch_labor_disruptions", "labor_disruptions", "work_stoppages"),
        ("fetch_migration_flows", "migration_flows", "displacement"),
        ("fetch_polymarket_whales", "polymarket_whales", "recent_signals"),
        ("fetch_satellite_activity", "satellite_activity", "fire"),
        ("fetch_transport_throughput", "transport_throughput", "recent"),
        ("fetch_treasury_receipts", "treasury_receipts", "cash_balance"),
        ("fetch_weather_alerts", "weather_alerts", "summary"),
    ]

    def test_all_23_nodes_present(self, dag):
        for node_id, _, _ in self._PHASE_453_NODES:
            assert node_id in dag.nodes, f"Missing node: {node_id}"

    @pytest.mark.parametrize("node_id,operator,mode", _PHASE_453_NODES)
    def test_node_operator_and_mode(self, dag, node_id, operator, mode):
        n = dag.nodes[node_id]
        assert n.operator == operator, f"{node_id}: operator mismatch"
        assert n.params.get("mode") == mode, f"{node_id}: mode mismatch"

    def test_all_phase_453_nodes_independent(self, dag):
        for node_id, _, _ in self._PHASE_453_NODES:
            assert dag.nodes[node_id].depends_on == []

    def test_total_node_count_52(self, dag):
        # 52 -> 56 on 2026-08-26 — see test_node_count in
        # TestDailyCollectionStructure for what was added; kept this test's
        # name to avoid churning its history, the assertion is what matters.
        assert len(dag.nodes) == 56

    def test_all_nodes_single_parallel_layer(self, dag):
        # 2026-08-26: ingest_evidence_from_gdelt depends on fetch_gdelt by
        # design — see TestDailyCollectionStructure.test_single_parallel_layer.
        layers = dag.topo_sort()
        assert len(layers) == 2
        assert len(layers[0]) == 55
        assert len(layers[1]) == 1
