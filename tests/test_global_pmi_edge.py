"""
Edge-case tests for agent/tools/global_pmi.py

Covers: all 3 modes (cli, bci, cci), invalid mode, country validation,
period validation, CSV parsing, empty data, HTTP errors, signal computation
(momentum, regime, spreads), cache integration, tool schema.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import httpx

# ──────────────────────────────────────────────────────────────────
# Synthetic data factories
# ──────────────────────────────────────────────────────────────────


def _make_csv(rows=None, header=None):
    """Build OECD-style CSV with labels."""
    if header is None:
        header = "REF_AREA,FREQ,MEASURE,TIME_PERIOD,OBS_VALUE"
    if rows is None:
        rows = [
            "USA,M,LI,2025-06,99.50",
            "USA,M,LI,2025-07,99.60",
            "USA,M,LI,2025-08,99.80",
            "USA,M,LI,2025-09,100.10",
            "USA,M,LI,2025-10,100.30",
            "USA,M,LI,2025-11,100.50",
            "USA,M,LI,2025-12,100.80",
            "USA,M,LI,2026-01,101.00",
            "CHN,M,LI,2025-06,98.00",
            "CHN,M,LI,2025-07,98.20",
            "CHN,M,LI,2025-08,98.50",
            "CHN,M,LI,2025-09,98.80",
            "CHN,M,LI,2025-10,99.00",
            "CHN,M,LI,2025-11,99.20",
            "CHN,M,LI,2025-12,99.50",
            "CHN,M,LI,2026-01,99.80",
        ]
    return header + "\n" + "\n".join(rows) + "\n"


def _make_sparse_csv():
    """CSV with missing OBS_VALUE entries."""
    header = "REF_AREA,FREQ,MEASURE,TIME_PERIOD,OBS_VALUE"
    rows = [
        "USA,M,LI,2025-12,100.50",
        "USA,M,LI,2026-01,",
        "DEU,M,LI,2026-01,99.00",
    ]
    return header + "\n" + "\n".join(rows) + "\n"


def _make_single_country_csv(country="USA", values=None):
    """CSV for a single country."""
    header = "REF_AREA,FREQ,MEASURE,TIME_PERIOD,OBS_VALUE"
    if values is None:
        values = [("2025-12", "100.50"), ("2026-01", "100.80")]
    rows = [f"{country},M,LI,{p},{v}" for p, v in values]
    return header + "\n" + "\n".join(rows) + "\n"


def _mock_response(text, status=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = text
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(f"HTTP {status}", request=MagicMock(), response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestGlobalPmiTool(unittest.TestCase):
    """Edge-case tests for GlobalPmiTool."""

    def _make_tool(self, cache=None):
        from agent.tools.global_pmi import GlobalPmiTool

        return GlobalPmiTool(cache=cache)

    # ── Mode validation ─────────────────────────────────────────────

    def test_invalid_mode(self):
        tool = self._make_tool()
        result = tool.execute(mode="pmi")
        self.assertFalse(result.success)
        self.assertIn("Invalid mode", result.output)

    def test_empty_mode(self):
        tool = self._make_tool()
        result = tool.execute(mode="")
        self.assertFalse(result.success)

    # ── Period validation ───────────────────────────────────────────

    def test_invalid_start_period_format(self):
        tool = self._make_tool()
        result = tool.execute(mode="cli", start_period="2026-1")
        self.assertFalse(result.success)
        self.assertIn("YYYY-MM", result.output)

    def test_invalid_end_period_format(self):
        tool = self._make_tool()
        result = tool.execute(mode="cli", end_period="March2026")
        self.assertFalse(result.success)

    def test_invalid_month_13(self):
        tool = self._make_tool()
        result = tool.execute(mode="cli", start_period="2026-13")
        self.assertFalse(result.success)

    def test_invalid_month_00(self):
        tool = self._make_tool()
        result = tool.execute(mode="cli", start_period="2026-00")
        self.assertFalse(result.success)

    def test_start_after_end_period(self):
        tool = self._make_tool()
        result = tool.execute(mode="cli", start_period="2026-06", end_period="2026-01")
        self.assertFalse(result.success)
        self.assertIn("after", result.output)

    def test_valid_periods_accepted(self):
        tool = self._make_tool()
        with patch("agent.tools.global_pmi.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value.__enter__ = MagicMock(return_value=client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            client.get.return_value = _mock_response(_make_csv())

            result = tool.execute(mode="cli", start_period="2025-06", end_period="2026-01")
            self.assertTrue(result.success)

    # ── CLI mode basic ──────────────────────────────────────────────

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_cli_basic(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_csv())

        tool = self._make_tool()
        result = tool.execute(mode="cli")

        self.assertTrue(result.success)
        self.assertEqual(result.data["mode"], "cli")
        self.assertIn("USA", result.data["by_country"])
        self.assertIn("CHN", result.data["by_country"])

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_cli_latest_values(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_csv())

        tool = self._make_tool()
        result = tool.execute(mode="cli")

        usa_latest = result.data["by_country"]["USA"][-1]
        self.assertEqual(usa_latest["value"], 101.0)
        self.assertEqual(usa_latest["period"], "2026-01")

    # ── Signal computation ──────────────────────────────────────────

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_regime_expanding(self, mock_cls):
        """USA CLI > 100 and rising → expanding."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_csv())

        tool = self._make_tool()
        result = tool.execute(mode="cli", include_signals=True)

        usa_signals = result.data["signals"].get("USA", {})
        self.assertEqual(usa_signals.get("regime"), "expanding")

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_regime_contracting(self, mock_cls):
        """Country with CLI < 100 and declining → contracting."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        csv = _make_single_country_csv("DEU", [("2025-11", "99.50"), ("2025-12", "99.20"), ("2026-01", "98.90")])
        client.get.return_value = _mock_response(csv)

        tool = self._make_tool()
        result = tool.execute(mode="cli", countries="DEU")

        deu_signals = result.data["signals"].get("DEU", {})
        self.assertEqual(deu_signals.get("regime"), "contracting")

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_momentum_6m_computed(self, mock_cls):
        """6-month momentum computed when >=7 data points exist."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_csv())

        tool = self._make_tool()
        result = tool.execute(mode="cli", include_signals=True)

        usa_signals = result.data["signals"].get("USA", {})
        # 6m momentum: (101.0 - 99.60) / 99.60 * 100 = 1.406...
        self.assertIn("momentum_6m", usa_signals)
        self.assertAlmostEqual(usa_signals["momentum_6m"], 1.41, places=1)

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_momentum_not_computed_few_points(self, mock_cls):
        """6-month momentum NOT computed when < 7 data points."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        csv = _make_single_country_csv("USA", [("2025-12", "100.50"), ("2026-01", "100.80")])
        client.get.return_value = _mock_response(csv)

        tool = self._make_tool()
        result = tool.execute(mode="cli", countries="USA")

        usa_signals = result.data["signals"].get("USA", {})
        self.assertNotIn("momentum_6m", usa_signals)

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_cross_country_spreads(self, mock_cls):
        """USA-CHN spread computed when both present."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_csv())

        tool = self._make_tool()
        result = tool.execute(mode="cli", countries="USA,CHN")

        spreads = result.data["signals"].get("_spreads", {})
        # USA 101.0 - CHN 99.8 = 1.2
        self.assertIn("USA-CHN", spreads)
        self.assertAlmostEqual(spreads["USA-CHN"], 1.2, places=1)

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_signals_disabled(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_csv())

        tool = self._make_tool()
        result = tool.execute(mode="cli", include_signals=False)

        self.assertTrue(result.success)
        self.assertEqual(result.data["signals"], {})

    # ── BCI / CCI modes ─────────────────────────────────────────────

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_bci_mode(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_single_country_csv())

        tool = self._make_tool()
        result = tool.execute(mode="bci", countries="USA")
        self.assertTrue(result.success)
        self.assertEqual(result.data["mode"], "bci")

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_cci_mode(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_single_country_csv())

        tool = self._make_tool()
        result = tool.execute(mode="cci", countries="USA")
        self.assertTrue(result.success)
        self.assertEqual(result.data["mode"], "cci")

    # ── CSV parsing edge cases ──────────────────────────────────────

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_sparse_csv_missing_values(self, mock_cls):
        """Missing OBS_VALUE rows are skipped gracefully."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_sparse_csv())

        tool = self._make_tool()
        result = tool.execute(mode="cli", countries="USA,DEU")

        self.assertTrue(result.success)
        # USA should have 1 row (the blank OBS_VALUE skipped), DEU has 1
        usa_rows = result.data["by_country"].get("USA", [])
        deu_rows = result.data["by_country"].get("DEU", [])
        self.assertEqual(len(usa_rows), 1)
        self.assertEqual(len(deu_rows), 1)

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_empty_csv_response(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response("")

        tool = self._make_tool()
        result = tool.execute(mode="cli")

        self.assertTrue(result.success)
        self.assertIn("No data", result.output)

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_csv_header_only(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response("REF_AREA,FREQ,MEASURE,TIME_PERIOD,OBS_VALUE\n")

        tool = self._make_tool()
        result = tool.execute(mode="cli")
        self.assertTrue(result.success)
        self.assertIn("No data", result.output)

    # ── HTTP errors ─────────────────────────────────────────────────

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_http_404_error(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response("", status=404)

        tool = self._make_tool()
        result = tool.execute(mode="cli")
        self.assertFalse(result.success)

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_http_500_error(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response("", status=500)

        tool = self._make_tool()
        result = tool.execute(mode="cli")
        self.assertFalse(result.success)

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_network_timeout(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = httpx.ReadTimeout("Read timed out")

        tool = self._make_tool()
        result = tool.execute(mode="cli")
        self.assertFalse(result.success)
        self.assertIn("error", result.output.lower())

    # ── Country parsing ─────────────────────────────────────────────

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_single_country(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_single_country_csv("JPN"))

        tool = self._make_tool()
        result = tool.execute(mode="cli", countries="JPN")
        self.assertTrue(result.success)
        self.assertIn("JPN", result.data["by_country"])

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_lowercase_countries_normalized(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_single_country_csv("USA"))

        tool = self._make_tool()
        result = tool.execute(mode="cli", countries="usa")
        self.assertTrue(result.success)

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_empty_countries_uses_default(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_csv())

        tool = self._make_tool()
        result = tool.execute(mode="cli", countries="")
        self.assertTrue(result.success)

    # ── Cache integration ───────────────────────────────────────────

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_cache_miss_stores(self, mock_cls):
        cache = MagicMock()
        cache.get.return_value = None

        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_csv())

        tool = self._make_tool(cache=cache)
        result = tool.execute(mode="cli")

        self.assertTrue(result.success)
        cache.put.assert_called_once()

    def test_cache_hit_skips_http(self):
        from agent.tools.global_pmi import GlobalPmiTool

        cache = MagicMock()
        # Return pre-parsed rows
        cache.get.return_value = [
            {"country": "USA", "period": "2026-01", "value": 101.0},
        ]

        tool = GlobalPmiTool(cache=cache)
        result = tool.execute(mode="cli", countries="USA")

        self.assertTrue(result.success)
        cache.get.assert_called_once()

    # ── Tool schema ─────────────────────────────────────────────────

    def test_tool_schema(self):
        tool = self._make_tool()
        self.assertEqual(tool.name, "global_pmi")
        self.assertIn("mode", tool.parameters["properties"])
        self.assertIn("countries", tool.parameters["properties"])
        schema = tool.to_openai_tool()
        self.assertEqual(schema["function"]["name"], "global_pmi")

    def test_valid_modes_constant(self):
        from agent.tools.global_pmi import VALID_MODES

        self.assertEqual(VALID_MODES, frozenset({"cli", "bci", "cci"}))

    # ── Regime edge cases ───────────────────────────────────────────

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_regime_peaking(self, mock_cls):
        """CLI > 100 but declining → peaking."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        csv = _make_single_country_csv("USA", [("2025-12", "101.50"), ("2026-01", "101.20")])
        client.get.return_value = _mock_response(csv)

        tool = self._make_tool()
        result = tool.execute(mode="cli", countries="USA")

        signals = result.data["signals"].get("USA", {})
        self.assertEqual(signals.get("regime"), "peaking")

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_regime_troughing(self, mock_cls):
        """CLI < 100 but rising → troughing."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        csv = _make_single_country_csv("USA", [("2025-12", "98.50"), ("2026-01", "98.80")])
        client.get.return_value = _mock_response(csv)

        tool = self._make_tool()
        result = tool.execute(mode="cli", countries="USA")

        signals = result.data["signals"].get("USA", {})
        self.assertEqual(signals.get("regime"), "troughing")

    # ── kwargs passthrough ──────────────────────────────────────────

    @patch("agent.tools.global_pmi.httpx.Client")
    def test_extra_kwargs_ignored(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_csv())

        tool = self._make_tool()
        result = tool.execute(mode="cli", unknown_param="hello")
        self.assertTrue(result.success)


# ──────────────────────────────────────────────────────────────────
# Phase 28: L2 economic-activity entity persistence
# ──────────────────────────────────────────────────────────────────


def _make_store_mock():
    """Build a mock PipelineStore for L2 persistence testing."""
    store = MagicMock()
    store.register_entity = MagicMock(side_effect=lambda **kw: kw["entity_id"])
    store.store_entity_observation = MagicMock(return_value=1)
    return store


class TestL2PersistenceGuards(unittest.TestCase):
    """Persistence guards: no store or no entity_id_from_key → no-op."""

    def test_no_store_returns_zeros(self):
        from agent.tools.global_pmi import GlobalPmiTool

        tool = GlobalPmiTool()
        tool._store = None
        counts = tool._persist_entities({"by_country": {}, "signals": {}}, "cli")
        self.assertEqual(counts, {"economic_activity_obs": 0})

    def test_no_entity_id_fn_returns_zeros(self):
        import agent.tools.global_pmi as gp_mod
        from agent.tools.global_pmi import GlobalPmiTool

        tool = GlobalPmiTool()
        tool._store = _make_store_mock()
        original = gp_mod._entity_id_from_key
        try:
            gp_mod._entity_id_from_key = None
            counts = tool._persist_entities({"by_country": {}, "signals": {}}, "cli")
            self.assertEqual(counts, {"economic_activity_obs": 0})
        finally:
            gp_mod._entity_id_from_key = original

    def test_inner_exception_returns_zeros(self):
        from agent.tools.global_pmi import GlobalPmiTool

        tool = GlobalPmiTool()
        store = _make_store_mock()
        store.register_entity.side_effect = RuntimeError("DB down")
        tool._store = store
        data = {
            "by_country": {"USA": [{"value": 101.0, "period": "2026-01"}]},
            "signals": {},
        }
        counts = tool._persist_entities(data, "cli")
        self.assertEqual(counts, {"economic_activity_obs": 0})


class TestL2PersistencePerCountry(unittest.TestCase):
    """economic_activity obs persisted per-country with ISO3→ISO2 mapping."""

    def test_multiple_countries_persisted(self):
        from agent.tools.global_pmi import GlobalPmiTool

        tool = GlobalPmiTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "by_country": {
                "USA": [{"value": 101.0, "period": "2026-01"}],
                "DEU": [{"value": 99.0, "period": "2026-01"}],
                "JPN": [{"value": 100.5, "period": "2026-01"}],
            },
            "signals": {
                "USA": {"regime": "expanding", "momentum_6m": 1.4},
                "DEU": {"regime": "contracting"},
                "JPN": {"regime": "troughing", "momentum_6m": 0.5},
            },
        }
        counts = tool._persist_entities(data, "cli")
        self.assertEqual(counts["economic_activity_obs"], 3)

    def test_obs_type_and_depth(self):
        from agent.tools.global_pmi import GlobalPmiTool

        tool = GlobalPmiTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "by_country": {"USA": [{"value": 101.0, "period": "2026-01"}]},
            "signals": {"USA": {"regime": "expanding"}},
        }
        tool._persist_entities(data, "cli")
        obs = store.store_entity_observation.call_args_list[0]
        self.assertEqual(obs.kwargs["observation_type"], "economic_activity")
        self.assertEqual(obs.kwargs["depth_level"], 2)
        self.assertEqual(obs.kwargs["source_tool"], "global_pmi")

    def test_targets_correct_country_entity(self):
        from agent.pipeline.entity import entity_id_from_key
        from agent.tools.global_pmi import GlobalPmiTool

        tool = GlobalPmiTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "by_country": {"DEU": [{"value": 99.0, "period": "2026-01"}]},
            "signals": {},
        }
        tool._persist_entities(data, "cli")
        de_eid = entity_id_from_key("country", "DE")
        obs = store.store_entity_observation.call_args_list[0]
        self.assertEqual(obs.kwargs["entity_id"], de_eid)

    def test_value_fields_complete(self):
        from agent.tools.global_pmi import GlobalPmiTool

        tool = GlobalPmiTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "by_country": {"USA": [{"value": 101.0, "period": "2026-01"}]},
            "signals": {"USA": {"regime": "expanding", "momentum_6m": 1.41}},
        }
        tool._persist_entities(data, "cli")
        val = store.store_entity_observation.call_args_list[0].kwargs["value"]
        self.assertEqual(val["indicator"], "cli")
        self.assertEqual(val["value"], 101.0)
        self.assertEqual(val["period"], "2026-01")
        self.assertEqual(val["regime"], "expanding")
        self.assertAlmostEqual(val["momentum_6m"], 1.41)


class TestL2PersistenceAggregatesSkipped(unittest.TestCase):
    """Aggregates like OECD, G-7, EA19, G-20 have no ISO2 → skipped."""

    def test_oecd_aggregate_skipped(self):
        from agent.tools.global_pmi import GlobalPmiTool

        tool = GlobalPmiTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "by_country": {
                "OECD": [{"value": 100.2, "period": "2026-01"}],
                "G-7": [{"value": 100.1, "period": "2026-01"}],
            },
            "signals": {},
        }
        counts = tool._persist_entities(data, "cli")
        self.assertEqual(counts["economic_activity_obs"], 0)
        store.store_entity_observation.assert_not_called()

    def test_mix_aggregate_and_country(self):
        from agent.tools.global_pmi import GlobalPmiTool

        tool = GlobalPmiTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "by_country": {
                "OECD": [{"value": 100.2, "period": "2026-01"}],
                "USA": [{"value": 101.0, "period": "2026-01"}],
                "EA19": [{"value": 99.8, "period": "2026-01"}],
            },
            "signals": {},
        }
        counts = tool._persist_entities(data, "cli")
        self.assertEqual(counts["economic_activity_obs"], 1)  # only USA


class TestL2PersistenceISO3Map(unittest.TestCase):
    """ISO3_TO_ISO2 mapping sanity."""

    def test_iso3_map_all_uppercase(self):
        from agent.tools.global_pmi import ISO3_TO_ISO2

        for k, v in ISO3_TO_ISO2.items():
            self.assertEqual(k, k.upper(), f"Key {k} not uppercase")
            self.assertEqual(v, v.upper(), f"Value {v} not uppercase")
            self.assertEqual(len(v), 2, f"Value {v} not 2-char ISO-2")

    def test_iso3_map_has_major_countries(self):
        from agent.tools.global_pmi import ISO3_TO_ISO2

        for iso3 in ("USA", "GBR", "DEU", "FRA", "JPN", "CHN", "KOR", "AUS", "CAN"):
            self.assertIn(iso3, ISO3_TO_ISO2)

    def test_bci_mode_persists_same_way(self):
        """BCI mode uses same persistence path as CLI."""
        from agent.tools.global_pmi import GlobalPmiTool

        tool = GlobalPmiTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "by_country": {"FRA": [{"value": 100.5, "period": "2026-01"}]},
            "signals": {"FRA": {"regime": "expanding"}},
        }
        counts = tool._persist_entities(data, "bci")
        self.assertEqual(counts["economic_activity_obs"], 1)
        val = store.store_entity_observation.call_args_list[0].kwargs["value"]
        self.assertEqual(val["indicator"], "bci")


if __name__ == "__main__":
    unittest.main()
