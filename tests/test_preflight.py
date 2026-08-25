"""Tests for agent/preflight.py — Feature Preflight System.

Covers:
    1.  PreflightResult.passed() has ok=True, reason=OK
    2.  PreflightResult.missing_config() has ok=False, MISSING_CONFIG
    3.  PreflightResult.no_data() has ok=False, NO_DATA
    4.  PreflightResult.model_not_ready() has ok=False, MODEL_NOT_READY
    5.  PreflightResult.user_message includes fix hint on failure
    6.  PreflightResult.user_message is plain "passed" when ok
    7.  for_nightlight: fails MISSING_CONFIG when key absent (nightlight mode)
    8.  for_nightlight: passes when key present (nightlight mode)
    9.  for_nightlight: passes for ndvi mode even without FIRMS key
    10. for_nightlight: fails NO_DATA when data is stale
    11. for_nightlight: passes when data is fresh
    12. for_attribution: fails MODEL_NOT_READY when model=None
    13. for_attribution: passes when model is provided
    14. for_attribution: fails NO_DATA when no instrument nodes
    15. for_attribution: passes when instrument nodes present
    16. for_portfolio: fails NO_DATA when fewer than min_assets predictions
    17. for_portfolio: passes with sufficient predictions
    18. for_data_catalog: fails MISSING_CONFIG when flag=False
    19. for_data_catalog: fails MISSING_CONFIG when store=None
    20. for_data_catalog: passes when flag=True and store present
    21. for_gnn_inference: fails MODEL_NOT_READY when model=None
    22. for_gnn_inference: passes when model provided
    23. for_api_key: fails MISSING_CONFIG when key absent
    24. for_api_key: passes when key present
    25. for_api_key: checks env var as fallback
    26. _check_data_staleness: returns NO_DATA when no rows
    27. _check_data_staleness: returns NO_DATA when data too old
    28. _check_data_staleness: returns OK when data fresh
    29. _check_data_staleness: returns OK on DB error (fail-open)
    30. NightlightActivityTool: preflight stops execution for missing FIRMS key in nightlight mode
    31. compute_attribution: preflight returns {} when model=None
    32. check_data_freshness: preflight returns None when use_data_catalog=False
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.preflight import (
    FailureReason,
    FeaturePreflight,
    PreflightResult,
)
from agent.models.gnn.trainer import Trainer, TrainerConfig, SyntheticGraphGenerator
from agent.pipeline.store import PipelineStore
from agent.tools.nightlight_activity import NightlightActivityTool


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_store(tmp_path: Path, name: str = "pf.db") -> PipelineStore:
    return PipelineStore(str(tmp_path / name))


def _add_obs(store: PipelineStore, tool: str, entity_id: str, ts: float) -> None:
    store.store_entity_observation(
        entity_id=entity_id,
        source_tool=tool,
        observed_at=ts,
        observation_type="price",
        value=1.0,
    )


def _make_trainer(tmp_path: Path, **cfg_kwargs) -> Trainer:
    store = _make_store(tmp_path, "trainer_pf.db")
    gen = SyntheticGraphGenerator(
        num_companies=2, num_countries=1,
        time_span=3600.0 * 3, base_event_rate=0.001, seed=99,
    )
    gen.generate(store)
    cfg = TrainerConfig(
        hidden_dim=16, memory_dim=16, message_dim=16, time_dim=8,
        num_heads=1, num_layers=1, **cfg_kwargs,
    )
    return Trainer(store, cfg)


# ═══════════════════════════════════════════════════════════════
# 1–6. PreflightResult
# ═══════════════════════════════════════════════════════════════

class TestPreflightResult:

    def test_passed_ok_true(self):
        r = PreflightResult.passed()
        assert r.ok is True
        assert r.reason == FailureReason.OK

    def test_missing_config_ok_false(self):
        r = PreflightResult.missing_config("no key", "set it")
        assert r.ok is False
        assert r.reason == FailureReason.MISSING_CONFIG

    def test_no_data_ok_false(self):
        r = PreflightResult.no_data("no rows")
        assert r.ok is False
        assert r.reason == FailureReason.NO_DATA

    def test_model_not_ready_ok_false(self):
        r = PreflightResult.model_not_ready("not built")
        assert r.ok is False
        assert r.reason == FailureReason.MODEL_NOT_READY

    def test_user_message_includes_fix(self):
        r = PreflightResult.missing_config("key absent", "export KEY=xxx")
        assert "export KEY=xxx" in r.user_message
        assert "MISSING_CONFIG" in r.user_message

    def test_user_message_passed_is_plain(self):
        r = PreflightResult.passed()
        assert "passed" in r.user_message.lower()


# ═══════════════════════════════════════════════════════════════
# 7–11. for_nightlight
# ═══════════════════════════════════════════════════════════════

class TestForNightlight:

    def test_fails_missing_config_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove FIRMS_API_KEY if set
            os.environ.pop("FIRMS_API_KEY", None)
            ok, r = FeaturePreflight.for_nightlight(firms_api_key="", mode="nightlight")
        assert ok is False
        assert r.reason == FailureReason.MISSING_CONFIG

    def test_passes_with_key(self):
        ok, r = FeaturePreflight.for_nightlight(firms_api_key="MY_KEY", mode="nightlight")
        assert ok is True

    def test_passes_ndvi_mode_without_firms_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("FIRMS_API_KEY", None)
            ok, r = FeaturePreflight.for_nightlight(firms_api_key="", mode="ndvi")
        assert ok is True

    def test_fails_no_data_when_stale(self, tmp_path):
        store = _make_store(tmp_path, "nl_stale.db")
        old_ts = time.time() - 400 * 3600  # 400h ago
        _add_obs(store, "nightlight_activity", "zone1", old_ts)
        ok, r = FeaturePreflight.for_nightlight(
            firms_api_key="KEY", store=store, mode="ndvi", max_stale_hours=336.0
        )
        assert ok is False
        assert r.reason == FailureReason.NO_DATA

    def test_passes_when_data_fresh(self, tmp_path):
        store = _make_store(tmp_path, "nl_fresh.db")
        _add_obs(store, "nightlight_activity", "zone1", time.time())
        ok, r = FeaturePreflight.for_nightlight(
            firms_api_key="KEY", store=store, mode="ndvi", max_stale_hours=336.0
        )
        assert ok is True


# ═══════════════════════════════════════════════════════════════
# 12–15. for_attribution
# ═══════════════════════════════════════════════════════════════

class TestForAttribution:

    def test_fails_model_none(self):
        ok, r = FeaturePreflight.for_attribution(model=None)
        assert ok is False
        assert r.reason == FailureReason.MODEL_NOT_READY

    def test_passes_with_model(self):
        model = MagicMock()
        ok, r = FeaturePreflight.for_attribution(model=model)
        assert ok is True

    def test_fails_no_instrument_nodes(self):
        model = MagicMock()
        id_map = MagicMock()
        id_map.type_local = {}  # no instruments
        ok, r = FeaturePreflight.for_attribution(
            model=model, id_map=id_map, min_instrument_nodes=1
        )
        assert ok is False
        assert r.reason == FailureReason.NO_DATA

    def test_passes_with_instrument_nodes(self):
        model = MagicMock()
        id_map = MagicMock()
        id_map.type_local = {"instrument": {"e1": 0, "e2": 1}}
        ok, r = FeaturePreflight.for_attribution(
            model=model, id_map=id_map, min_instrument_nodes=1
        )
        assert ok is True


# ═══════════════════════════════════════════════════════════════
# 16–17. for_portfolio
# ═══════════════════════════════════════════════════════════════

class TestForPortfolio:

    def test_fails_too_few_predictions(self):
        ok, r = FeaturePreflight.for_portfolio(
            store=None, return_preds={"e1": 0.01}, min_assets=2
        )
        assert ok is False
        assert r.reason == FailureReason.NO_DATA

    def test_passes_with_enough_predictions(self):
        ok, r = FeaturePreflight.for_portfolio(
            store=None, return_preds={"e1": 0.01, "e2": -0.02}, min_assets=2
        )
        assert ok is True


# ═══════════════════════════════════════════════════════════════
# 18–20. for_data_catalog
# ═══════════════════════════════════════════════════════════════

class TestForDataCatalog:

    def test_fails_when_flag_false(self):
        ok, r = FeaturePreflight.for_data_catalog(store=MagicMock(), use_data_catalog=False)
        assert ok is False
        assert r.reason == FailureReason.MISSING_CONFIG

    def test_fails_when_store_none(self):
        ok, r = FeaturePreflight.for_data_catalog(store=None, use_data_catalog=True)
        assert ok is False
        assert r.reason == FailureReason.MISSING_CONFIG

    def test_passes_when_enabled_with_store(self):
        ok, r = FeaturePreflight.for_data_catalog(store=MagicMock(), use_data_catalog=True)
        assert ok is True


# ═══════════════════════════════════════════════════════════════
# 21–22. for_gnn_inference
# ═══════════════════════════════════════════════════════════════

class TestForGnnInference:

    def test_fails_model_none(self):
        ok, r = FeaturePreflight.for_gnn_inference(model=None, store=None)
        assert ok is False
        assert r.reason == FailureReason.MODEL_NOT_READY

    def test_passes_with_model(self):
        ok, r = FeaturePreflight.for_gnn_inference(model=MagicMock(), store=None)
        assert ok is True


# ═══════════════════════════════════════════════════════════════
# 23–25. for_api_key
# ═══════════════════════════════════════════════════════════════

class TestForApiKey:

    def test_fails_no_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MY_TEST_KEY", None)
            ok, r = FeaturePreflight.for_api_key(
                key_value="", env_var="MY_TEST_KEY", tool_name="TestTool"
            )
        assert ok is False
        assert r.reason == FailureReason.MISSING_CONFIG

    def test_passes_with_direct_key(self):
        ok, r = FeaturePreflight.for_api_key(
            key_value="abc123", env_var="MY_TEST_KEY", tool_name="TestTool"
        )
        assert ok is True

    def test_passes_via_env_var(self):
        with patch.dict(os.environ, {"MY_TEST_KEY": "env_val"}):
            ok, r = FeaturePreflight.for_api_key(
                key_value=None, env_var="MY_TEST_KEY", tool_name="TestTool"
            )
        assert ok is True


# ═══════════════════════════════════════════════════════════════
# 26–29. _check_data_staleness
# ═══════════════════════════════════════════════════════════════

class TestCheckDataStaleness:

    def test_no_rows_returns_no_data(self, tmp_path):
        store = _make_store(tmp_path, "stale_a.db")
        r = FeaturePreflight._check_data_staleness(store, "some_tool", 24.0)
        assert r.reason == FailureReason.NO_DATA

    def test_too_old_returns_no_data(self, tmp_path):
        store = _make_store(tmp_path, "stale_b.db")
        _add_obs(store, "some_tool", "e1", time.time() - 100 * 3600)
        r = FeaturePreflight._check_data_staleness(store, "some_tool", 24.0)
        assert r.reason == FailureReason.NO_DATA

    def test_fresh_returns_ok(self, tmp_path):
        store = _make_store(tmp_path, "stale_c.db")
        _add_obs(store, "some_tool", "e1", time.time())
        r = FeaturePreflight._check_data_staleness(store, "some_tool", 24.0)
        assert r.ok is True

    def test_db_error_is_fail_open(self):
        mock_store = MagicMock()
        mock_store._get_conn.side_effect = RuntimeError("db gone")
        r = FeaturePreflight._check_data_staleness(mock_store, "tool", 24.0)
        assert r.ok is True  # fail-open, don't block on DB errors


# ═══════════════════════════════════════════════════════════════
# 30–32. Integration with NightlightActivityTool + Trainer
# ═══════════════════════════════════════════════════════════════

class TestIntegration:

    def test_nightlight_tool_stops_on_missing_firms_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FIRMS_API_KEY", None)
            tool = NightlightActivityTool(firms_api_key="")
            result = tool.execute(mode="nightlight")
        assert result.success is False
        assert "FIRMS_API_KEY" in result.output

    def test_compute_attribution_returns_empty_when_model_none(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        # Don't call build_model()
        result = trainer.compute_attribution()
        assert result == {}

    def test_check_data_freshness_returns_none_when_disabled(self, tmp_path):
        trainer = _make_trainer(tmp_path, use_data_catalog=False)
        assert trainer.check_data_freshness() is None
