"""Tests for Idea 13 — Data Governance Catalog (catalog.py).

Covers:
    1.  ToolMeta is frozen (immutable)
    2.  _TOOL_MANIFEST covers expected categories
    3.  _MANIFEST_BY_NAME keyed by tool name
    4.  DataCatalog defaults (max_tools=200)
    5.  DataCatalog.register() adds a new tool
    6.  DataCatalog.register() overwrites an existing tool
    7.  DataCatalog.tool_names sorted alphabetically
    8.  DataCatalog.get_meta() returns correct ToolMeta
    9.  DataCatalog.get_meta() returns None for unknown tool
    10. DataCatalog.schema_fingerprint() is deterministic
    11. DataCatalog.schema_fingerprint() changes on metadata edit
    12. DataCatalog.schema_fingerprint() returns empty for unknown
    13. FreshnessStatus.is_breach when freshness > sla
    14. FreshnessStatus.is_breach False when freshness <= sla
    15. FreshnessStatus.hours_overdue correct
    16. check_freshness: never-seen tool gets inf freshness + breach
    17. check_freshness: fresh tool (observed now) is not breached
    18. check_freshness: stale tool (observed past SLA) is breached
    19. check_freshness: CatalogReport counts match statuses
    20. check_freshness: breached_tools property correct
    21. check_freshness: never_seen_tools property correct
    22. check_freshness: max_tools cap respected
    23. get_lineage: returns correct source tools for entity
    24. get_lineage: returns empty list for unknown entity
    25. get_lineage: handles store error gracefully
    26. store_freshness_signals: uses correct signal names
    27. store_freshness_signals: skips never-seen + no-breach tools
    28. store_freshness_signals: handles store error gracefully
    29. TrainerConfig.use_data_catalog defaults False
    30. TrainerConfig.catalog_max_tools defaults 200
    31. TrainerConfig.catalog_sla_multiplier defaults 1.0
    32. Trainer.check_data_freshness returns None when flag=False
    33. Trainer.check_data_freshness returns CatalogReport when flag=True
    34. SLA multiplier doubles all SLA thresholds correctly
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from agent.data_catalog.catalog import (
    CatalogReport,
    DataCatalog,
    FreshnessStatus,
    ToolMeta,
    _MANIFEST_BY_NAME,
    _TOOL_MANIFEST,
)
from agent.models.gnn.trainer import Trainer, TrainerConfig, SyntheticGraphGenerator
from agent.pipeline.store import PipelineStore

# ── Helpers ───────────────────────────────────────────────────────────────


def _make_store(tmp_path: Path, name: str = "cat.db") -> PipelineStore:
    return PipelineStore(str(tmp_path / name))


def _add_observation(
    store: PipelineStore, tool: str, entity_id: str, ts: float
) -> None:
    store.store_entity_observation(
        entity_id=entity_id,
        source_tool=tool,
        observed_at=ts,
        observation_type="price",
        value=1.0,
    )


def _make_trainer(tmp_path: Path, use_catalog: bool = False) -> Trainer:
    store = _make_store(tmp_path, "trainer.db")
    gen = SyntheticGraphGenerator(
        num_companies=2,
        num_countries=1,
        time_span=3600.0 * 3,
        base_event_rate=0.001,
        seed=42,
    )
    gen.generate(store)
    cfg = TrainerConfig(
        hidden_dim=16,
        memory_dim=16,
        message_dim=16,
        time_dim=8,
        num_heads=1,
        num_layers=1,
        use_data_catalog=use_catalog,
    )
    return Trainer(store, cfg)


# ═══════════════════════════════════════════════════════════════
# 1–3. ToolMeta & manifest
# ═══════════════════════════════════════════════════════════════


class TestToolMeta:

    def test_frozen(self):
        tm = ToolMeta("x", "cat", 24.0, 48.0)
        with pytest.raises((AttributeError, TypeError)):
            tm.name = "y"  # type: ignore[misc]

    def test_manifest_covers_maritime(self):
        categories = {m.category for m in _TOOL_MANIFEST}
        assert "maritime" in categories

    def test_manifest_covers_macro(self):
        categories = {m.category for m in _TOOL_MANIFEST}
        assert "macro" in categories

    def test_manifest_by_name_keyed_correctly(self):
        for name, meta in _MANIFEST_BY_NAME.items():
            assert meta.name == name

    def test_manifest_no_duplicate_names(self):
        names = [m.name for m in _TOOL_MANIFEST]
        assert len(names) == len(set(names))

    def test_all_sla_geq_frequency(self):
        for m in _TOOL_MANIFEST:
            assert (
                m.sla_hours >= m.frequency_hours
            ), f"{m.name}: sla_hours {m.sla_hours} < frequency {m.frequency_hours}"


# ═══════════════════════════════════════════════════════════════
# 4–12. DataCatalog construction & registry
# ═══════════════════════════════════════════════════════════════


class TestDataCatalog:

    def test_defaults(self):
        dc = DataCatalog()
        assert dc._max_tools == 200
        assert len(dc._registry) == len(_TOOL_MANIFEST)

    def test_register_new_tool(self):
        dc = DataCatalog()
        tm = ToolMeta("my_new_tool", "custom", 12.0, 24.0)
        dc.register(tm)
        assert "my_new_tool" in dc._registry

    def test_register_overwrites(self):
        dc = DataCatalog()
        tm = ToolMeta("cftc", "custom", 1.0, 2.0)
        dc.register(tm)
        assert dc._registry["cftc"].sla_hours == pytest.approx(2.0)

    def test_tool_names_sorted(self):
        dc = DataCatalog()
        names = dc.tool_names
        assert names == sorted(names)

    def test_get_meta_known(self):
        dc = DataCatalog()
        meta = dc.get_meta("ais_vessel")
        assert meta is not None
        assert meta.name == "ais_vessel"
        assert meta.category == "maritime"

    def test_get_meta_unknown_returns_none(self):
        dc = DataCatalog()
        assert dc.get_meta("nonexistent_tool_xyz") is None

    def test_schema_fingerprint_deterministic(self):
        dc = DataCatalog()
        fp1 = dc.schema_fingerprint("cftc")
        fp2 = dc.schema_fingerprint("cftc")
        assert fp1 == fp2 and len(fp1) == 16

    def test_schema_fingerprint_changes_on_edit(self):
        dc = DataCatalog()
        fp_original = dc.schema_fingerprint("cftc")
        dc.register(ToolMeta("cftc", "positioning", 100.0, 200.0))
        fp_modified = dc.schema_fingerprint("cftc")
        assert fp_original != fp_modified

    def test_schema_fingerprint_unknown_returns_empty(self):
        dc = DataCatalog()
        assert dc.schema_fingerprint("no_such_tool") == ""


# ═══════════════════════════════════════════════════════════════
# 13–22. check_freshness
# ═══════════════════════════════════════════════════════════════


class TestCheckFreshness:

    def test_freshness_breach_when_stale(self, tmp_path):
        store = _make_store(tmp_path, "stale.db")
        dc = DataCatalog()
        # Add a very old observation (1000 hours ago)
        old_ts = time.time() - 1000 * 3600
        _add_observation(store, "cftc", "crude", old_ts)
        report = dc.check_freshness(store)
        assert report.statuses["cftc"].is_breach is True

    def test_freshness_healthy_when_recent(self, tmp_path):
        store = _make_store(tmp_path, "fresh.db")
        dc = DataCatalog()
        _add_observation(store, "cftc", "crude", time.time())
        report = dc.check_freshness(store)
        assert report.statuses["cftc"].is_breach is False

    def test_never_seen_tool_is_breach(self, tmp_path):
        store = _make_store(tmp_path, "ns.db")
        dc = DataCatalog()
        report = dc.check_freshness(store)
        s = report.statuses["cftc"]
        assert s.last_seen_at == 0.0
        assert s.is_breach is True

    def test_hours_overdue_correct(self, tmp_path):
        store = _make_store(tmp_path, "od.db")
        dc = DataCatalog()
        sla_hours = dc.get_meta("cftc").sla_hours  # 336
        now = time.time()
        overdue_by = 24.0  # 24h past SLA
        old_ts = now - (sla_hours + overdue_by) * 3600
        _add_observation(store, "cftc", "crude", old_ts)
        report = dc.check_freshness(store, now=now)
        assert report.statuses["cftc"].hours_overdue == pytest.approx(
            overdue_by, abs=0.1
        )

    def test_report_counts_match(self, tmp_path):
        store = _make_store(tmp_path, "cnt.db")
        dc = DataCatalog()
        report = dc.check_freshness(store)
        total = report.n_breached + report.n_healthy
        assert total == len(report.statuses)

    def test_breached_tools_property(self, tmp_path):
        store = _make_store(tmp_path, "bt.db")
        dc = DataCatalog()
        _add_observation(store, "cftc", "crude", time.time())
        report = dc.check_freshness(store)
        assert "cftc" not in report.breached_tools

    def test_never_seen_tools_property(self, tmp_path):
        store = _make_store(tmp_path, "nv.db")
        dc = DataCatalog()
        report = dc.check_freshness(store)
        assert "cftc" in report.never_seen_tools

    def test_max_tools_cap(self, tmp_path):
        store = _make_store(tmp_path, "cap.db")
        dc = DataCatalog(max_tools=3)
        report = dc.check_freshness(store)
        assert len(report.statuses) <= 3


# ═══════════════════════════════════════════════════════════════
# 23–25. get_lineage
# ═══════════════════════════════════════════════════════════════


class TestGetLineage:

    def test_returns_correct_tools(self, tmp_path):
        store = _make_store(tmp_path, "lin.db")
        now = time.time()
        _add_observation(store, "cftc", "crude", now)
        _add_observation(store, "ais_vessel", "crude", now - 10)
        dc = DataCatalog()
        lineage = dc.get_lineage(store, "crude")
        assert "ais_vessel" in lineage
        assert "cftc" in lineage

    def test_empty_for_unknown_entity(self, tmp_path):
        store = _make_store(tmp_path, "lin2.db")
        dc = DataCatalog()
        assert dc.get_lineage(store, "entity_that_never_existed") == []

    def test_handles_store_error_gracefully(self):
        mock_store = MagicMock()
        mock_store._backend.fetchall.side_effect = RuntimeError("db error")
        dc = DataCatalog()
        result = dc.get_lineage(mock_store, "some_entity")
        assert result == []


# ═══════════════════════════════════════════════════════════════
# 26–28. store_freshness_signals
# ═══════════════════════════════════════════════════════════════


class TestStoreFreshnessSignals:

    def _make_report(self, tool_name, freshness_h, sla_h) -> CatalogReport:
        is_breach = freshness_h > sla_h
        s = FreshnessStatus(
            tool_name=tool_name,
            last_seen_at=time.time() - freshness_h * 3600,
            freshness_hours=freshness_h,
            sla_hours=sla_h,
            is_breach=is_breach,
            hours_overdue=max(0.0, freshness_h - sla_h),
            checked_at=time.time(),
        )
        n_breached = 1 if is_breach else 0
        return CatalogReport(
            statuses={tool_name: s},
            n_breached=n_breached,
            n_never_seen=0,
            n_healthy=1 - n_breached,
            generated_at=time.time(),
        )

    def test_correct_signal_names_on_breach(self):
        mock_store = MagicMock()
        dc = DataCatalog()
        report = self._make_report("cftc", freshness_h=400, sla_h=336)
        dc.store_freshness_signals(mock_store, report)
        names = {
            c.kwargs["signal_name"] for c in mock_store.store_signal.call_args_list
        }
        assert "catalog.cftc.freshness_hours" in names
        assert "catalog.cftc.sla_breach" in names

    def test_skips_healthy_never_seen(self):
        mock_store = MagicMock()
        dc = DataCatalog()
        # A tool that has NEVER been seen AND has no breach → skip to avoid noise
        s = FreshnessStatus(
            tool_name="cftc",
            last_seen_at=0.0,
            freshness_hours=float("inf"),
            sla_hours=336.0,
            is_breach=True,
            hours_overdue=0.0,  # never seen so overdue=0 but still breach
            checked_at=time.time(),
        )
        # Manually force is_breach=False + last_seen=0 to exercise the skip path
        s2 = FreshnessStatus(
            tool_name="cftc",
            last_seen_at=0.0,
            freshness_hours=float("inf"),
            sla_hours=336.0,
            is_breach=False,
            hours_overdue=0.0,
            checked_at=time.time(),
        )
        from agent.data_catalog.catalog import CatalogReport

        report = CatalogReport(
            statuses={"cftc": s2},
            n_breached=0,
            n_never_seen=1,
            n_healthy=1,
            generated_at=time.time(),
        )
        n = dc.store_freshness_signals(mock_store, report)
        assert n == 0  # not_breach AND never_seen → skip

    def test_handles_store_error_gracefully(self):
        mock_store = MagicMock()
        mock_store.store_signal.side_effect = RuntimeError("disk full")
        dc = DataCatalog()
        report = self._make_report("cftc", freshness_h=1000, sla_h=336)
        n = dc.store_freshness_signals(mock_store, report)
        assert n == 0


# ═══════════════════════════════════════════════════════════════
# 29–31. TrainerConfig defaults
# ═══════════════════════════════════════════════════════════════


class TestTrainerConfigDefaults:

    def test_use_data_catalog_false(self):
        assert TrainerConfig().use_data_catalog is False

    def test_catalog_max_tools_200(self):
        assert TrainerConfig().catalog_max_tools == 200

    def test_catalog_sla_multiplier_one(self):
        assert TrainerConfig().catalog_sla_multiplier == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════
# 32–34. Trainer.check_data_freshness
# ═══════════════════════════════════════════════════════════════


class TestTrainerCheckDataFreshness:

    def test_returns_none_when_disabled(self, tmp_path):
        trainer = _make_trainer(tmp_path, use_catalog=False)
        assert trainer.check_data_freshness() is None

    def test_returns_catalog_report_when_enabled(self, tmp_path):
        trainer = _make_trainer(tmp_path, use_catalog=True)
        report = trainer.check_data_freshness()
        assert isinstance(report, CatalogReport)
        assert len(report.statuses) > 0

    def test_sla_multiplier_doubles_thresholds(self, tmp_path):
        store = _make_store(tmp_path, "mul.db")
        gen = SyntheticGraphGenerator(
            num_companies=2,
            num_countries=1,
            time_span=3600.0 * 3,
            base_event_rate=0.001,
            seed=3,
        )
        gen.generate(store)
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_data_catalog=True,
            catalog_sla_multiplier=2.0,
        )
        trainer = Trainer(store, cfg)
        report = trainer.check_data_freshness()
        dc_base = DataCatalog()
        # For each tool, scaled SLA should be 2× base
        for tool_name, status in report.statuses.items():
            base_meta = dc_base.get_meta(tool_name)
            if base_meta:
                assert status.sla_hours == pytest.approx(
                    base_meta.sla_hours * 2.0, abs=0.01
                )
