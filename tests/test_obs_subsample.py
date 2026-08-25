"""Tests for training observation subsampling."""

from __future__ import annotations

from agent.models.gnn.obs_subsample import (
    RAW_SOURCE_TOOLS_FLOOR,
    apply_training_obs_subsample,
)


def _obs(obs_type: str, source: str, ts: float = 1.0) -> dict:
    return {
        "entity_id": "e1",
        "observation_type": obs_type,
        "source_tool": source,
        "observed_at": ts,
    }


def test_floor_keeps_ais_at_full_rate():
    obs = [_obs("vessel_position", "ais_vessel", float(i)) for i in range(20)]
    obs += [_obs("tvl_change", "defi_flows", 100.0 + i) for i in range(100)]
    kept, stats = apply_training_obs_subsample(
        obs,
        gdelt_subsample_frac=1.0,
        defi_subsample_frac=0.05,
        seed=42,
    )
    assert len([o for o in kept if o["source_tool"] == "ais_vessel"]) == 20
    assert stats["floor_kept"] == 20


def test_defi_thinned_gdelt_thinned():
    obs = (
        [_obs("geopolitical_event", "gdelt", float(i)) for i in range(100)]
        + [_obs("tvl_change", "defi_flows", 200.0 + i) for i in range(100)]
        + [_obs("instrument_daily", "instrument_universe", 300.0 + i) for i in range(10)]
    )
    kept, stats = apply_training_obs_subsample(
        obs,
        gdelt_subsample_frac=0.05,
        defi_subsample_frac=0.05,
        seed=42,
    )
    assert stats["other_kept"] == 10
    assert 0 < stats.get("gdelt_kept", 0) < 100
    assert 0 < stats.get("defi_kept", 0) < 100
    assert len(kept) == stats["kept_total"]


def test_cftc_in_floor_set():
    assert "cftc" in RAW_SOURCE_TOOLS_FLOOR
    assert "ais_vessel" in RAW_SOURCE_TOOLS_FLOOR
