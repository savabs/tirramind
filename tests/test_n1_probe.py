"""Tests for combined N1 probe builder."""

from agent.quant.n1_probe import build_n1_combined_probe


def _daily(entity_id: str, n: int = 80):
    import math

    obs = []
    close = 100.0
    for i in range(n):
        ret = 0.01 * math.sin(i / 5.0)
        close *= math.exp(ret)
        obs.append(
            {
                "entity_id": entity_id,
                "observation_type": "instrument_daily",
                "observed_at": float(1_700_000_000 + i * 86400),
                "value": {
                    "close": close,
                    "log_return": ret,
                    "volume": 1_000_000.0,
                },
            }
        )
    return obs


def test_build_n1_combined_probe_minimal():
    eid = "test_cl"
    obs = _daily(eid)
    supply = {
        "avg_goldstein": -2.5,
        "stress_level": "MODERATE",
        "countries_with_data": 3,
        "top_stress_country": ("RU", -4.0),
        "event_count_total": 100.0,
    }
    probe = build_n1_combined_probe(
        "CL=F",
        eid,
        "WTI Crude Oil",
        obs,
        cftc_rank=0.85,
        cftc_extras={"direction_change": True, "mm_net_pct_oi": 12.0},
        supply_countries=[],
        supply_risk=supply,
    )
    assert probe is not None
    assert probe.ticker == "CL=F"
    assert probe.sector == "Energy"
    assert probe.composite_priority >= 1
    assert "POS_" in "".join(probe.composite_flags) or probe.positioning["label"]
    assert "Producer-country" in probe.chain_narrative
