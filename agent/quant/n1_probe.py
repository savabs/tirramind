"""
Combined N1 playground probe: microstructure + CFTC positioning + supply GDELT.

Standalone raw intelligence (no GNN). See [[n1_n4_playground_spec]] hero readouts.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.quant.microstructure_signals import (
    InstrumentMicroPanel,
    MicroThresholds,
    build_instrument_panel,
    classify_cftc_positioning,
    list_instruments_by_asset_class,
    load_cftc_ranks_by_ticker,
    normalize_cftc_rank,
)
from agent.quant.n1_geo import (
    aggregate_supply_risk,
    build_gdelt_entity_map,
    load_gdelt_by_iso2,
    load_gdelt_sentiment,
    load_supply_countries,
    resolve_iso2_to_gdelt,
)

# Sector tags (aligned with cot_signal_report)
SECTOR_MAP: dict[str, str] = {
    "CRUDE OIL": "Energy",
    "NAT GAS": "Energy",
    "NATURAL GAS": "Energy",
    "GASOLINE": "Energy",
    "HEATING OIL": "Energy",
    "BRENT": "Energy",
    "WTI": "Energy",
    "GOLD": "Metals",
    "SILVER": "Metals",
    "COPPER": "Metals",
    "PLATINUM": "Metals",
    "PALLADIUM": "Metals",
    "CORN": "Agriculture",
    "SOYBEAN": "Agriculture",
    "WHEAT": "Agriculture",
    "SUGAR": "Agriculture",
    "COFFEE": "Agriculture",
    "COCOA": "Agriculture",
    "COTTON": "Agriculture",
    "ORANGE JUICE": "Agriculture",
    "LIVE CATTLE": "Livestock",
    "LEAN HOG": "Livestock",
}


def classify_sector(name: str) -> str:
    upper = name.upper()
    for fragment, sector in SECTOR_MAP.items():
        if fragment in upper:
            return sector
    return "Other"


@dataclass(frozen=True)
class N1Thresholds:
    """All tunable cutoffs for combined N1 probe (defaults for first pass)."""

    micro: MicroThresholds = field(default_factory=MicroThresholds)
    gdelt_lookback_days: int = 30
    supply_stress_high_goldstein: float = -3.0
    supply_stress_moderate_goldstein: float = -1.5
    priority_min_display: int = 2


@dataclass(frozen=True)
class N1CombinedProbe:
    """One commodity future with fused N1 readouts."""

    ticker: str
    entity_id: str
    canonical_name: str
    sector: str

    micro: dict[str, Any]
    micro_alerts: tuple[dict[str, Any], ...]
    positioning: dict[str, Any]
    supply: dict[str, Any]

    composite_priority: int
    composite_flags: tuple[str, ...]
    chain_narrative: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "sector": self.sector,
            "composite_priority": self.composite_priority,
            "composite_flags": list(self.composite_flags),
            "chain_narrative": self.chain_narrative,
            "micro": self.micro,
            "micro_alerts": list(self.micro_alerts),
            "positioning": self.positioning,
            "supply": self.supply,
        }


def _load_cftc_extras(
    con: sqlite3.Connection,
    observations: list[dict[str, Any]],
    entities: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per ticker: direction_change, mm_net_pct_oi, oi_vs_52w from latest derived/raw."""
    import json as _json

    from agent.tools.instrument_universe import cftc_code_to_ticker

    code_to_ticker = cftc_code_to_ticker()
    code_by_eid: dict[str, str] = {}
    for e in entities:
        if e.get("entity_type") != "cftc_contract":
            continue
        meta = e.get("metadata") or {}
        if isinstance(meta, str):
            meta = _json.loads(meta)
        if meta.get("cftc_code"):
            code_by_eid[e["entity_id"]] = str(meta["cftc_code"])

    latest_derived: dict[str, dict] = {}
    latest_raw: dict[str, dict] = {}

    for o in observations:
        ot = o.get("observation_type")
        eid = o.get("entity_id", "")
        if eid not in code_by_eid:
            continue
        ts = float(o.get("observed_at", 0))
        v = o.get("value", {})
        if not isinstance(v, dict):
            continue
        if ot == "futures_positioning_derived":
            prev = latest_derived.get(eid)
            if prev is None or ts >= prev["_ts"]:
                latest_derived[eid] = {**v, "_ts": ts}
        elif ot == "futures_positioning":
            prev = latest_raw.get(eid)
            if prev is None or ts >= prev["_ts"]:
                latest_raw[eid] = {**v, "_ts": ts}

    out: dict[str, dict[str, Any]] = {}
    for eid, code in code_by_eid.items():
        ticker = code_to_ticker.get(code)
        if not ticker:
            continue
        d = latest_derived.get(eid, {})
        r = latest_raw.get(eid, {})
        out[ticker] = {
            "direction_change": bool(d.get("cftc_mm_direction_change", 0) != 0),
            "mm_net_pct_oi": r.get("mm_net_pct_oi"),
            "mm_weekly_flow": r.get("mm_weekly_flow"),
            "oi_vs_52w_avg": d.get("cftc_oi_vs_52w_avg"),
            "open_interest": r.get("open_interest"),
        }
    return out


def _build_chain_narrative(
    ticker: str,
    positioning_label: str | None,
    supply: dict[str, Any],
    micro_alerts: list[dict[str, Any]],
) -> tuple[tuple[str, ...], str]:
    """Cross-domain chain text + flag tokens."""
    flags: list[str] = []
    parts: list[str] = []

    stress = supply.get("stress_level", "NO_DATA")
    if stress in ("HIGH", "MODERATE"):
        flags.append(f"SUPPLY_{stress}")
        g = supply.get("avg_goldstein")
        top = supply.get("top_stress_country")
        top_s = f" ({top[0]} goldstein={top[1]:+.1f})" if top else ""
        parts.append(
            f"Producer-country event field stress={stress}"
            f" (avg goldstein={g}{top_s})"
        )

    if positioning_label and positioning_label != "NEUTRAL":
        flags.append(f"POS_{positioning_label}")
        parts.append(f"CFTC positioning={positioning_label}")

    strong_micro = [a for a in micro_alerts if a.get("severity") == "STRONG"]
    if strong_micro:
        codes = ",".join(a["code"] for a in strong_micro)
        flags.append("MICRO_STRONG")
        parts.append(f"Micro alerts: {codes}")

    if not parts:
        return tuple(flags), f"{ticker}: no elevated N1 fuse signals."

    chain = " → ".join(parts) + f" → watch {ticker}"
    return tuple(flags), chain


def _composite_priority(
    panel: InstrumentMicroPanel,
    positioning_label: str | None,
    supply: dict[str, Any],
    direction_change: bool,
) -> int:
    score = 0
    for a in panel.alerts:
        if a.severity == "STRONG":
            score += 3
        elif a.severity == "WATCH":
            score += 1

    if positioning_label in ("CROWDED_LONG", "CROWDED_SHORT"):
        score += 3
    elif positioning_label in ("APPROACHING_LONG", "APPROACHING_SHORT"):
        score += 1

    stress = supply.get("stress_level")
    if stress == "HIGH":
        score += 3
    elif stress == "MODERATE":
        score += 2
    elif stress == "LOW":
        score += 1

    if direction_change:
        score += 1

    return score


def build_n1_combined_probe(
    ticker: str,
    entity_id: str,
    canonical_name: str,
    observations: list[dict[str, Any]],
    *,
    cftc_rank: float | None,
    cftc_extras: dict[str, Any],
    supply_countries: list[tuple[str, str]],
    supply_risk: dict[str, Any],
    thresholds: N1Thresholds | None = None,
    min_days: int = 30,
) -> N1CombinedProbe | None:
    th = thresholds or N1Thresholds()
    panel = build_instrument_panel(
        ticker,
        entity_id,
        observations,
        cftc_rank=cftc_rank,
        min_days=min_days,
        thresholds=th.micro,
    )
    if panel is None:
        return None

    rank_pct = (
        round(normalize_cftc_rank(cftc_rank) * 100.0, 1)
        if cftc_rank is not None
        else None
    )
    pos_label = panel.cftc_positioning_label

    alert_dicts = [a.to_dict() for a in panel.alerts]
    flags, narrative = _build_chain_narrative(ticker, pos_label, supply_risk, alert_dicts)
    priority = _composite_priority(
        panel, pos_label, supply_risk, cftc_extras.get("direction_change", False)
    )

    positioning = {
        "cftc_mm_pct_52w_rank": cftc_rank,
        "cftc_mm_pct_52w_rank_pct": rank_pct,
        "label": pos_label,
        **{k: v for k, v in cftc_extras.items() if k != "direction_change"},
        "direction_change": cftc_extras.get("direction_change", False),
    }

    return N1CombinedProbe(
        ticker=ticker,
        entity_id=entity_id,
        canonical_name=canonical_name,
        sector=classify_sector(canonical_name),
        micro=panel.snapshot.to_dict(),
        micro_alerts=tuple(alert_dicts),
        positioning=positioning,
        supply=supply_risk,
        composite_priority=priority,
        composite_flags=flags,
        chain_narrative=narrative,
    )


def build_all_n1_probes(
    db_path: str,
    *,
    asset_class: str = "commodity_future",
    gdelt_lookback_days: int = 30,
    min_days: int = 30,
    thresholds: N1Thresholds | None = None,
) -> list[N1CombinedProbe]:
    """Build combined probes for all instruments in asset_class."""
    from agent.pipeline.store import PipelineStore

    th = thresholds or N1Thresholds()
    th = N1Thresholds(
        micro=th.micro,
        gdelt_lookback_days=gdelt_lookback_days,
        supply_stress_high_goldstein=th.supply_stress_high_goldstein,
        supply_stress_moderate_goldstein=th.supply_stress_moderate_goldstein,
        priority_min_display=th.priority_min_display,
    )

    store = PipelineStore(db_path)
    con = sqlite3.connect(db_path)
    try:
        entities = store.query_all_entities()
        observations = store.query_all_observations()
        cftc_ranks = load_cftc_ranks_by_ticker(observations, entities)
        cftc_extras = _load_cftc_extras(con, observations, entities)

        instruments = list_instruments_by_asset_class(entities, asset_class)
        inst_ids = [eid for eid, _ in instruments]
        supply_map = load_supply_countries(con, inst_ids)
        gdelt_map = build_gdelt_entity_map(con)

        all_gdelt_ids: list[str] = []
        for clist in supply_map.values():
            for _cid, iso2 in clist:
                geid = resolve_iso2_to_gdelt(iso2, gdelt_map)
                if geid:
                    all_gdelt_ids.append(geid)
        gdelt = load_gdelt_sentiment(
            con, list(set(all_gdelt_ids)), lookback_days=gdelt_lookback_days
        )
        gdelt_by_iso2 = load_gdelt_by_iso2(con, lookback_days=gdelt_lookback_days)

        probes: list[N1CombinedProbe] = []
        name_by_eid = {
            e["entity_id"]: e.get("canonical_name", "")
            for e in entities
            if e.get("entity_type") == "instrument"
        }

        for eid, ticker in instruments:
            supply_risk = aggregate_supply_risk(
                supply_map.get(eid, []),
                gdelt,
                gdelt_map,
                gdelt_by_iso2=gdelt_by_iso2,
            )
            probe = build_n1_combined_probe(
                ticker,
                eid,
                name_by_eid.get(eid, ticker),
                observations,
                cftc_rank=cftc_ranks.get(ticker),
                cftc_extras=cftc_extras.get(ticker, {}),
                supply_countries=supply_map.get(eid, []),
                supply_risk=supply_risk,
                thresholds=th,
                min_days=min_days,
            )
            if probe is not None:
                probes.append(probe)

        probes.sort(key=lambda p: (-p.composite_priority, p.ticker))
        return probes
    finally:
        con.close()
        store.close()


def default_thresholds_dict() -> dict[str, Any]:
    """Serializable defaults for threshold tuning file."""
    th = N1Thresholds()
    return {
        "micro": asdict(th.micro),
        "gdelt_lookback_days": th.gdelt_lookback_days,
        "supply_stress_high_goldstein": th.supply_stress_high_goldstein,
        "supply_stress_moderate_goldstein": th.supply_stress_moderate_goldstein,
        "priority_min_display": th.priority_min_display,
    }


def thresholds_from_dict(d: dict[str, Any]) -> N1Thresholds:
    micro_d = d.get("micro", {})
    return N1Thresholds(
        micro=MicroThresholds(**micro_d) if micro_d else MicroThresholds(),
        gdelt_lookback_days=int(d.get("gdelt_lookback_days", 30)),
        supply_stress_high_goldstein=float(
            d.get("supply_stress_high_goldstein", -3.0)
        ),
        supply_stress_moderate_goldstein=float(
            d.get("supply_stress_moderate_goldstein", -1.5)
        ),
        priority_min_display=int(d.get("priority_min_display", 2)),
    )
