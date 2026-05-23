#!/usr/bin/env python3
"""
COT Signal Report — weekly commodity intelligence.
CFTC positioning percentile × GDELT supply-country sentiment × direction regime.

Usage:
    python3 scripts/cot_signal_report.py
    python3 scripts/cot_signal_report.py --db .tirra_pipeline/pipeline.db
    python3 scripts/cot_signal_report.py --output reports/cot_2026-05-04.md
    python3 scripts/cot_signal_report.py --gdelt-days 14   # wider GDELT window

Output: Rich terminal report + markdown file (default: reports/cot_YYYY-MM-DD.md)

Signal definitions:
  CROWDED LONG   : mm_pct_52w_rank >= 80 (top quintile — crowding, potential reversal)
  CROWDED SHORT  : mm_pct_52w_rank <= 20 (bottom quintile — crowding, squeeze potential)
  MOMENTUM SHIFT : mm_pct_52w_rank 35-65 AND direction_change == 1 in past 2 weeks
  APPROACHING    : 65-79 (long watch) or 21-34 (short watch)
  NEUTRAL        : 35-64, no direction change
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ── optional rich ─────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box as rich_box
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None  # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────────
# Sector classification (by CFTC canonical name fragment)
# ─────────────────────────────────────────────────────────────────────────────
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

SECTOR_ORDER = ["Energy", "Metals", "Agriculture", "Livestock", "Other"]

# ─────────────────────────────────────────────────────────────────────────────
# ISO-2 → ISO-3 bridge for produced_in country codes → GDELT fips_code lookup
# ─────────────────────────────────────────────────────────────────────────────
_ISO2_TO_ISO3: dict[str, str] = {
    "AE": "ARE", "AR": "ARG", "AU": "AUS", "BR": "BRA", "CA": "CAN",
    "CI": "CIV", "CL": "CHL", "CM": "CMR", "CN": "CHN", "CO": "COL",
    "DZ": "DZA", "EC": "ECU", "ET": "ETH", "FR": "FRA", "GH": "GHA",
    "HN": "HND", "ID": "IDN", "IN": "IND", "IQ": "IRQ", "IR": "IRN",
    "KW": "KWT", "KZ": "KAZ", "LY": "LBY", "MX": "MEX", "NG": "NGA",
    "NO": "NOR", "PE": "PER", "PK": "PAK", "PL": "POL", "PY": "PRY",
    "QA": "QAT", "RU": "RUS", "SA": "SAU", "TH": "THA", "UA": "UKR",
    "US": "USA", "UZ": "UZB", "VE": "VEN", "VN": "VNM", "ZA": "ZAF",
    "ZM": "ZMB", "ZW": "ZWE",
}


def classify_sector(name: str) -> str:
    upper = name.upper()
    for fragment, sector in SECTOR_MAP.items():
        if fragment in upper:
            return sector
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# Signal logic
# ─────────────────────────────────────────────────────────────────────────────
def classify_signal(rank: float, direction_change: bool) -> tuple[str, str]:
    """
    Returns (signal_label, signal_severity).
    severity: STRONG | WATCH | NEUTRAL
    """
    if rank >= 80:
        return ("CROWDED LONG", "STRONG")
    if rank <= 20:
        return ("CROWDED SHORT", "STRONG")
    if 65 <= rank < 80:
        return ("APPROACHING CROWDED LONG", "WATCH")
    if 21 < rank <= 35:
        return ("APPROACHING CROWDED SHORT", "WATCH")
    if direction_change and 35 < rank < 65:
        return ("MOMENTUM SHIFT", "WATCH")
    return ("NEUTRAL", "NEUTRAL")


def percentile_bar(rank: float, width: int = 20) -> str:
    """ASCII bar from 0–100."""
    filled = round(rank / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {rank:5.1f}th"


# ─────────────────────────────────────────────────────────────────────────────
# DB queries
# ─────────────────────────────────────────────────────────────────────────────
def load_instruments(con: sqlite3.Connection) -> dict[str, dict]:
    """
    Build instrument registry from cftc_tracks links.
    Returns { cftc_entity_id: { name, ticker, asset_class, instrument_entity_id } }
    """
    rows = con.execute("""
        SELECT el.entity_id_a, el.entity_id_b,
               src.canonical_name AS cftc_name,
               tgt.canonical_name AS instrument_name,
               tgt.metadata_json
        FROM entity_links el
        JOIN entities src ON src.entity_id = el.entity_id_a
        JOIN entities tgt ON tgt.entity_id = el.entity_id_b
        WHERE el.link_type = 'cftc_tracks'
        ORDER BY tgt.canonical_name
    """).fetchall()

    instruments: dict[str, dict] = {}
    for r in rows:
        cftc_id, inst_id, cftc_name, inst_name, meta_json = r
        meta = json.loads(meta_json) if meta_json else {}
        instruments[cftc_id] = {
            "cftc_name": cftc_name,
            "name": inst_name,
            "ticker": meta.get("ticker", "?"),
            "asset_class": meta.get("asset_class", "unknown"),
            "sector": classify_sector(cftc_name),
            "instrument_entity_id": inst_id,
        }
    return instruments


def load_supply_countries(
    con: sqlite3.Connection, instrument_entity_ids: list[str]
) -> dict[str, list[tuple[str, str]]]:
    """
    Returns { instrument_entity_id: [(country_entity_id, country_name), ...] }
    """
    if not instrument_entity_ids:
        return {}
    placeholders = ",".join("?" * len(instrument_entity_ids))
    rows = con.execute(f"""
        SELECT el.entity_id_a, el.entity_id_b, tgt.canonical_name
        FROM entity_links el
        JOIN entities tgt ON tgt.entity_id = el.entity_id_b
        WHERE el.link_type = 'produced_in'
          AND el.entity_id_a IN ({placeholders})
    """, instrument_entity_ids).fetchall()

    result: dict[str, list] = defaultdict(list)
    for inst_id, country_id, country_name in rows:
        result[inst_id].append((country_id, country_name))
    return dict(result)


def load_latest_positioning(
    con: sqlite3.Connection, cftc_entity_ids: list[str]
) -> dict[str, dict]:
    """Latest futures_positioning per entity (raw CFTC fields)."""
    if not cftc_entity_ids:
        return {}
    placeholders = ",".join("?" * len(cftc_entity_ids))
    rows = con.execute(f"""
        SELECT entity_id, MAX(observed_at), value_json
        FROM entity_observations
        WHERE observation_type = 'futures_positioning'
          AND entity_id IN ({placeholders})
        GROUP BY entity_id
    """, cftc_entity_ids).fetchall()

    result = {}
    for entity_id, ts, value_json in rows:
        d = json.loads(value_json) if value_json else {}
        d["_observed_at"] = ts
        result[entity_id] = d
    return result


def load_latest_derived(
    con: sqlite3.Connection, cftc_entity_ids: list[str]
) -> dict[str, dict]:
    """Latest futures_positioning_derived per entity."""
    if not cftc_entity_ids:
        return {}
    placeholders = ",".join("?" * len(cftc_entity_ids))
    rows = con.execute(f"""
        SELECT entity_id, MAX(observed_at), value_json
        FROM entity_observations
        WHERE observation_type = 'futures_positioning_derived'
          AND entity_id IN ({placeholders})
        GROUP BY entity_id
    """, cftc_entity_ids).fetchall()

    result = {}
    for entity_id, ts, value_json in rows:
        d = json.loads(value_json) if value_json else {}
        d["_observed_at"] = ts
        result[entity_id] = d
    return result


def load_recent_direction_change(
    con: sqlite3.Connection, cftc_entity_ids: list[str], lookback_days: int = 14
) -> dict[str, bool]:
    """True if any direction_change==1 in the past N days per entity."""
    if not cftc_entity_ids:
        return {}
    cutoff_ts = (
        datetime.now(timezone.utc) - timedelta(days=lookback_days)
    ).timestamp()
    placeholders = ",".join("?" * len(cftc_entity_ids))
    rows = con.execute(f"""
        SELECT entity_id, value_json
        FROM entity_observations
        WHERE observation_type = 'futures_positioning_derived'
          AND entity_id IN ({placeholders})
          AND observed_at >= ?
        ORDER BY entity_id, observed_at DESC
    """, cftc_entity_ids + [cutoff_ts]).fetchall()

    changed: dict[str, bool] = {eid: False for eid in cftc_entity_ids}
    for entity_id, value_json in rows:
        if changed.get(entity_id):
            continue
        d = json.loads(value_json) if value_json else {}
        if d.get("cftc_mm_direction_change", 0) == 1:
            changed[entity_id] = True
    return changed


def build_gdelt_entity_map(con: sqlite3.Connection) -> dict[str, str]:
    """
    Returns { iso3_fips_code: entity_id } for all GDELT country entities.
    Used to bridge ISO-2 produced_in codes → ISO-3 → GDELT entity IDs.
    """
    rows = con.execute("""
        SELECT entity_id, metadata_json
        FROM entities
        WHERE entity_type = 'country'
          AND metadata_json IS NOT NULL
    """).fetchall()
    result: dict[str, str] = {}
    for entity_id, meta_json in rows:
        try:
            meta = json.loads(meta_json)
            fips = meta.get("fips_code", "")
            if fips and len(fips) == 3:
                result[fips] = entity_id
        except (json.JSONDecodeError, AttributeError):
            pass
    return result


def resolve_iso2_to_gdelt(
    iso2_code: str, gdelt_map: dict[str, str]
) -> str | None:
    """Convert ISO-2 country code to GDELT entity_id via ISO-3 FIPS bridge."""
    iso3 = _ISO2_TO_ISO3.get(iso2_code)
    if not iso3:
        return None
    return gdelt_map.get(iso3)


def load_gdelt_sentiment(
    con: sqlite3.Connection,
    country_entity_ids: list[str],
    lookback_days: int = 30,
) -> dict[str, dict[str, float]]:
    """
    Returns { country_entity_id: { avg_goldstein, event_count, avg_quad_class } }
    covering the past N days.
    """
    if not country_entity_ids:
        return {}
    cutoff_ts = (
        datetime.now(timezone.utc) - timedelta(days=lookback_days)
    ).timestamp()
    placeholders = ",".join("?" * len(country_entity_ids))
    rows = con.execute(f"""
        SELECT entity_id, value_json
        FROM entity_observations
        WHERE observation_type = 'geopolitical_event'
          AND entity_id IN ({placeholders})
          AND observed_at >= ?
    """, country_entity_ids + [cutoff_ts]).fetchall()

    buckets: dict[str, list] = defaultdict(list)
    for entity_id, value_json in rows:
        d = json.loads(value_json) if value_json else {}
        goldstein = d.get("goldstein")
        quad_class = d.get("quad_class")
        if goldstein is not None:
            buckets[entity_id].append(
                {"goldstein": goldstein, "quad_class": quad_class or 0}
            )

    result = {}
    for eid, events in buckets.items():
        if events:
            avg_g = sum(e["goldstein"] for e in events) / len(events)
            avg_q = sum(e["quad_class"] for e in events) / len(events)
            result[eid] = {
                "avg_goldstein": round(avg_g, 2),
                "avg_quad_class": round(avg_q, 2),
                "event_count": len(events),
            }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Supply risk aggregation
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_supply_risk(
    country_ids: list[tuple[str, str]],
    gdelt: dict[str, dict],
    gdelt_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Aggregate GDELT sentiment across all supply countries for an instrument.
    country_ids: list of (country_entity_id, iso2_code)
    gdelt: { gdelt_entity_id: { avg_goldstein, ... } }
    gdelt_map: { iso3: gdelt_entity_id } — used to bridge ISO-2 produced_in to GDELT
    Returns { avg_goldstein, stress_level, countries_with_data, top_stress_country }.
    """
    scores = []
    stressed_countries = []
    for cid, iso2 in country_ids:
        # Try direct entity ID match first, then ISO-2 → ISO-3 → GDELT bridge
        gdelt_eid = cid if cid in gdelt else None
        if gdelt_eid is None and gdelt_map is not None:
            gdelt_eid = resolve_iso2_to_gdelt(iso2, gdelt_map)
        if gdelt_eid and gdelt_eid in gdelt:
            g = gdelt[gdelt_eid]["avg_goldstein"]
            scores.append(g)
            if g < -1.5:
                stressed_countries.append((iso2, g))

    if not scores:
        return {
            "avg_goldstein": None,
            "stress_level": "NO DATA",
            "countries_with_data": 0,
            "top_stress_country": None,
        }

    avg = sum(scores) / len(scores)
    stressed_countries.sort(key=lambda x: x[1])

    if avg < -3:
        stress = "HIGH"
    elif avg < -1.5:
        stress = "MODERATE"
    elif avg < 0:
        stress = "LOW"
    else:
        stress = "STABLE"

    return {
        "avg_goldstein": round(avg, 2),
        "stress_level": stress,
        "countries_with_data": len(scores),
        "top_stress_country": stressed_countries[0] if stressed_countries else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composite instrument record
# ─────────────────────────────────────────────────────────────────────────────
def build_instrument_record(
    cftc_id: str,
    meta: dict,
    raw: dict,
    derived: dict,
    direction_changed: bool,
    supply_countries: list[tuple[str, str]],
    supply_risk: dict,
) -> dict[str, Any]:
    rank = derived.get("cftc_mm_pct_52w_rank", 50.0)
    signal, severity = classify_signal(rank, direction_changed)
    obs_ts = derived.get("_observed_at") or raw.get("_observed_at")
    obs_date = (
        datetime.fromtimestamp(obs_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if obs_ts
        else "unknown"
    )

    return {
        "cftc_id": cftc_id,
        "name": meta["name"],
        "cftc_name": meta["cftc_name"],
        "ticker": meta["ticker"],
        "sector": meta["sector"],
        # positioning
        "rank": rank,
        "mm_net_pct_oi": raw.get("mm_net_pct_oi"),
        "mm_weekly_flow": raw.get("mm_weekly_flow"),
        "open_interest": raw.get("open_interest"),
        "oi_vs_52w_avg": derived.get("cftc_oi_vs_52w_avg"),
        "direction_changed": direction_changed,
        "obs_date": obs_date,
        # signal
        "signal": signal,
        "severity": severity,
        # supply
        "supply_countries": supply_countries,
        "supply_risk": supply_risk,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_COLOR = {
    "STRONG": "bold red",
    "WATCH": "bold yellow",
    "NEUTRAL": "dim",
}

SUPPLY_STRESS_COLOR = {
    "HIGH": "bold red",
    "MODERATE": "yellow",
    "LOW": "green",
    "STABLE": "dim green",
    "NO DATA": "dim",
}


def _flow_arrow(flow: float | None) -> str:
    """Display mm_weekly_flow (raw contracts) in thousands."""
    if flow is None:
        return "n/a"
    k = flow / 1000
    if flow > 0:
        return f"+{k:.1f}k contracts ▲"
    if flow < 0:
        return f"{k:.1f}k contracts ▼"
    return "flat"


def _oi_vs_52w(ratio: float | None) -> str:
    if ratio is None:
        return "n/a"
    pct = (ratio - 1.0) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}% vs 52w avg"


def render_terminal(records: list[dict], report_date: str, gdelt_days: int) -> None:
    if not RICH:
        _render_plain(records, report_date)
        return

    console.print()
    console.rule(
        f"[bold white] COT SIGNAL REPORT — {report_date}  "
        f"[dim](CFTC data + {gdelt_days}d GDELT supply sentiment)[/dim]",
        style="white",
    )
    console.print()

    # Top signals summary
    strong = [r for r in records if r["severity"] == "STRONG"]
    watch = [r for r in records if r["severity"] == "WATCH"]

    if strong:
        console.print("[bold red]━━ STRONG SIGNALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")
        for r in strong:
            direction_flag = " ⚡ DIR CHANGE" if r["direction_changed"] else ""
            console.print(
                f"  [bold]{r['name']:20s}[/bold] ({r['ticker']:6s})  "
                f"[bold red]{r['signal']}[/bold red]{direction_flag}"
            )
        console.print()

    if watch:
        console.print("[bold yellow]━━ WATCH LIST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold yellow]")
        for r in watch:
            direction_flag = " ⚡" if r["direction_changed"] else ""
            console.print(
                f"  [bold]{r['name']:20s}[/bold] ({r['ticker']:6s})  "
                f"[yellow]{r['signal']}[/yellow]{direction_flag}"
            )
        console.print()

    # Per-sector breakdown
    by_sector: dict[str, list] = defaultdict(list)
    for r in records:
        by_sector[r["sector"]].append(r)

    for sector in SECTOR_ORDER:
        sector_records = by_sector.get(sector, [])
        if not sector_records:
            continue

        console.print(
            f"[bold blue]━━ {sector.upper()}"
            f" {'━' * max(0, 60 - len(sector))}[/bold blue]"
        )
        console.print()

        for r in sorted(sector_records, key=lambda x: -x["rank"]):
            signal_color = SEVERITY_COLOR.get(r["severity"], "")
            bar = percentile_bar(r["rank"])

            flow_str = _flow_arrow(r.get("mm_weekly_flow"))
            oi_str = _oi_vs_52w(r.get("oi_vs_52w_avg"))
            mm_net = r.get("mm_net_pct_oi")
            mm_str = f"{mm_net:+.1f}% OI" if mm_net is not None else "n/a"

            oi_raw = r.get("open_interest")
            oi_label = f"{oi_raw:,.0f}" if oi_raw is not None else "n/a"

            direction_flag = "  ⚡ DIRECTION FLIP" if r["direction_changed"] else ""

            # Supply risk
            sr = r["supply_risk"]
            if sr["avg_goldstein"] is not None:
                stress_color = SUPPLY_STRESS_COLOR.get(sr["stress_level"], "")
                supply_str = (
                    f"[{stress_color}]{sr['stress_level']}[/{stress_color}]"
                    f"  (goldstein={sr['avg_goldstein']:+.1f},"
                    f" {sr['countries_with_data']} {'country' if sr['countries_with_data'] == 1 else 'countries'})"
                )
                top_stress = sr.get("top_stress_country")
                if top_stress:
                    supply_str += f"  ← {top_stress[0]} ({top_stress[1]:+.1f})"
            else:
                supply_str = "[dim]no GDELT data[/dim]"

            console.print(
                f"  [bold]{r['name']:22s}[/bold]  [{signal_color}]{r['ticker']:6s}[/{signal_color}]"
                f"  [{signal_color}]{r['signal']}[/{signal_color}]{direction_flag}"
            )
            console.print(
                f"    Positioning  : {bar}  (data: {r['obs_date']})"
            )
            console.print(
                f"    MM Net       : {mm_str:12s}  Flow: {flow_str}  OI: {oi_label}"
            )
            console.print(f"    OI vs history: {oi_str}")
            console.print(f"    Supply risk  : {supply_str}")
            console.print()

    console.rule("[dim]end of report[/dim]", style="dim")


def _render_plain(records: list[dict], report_date: str) -> None:
    print(f"\n{'='*70}")
    print(f"COT SIGNAL REPORT — {report_date}")
    print("=" * 70)
    by_sector: dict[str, list] = defaultdict(list)
    for r in records:
        by_sector[r["sector"]].append(r)
    for sector in SECTOR_ORDER:
        sector_records = by_sector.get(sector, [])
        if not sector_records:
            continue
        print(f"\n── {sector.upper()} ──")
        for r in sorted(sector_records, key=lambda x: -x["rank"]):
            bar = percentile_bar(r["rank"])
            print(f"  {r['name']:22s} ({r['ticker']:6s})  {bar}  {r['signal']}")
            sr = r["supply_risk"]
            g = f"goldstein={sr['avg_goldstein']:+.1f}" if sr["avg_goldstein"] is not None else "no GDELT"
            print(f"    supply risk: {sr['stress_level']:8s}  {g}")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown output
# ─────────────────────────────────────────────────────────────────────────────
def render_markdown(records: list[dict], report_date: str, gdelt_days: int) -> str:
    lines = [
        f"# COT Signal Report — {report_date}",
        "",
        f"*CFTC Commitment of Traders data + {gdelt_days}-day GDELT supply-country sentiment.*",
        f"*Crowding signal: 52-week percentile rank of managed-money net positioning.*",
        "",
    ]

    strong = [r for r in records if r["severity"] == "STRONG"]
    watch = [r for r in records if r["severity"] == "WATCH"]

    if strong:
        lines.append("## 🔴 Strong Signals")
        lines.append("")
        lines.append("| Instrument | Ticker | Signal | Rank | Dir Change |")
        lines.append("|---|---|---|---|---|")
        for r in strong:
            dc = "⚡ YES" if r["direction_changed"] else "—"
            lines.append(
                f"| {r['name']} | `{r['ticker']}` | **{r['signal']}** "
                f"| {r['rank']:.0f}th pct | {dc} |"
            )
        lines.append("")

    if watch:
        lines.append("## 🟡 Watch List")
        lines.append("")
        lines.append("| Instrument | Ticker | Signal | Rank | Dir Change |")
        lines.append("|---|---|---|---|---|")
        for r in watch:
            dc = "⚡ YES" if r["direction_changed"] else "—"
            lines.append(
                f"| {r['name']} | `{r['ticker']}` | {r['signal']} "
                f"| {r['rank']:.0f}th pct | {dc} |"
            )
        lines.append("")

    by_sector: dict[str, list] = defaultdict(list)
    for r in records:
        by_sector[r["sector"]].append(r)

    for sector in SECTOR_ORDER:
        sector_records = by_sector.get(sector, [])
        if not sector_records:
            continue
        lines.append(f"## {sector}")
        lines.append("")

        for r in sorted(sector_records, key=lambda x: -x["rank"]):
            bar = percentile_bar(r["rank"])
            mm_net = r.get("mm_net_pct_oi")
            mm_str = f"{mm_net:+.1f}% OI" if mm_net is not None else "n/a"
            flow_str = _flow_arrow(r.get("mm_weekly_flow"))
            oi_raw = r.get("open_interest")
            oi_label = f"{oi_raw:,.0f}" if oi_raw is not None else "n/a"
            oi_str = _oi_vs_52w(r.get("oi_vs_52w_avg"))
            dc_flag = " ⚡ direction flip" if r["direction_changed"] else ""

            sr = r["supply_risk"]
            if sr["avg_goldstein"] is not None:
                supply_md = (
                    f"supply risk: **{sr['stress_level']}** "
                    f"(goldstein={sr['avg_goldstein']:+.1f}, "
                    f"{sr['countries_with_data']} countries)"
                )
                top = sr.get("top_stress_country")
                if top:
                    supply_md += f" ← `{top[0]}` most stressed ({top[1]:+.1f})"
            else:
                supply_md = "supply risk: no GDELT data"

            signal_prefix = {
                "STRONG": "🔴",
                "WATCH": "🟡",
                "NEUTRAL": "⚪",
            }.get(r["severity"], "⚪")

            lines += [
                f"### {signal_prefix} {r['name']} (`{r['ticker']}`){dc_flag}",
                "",
                f"- **Signal**: {r['signal']}",
                f"- **Positioning**: `{bar}`",
                f"- MM Net: {mm_str} | Flow: {flow_str} | OI: {oi_label}",
                f"- OI vs 52w history: {oi_str}",
                f"- {supply_md}",
                "",
            ]

    lines += [
        "---",
        "",
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "**Signal methodology:**",
        "- CROWDED LONG: MM net positioning ≥ 80th percentile vs past 52 weeks → contrarian bearish pressure",
        "- CROWDED SHORT: ≤ 20th percentile → potential short squeeze",
        "- MOMENTUM SHIFT: direction flip detected in past 14 days (inside neutral territory)",
        "- Supply sentiment: GDELT Goldstein scale (−10 hostile → +10 cooperative) averaged across production countries",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="COT Signal Report")
    parser.add_argument(
        "--db",
        default=".tirra_pipeline/pipeline.db",
        help="Path to pipeline.db (default: .tirra_pipeline/pipeline.db)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output markdown path (default: reports/cot_YYYY-MM-DD.md)",
    )
    parser.add_argument(
        "--gdelt-days",
        type=int,
        default=30,
        help="GDELT lookback window in days (default: 30)",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip saving markdown output",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        sys.exit(f"ERROR: database not found at {db_path}")

    con = sqlite3.connect(str(db_path))
    con.row_factory = None  # raw tuples for speed

    # ── load instruments ───────────────────────────────────────────────────
    instruments = load_instruments(con)
    if not instruments:
        sys.exit("ERROR: no cftc_tracks links found in DB")

    cftc_ids = list(instruments.keys())
    instrument_entity_ids = [v["instrument_entity_id"] for v in instruments.values()]

    if RICH:
        console.print(
            f"[dim]Loaded {len(instruments)} CFTC-tracked instruments.[/dim]"
        )

    # ── load positioning ───────────────────────────────────────────────────
    raw_pos = load_latest_positioning(con, cftc_ids)
    derived_pos = load_latest_derived(con, cftc_ids)
    direction_changes = load_recent_direction_change(con, cftc_ids)

    # ── supply countries + GDELT ───────────────────────────────────────────
    supply_map = load_supply_countries(con, instrument_entity_ids)

    # Build ISO-3 → GDELT entity_id map for bridging ISO-2 produced_in codes
    gdelt_entity_map = build_gdelt_entity_map(con)

    # Collect GDELT entity IDs via ISO-2 → ISO-3 → entity_id bridge
    all_gdelt_ids: list[str] = []
    for country_list in supply_map.values():
        for _cid, iso2 in country_list:
            geid = resolve_iso2_to_gdelt(iso2, gdelt_entity_map)
            if geid:
                all_gdelt_ids.append(geid)
    all_gdelt_ids = list(set(all_gdelt_ids))

    gdelt = load_gdelt_sentiment(con, all_gdelt_ids, lookback_days=args.gdelt_days)

    if RICH:
        console.print(
            f"[dim]GDELT: {len(gdelt)} of {len(all_gdelt_ids)} bridged supply countries "
            f"have data in past {args.gdelt_days}d.[/dim]"
        )

    # ── build records ──────────────────────────────────────────────────────
    records: list[dict] = []
    for cftc_id, meta in instruments.items():
        raw = raw_pos.get(cftc_id, {})
        derived = derived_pos.get(cftc_id, {})
        if not derived:
            continue  # skip if no derived features

        direction_changed = direction_changes.get(cftc_id, False)
        inst_id = meta["instrument_entity_id"]
        country_list = supply_map.get(inst_id, [])
        supply_risk = aggregate_supply_risk(country_list, gdelt, gdelt_entity_map)

        rec = build_instrument_record(
            cftc_id=cftc_id,
            meta=meta,
            raw=raw,
            derived=derived,
            direction_changed=direction_changed,
            supply_countries=country_list,
            supply_risk=supply_risk,
        )
        records.append(rec)

    if not records:
        sys.exit("ERROR: no positioning data found for any instrument")

    records.sort(key=lambda x: (x["sector"], -x["rank"]))

    # ── report date ────────────────────────────────────────────────────────
    # Use the latest observed_at across all instruments as the report date
    latest_ts = max(
        (r.get("_observed_at", 0) for r in {**raw_pos, **derived_pos}.values()),
        default=datetime.now(timezone.utc).timestamp(),
    )
    report_date = datetime.fromtimestamp(latest_ts, tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )

    # ── render ─────────────────────────────────────────────────────────────
    render_terminal(records, report_date, args.gdelt_days)

    if not args.no_markdown:
        md_content = render_markdown(records, report_date, args.gdelt_days)
        out_path = (
            Path(args.output)
            if args.output
            else Path("reports") / f"cot_{report_date}.md"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_content, encoding="utf-8")
        if RICH:
            console.print(f"\n[dim]Markdown saved → {out_path}[/dim]\n")
        else:
            print(f"\nMarkdown saved → {out_path}")


if __name__ == "__main__":
    main()
