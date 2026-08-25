"""
N1 supply-country GDELT fusion (no GNN).

Event-rate / Goldstein aggregates over producer countries linked via produced_in.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

# ISO-2 → ISO-3 for GDELT country entity bridge
# Map ISO-2 codes to GDELT country entity canonical_name fragments
_ISO2_NAME_FRAGMENTS: dict[str, list[str]] = {
    "US": ["UNITED STATES", "U.S.", "AMERICA"],
    "SA": ["SAUDI"],
    "RU": ["RUSSIA", "RUSSIAN"],
    "IQ": ["IRAQ"],
    "AE": ["EMIRATES", "UAE"],
    "CA": ["CANADA"],
    "KW": ["KUWAIT"],
    "IR": ["IRAN"],
    "NG": ["NIGERIA"],
    "KZ": ["KAZAKH"],
    "LY": ["LIBYA"],
    "VE": ["VENEZUELA"],
    "NO": ["NORWAY"],
    "BR": ["BRAZIL"],
    "AR": ["ARGENTINA"],
    "UA": ["UKRAINE"],
    "AU": ["AUSTRALIA"],
    "CN": ["CHINA", "CHINESE"],
    "ZA": ["SOUTH AFRICA"],
    "CL": ["CHILE"],
    "PE": ["PERU"],
    "ID": ["INDONESIA"],
    "IN": ["INDIA"],
    "MX": ["MEXICO"],
    "ET": ["ETHIOPIA"],
    "VN": ["VIETNAM"],
    "TH": ["THAILAND"],
    "CO": ["COLOMBIA"],
}

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


def load_supply_countries(
    con: sqlite3.Connection, instrument_entity_ids: list[str]
) -> dict[str, list[tuple[str, str]]]:
    """{ instrument_entity_id: [(country_entity_id, iso2), ...] }"""
    if not instrument_entity_ids:
        return {}
    placeholders = ",".join("?" * len(instrument_entity_ids))
    rows = con.execute(
        f"""
        SELECT el.entity_id_a, el.entity_id_b, tgt.canonical_name
        FROM entity_links el
        JOIN entities tgt ON tgt.entity_id = el.entity_id_b
        WHERE el.link_type = 'produced_in'
          AND el.entity_id_a IN ({placeholders})
        """,
        instrument_entity_ids,
    ).fetchall()

    result: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for inst_id, country_id, iso2 in rows:
        result[inst_id].append((country_id, str(iso2)))
    return dict(result)


def build_gdelt_entity_map(con: sqlite3.Connection) -> dict[str, str]:
    """{ iso3_fips: gdelt_country_entity_id }"""
    rows = con.execute(
        """
        SELECT entity_id, metadata_json
        FROM entities
        WHERE entity_type = 'country' AND metadata_json IS NOT NULL
        """
    ).fetchall()
    out: dict[str, str] = {}
    for entity_id, meta_json in rows:
        try:
            meta = json.loads(meta_json) if isinstance(meta_json, str) else meta_json
            fips = meta.get("fips_code", "")
            if fips and len(fips) == 3:
                out[fips] = entity_id
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return out


def resolve_iso2_to_gdelt(iso2: str, gdelt_map: dict[str, str]) -> str | None:
    iso3 = _ISO2_TO_ISO3.get(iso2)
    if not iso3:
        return None
    return gdelt_map.get(iso3)


def load_gdelt_sentiment(
    con: sqlite3.Connection,
    country_entity_ids: list[str],
    lookback_days: int = 30,
) -> dict[str, dict[str, float]]:
    """{ country_entity_id: { avg_goldstein, event_count, avg_quad_class } }"""
    if not country_entity_ids:
        return {}
    cutoff_ts = _gdelt_cutoff_ts(con, lookback_days)
    placeholders = ",".join("?" * len(country_entity_ids))
    rows = con.execute(
        f"""
        SELECT entity_id, value_json
        FROM entity_observations
        WHERE observation_type = 'geopolitical_event'
          AND entity_id IN ({placeholders})
          AND observed_at >= ?
        """,
        country_entity_ids + [cutoff_ts],
    ).fetchall()

    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    for entity_id, value_json in rows:
        d = json.loads(value_json) if value_json else {}
        goldstein = d.get("goldstein")
        if goldstein is not None:
            buckets[entity_id].append(
                {
                    "goldstein": float(goldstein),
                    "quad_class": float(d.get("quad_class") or 0),
                }
            )

    result: dict[str, dict[str, float]] = {}
    for eid, events in buckets.items():
        if events:
            result[eid] = {
                "avg_goldstein": round(
                    sum(e["goldstein"] for e in events) / len(events), 2
                ),
                "avg_quad_class": round(
                    sum(e["quad_class"] for e in events) / len(events), 2
                ),
                "event_count": float(len(events)),
            }
    return result


def _gdelt_cutoff_ts(con: sqlite3.Connection, lookback_days: int) -> float:
    """Cutoff using latest GDELT obs in DB (not wall clock) for backfilled pipelines."""
    row = con.execute(
        "SELECT MAX(observed_at) FROM entity_observations "
        "WHERE observation_type = 'geopolitical_event'"
    ).fetchone()
    max_ts = float(row[0] or 0.0)
    if max_ts <= 0:
        return (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).timestamp()
    return max_ts - lookback_days * 86400.0


def load_gdelt_by_iso2(
    con: sqlite3.Connection,
    lookback_days: int = 30,
) -> dict[str, dict[str, float]]:
    """
    Aggregate geopolitical_event by ISO-2 country code.

    Works with seed_producer_links countries (canonical_name = ISO2) and
    GDELT-registered countries (fips_code in metadata).
    """
    cutoff_ts = _gdelt_cutoff_ts(con, lookback_days)

    def _iso_from_country_name(cname: str) -> str | None:
        upper = cname.upper().strip()
        if len(upper) == 2 and upper.isalpha():
            return upper
        for iso2, frags in _ISO2_NAME_FRAGMENTS.items():
            for frag in frags:
                if frag in upper:
                    return iso2
        return None

    # entity_id -> iso2 (seed ISO entities + GDELT full-name entities)
    eid_to_iso: dict[str, str] = {}
    for eid, etype, cname, meta_json in con.execute(
        "SELECT entity_id, entity_type, canonical_name, metadata_json FROM entities"
    ):
        if etype != "country":
            continue
        iso = _iso_from_country_name(str(cname or ""))
        if not iso and meta_json:
            try:
                meta = json.loads(meta_json) if isinstance(meta_json, str) else meta_json
                fips = meta.get("fips_code", "")
                if fips and len(fips) == 3:
                    for i2, i3 in _ISO2_TO_ISO3.items():
                        if i3 == fips:
                            iso = i2
                            break
            except (json.JSONDecodeError, TypeError):
                pass
        if iso:
            eid_to_iso[eid] = iso

    buckets: dict[str, list[float]] = defaultdict(list)
    rows = con.execute(
        """
        SELECT entity_id, value_json FROM entity_observations
        WHERE observation_type = 'geopolitical_event' AND observed_at >= ?
        """,
        (cutoff_ts,),
    )
    for entity_id, value_json in rows:
        iso = eid_to_iso.get(entity_id)
        if not iso:
            continue
        d = json.loads(value_json) if value_json else {}
        g = d.get("goldstein")
        if g is not None:
            buckets[iso].append(float(g))

    out: dict[str, dict[str, float]] = {}
    for iso, vals in buckets.items():
        out[iso] = {
            "avg_goldstein": round(sum(vals) / len(vals), 2),
            "event_count": float(len(vals)),
        }
    return out


def aggregate_supply_risk(
    country_ids: list[tuple[str, str]],
    gdelt: dict[str, dict[str, float]],
    gdelt_map: dict[str, str] | None = None,
    gdelt_by_iso2: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Aggregate GDELT across producer countries for one instrument."""
    scores: list[float] = []
    stressed: list[tuple[str, float]] = []
    event_total = 0.0

    for cid, iso2 in country_ids:
        g_val: dict[str, float] | None = None
        if gdelt_by_iso2 and iso2 in gdelt_by_iso2:
            g_val = gdelt_by_iso2[iso2]
        else:
            gdelt_eid = cid if cid in gdelt else None
            if gdelt_eid is None and gdelt_map is not None:
                gdelt_eid = resolve_iso2_to_gdelt(iso2, gdelt_map)
            if gdelt_eid and gdelt_eid in gdelt:
                g_val = gdelt[gdelt_eid]

        if g_val:
            g = g_val["avg_goldstein"]
            scores.append(g)
            event_total += g_val.get("event_count", 0.0)
            if g < -1.5:
                stressed.append((iso2, g))

    if not scores:
        return {
            "avg_goldstein": None,
            "stress_level": "NO_DATA",
            "countries_with_data": 0,
            "top_stress_country": None,
            "event_count_total": 0.0,
        }

    avg = sum(scores) / len(scores)
    stressed.sort(key=lambda x: x[1])

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
        "top_stress_country": stressed[0] if stressed else None,
        "event_count_total": event_total,
    }
