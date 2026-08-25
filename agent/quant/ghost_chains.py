"""Cross-domain ghost pattern chain matching (Phase A).

Loads YAML chain templates, scores node anomalies from pipeline.db observations,
and emits archive-ready alert dicts when all nodes fire within lag windows.

Layer 2: Feature Engineering — no fetching, no LLM calls.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from agent.tools.instrument_universe import INSTRUMENTS, cftc_code_to_ticker

_DEFAULT_DB = Path(".tirra_pipeline/pipeline.db")
_Z_WINDOW = 52  # weeks for CFTC/EIA; days for AIS/GDELT use min(52, n)

# CFTC disaggregated report uses multiple WTI contract rows; 06765A (financial)
# often has zero mm_net in our feed — prefer WTI-PHYSICAL for CL=F readouts.
_CFTC_TICKER_ENTITY_HINTS: dict[str, tuple[str, ...]] = {
    "CL=F": ("WTI-PHYSICAL",),
}

# GDELT country entities in pipeline.db use canonical names, not ISO2.
_ISO2_TO_COUNTRY_NAME: dict[str, str] = {
    "SA": "SAUDI",
    "RU": "RUSSIA",
    "IQ": "IRAQ",
    "AE": "UAE",
    "NG": "NIGERIA",
    "NO": "NORWAY",
    "US": "UNITED STATES",
    "IR": "IRAN",
    "KW": "KUWAIT",
}


@dataclass(frozen=True)
class ChainNodeSpec:
    id: str
    label: str
    source_tool: str
    observation_type: str
    metric: str
    min_zscore: float
    lag_days: tuple[int, int]
    source_url: str
    direction: str | None = None
    instrument_ticker: str | None = None
    value_filter: dict[str, str] = field(default_factory=dict)
    producer_countries: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChainTemplate:
    id: str
    micro_playground: str
    description: str
    readout_instrument: str
    evaluation_window_days: int
    max_chain_lag_days: int
    min_chain_score: float
    nodes: tuple[ChainNodeSpec, ...]


@dataclass
class NodeMatch:
    node_id: str
    entity: str
    obs: str
    z: float
    value: float
    observed_at: str
    source_url: str


@dataclass
class ChainMatch:
    template: ChainTemplate
    nodes: list[NodeMatch]
    chain_score: float
    issued_at: str


def load_chain_template(path: str | Path) -> ChainTemplate:
    """Load a YAML chain template."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    nodes = tuple(
        ChainNodeSpec(
            id=n["id"],
            label=n.get("label", n["id"]),
            source_tool=n["source_tool"],
            observation_type=n["observation_type"],
            metric=n["metric"],
            min_zscore=float(n.get("min_zscore", 2.0)),
            lag_days=tuple(n.get("lag_days", [0, 14])),
            source_url=n["source_url"],
            direction=n.get("direction"),
            instrument_ticker=n.get("instrument_ticker"),
            value_filter=dict(n.get("value_filter") or {}),
            producer_countries=tuple(n.get("producer_countries") or ()),
        )
        for n in data["nodes"]
    )
    return ChainTemplate(
        id=data["id"],
        micro_playground=data["micro_playground"],
        description=(data.get("description") or "").strip(),
        readout_instrument=data["readout_instrument"],
        evaluation_window_days=int(data.get("evaluation_window_days", 21)),
        max_chain_lag_days=int(data.get("max_chain_lag_days", 14)),
        min_chain_score=float(data.get("min_chain_score", 2.0)),
        nodes=nodes,
    )


def rolling_zscore(series: np.ndarray, idx: int, window: int) -> float:
    """Z-score at index `idx` using prior `window` values (inclusive)."""
    if idx < 1 or len(series) == 0:
        return 0.0
    start = max(0, idx - window + 1)
    window_vals = series[start:idx]
    window_vals = window_vals[np.isfinite(window_vals)]
    if len(window_vals) < 3:
        return 0.0
    mu = float(np.mean(window_vals))
    sigma = float(np.std(window_vals, ddof=1))
    if sigma < 1e-12:
        return 0.0
    cur = float(series[idx])
    if not np.isfinite(cur):
        return 0.0
    return (cur - mu) / sigma


def _parse_ts(ts: str | float | int) -> datetime:
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    s = str(ts).strip()
    if s.replace(".", "", 1).isdigit():
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    s = s.replace("Z", "+00:00")
    if "T" not in s and len(s) >= 10 and s[4] == "-":
        s = s[:10] + "T00:00:00+00:00"
    return datetime.fromisoformat(s)


def _day_key(ts: str | float | int) -> str:
    return _parse_ts(ts).strftime("%Y-%m-%d")


def _cftc_contract_for_ticker(con: sqlite3.Connection, ticker: str) -> str | None:
    for hint in _CFTC_TICKER_ENTITY_HINTS.get(ticker, ()):
        row = con.execute(
            "SELECT entity_id FROM entities WHERE entity_type='cftc_contract' "
            "AND canonical_name LIKE ? LIMIT 1",
            (f"%{hint}%",),
        ).fetchone()
        if row:
            return row[0]

    code_map = cftc_code_to_ticker()
    code = next((c for c, t in code_map.items() if t == ticker), None)
    if not code:
        for inst in INSTRUMENTS:
            if inst.ticker == ticker and inst.cftc_code:
                code = inst.cftc_code
                break
    if not code:
        return None
    row = con.execute(
        "SELECT entity_id FROM entities WHERE entity_type='cftc_contract' "
        "AND metadata_json LIKE ? LIMIT 1",
        (f'%"cftc_code": "{code}"%',),
    ).fetchone()
    if row:
        return row[0]
    from agent.pipeline.entity import entity_id_from_key

    return entity_id_from_key("cftc_contract", code)


def _load_cftc_series(
    con: sqlite3.Connection, ticker: str, metric: str
) -> tuple[list[datetime], np.ndarray, str]:
    eid = _cftc_contract_for_ticker(con, ticker)
    if not eid:
        return [], np.array([]), ticker
    rows = con.execute(
        """
        SELECT observed_at, value_json FROM entity_observations
        WHERE source_tool='cftc' AND entity_id=? AND observation_type='futures_positioning'
        ORDER BY observed_at ASC
        """,
        (eid,),
    ).fetchall()
    # CFTC ingest can duplicate the same report date — keep the last row per day.
    by_day: dict[str, tuple[datetime, float]] = {}
    name = ticker
    name_row = con.execute(
        "SELECT canonical_name FROM entities WHERE entity_id=? LIMIT 1", (eid,)
    ).fetchone()
    if name_row and name_row[0]:
        name = name_row[0].split(" - ")[0].strip()
    for ts, vj in rows:
        try:
            val = json.loads(vj) if isinstance(vj, str) else vj
            v = val.get(metric)
            if v is None:
                continue
            day = _day_key(ts)
            by_day[day] = (_parse_ts(ts), float(v))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    if not by_day:
        return [], np.array([]), name
    days = sorted(by_day)
    times = [by_day[d][0] for d in days]
    values = np.array([by_day[d][1] for d in days], dtype=float)
    return times, values, name


def _load_eia_series(
    con: sqlite3.Connection, series_key: str, metric: str
) -> tuple[list[datetime], np.ndarray, str]:
    rows = con.execute(
        """
        SELECT observed_at, value_json FROM entity_observations
        WHERE source_tool='energy_supply' AND observation_type='petroleum_inventory'
        ORDER BY observed_at ASC
        """
    ).fetchall()
    times: list[datetime] = []
    values: list[float] = []
    for ts, vj in rows:
        try:
            val = json.loads(vj) if isinstance(vj, str) else vj
            if val.get("series") != series_key:
                continue
            if metric == "weekly_change":
                v = float(val.get("value", 0))
            else:
                v = float(val.get("value", 0))
            times.append(_parse_ts(ts))
            values.append(v)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    if metric == "weekly_change" and len(values) > 1:
        arr = np.array(values, dtype=float)
        changes = np.diff(arr)
        return times[1:], changes, f"US crude ({series_key}) weekly Δ"
    return times, np.array(values, dtype=float), f"US crude ({series_key})"


def _ais_metric_from_value(val: dict[str, Any]) -> float:
    for key in ("tanker_count", "series_count", "vessel_count"):
        if key in val and val[key] is not None:
            return float(val[key])
    return 0.0


def _load_ais_obs_series(
    con: sqlite3.Connection, observation_type: str
) -> tuple[list[datetime], np.ndarray, str]:
    rows = con.execute(
        """
        SELECT observed_at, value_json FROM entity_observations
        WHERE source_tool='ais_vessel' AND observation_type=?
        ORDER BY observed_at ASC
        """,
        (observation_type,),
    ).fetchall()
    day_values: dict[str, float] = {}
    for ts, vj in rows:
        day = _day_key(ts)
        try:
            val = json.loads(vj) if isinstance(vj, str) else vj
            day_values[day] = _ais_metric_from_value(val or {})
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    if not day_values:
        return [], np.array([]), ""
    day_order = sorted(day_values.keys())
    times = [
        datetime.fromisoformat(d + "T12:00:00+00:00").replace(tzinfo=timezone.utc)
        for d in day_order
    ]
    values = np.array([day_values[d] for d in day_order], dtype=float)
    return times, values, ""


def _load_ais_daily_counts(con: sqlite3.Connection) -> tuple[list[datetime], np.ndarray, str]:
    live_times, live_values, _ = _load_ais_obs_series(con, "area_daily_activity")
    if len(live_times) >= 30:
        return live_times, live_values, "AIS Baltic tanker count (live)"

    proxy_times, proxy_values, _ = _load_ais_obs_series(con, "baltic_activity_proxy")
    if len(proxy_times) >= 7:
        return proxy_times, proxy_values, "AIS Baltic gateway tanker port calls (proxy)"

    if len(live_times) > 0:
        return live_times, live_values, "AIS Baltic tanker count (live)"

    rows = con.execute(
        """
        SELECT observed_at FROM entity_observations
        WHERE source_tool='ais_vessel' AND observation_type='vessel_position'
        ORDER BY observed_at ASC
        """
    ).fetchall()
    if not rows:
        return [], np.array([]), "AIS tanker positions (Baltic)"
    day_counts: dict[str, int] = {}
    for (ts,) in rows:
        day = _day_key(ts)
        day_counts[day] = day_counts.get(day, 0) + 1
    day_order = sorted(day_counts.keys())
    times = [
        datetime.fromisoformat(d + "T12:00:00+00:00").replace(tzinfo=timezone.utc)
        for d in day_order
    ]
    values = np.array([day_counts[d] for d in day_order], dtype=float)
    return times, values, "AIS tanker positions (Baltic)"


def _country_entity_ids(con: sqlite3.Connection, iso2_list: tuple[str, ...]) -> list[str]:
    if not iso2_list:
        return []
    names: list[str] = []
    for code in iso2_list:
        names.append(code)
        if code in _ISO2_TO_COUNTRY_NAME:
            names.append(_ISO2_TO_COUNTRY_NAME[code])
    unique_names = list(dict.fromkeys(names))
    placeholders = ",".join("?" * len(unique_names))
    rows = con.execute(
        f"""
        SELECT entity_id FROM entities
        WHERE entity_type='country' AND canonical_name IN ({placeholders})
        """,
        unique_names,
    ).fetchall()
    if rows:
        return [r[0] for r in rows]
    # Fallback: metadata iso codes
    out: list[str] = []
    for iso2 in iso2_list:
        row = con.execute(
            "SELECT entity_id FROM entities WHERE entity_type='country' "
            "AND (canonical_name=? OR metadata_json LIKE ?) LIMIT 1",
            (iso2, f'%"iso2": "{iso2}"%'),
        ).fetchone()
        if row:
            out.append(row[0])
    return out


def _load_gdelt_producer_stress(
    con: sqlite3.Connection, iso2_list: tuple[str, ...]
) -> tuple[list[datetime], np.ndarray, str]:
    eids = _country_entity_ids(con, iso2_list)
    if not eids:
        return [], np.array([]), "GDELT producer countries"
    placeholders = ",".join("?" * len(eids))
    rows = con.execute(
        f"""
        SELECT observed_at, value_json FROM entity_observations
        WHERE source_tool='gdelt' AND observation_type='geopolitical_event'
          AND entity_id IN ({placeholders})
        ORDER BY observed_at ASC
        """,
        eids,
    ).fetchall()
    day_gold: dict[str, list[float]] = {}
    day_order: list[str] = []
    for ts, vj in rows:
        day = _day_key(ts)
        try:
            val = json.loads(vj) if isinstance(vj, str) else vj
            g = abs(float(val.get("goldstein", 0)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if day not in day_gold:
            day_gold[day] = []
            day_order.append(day)
        day_gold[day].append(g)
    if not day_order:
        return [], np.array([]), "GDELT producer countries"
    times = [datetime.fromisoformat(d + "T12:00:00+00:00") for d in day_order]
    values = np.array([float(np.mean(day_gold[d])) for d in day_order], dtype=float)
    return times, values, "GDELT producer stress"


def _series_for_node(
    con: sqlite3.Connection, node: ChainNodeSpec
) -> tuple[list[datetime], np.ndarray, str]:
    if node.source_tool == "cftc" and node.instrument_ticker:
        return _load_cftc_series(con, node.instrument_ticker, node.metric)
    if node.source_tool == "energy_supply":
        series = node.value_filter.get("series", "crude_excl_spr")
        return _load_eia_series(con, series, node.metric)
    if node.source_tool == "ais_vessel":
        return _load_ais_daily_counts(con)
    if node.source_tool == "gdelt":
        return _load_gdelt_producer_stress(con, node.producer_countries)
    return [], np.array([]), node.label


def _z_passes(z: float, min_z: float, direction: str | None) -> bool:
    if direction == "below":
        return z <= -min_z
    if direction == "above":
        return z >= min_z
    return abs(z) >= min_z


def _best_anomaly_in_window(
    times: list[datetime],
    values: np.ndarray,
    as_of: datetime,
    lag: tuple[int, int],
    min_z: float,
    direction: str | None,
    window: int,
) -> tuple[float, float, datetime, int] | None:
    if len(times) == 0:
        return None
    lo = as_of - timedelta(days=lag[1])
    hi = as_of - timedelta(days=lag[0])
    best: tuple[float, float, datetime, int] | None = None
    for i, t in enumerate(times):
        if t < lo or t > hi:
            continue
        z = rolling_zscore(values, i, window)
        if not _z_passes(z, min_z, direction):
            continue
        score = abs(z)
        if best is None or score > best[0]:
            best = (score, z, t, i)
    return best


def evaluate_node(
    con: sqlite3.Connection,
    node: ChainNodeSpec,
    as_of: datetime | None = None,
) -> NodeMatch | None:
    """Return the strongest anomaly for one template node, or None."""
    as_of = as_of or datetime.now(timezone.utc)
    times, values, entity_label = _series_for_node(con, node)
    if len(times) == 0:
        return None
    window = min(_Z_WINDOW, max(3, len(values) - 1))
    hit = _best_anomaly_in_window(
        times, values, as_of, node.lag_days, node.min_zscore, node.direction, window
    )
    if hit is None:
        return None
    _score, z, obs_dt, idx = hit
    obs_at = obs_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return NodeMatch(
        node_id=node.id,
        entity=entity_label,
        obs=f"{node.source_tool}/{node.observation_type}/{node.metric}",
        z=round(z, 3),
        value=round(float(values[idx]), 4),
        observed_at=obs_at,
        source_url=node.source_url,
    )


def match_chain(
    con: sqlite3.Connection,
    template: ChainTemplate,
    as_of: datetime | None = None,
) -> ChainMatch | None:
    """Match all nodes in a template. Returns None if any node fails."""
    as_of = as_of or datetime.now(timezone.utc)
    matches: list[NodeMatch] = []
    for node in template.nodes:
        m = evaluate_node(con, node, as_of)
        if m is None:
            return None
        matches.append(m)
    # Chain score = geometric mean of |z| (penalizes weak nodes)
    zs = [max(0.01, abs(n.z)) for n in matches]
    chain_score = float(np.exp(np.mean(np.log(zs))))
    if chain_score < template.min_chain_score:
        return None
    return ChainMatch(
        template=template,
        nodes=matches,
        chain_score=round(chain_score, 3),
        issued_at=as_of.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )


def chain_to_alert(match: ChainMatch, seq: int = 1) -> dict[str, Any]:
    """Convert a ChainMatch to ghost archive JSON."""
    day = match.issued_at[:10]
    alert_id = f"{day}_{match.template.micro_playground}_{match.template.id.upper()}_{seq:03d}"
    return {
        "alert_id": alert_id,
        "micro_playground": match.template.micro_playground,
        "chain_template": match.template.id,
        "nodes": [
            {
                "entity": n.entity,
                "obs": n.obs,
                "z": n.z,
                "value": n.value,
                "source_url": n.source_url,
                "observed_at": n.observed_at,
            }
            for n in match.nodes
        ],
        "issued_at": match.issued_at,
        "readout_instrument": match.template.readout_instrument,
        "evaluation_window_days": match.template.evaluation_window_days,
        "chain_score": match.chain_score,
        "outcome": None,
    }


def scan_templates(
    con: sqlite3.Connection,
    template_paths: list[Path],
    as_of: datetime | None = None,
) -> list[dict[str, Any]]:
    """Scan templates and return alert dicts for all matches."""
    alerts: list[dict[str, Any]] = []
    for i, path in enumerate(template_paths, start=1):
        tmpl = load_chain_template(path)
        m = match_chain(con, tmpl, as_of)
        if m is not None:
            alerts.append(chain_to_alert(m, seq=i))
    return alerts


def list_mp1_templates(root: Path | None = None) -> list[Path]:
    root = root or Path("templates/ghost_chains/mp1")
    return sorted(root.glob("*.yaml"))
