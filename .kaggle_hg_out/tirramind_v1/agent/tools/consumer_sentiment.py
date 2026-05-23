"""
Tool: Consumer Sentiment Monitor — Eurostat + FRED + BLS CPI

Cross-source consumer confidence and inflation perception tracking.
Eurostat provides EU-wide + per-country confidence (free, no auth).
FRED provides US UMichigan sentiment (requires TIRRA_FRED_API_KEY).
BLS provides actual CPI for reality-checking inflation expectations.

Modes:
  eu_confidence   — Eurostat ei_bsco_m: EU consumer confidence by country.
                    Balance % indicator (positive = more optimists than pessimists).
  us_sentiment    — FRED: UMichigan headline sentiment + inflation expectations.
                    Requires TIRRA_FRED_API_KEY.  Graceful degradation without key.
  inflation_reality — BLS CPI actual + optional FRED inflation expectations gap.
                    Free, no auth for CPI portion.

Signal theory:
  - EU confidence dropping while US stable → transatlantic divergence = EUR/USD signal
  - UMich inflation expectations unanchored (>4%) → CB credibility at risk
  - CPI actual MoM diverging from MICH expectations → expectation gap
  - Multi-country EU confidence plunging simultaneously → synchronized recession
  - Financial situation + major purchases both sinking → demand cliff forming
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC
from typing import TYPE_CHECKING, Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key as _entity_id_from_key
except ImportError:  # pragma: no cover -- optional dependency
    _entity_id_from_key = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_UA = "TirraMind/0.1 (research)"
_TIMEOUT = 15
_CACHE_TTL = 21600  # 6 hours — monthly data

# ── Eurostat ────────────────────────────────────────────────────
_EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsco_m"

# Known EU geo codes
_EU_GEOS = frozenset(
    {
        "EU27_2020",
        "EA20",
        "DE",
        "FR",
        "IT",
        "ES",
        "NL",
        "BE",
        "AT",
        "PT",
        "GR",
        "FI",
        "IE",
        "SE",
        "DK",
        "PL",
        "CZ",
        "RO",
        "HU",
        "BG",
        "HR",
        "SK",
        "SI",
        "LT",
        "LV",
        "EE",
        "CY",
        "LU",
        "MT",
    }
)

_DEFAULT_EU_COUNTRIES = "EU27_2020,DE,FR,IT,ES"
_EU_AGGREGATES = frozenset({"EU27_2020", "EA20"})

# ── FRED ────────────────────────────────────────────────────────
_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

_FRED_SERIES = {
    "UMCSENT": "UMichigan Consumer Sentiment (headline)",
    "MICH": "UMichigan 1-Year Inflation Expectations",
}

# ── BLS ─────────────────────────────────────────────────────────
_BLS_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

_BLS_CPI_SA = "CUSR0000SA0"  # CPI-U seasonally adjusted
_BLS_CPI_NSA = "CUUR0000SA0"  # CPI-U not seasonally adjusted

VALID_MODES = frozenset({"eu_confidence", "us_sentiment", "inflation_reality"})


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s in (".", "NaN", "nan", "null", ""):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _get_fred_key() -> str | None:
    """Return FRED API key if configured and not a placeholder."""
    key = os.environ.get("TIRRA_FRED_API_KEY", "")
    if not key or key.startswith("your-"):
        return None
    return key


class ConsumerSentimentTool(Tool):
    """Consumer confidence + inflation expectations across US and EU."""

    def __init__(
        self,
        cache: DataCache | None = None,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    @property
    def name(self) -> str:
        return "consumer_sentiment"

    @property
    def description(self) -> str:
        return (
            "Track consumer confidence and inflation expectations. "
            "Modes: eu_confidence (Eurostat, 28 EU countries, free), "
            "us_sentiment (UMichigan via FRED, requires API key), "
            "inflation_reality (BLS CPI actual vs expectations). "
            "Detects cross-country sentiment divergence and inflation expectation gaps."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": sorted(VALID_MODES),
                    "description": (
                        "eu_confidence: Eurostat consumer confidence by EU country. "
                        "us_sentiment: UMichigan headline + inflation expectations (needs FRED key). "
                        "inflation_reality: BLS CPI actual + expectation gap analysis."
                    ),
                },
                "countries": {
                    "type": "string",
                    "default": _DEFAULT_EU_COUNTRIES,
                    "description": (
                        "Comma-separated Eurostat geo codes (eu_confidence mode). "
                        f"Available: {', '.join(sorted(_EU_GEOS))}. "
                        f"Default: {_DEFAULT_EU_COUNTRIES}"
                    ),
                },
                "months": {
                    "type": "integer",
                    "default": 6,
                    "description": "Number of months of data to fetch (1-24). Default: 6.",
                },
            },
            "required": ["mode"],
        }

    # ── Public execute ──────────────────────────────────────────

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = kwargs.get("mode", "")
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Choose from: {', '.join(sorted(VALID_MODES))}.",
            )

        months = kwargs.get("months") or 6
        if not isinstance(months, int):
            try:
                months = int(months)
            except (ValueError, TypeError):
                months = 6
        months = max(1, min(months, 24))

        if mode == "eu_confidence":
            countries_str = kwargs.get("countries") or _DEFAULT_EU_COUNTRIES
            result = self._handle_eu_confidence(countries_str, months)
        elif mode == "us_sentiment":
            result = self._handle_us_sentiment(months)
        else:
            result = self._handle_inflation_reality(months)

        if result.success and result.data:
            self._persist_entities(result.data, mode)

        return result

    # ── EU Confidence ───────────────────────────────────────────

    def _handle_eu_confidence(self, countries_str: str, months: int) -> ToolResult:
        geos = [g.strip().upper() for g in countries_str.split(",") if g.strip()]
        valid_geos = [g for g in geos if g in _EU_GEOS]
        if not valid_geos:
            return ToolResult(
                success=False,
                output=f"No valid Eurostat geo codes in '{countries_str}'. Available: {', '.join(sorted(_EU_GEOS))}",
            )

        cache_key = f"consumer_sentiment:eu:{','.join(sorted(valid_geos))}:{months}"
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(success=True, output=hit["output"], data=hit["data"])

        data, err = _fetch_eurostat(valid_geos, months)
        if err:
            return ToolResult(success=False, output=err)

        signals = _compute_eu_signals(data, valid_geos)
        summary = _format_eu_summary(data, signals, valid_geos, months)

        result_data = {
            "mode": "eu_confidence",
            "countries": valid_geos,
            "months": months,
            "data": data,
            "signals": signals,
        }

        if self._cache:
            self._cache.set(cache_key, {"output": summary, "data": result_data}, ttl=_CACHE_TTL)

        return ToolResult(success=True, output=summary, data=result_data)

    # ── US Sentiment ────────────────────────────────────────────

    def _handle_us_sentiment(self, months: int) -> ToolResult:
        fred_key = _get_fred_key()
        if not fred_key:
            return ToolResult(
                success=False,
                output=(
                    "us_sentiment mode requires a FRED API key. "
                    "Set TIRRA_FRED_API_KEY in your environment. "
                    "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
                ),
            )

        cache_key = f"consumer_sentiment:us:{months}"
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(success=True, output=hit["output"], data=hit["data"])

        series_data = {}
        for sid, label in _FRED_SERIES.items():
            records, err = _fetch_fred_series(fred_key, sid, months)
            if err:
                series_data[sid] = {"label": label, "error": err, "records": []}
            else:
                series_data[sid] = {"label": label, "records": records}

        signals = _compute_us_signals(series_data)
        summary = _format_us_summary(series_data, signals, months)

        result_data = {
            "mode": "us_sentiment",
            "months": months,
            "series": series_data,
            "signals": signals,
        }

        if self._cache:
            self._cache.set(cache_key, {"output": summary, "data": result_data}, ttl=_CACHE_TTL)

        return ToolResult(success=True, output=summary, data=result_data)

    # ── Inflation Reality ───────────────────────────────────────

    def _handle_inflation_reality(self, months: int) -> ToolResult:
        cache_key = f"consumer_sentiment:cpi:{months}"
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(success=True, output=hit["output"], data=hit["data"])

        cpi_records, err = _fetch_bls_cpi(months)
        if err:
            return ToolResult(success=False, output=err)

        # Optionally fetch FRED inflation expectations for gap analysis
        fred_key = _get_fred_key()
        mich_records = []
        if fred_key:
            mich_records, _ = _fetch_fred_series(fred_key, "MICH", months)

        signals = _compute_cpi_signals(cpi_records, mich_records)
        summary = _format_cpi_summary(cpi_records, mich_records, signals, months)

        result_data = {
            "mode": "inflation_reality",
            "months": months,
            "cpi": cpi_records,
            "inflation_expectations": mich_records,
            "has_expectations": bool(mich_records),
            "signals": signals,
        }

        if self._cache:
            self._cache.set(cache_key, {"output": summary, "data": result_data}, ttl=_CACHE_TTL)

        return ToolResult(success=True, output=summary, data=result_data)

    # ── L2 entity persistence (Phase 31) ──────────────────────

    def _persist_entities(
        self,
        data: dict[str, Any],
        mode: str,
    ) -> dict[str, int]:
        if self._store is None or _entity_id_from_key is None:
            return {"consumer_confidence_obs": 0}
        try:
            return self._persist_entities_inner(data, mode)
        except Exception:
            log.exception("Consumer sentiment entity persistence failed (non-fatal)")
            return {"consumer_confidence_obs": 0}

    def _persist_entities_inner(
        self,
        data: dict[str, Any],
        mode: str,
    ) -> dict[str, int]:
        assert self._store is not None  # noqa: S101 -- guarded
        assert _entity_id_from_key is not None  # noqa: S101

        store = self._store
        counts = {"consumer_confidence_obs": 0}
        now_ts = time.time()

        if mode == "eu_confidence":
            by_geo = data.get("data", {})
            country_signals = data.get("signals", {}).get("countries", {})
            for geo, series in by_geo.items():
                if geo in _EU_AGGREGATES or len(geo) != 2 or not series:
                    continue
                latest = series[-1]
                signal = country_signals.get(geo, {})
                country_eid = _entity_id_from_key("country", geo)
                store.register_entity("country", geo, country_eid)
                store.store_entity_observation(
                    entity_id=country_eid,
                    source_tool="consumer_sentiment",
                    observed_at=now_ts,
                    observation_type="consumer_confidence",
                    value={
                        "mode": "eu_confidence",
                        "source": "eurostat",
                        "latest": latest.get("value"),
                        "period": latest.get("period"),
                        "mom_change": signal.get("mom_change"),
                        "trend": signal.get("trend"),
                        "consecutive_decline": signal.get("consecutive_decline"),
                    },
                    depth_level=2,
                )
                counts["consumer_confidence_obs"] += 1

        elif mode == "us_sentiment":
            signals = data.get("signals", {})
            country_eid = _entity_id_from_key("country", "US")
            store.register_entity("country", "US", country_eid)
            store.store_entity_observation(
                entity_id=country_eid,
                source_tool="consumer_sentiment",
                observed_at=now_ts,
                observation_type="consumer_confidence",
                value={
                    "mode": "us_sentiment",
                    "source": "fred",
                    "latest": signals.get("sentiment_latest"),
                    "period": signals.get("sentiment_date"),
                    "mom_change": signals.get("sentiment_mom"),
                    "trend": signals.get("sentiment_alert"),
                    "inflation_exp_1yr": signals.get("inflation_exp_1yr"),
                    "inflation_anchor": signals.get("inflation_anchor"),
                },
                depth_level=2,
            )
            counts["consumer_confidence_obs"] += 1

        elif mode == "inflation_reality":
            signals = data.get("signals", {})
            country_eid = _entity_id_from_key("country", "US")
            store.register_entity("country", "US", country_eid)
            store.store_entity_observation(
                entity_id=country_eid,
                source_tool="consumer_sentiment",
                observed_at=now_ts,
                observation_type="consumer_confidence",
                value={
                    "mode": "inflation_reality",
                    "source": "bls",
                    "latest": signals.get("cpi_latest"),
                    "period": signals.get("cpi_period"),
                    "mom_change": signals.get("cpi_mom_pct"),
                    "trend": signals.get("gap_signal"),
                    "cpi_yoy_pct": signals.get("cpi_yoy_pct"),
                    "expectation_gap": signals.get("expectation_gap"),
                },
                depth_level=2,
            )
            counts["consumer_confidence_obs"] += 1

        if counts["consumer_confidence_obs"]:
            log.info(
                "Consumer sentiment L2: %d consumer_confidence obs persisted",
                counts["consumer_confidence_obs"],
            )
        return counts


# ── Eurostat fetch & parse ──────────────────────────────────────


def _fetch_eurostat(
    geos: list[str],
    months: int,
) -> tuple[dict[str, list[dict]], str | None]:
    """Fetch Eurostat ei_bsco_m consumer confidence. Returns {geo: [{period, value}]}."""
    params = {
        "s_adj": "SA",
        "indic": "BS-CSMCI",
        "lastTimePeriod": str(months),
    }
    # Eurostat wants repeated geo params
    geo_params = "&".join(f"geo={g}" for g in geos)
    url = f"{_EUROSTAT_BASE}?{geo_params}"

    try:
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}) as client:
            resp = client.get(url, params=params)
    except httpx.TimeoutException:
        return {}, "Eurostat API timed out."
    except httpx.HTTPError as exc:
        return {}, f"Eurostat HTTP error: {exc}"

    if resp.status_code != 200:
        return {}, f"Eurostat returned HTTP {resp.status_code}"

    try:
        body = resp.json()
    except Exception:
        return {}, "Failed to parse Eurostat response."

    return _parse_eurostat_jsonstat(body, geos), None


def _parse_eurostat_jsonstat(
    body: dict,
    requested_geos: list[str],
) -> dict[str, list[dict]]:
    """Parse JSON-stat 2.0 response into {geo: [{period, value}]}."""
    values = body.get("value", {})
    if not values:
        return {}

    dimensions = body.get("id", [])
    sizes = body.get("size", [])

    # Find geo and time dimension positions
    dim_info = body.get("dimension", {})

    geo_dim = dim_info.get("geo", {})
    geo_index = geo_dim.get("category", {}).get("index", {})
    geo_labels = geo_dim.get("category", {}).get("label", {})

    time_dim = dim_info.get("time", {})
    time_index = time_dim.get("category", {}).get("index", {})

    if not geo_index or not time_index:
        return {}

    # Build reverse index: position → geo/time label
    geo_by_pos = {v: k for k, v in geo_index.items()}
    time_by_pos = {v: k for k, v in time_index.items()}

    n_geos = len(geo_index)
    n_times = len(time_index)

    result: dict[str, list[dict]] = {}

    for flat_idx_str, val in values.items():
        flat_idx = int(flat_idx_str)
        # JSON-stat flat index: multiply sizes in order
        # For 2D (after collapsing other dims): geo * n_times + time
        # But actual structure may have more dims — compute from sizes
        # Eurostat ei_bsco_m dimensions: freq, s_adj, indic, geo, time
        # After filtering freq=M, s_adj=SA, indic=BS-CSMCI, the remaining dims
        # are geo × time. The flat index maps: geo_pos * n_times + time_pos

        geo_pos = flat_idx // n_times
        time_pos = flat_idx % n_times

        geo_code = geo_by_pos.get(geo_pos, "")
        time_label = time_by_pos.get(time_pos, "")

        if not geo_code or not time_label:
            continue

        fval = _safe_float(val)
        if fval is None:
            continue

        if geo_code not in result:
            result[geo_code] = []
        result[geo_code].append(
            {
                "period": time_label,
                "value": fval,
                "geo_label": geo_labels.get(geo_code, geo_code),
            }
        )

    # Sort each series chronologically
    for geo in result:
        result[geo].sort(key=lambda r: r["period"])

    return result


# ── FRED fetch ──────────────────────────────────────────────────


def _fetch_fred_series(
    api_key: str,
    series_id: str,
    limit: int,
) -> tuple[list[dict], str | None]:
    """Fetch FRED series observations."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": str(limit),
    }

    try:
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}) as client:
            resp = client.get(_FRED_BASE, params=params)
    except httpx.TimeoutException:
        return [], "FRED API timed out."
    except httpx.HTTPError as exc:
        return [], f"FRED HTTP error: {exc}"

    if resp.status_code == 400:
        return [], "FRED returned 400 — check API key validity."
    if resp.status_code != 200:
        return [], f"FRED returned HTTP {resp.status_code}"

    try:
        body = resp.json()
    except Exception:
        return [], "Failed to parse FRED response."

    if "error_message" in body:
        return [], f"FRED error: {body['error_message']}"

    observations = body.get("observations", [])
    records = []
    for obs in observations:
        val = _safe_float(obs.get("value"))
        if val is None:
            continue
        records.append(
            {
                "date": obs.get("date", ""),
                "value": val,
            }
        )

    # Sort chronologically (FRED returns desc)
    records.sort(key=lambda r: r["date"])
    return records, None


# ── BLS CPI fetch ───────────────────────────────────────────────


def _fetch_bls_cpi(months: int) -> tuple[list[dict], str | None]:
    """Fetch BLS CPI-U seasonally adjusted."""
    from datetime import datetime

    now = datetime.now(UTC)
    # BLS requires year range; we need enough years to cover months
    end_year = now.year
    start_year = max(end_year - 2, end_year - (months // 12 + 1))

    payload = {
        "seriesid": [_BLS_CPI_SA],
        "startyear": str(start_year),
        "endyear": str(end_year),
    }

    try:
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}) as client:
            resp = client.post(_BLS_BASE, json=payload)
    except httpx.TimeoutException:
        return [], "BLS API timed out."
    except httpx.HTTPError as exc:
        return [], f"BLS HTTP error: {exc}"

    if resp.status_code == 429:
        return [], "BLS API rate limit reached. Retry later."
    if resp.status_code != 200:
        return [], f"BLS returned HTTP {resp.status_code}"

    try:
        body = resp.json()
    except Exception:
        return [], "Failed to parse BLS response."

    if body.get("status") != "REQUEST_SUCCEEDED":
        msg = "; ".join(body.get("message", ["Unknown error"]))
        return [], f"BLS request failed: {msg}"

    series_list = body.get("Results", {}).get("series", [])
    if not series_list:
        return [], "No CPI series returned."

    raw = series_list[0].get("data", [])
    records = []
    for entry in raw:
        period = entry.get("period", "")
        if period == "M13":  # Annual average — skip
            continue
        val = _safe_float(entry.get("value"))
        if val is None:
            continue
        records.append(
            {
                "year": entry.get("year", ""),
                "period": period,
                "value": val,
            }
        )

    records.sort(key=lambda r: (r["year"], r["period"]))

    # Trim to requested months
    if len(records) > months:
        records = records[-months:]

    return records, None


# ── Signal computation ──────────────────────────────────────────


def _compute_eu_signals(
    data: dict[str, list[dict]],
    geos: list[str],
) -> dict[str, Any]:
    """Compute signals from Eurostat consumer confidence data."""
    if not data:
        return {"status": "NO_DATA"}

    signals: dict[str, Any] = {"countries": {}}

    latest_values: dict[str, float] = {}

    for geo in geos:
        series = data.get(geo, [])
        if not series:
            continue

        latest = series[-1]["value"]
        latest_values[geo] = latest

        country_sig: dict[str, Any] = {
            "latest": latest,
            "latest_period": series[-1]["period"],
        }

        if len(series) >= 2:
            prev = series[-2]["value"]
            mom_change = latest - prev
            country_sig["mom_change"] = round(mom_change, 1)

            if mom_change > 1.0:
                country_sig["trend"] = "IMPROVING"
            elif mom_change < -1.0:
                country_sig["trend"] = "DETERIORATING"
            else:
                country_sig["trend"] = "STABLE"
        else:
            country_sig["mom_change"] = None
            country_sig["trend"] = "INSUFFICIENT_DATA"

        if len(series) >= 3:
            recent_3 = [r["value"] for r in series[-3:]]
            consecutive_drops = all(recent_3[i] < recent_3[i - 1] for i in range(1, len(recent_3)))
            country_sig["consecutive_decline"] = consecutive_drops
        else:
            country_sig["consecutive_decline"] = False

        signals["countries"][geo] = country_sig

    # Cross-country divergence
    if len(latest_values) >= 2:
        vals = list(latest_values.values())
        spread = max(vals) - min(vals)
        signals["cross_country_spread"] = round(spread, 1)
        signals["most_optimistic"] = max(latest_values, key=latest_values.get)  # type: ignore[arg-type]
        signals["most_pessimistic"] = min(latest_values, key=latest_values.get)  # type: ignore[arg-type]

        # Synchronized decline: all countries deteriorating
        trends = [signals["countries"][g].get("trend") for g in geos if g in signals["countries"]]
        signals["synchronized_decline"] = all(t == "DETERIORATING" for t in trends) and len(trends) >= 2
    else:
        signals["cross_country_spread"] = None
        signals["synchronized_decline"] = False

    return signals


def _compute_us_signals(series_data: dict) -> dict[str, Any]:
    """Compute signals from FRED UMichigan data."""
    signals: dict[str, Any] = {}

    # Headline sentiment
    umcsent = series_data.get("UMCSENT", {})
    records = umcsent.get("records", [])
    if records:
        latest = records[-1]["value"]
        signals["sentiment_latest"] = latest
        signals["sentiment_date"] = records[-1]["date"]

        if len(records) >= 2:
            prev = records[-2]["value"]
            signals["sentiment_mom"] = round(latest - prev, 1)
        else:
            signals["sentiment_mom"] = None

        avg = sum(r["value"] for r in records) / len(records)
        signals["sentiment_avg"] = round(avg, 1)
        signals["sentiment_vs_avg"] = round(latest - avg, 1)

        if latest < 60:
            signals["sentiment_alert"] = "CRITICAL_LOW — recession-level pessimism"
        elif latest < 70:
            signals["sentiment_alert"] = "WARNING — below-normal confidence"
        else:
            signals["sentiment_alert"] = None
    else:
        signals["sentiment_latest"] = None

    # Inflation expectations
    mich = series_data.get("MICH", {})
    records = mich.get("records", [])
    if records:
        latest_exp = records[-1]["value"]
        signals["inflation_exp_1yr"] = latest_exp
        signals["inflation_exp_date"] = records[-1]["date"]

        if latest_exp > 4.0:
            signals["inflation_anchor"] = "UNANCHORED — expectations above 4%"
        elif latest_exp > 3.0:
            signals["inflation_anchor"] = "ELEVATED — expectations above 3%"
        else:
            signals["inflation_anchor"] = "ANCHORED"
    else:
        signals["inflation_exp_1yr"] = None

    return signals


def _compute_cpi_signals(
    cpi_records: list[dict],
    mich_records: list[dict],
) -> dict[str, Any]:
    """Compute CPI signals and expectation gap."""
    if not cpi_records:
        return {"status": "NO_DATA"}

    signals: dict[str, Any] = {}

    latest = cpi_records[-1]
    signals["cpi_latest"] = latest["value"]
    signals["cpi_period"] = f"{latest['year']}-{latest['period']}"

    if len(cpi_records) >= 2:
        prev = cpi_records[-2]["value"]
        if prev > 0:
            mom_pct = ((latest["value"] / prev) - 1) * 100
            signals["cpi_mom_pct"] = round(mom_pct, 2)
            signals["cpi_annualized"] = round(mom_pct * 12, 1)
        else:
            signals["cpi_mom_pct"] = None
            signals["cpi_annualized"] = None
    else:
        signals["cpi_mom_pct"] = None
        signals["cpi_annualized"] = None

    # YoY (12 months back)
    if len(cpi_records) >= 13:
        yr_ago = cpi_records[-13]["value"]
        if yr_ago > 0:
            yoy = ((latest["value"] / yr_ago) - 1) * 100
            signals["cpi_yoy_pct"] = round(yoy, 2)
        else:
            signals["cpi_yoy_pct"] = None
    else:
        signals["cpi_yoy_pct"] = None

    # Expectation gap
    if mich_records and signals.get("cpi_yoy_pct") is not None:
        # Compare UMich 1yr expectations with actual YoY CPI
        latest_exp = mich_records[-1]["value"]
        actual_yoy = signals["cpi_yoy_pct"]
        gap = latest_exp - actual_yoy
        signals["expectation_gap"] = round(gap, 2)
        if gap > 1.5:
            signals["gap_signal"] = "EXPECTATIONS_ABOVE_REALITY — consumers more worried than warranted"
        elif gap < -1.0:
            signals["gap_signal"] = "EXPECTATIONS_BELOW_REALITY — inflation underestimated"
        else:
            signals["gap_signal"] = "ALIGNED"
    else:
        signals["expectation_gap"] = None
        signals["gap_signal"] = None

    return signals


# ── Formatting ──────────────────────────────────────────────────


def _format_eu_summary(
    data: dict[str, list[dict]],
    signals: dict,
    geos: list[str],
    months: int,
) -> str:
    lines = [f"EU Consumer Confidence ({months} months, Eurostat ei_bsco_m):\n"]

    for geo in geos:
        csig = signals.get("countries", {}).get(geo, {})
        if not csig:
            lines.append(f"  {geo}: no data")
            continue

        latest = csig.get("latest", "N/A")
        period = csig.get("latest_period", "")
        trend = csig.get("trend", "")
        mom = csig.get("mom_change")
        mom_str = f" ({'+' if mom and mom > 0 else ''}{mom})" if mom is not None else ""
        decline = " ⚠ consecutive decline" if csig.get("consecutive_decline") else ""
        lines.append(f"  {geo}: {latest}{mom_str} [{trend}] ({period}){decline}")

    spread = signals.get("cross_country_spread")
    if spread is not None:
        best = signals.get("most_optimistic", "?")
        worst = signals.get("most_pessimistic", "?")
        lines.append(f"\n  Divergence: spread={spread} pts (best: {best}, worst: {worst})")

    if signals.get("synchronized_decline"):
        lines.append("  ⚠ SYNCHRONIZED DECLINE — all tracked countries deteriorating")

    return "\n".join(lines)


def _format_us_summary(
    series_data: dict,
    signals: dict,
    months: int,
) -> str:
    lines = [f"US Consumer Sentiment ({months} months, UMichigan via FRED):\n"]

    sentinel_val = signals.get("sentiment_latest")
    if sentinel_val is not None:
        mom = signals.get("sentiment_mom")
        mom_str = f" (MoM: {'+' if mom and mom > 0 else ''}{mom})" if mom is not None else ""
        vs_avg = signals.get("sentiment_vs_avg")
        vs_str = f", vs {months}mo avg: {'+' if vs_avg and vs_avg > 0 else ''}{vs_avg}" if vs_avg is not None else ""
        lines.append(f"  Headline: {sentinel_val}{mom_str}{vs_str}")
        alert = signals.get("sentiment_alert")
        if alert:
            lines.append(f"  ⚠ {alert}")
    else:
        lines.append("  Headline: no data")

    inf_exp = signals.get("inflation_exp_1yr")
    if inf_exp is not None:
        anchor = signals.get("inflation_anchor", "")
        lines.append(f"  1yr Inflation Expectations: {inf_exp}% [{anchor}]")
    else:
        lines.append("  1yr Inflation Expectations: no data")

    return "\n".join(lines)


def _format_cpi_summary(
    cpi_records: list[dict],
    mich_records: list[dict],
    signals: dict,
    months: int,
) -> str:
    lines = [f"Inflation Reality Check ({months} months, BLS CPI-U SA):\n"]

    cpi_val = signals.get("cpi_latest")
    if cpi_val is None:
        lines.append("  CPI: no data")
        return "\n".join(lines)

    period = signals.get("cpi_period", "")
    mom = signals.get("cpi_mom_pct")
    ann = signals.get("cpi_annualized")
    yoy = signals.get("cpi_yoy_pct")

    lines.append(f"  CPI-U SA: {cpi_val} ({period})")
    if mom is not None:
        lines.append(f"  MoM: {'+' if mom > 0 else ''}{mom}% (annualized: {'+' if ann and ann > 0 else ''}{ann}%)")
    if yoy is not None:
        lines.append(f"  YoY: {'+' if yoy > 0 else ''}{yoy}%")

    gap = signals.get("expectation_gap")
    if gap is not None:
        gap_sig = signals.get("gap_signal", "")
        exp = mich_records[-1]["value"] if mich_records else "?"
        lines.append(
            f"\n  Expectation Gap: UMich expects {exp}%, actual YoY {yoy}% → gap {'+' if gap > 0 else ''}{gap}pp"
        )
        lines.append(f"  [{gap_sig}]")
    elif not mich_records:
        lines.append("\n  (FRED key not available — expectation gap analysis skipped)")

    return "\n".join(lines)
