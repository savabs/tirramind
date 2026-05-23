"""
Tool: Internet Infrastructure Monitor — IODA + OONI

Global internet outage detection and censorship monitoring using two
commercially safe data sources:
  - IODA (Georgia Tech) — BGP visibility, active probing, Google Transparency
  - OONI (Open Observatory of Network Interference) — CC BY 4.0

Modes:
  outages    — IODA alerts + events: country-level internet outage detection.
               Detects BGP prefix visibility drops, active-probing loss,
               and Google Transparency traffic anomalies.
  censorship — OONI aggregation: daily blocking rates by country and test type.
               Covers web_connectivity, telegram, whatsapp, signal, tor, etc.
               Returns trend analysis (rising/falling/stable censorship).
  signals    — IODA normalized connectivity timeseries for a country.
               gtr-norm = fraction-of-normal traffic (1.0 = healthy).
               Flags drops below configurable threshold.
  incidents  — OONI: major ongoing censorship/blocking events worldwide.

Signal theory:
  - Country internet going dark → coup, natural disaster, cable cut, censorship
  - BGP prefix drop → physical infrastructure failure or state-directed reroute
  - Censorship escalation (anomaly_rate rising) → political instability forming
  - Multiple-country simultaneous outage → correlated infrastructure event
  - Messaging-app block (telegram/whatsapp/signal) → protest suppression signal
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from agent.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IODA_BASE = "https://api.ioda.inetintel.cc.gatech.edu/v2"
OONI_BASE = "https://api.ooni.io/api/v1"

VALID_MODES = ("outages", "censorship", "signals", "incidents")

OONI_TEST_TYPES = (
    "web_connectivity",
    "telegram",
    "whatsapp",
    "signal",
    "tor",
    "facebook_messenger",
    "psiphon",
    "ndt",
)

# Cache TTLs (seconds)
CACHE_IODA_ALERTS = 600  # 10 min — outages are time-critical
CACHE_IODA_SIGNALS = 1800  # 30 min — matches data resolution
CACHE_OONI_AGGREGATION = 3600  # 1 hr — daily data
CACHE_OONI_INCIDENTS = 3600  # 1 hr

# Scoring thresholds
GTR_NORM_WARNING = 0.80  # below 80% of normal = warning
GTR_NORM_CRITICAL = 0.50  # below 50% of normal = critical

HTTP_TIMEOUT = 15.0


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _ts_to_iso(ts: int | float | None) -> str:
    """Convert unix timestamp to ISO date string."""
    if ts is None:
        return "unknown"
    try:
        return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(int(ts)))
    except (ValueError, TypeError, OSError):
        return "unknown"


def _severity_from_gtr_norm(val: float) -> str:
    if val < GTR_NORM_CRITICAL:
        return "critical"
    if val < GTR_NORM_WARNING:
        return "warning"
    return "normal"


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class InternetInfrastructureTool(Tool):
    name = "internet_infrastructure"
    description = (
        "Monitor global internet outages and censorship. "
        "Uses IODA (Georgia Tech) for BGP/connectivity outage detection "
        "and OONI for censorship/blocking measurement across 237 countries. "
        "Modes: outages (country-level internet disruptions), "
        "censorship (daily blocking rates by country), "
        "signals (normalized connectivity timeseries), "
        "incidents (major ongoing censorship events worldwide)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": list(VALID_MODES),
                "default": "outages",
                "description": (
                    "outages = IODA country-level internet outage alerts/events. "
                    "censorship = OONI daily blocking rates by country+test. "
                    "signals = IODA normalized connectivity timeseries. "
                    "incidents = OONI major ongoing censorship events."
                ),
            },
            "country": {
                "type": "string",
                "default": "",
                "description": (
                    "ISO-2 country code (e.g. US, RU, CN, IR, BR). "
                    "Required for censorship and signals modes. "
                    "Optional for outages (blank = global scan). "
                    "Not used for incidents."
                ),
            },
            "test": {
                "type": "string",
                "enum": list(OONI_TEST_TYPES),
                "default": "web_connectivity",
                "description": (
                    "OONI test type (censorship mode only). "
                    "web_connectivity = website blocking. "
                    "telegram/whatsapp/signal/facebook_messenger = messaging app blocking. "
                    "tor/psiphon = circumvention tool blocking. "
                    "ndt = network performance."
                ),
            },
            "hours_back": {
                "type": "integer",
                "default": 24,
                "description": "Lookback window in hours (outages/signals modes). Max 168 (7 days).",
            },
            "days_back": {
                "type": "integer",
                "default": 30,
                "description": "Lookback window in days (censorship mode). Max 90.",
            },
            "limit": {
                "type": "integer",
                "default": 20,
                "description": "Max results to return. Max 100.",
            },
        },
    }

    def __init__(self, cache: Any = None) -> None:
        self._cache = cache

    def _cached_get(self, url: str, ttl: int, params: dict | None = None) -> dict | list | None:
        """HTTP GET with optional caching."""
        cache_key = f"iinfra:{url}:{params}"
        if self._cache:
            cached = self._cache.get("internet_infrastructure", {"key": cache_key})
            if cached is not None:
                return cached

        try:
            r = httpx.get(
                url,
                params=params,
                timeout=HTTP_TIMEOUT,
                headers={"User-Agent": "TirraMind/1.0 (internet-infrastructure-monitor)"},
            )
            if r.status_code != 200:
                logger.warning("HTTP %d from %s", r.status_code, url)
                return None
            data = r.json()
            if self._cache and data is not None:
                self._cache.put("internet_infrastructure", {"key": cache_key}, data)
            return data
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Request failed for %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(
        self,
        *,
        mode: str = "outages",
        country: str = "",
        test: str = "web_connectivity",
        hours_back: int = 24,
        days_back: int = 30,
        limit: int = 20,
        _backfill: bool = False,
        **_: Any,
    ) -> ToolResult:
        mode = str(mode).lower().strip()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: {', '.join(VALID_MODES)}",
            )

        country = str(country).upper().strip() if country else ""
        test = str(test).lower().strip()
        hours_back = max(1, min(_safe_int(hours_back, 24), 168))
        if not _backfill:
            days_back = max(1, min(_safe_int(days_back, 30), 90))
        limit = max(1, min(_safe_int(limit, 20), 100))

        if mode == "censorship":
            if not country or len(country) != 2:
                return ToolResult(
                    success=False,
                    output="censorship mode requires a valid 2-letter country code.",
                )
            if test not in OONI_TEST_TYPES:
                return ToolResult(
                    success=False,
                    output=f"Invalid test '{test}'. Use: {', '.join(OONI_TEST_TYPES)}",
                )
            return self._execute_censorship(country=country, test=test, days_back=days_back, limit=limit)

        if mode == "signals":
            if not country or len(country) != 2:
                return ToolResult(
                    success=False,
                    output="signals mode requires a valid 2-letter country code.",
                )
            return self._execute_signals(country=country, hours_back=hours_back)

        if mode == "incidents":
            return self._execute_incidents(limit=limit)

        # mode == "outages"
        return self._execute_outages(country=country, hours_back=hours_back, limit=limit)

    # ------------------------------------------------------------------
    # Mode: outages (IODA)
    # ------------------------------------------------------------------

    def _execute_outages(self, *, country: str, hours_back: int, limit: int) -> ToolResult:
        now = int(time.time())
        from_ts = now - (hours_back * 3600)

        params: dict[str, Any] = {
            "from": str(from_ts),
            "until": str(now),
            "limit": str(limit),
            "entityType": "country",
        }
        if country:
            params["entityCode"] = country

        # Fetch alerts
        alerts_data = self._cached_get(
            f"{IODA_BASE}/outages/alerts",
            ttl=CACHE_IODA_ALERTS,
            params=params,
        )

        # Fetch events
        events_data = self._cached_get(
            f"{IODA_BASE}/outages/events",
            ttl=CACHE_IODA_ALERTS,
            params=params,
        )

        alerts = []
        if isinstance(alerts_data, dict):
            raw_alerts = alerts_data.get("data", [])
            if isinstance(raw_alerts, list):
                for a in raw_alerts:
                    if not isinstance(a, dict):
                        continue
                    entity = a.get("entity", {})
                    if not isinstance(entity, dict):
                        continue
                    level = a.get("level", "unknown")
                    if level == "normal":
                        continue  # skip recovery-to-normal transitions
                    alerts.append(
                        {
                            "type": "alert",
                            "country": entity.get("code", "??"),
                            "country_name": entity.get("name", "Unknown"),
                            "datasource": a.get("datasource", "unknown"),
                            "level": level,
                            "condition": a.get("condition", ""),
                            "value": _safe_float(a.get("value")),
                            "baseline": _safe_float(a.get("historyValue")),
                            "time": _ts_to_iso(a.get("time")),
                            "timestamp": _safe_int(a.get("time")),
                        }
                    )

        events = []
        if isinstance(events_data, dict):
            raw_events = events_data.get("data", [])
            if isinstance(raw_events, list):
                for e in raw_events:
                    if not isinstance(e, dict):
                        continue
                    location = e.get("location", "")
                    cc = location.split("/")[-1] if "/" in str(location) else str(location)
                    events.append(
                        {
                            "type": "event",
                            "country": cc.upper(),
                            "datasource": e.get("datasource", "unknown"),
                            "score": _safe_float(e.get("score")),
                            "start": _ts_to_iso(e.get("start")),
                            "duration_minutes": round(_safe_float(e.get("duration")) / 60, 1),
                            "method": e.get("method", "unknown"),
                            "status": e.get("status"),
                        }
                    )

        # Sort: alerts by timestamp desc, events by score desc
        alerts.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        events.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Format output
        lines = [f"## Internet Outage Monitor — last {hours_back}h"]
        if country:
            lines[0] += f" ({country})"
        lines.append("Source: IODA (Georgia Tech)\n")

        _outage_data = {
            "mode": "outages",
            "alerts": alerts,
            "events": events,
            "country": country,
        }

        if not alerts and not events:
            lines.append("No outage alerts or events detected in this window.")
            return ToolResult(success=True, output="\n".join(lines), data=_outage_data)

        if alerts:
            lines.append(f"### Active Alerts ({len(alerts)})")
            for a in alerts[:limit]:
                lines.append(
                    f"  {a['country']} ({a['country_name']}): "
                    f"{a['level'].upper()} via {a['datasource']} — "
                    f"value={a['value']:.0f} (baseline={a['baseline']:.0f}) "
                    f"at {a['time']}"
                )

        if events:
            lines.append(f"\n### Outage Events ({len(events)})")
            for e in events[:limit]:
                lines.append(
                    f"  {e['country']}: score={e['score']:.1f}, "
                    f"ds={e['datasource']}, dur={e['duration_minutes']}min, "
                    f"started {e['start']}"
                )

        # Summary signals
        critical_countries = {a["country"] for a in alerts if a.get("level") == "critical"}
        if critical_countries:
            lines.append(f"\n⚠ CRITICAL: {', '.join(sorted(critical_countries))}")

        return ToolResult(success=True, output="\n".join(lines), data=_outage_data)

    # ------------------------------------------------------------------
    # Mode: censorship (OONI)
    # ------------------------------------------------------------------

    def _execute_censorship(self, *, country: str, test: str, days_back: int, limit: int) -> ToolResult:
        until_date = time.strftime("%Y-%m-%d", time.gmtime())
        since_ts = time.time() - (days_back * 86400)
        since_date = time.strftime("%Y-%m-%d", time.gmtime(since_ts))

        data = self._cached_get(
            f"{OONI_BASE}/aggregation",
            ttl=CACHE_OONI_AGGREGATION,
            params={
                "probe_cc": country,
                "test_name": test,
                "since": since_date,
                "until": until_date,
                "axis_x": "measurement_start_day",
            },
        )

        if not isinstance(data, dict):
            return ToolResult(
                success=False,
                output=f"Failed to fetch OONI aggregation for {country}/{test}.",
            )

        result = data.get("result", [])
        if not isinstance(result, list) or not result:
            return ToolResult(
                success=True,
                output=f"No OONI {test} measurements for {country} in last {days_back} days.",
            )

        # Parse daily rows
        rows = []
        for row in result:
            if not isinstance(row, dict):
                continue
            ok = _safe_int(row.get("ok_count"))
            anomaly = _safe_int(row.get("anomaly_count"))
            confirmed = _safe_int(row.get("confirmed_count"))
            total = ok + anomaly + confirmed
            rate = (anomaly + confirmed) / total if total > 0 else 0.0
            rows.append(
                {
                    "date": row.get("measurement_start_day", "unknown"),
                    "ok": ok,
                    "anomaly": anomaly,
                    "confirmed": confirmed,
                    "total": total,
                    "anomaly_rate": round(rate, 4),
                }
            )

        # Trend analysis
        rates = [r["anomaly_rate"] for r in rows]
        trend = "stable"
        if len(rates) >= 7:
            first_half = sum(rates[: len(rates) // 2]) / max(len(rates) // 2, 1)
            second_half = sum(rates[len(rates) // 2 :]) / max(len(rates) - len(rates) // 2, 1)
            diff = second_half - first_half
            if diff > 0.02:
                trend = "rising"
            elif diff < -0.02:
                trend = "falling"

        avg_rate = sum(rates) / len(rates) if rates else 0
        max_rate = max(rates) if rates else 0
        max_day = next((r["date"] for r in rows if r["anomaly_rate"] == max_rate), "unknown")

        lines = [
            f"## Censorship Monitor — {country} ({test})",
            f"Source: OONI | Period: {since_date} to {until_date}\n",
            f"Days with data: {len(rows)}",
            f"Average anomaly rate: {avg_rate:.1%}",
            f"Peak anomaly rate: {max_rate:.1%} ({max_day})",
            f"Trend: {trend.upper()}",
        ]

        if trend == "rising":
            lines.append("\n⚠ RISING censorship detected — anomaly rate increasing")
        if avg_rate > 0.5:
            lines.append(f"\n🔴 HEAVY BLOCKING: {avg_rate:.0%} of {test} tests show anomalies")

        # Recent days detail
        lines.append(f"\n### Recent Daily Breakdown (last {min(limit, len(rows))} days)")
        for r in rows[-limit:]:
            marker = " ⚠" if r["anomaly_rate"] > 0.1 else ""
            lines.append(
                f"  {r['date']}: {r['anomaly_rate']:.1%} "
                f"(ok={r['ok']}, anomaly={r['anomaly']}, confirmed={r['confirmed']})"
                f"{marker}"
            )

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "censorship",
                "country": country,
                "test": test,
                "rows": rows,
                "trend": trend,
                "avg_rate": avg_rate,
                "max_rate": max_rate,
            },
        )

    # ------------------------------------------------------------------
    # Mode: signals (IODA)
    # ------------------------------------------------------------------

    def _execute_signals(self, *, country: str, hours_back: int) -> ToolResult:
        now = int(time.time())
        from_ts = now - (hours_back * 3600)

        data = self._cached_get(
            f"{IODA_BASE}/signals/raw/country/{country}",
            ttl=CACHE_IODA_SIGNALS,
            params={
                "from": str(from_ts),
                "until": str(now),
                "datasource": "gtr-norm",
            },
        )

        if not isinstance(data, dict):
            return ToolResult(
                success=False,
                output=f"Failed to fetch IODA signals for {country}.",
            )

        # Parse nested data structure: data is [[{...}, ...]]
        raw_data = data.get("data", [])
        series_list = []
        if isinstance(raw_data, list) and raw_data:
            items = raw_data[0] if isinstance(raw_data[0], list) else raw_data
            if isinstance(items, list):
                series_list = items
            elif isinstance(items, dict):
                series_list = [items]

        # Find gtr-norm series
        gtr_series = None
        for s in series_list:
            if isinstance(s, dict) and s.get("datasource") == "gtr-norm":
                gtr_series = s
                break

        if not gtr_series:
            return ToolResult(
                success=True,
                output=f"No gtr-norm signal data available for {country} in last {hours_back}h.",
            )

        values = gtr_series.get("values", [])
        step = _safe_int(gtr_series.get("step"), 1800)
        from_epoch = _safe_int(gtr_series.get("from"), from_ts)

        # Filter valid numeric values (exclude None and lists)
        valid_values = []
        timestamps = []
        for i, v in enumerate(values):
            if v is not None and isinstance(v, (int, float)):
                valid_values.append(float(v))
                timestamps.append(from_epoch + i * step)

        if not valid_values:
            return ToolResult(
                success=True,
                output=f"No valid gtr-norm data points for {country} in last {hours_back}h.",
            )

        avg_val = sum(valid_values) / len(valid_values)
        min_val = min(valid_values)
        max_val = max(valid_values)
        min_idx = valid_values.index(min_val)
        current = valid_values[-1]

        # Detect drops
        drops = []
        for i, v in enumerate(valid_values):
            if v < GTR_NORM_WARNING:
                drops.append(
                    {
                        "time": _ts_to_iso(timestamps[i]),
                        "value": v,
                        "severity": _severity_from_gtr_norm(v),
                    }
                )

        severity = _severity_from_gtr_norm(current)

        lines = [
            f"## Connectivity Signal — {country}",
            f"Source: IODA gtr-norm | Last {hours_back}h | {len(valid_values)} data points\n",
            f"Current level: {current:.4f} ({severity.upper()})",
            f"Average: {avg_val:.4f}",
            f"Min: {min_val:.4f} (at {_ts_to_iso(timestamps[min_idx])})",
            f"Max: {max_val:.4f}",
            f"Resolution: {step}s ({step // 60}min intervals)",
        ]

        if drops:
            lines.append(f"\n### Connectivity Drops ({len(drops)} below {GTR_NORM_WARNING})")
            for d in drops[:20]:
                lines.append(f"  {d['time']}: {d['value']:.4f} ({d['severity'].upper()})")

        if severity == "critical":
            lines.append(f"\n🔴 CRITICAL: {country} connectivity at {current:.1%} of normal")
        elif severity == "warning":
            lines.append(f"\n⚠ WARNING: {country} connectivity at {current:.1%} of normal")
        else:
            lines.append(f"\n✓ {country} connectivity nominal ({current:.1%} of normal)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "signals",
                "country": country,
                "current": current,
                "avg": avg_val,
                "min": min_val,
                "max": max_val,
                "severity": severity,
                "drops": drops,
                "data_points": len(valid_values),
            },
        )

    # ------------------------------------------------------------------
    # Mode: incidents (OONI)
    # ------------------------------------------------------------------

    def _execute_incidents(self, *, limit: int) -> ToolResult:
        data = self._cached_get(
            f"{OONI_BASE}/incidents/search",
            ttl=CACHE_OONI_INCIDENTS,
            params={"only_ongoing": "true", "limit": str(limit)},
        )

        if not isinstance(data, dict):
            return ToolResult(
                success=False,
                output="Failed to fetch OONI incidents.",
            )

        incidents = data.get("incidents", [])
        if not isinstance(incidents, list) or not incidents:
            return ToolResult(
                success=True,
                output="No ongoing censorship incidents reported by OONI.",
            )

        lines = [
            f"## Ongoing Censorship Incidents ({len(incidents)})",
            "Source: OONI\n",
        ]

        for inc in incidents[:limit]:
            if not isinstance(inc, dict):
                continue
            title = inc.get("title", "Unknown incident")
            countries = inc.get("CCs", [])
            cc_str = ", ".join(countries) if isinstance(countries, list) else str(countries)
            published = inc.get("published", False)
            start = inc.get("start_time", inc.get("create_time", "unknown"))
            lines.append(f"  [{cc_str}] {title}")
            if isinstance(start, str) and start != "unknown":
                lines.append(f"       Started: {start[:10]}")

        # Country frequency
        all_ccs: list[str] = []
        for inc in incidents:
            if isinstance(inc, dict):
                ccs = inc.get("CCs", [])
                if isinstance(ccs, list):
                    all_ccs.extend(ccs)
        country_frequency: dict[str, int] = {}
        if all_ccs:
            from collections import Counter

            freq = Counter(all_ccs).most_common(10)
            country_frequency = dict(freq)
            lines.append("\n### Most Affected Countries")
            for cc, count in freq:
                lines.append(f"  {cc}: {count} ongoing incidents")

        structured_incidents = []
        for inc in incidents[:limit]:
            if not isinstance(inc, dict):
                continue
            structured_incidents.append(
                {
                    "title": inc.get("title", "Unknown incident"),
                    "countries": (inc.get("CCs", []) if isinstance(inc.get("CCs"), list) else []),
                    "start": inc.get("start_time", inc.get("create_time", "unknown")),
                }
            )

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "incidents",
                "incidents": structured_incidents,
                "country_frequency": country_frequency,
            },
        )
