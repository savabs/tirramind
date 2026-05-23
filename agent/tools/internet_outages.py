"""
Tool: Internet Outages Monitor — OONI + RIPE Atlas

Global internet censorship detection, network health monitoring,
and outage analysis from two premier open measurement networks.

Sources:
  OONI (Open Observatory of Network Interference) — censorship
    measurements, blocking events, anomaly detection.  Free, no auth.
  RIPE Atlas — 58K+ distributed probes globally.  Connectivity
    status by country/ASN.  Free reads, no auth.

Modes:
  censorship       — OONI: censorship measurements by country.
                     Anomaly rates, confirmed blocking, test coverage.
  network_health   — RIPE Atlas: probe connectivity by country.
                     Probe dropout = infrastructure or political problem.
  outage_detection — OONI aggregation: anomaly/failure counts vs total.
                     Spike in anomalies = outage signal.

Signal theory:
  - Anomaly rate spike = active censorship or infrastructure failure
  - Confirmed blocking events = state-directed censorship escalation
  - RIPE probe dropout (connected→disconnected) = physical infra damage
  - Multiple probes in same ASN going dark = ISP-level outage
  - Country-wide probe dropout = national shutdown (coup, disaster)
  - Anomaly rate >50 pct = critical—possible internet blackout

Market relevance:
  Internet outages → operational disruption (cloud, fintech, e-commerce),
  political instability (shutdowns precede coups/elections), supply chain
  comms breakdown, flight-to-safety, VPN/cybersecurity demand spikes.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone; UTC = timezone.utc
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

_UA = "TirraMind/0.1"
_TIMEOUT = 15
_CACHE_TTL = 3600  # 1 hour — outage data is time-sensitive

_OONI_BASE = "https://api.ooni.io/api/v1"
_RIPE_BASE = "https://atlas.ripe.net/api/v2"

VALID_MODES = {"censorship", "network_health", "outage_detection"}

VALID_TESTS = {
    "web_connectivity",
    "dns_consistency",
    "http_header_field_manipulation",
    "http_invalid_request_line",
    "tcp_connect",
    "vanilla_tor",
    "whatsapp",
    "facebook_messenger",
    "telegram",
    "signal",
    "psiphon",
    "tor",
}


class InternetOutagesTool(Tool):
    """Monitor internet censorship and outages via OONI + RIPE Atlas."""

    def __init__(
        self,
        cache: DataCache | None = None,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    @property
    def name(self) -> str:
        return "internet_outages"

    @property
    def description(self) -> str:
        return (
            "Monitor global internet censorship and outages — OONI "
            "censorship measurements, RIPE Atlas probe connectivity, "
            "and outage detection via anomaly aggregation. Detects "
            "internet shutdowns, censorship escalation, and "
            "infrastructure failures across 200+ countries."
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
                        "censorship: OONI measurements by country. "
                        "network_health: RIPE Atlas probe status. "
                        "outage_detection: OONI aggregated anomaly counts."
                    ),
                },
                "country": {
                    "type": "string",
                    "description": ("ISO 3166-1 alpha-2 country code (e.g. 'US', 'IR', 'CN', 'RU')."),
                },
                "test_name": {
                    "type": "string",
                    "description": (
                        "OONI test name for censorship/outage modes. "
                        "Options: web_connectivity, whatsapp, telegram, "
                        "signal, tor, etc. Default: web_connectivity."
                    ),
                },
                "since": {
                    "type": "string",
                    "description": ("Start date YYYY-MM-DD (default: 7 days ago)."),
                },
                "until": {
                    "type": "string",
                    "description": "End date YYYY-MM-DD (default: today).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 50, max: 200).",
                },
            },
            "required": ["mode"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = kwargs.get("mode", "")
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=(f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}"),
            )

        country = (kwargs.get("country") or "").strip().upper()
        if country and len(country) != 2:
            return ToolResult(
                success=False,
                output=(f"Invalid country code '{country}'. Must be 2-letter ISO code."),
            )

        limit = min(kwargs.get("limit") or 50, 200)

        if mode == "censorship":
            result = self._handle_censorship(country, kwargs, limit)
        elif mode == "network_health":
            result = self._handle_network_health(country, kwargs, limit)
        else:
            result = self._handle_outage_detection(country, kwargs, limit)

        if result.success and result.data:
            self._persist_entities(result.data, mode)

        return result

    # ── Mode handlers ───────────────────────────────────────

    def _handle_censorship(
        self,
        country: str,
        kwargs: dict,
        limit: int,
    ) -> ToolResult:
        test_name = (kwargs.get("test_name") or "web_connectivity").strip().lower()
        if test_name not in VALID_TESTS:
            return ToolResult(
                success=False,
                output=(f"Invalid test_name '{test_name}'. Valid: {sorted(VALID_TESTS)}"),
            )

        since, until, err = _resolve_dates(kwargs)
        if err:
            return ToolResult(success=False, output=err)

        params: dict[str, str] = {
            "test_name": test_name,
            "since": since,
            "until": until,
            "limit": str(limit),
            "order_by": "measurement_start_time",
            "order": "desc",
        }
        if country:
            params["probe_cc"] = country

        cache_key = f"ooni:censorship:{country}:{test_name}:{since}:{until}:{limit}"
        return self._fetch_ooni_measurements(params, cache_key, country)

    def _handle_network_health(
        self,
        country: str,
        kwargs: dict,
        limit: int,
    ) -> ToolResult:
        if not country:
            return ToolResult(
                success=False,
                output="'country' is required for network_health mode.",
            )

        params: dict[str, str] = {
            "country_code": country,
            "limit": str(limit),
            "sort": "id",
        }
        cache_key = f"ripe:probes:{country}:{limit}"
        return self._fetch_ripe_probes(params, cache_key, country)

    def _handle_outage_detection(
        self,
        country: str,
        kwargs: dict,
        limit: int,
    ) -> ToolResult:
        test_name = (kwargs.get("test_name") or "web_connectivity").strip().lower()
        if test_name not in VALID_TESTS:
            return ToolResult(
                success=False,
                output=(f"Invalid test_name '{test_name}'. Valid: {sorted(VALID_TESTS)}"),
            )

        since, until, err = _resolve_dates(kwargs)
        if err:
            return ToolResult(success=False, output=err)

        params: dict[str, str] = {
            "test_name": test_name,
            "since": since,
            "until": until,
        }
        if country:
            params["probe_cc"] = country

        cache_key = f"ooni:aggregation:{country}:{test_name}:{since}:{until}"
        return self._fetch_ooni_aggregation(
            params,
            cache_key,
            country,
            test_name,
        )

    # ── OONI measurements fetch ─────────────────────────────

    def _fetch_ooni_measurements(
        self,
        params: dict,
        cache_key: str,
        country: str,
    ) -> ToolResult:
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(
                    success=True,
                    output=hit["output"],
                    data=hit["data"],
                )

        try:
            with httpx.Client(
                timeout=_TIMEOUT,
                headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(
                    f"{_OONI_BASE}/measurements",
                    params=params,
                )
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output="OONI API request timed out.",
            )
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"HTTP error: {exc}")

        if resp.status_code == 429:
            return ToolResult(
                success=False,
                output="OONI API rate limit reached. Retry later.",
            )
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                output=f"OONI API returned HTTP {resp.status_code}",
            )

        try:
            body = resp.json()
        except Exception:
            return ToolResult(
                success=False,
                output="Failed to parse OONI API response.",
            )

        raw = body.get("results", [])
        records, counts = _parse_ooni_measurements(raw)
        signals = _censorship_signals(counts, country)
        summary = _format_censorship(records, signals, country)

        result_data = {
            "records": records,
            "signals": signals,
            "country": country or "ALL",
            "test_name": params.get("test_name", ""),
        }

        if self._cache:
            self._cache.set(
                cache_key,
                {"output": summary, "data": result_data},
                ttl=_CACHE_TTL,
            )

        return ToolResult(success=True, output=summary, data=result_data)

    # ── OONI aggregation fetch ──────────────────────────────

    def _fetch_ooni_aggregation(
        self,
        params: dict,
        cache_key: str,
        country: str,
        test_name: str,
    ) -> ToolResult:
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(
                    success=True,
                    output=hit["output"],
                    data=hit["data"],
                )

        try:
            with httpx.Client(
                timeout=_TIMEOUT,
                headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(
                    f"{_OONI_BASE}/aggregation",
                    params=params,
                )
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output="OONI aggregation API timed out.",
            )
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"HTTP error: {exc}")

        if resp.status_code == 429:
            return ToolResult(
                success=False,
                output="OONI API rate limit reached.",
            )
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                output=f"OONI API returned HTTP {resp.status_code}",
            )

        try:
            body = resp.json()
        except Exception:
            return ToolResult(
                success=False,
                output="Failed to parse OONI aggregation response.",
            )

        agg = _extract_aggregation(body)
        signals = _aggregation_signals(agg)

        cc = country or "ALL"
        parts = [f"Internet Outage Detection — {cc} — {test_name}"]
        parts.append(f"Period: {params.get('since', '?')} to {params.get('until', '?')}")
        total = agg.get("total", 0)
        parts.append(f"Total measurements: {total:,}")
        parts.append(
            f"OK: {agg.get('ok', 0):,} | "
            f"Anomaly: {agg.get('anomaly', 0):,} | "
            f"Confirmed: {agg.get('confirmed', 0):,} | "
            f"Failure: {agg.get('failure', 0):,}"
        )
        parts.append(
            f"Anomaly rate: {signals.get('anomaly_rate_pct', 0):.1f}% | "
            f"Failure rate: {signals.get('failure_rate_pct', 0):.1f}%"
        )
        if "alert" in signals:
            parts.append(f"⚠ {signals['alert']}")

        summary = "\n".join(parts)

        result_data = {
            "aggregation": agg,
            "signals": signals,
            "country": cc,
            "test_name": test_name,
        }

        if self._cache:
            self._cache.set(
                cache_key,
                {"output": summary, "data": result_data},
                ttl=_CACHE_TTL,
            )

        return ToolResult(success=True, output=summary, data=result_data)

    # ── RIPE Atlas fetch ────────────────────────────────────

    def _fetch_ripe_probes(
        self,
        params: dict,
        cache_key: str,
        country: str,
    ) -> ToolResult:
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(
                    success=True,
                    output=hit["output"],
                    data=hit["data"],
                )

        try:
            with httpx.Client(
                timeout=_TIMEOUT,
                headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(f"{_RIPE_BASE}/probes/", params=params)
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output="RIPE Atlas API request timed out.",
            )
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"HTTP error: {exc}")

        if resp.status_code == 429:
            return ToolResult(
                success=False,
                output="RIPE Atlas API rate limit reached.",
            )
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                output=f"RIPE Atlas API returned HTTP {resp.status_code}",
            )

        try:
            body = resp.json()
        except Exception:
            return ToolResult(
                success=False,
                output="Failed to parse RIPE Atlas response.",
            )

        probes_raw = body.get("results", [])
        total_count = body.get("count", len(probes_raw))
        records, status_counts, asns = _parse_ripe_probes(
            probes_raw,
            country,
        )
        signals = _network_health_signals(
            status_counts,
            asns,
            total_count,
            country,
        )
        summary = _format_network_health(
            records,
            signals,
            total_count,
            country,
            asns,
        )

        result_data = {
            "records": records,
            "signals": signals,
            "country": country,
        }

        if self._cache:
            self._cache.set(
                cache_key,
                {"output": summary, "data": result_data},
                ttl=_CACHE_TTL,
            )

        return ToolResult(success=True, output=summary, data=result_data)

    # ── L2 entity persistence (Phase 31) ──────────────────────

    def _persist_entities(
        self,
        data: dict[str, Any],
        mode: str,
    ) -> dict[str, int]:
        if self._store is None or _entity_id_from_key is None:
            return {"internet_disruption_obs": 0}
        try:
            return self._persist_entities_inner(data, mode)
        except Exception:
            log.exception("Internet outages entity persistence failed (non-fatal)")
            return {"internet_disruption_obs": 0}

    def _persist_entities_inner(
        self,
        data: dict[str, Any],
        mode: str,
    ) -> dict[str, int]:
        assert self._store is not None  # noqa: S101 -- guarded
        assert _entity_id_from_key is not None  # noqa: S101

        country = str(data.get("country", "")).upper()
        if country in {"", "ALL"} or len(country) != 2:
            return {"internet_disruption_obs": 0}

        signals = data.get("signals", {})
        country_eid = _entity_id_from_key("country", country)
        self._store.register_entity("country", country, country_eid)
        self._store.store_entity_observation(
            entity_id=country_eid,
            source_tool="internet_outages",
            observed_at=time.time(),
            observation_type="internet_disruption",
            value={
                "mode": mode,
                "test_name": data.get("test_name"),
                "anomaly_rate_pct": signals.get("anomaly_rate_pct"),
                "disconnect_rate_pct": signals.get("disconnect_rate_pct"),
                "confirmed_count": signals.get("confirmed_count"),
                "failure_count": signals.get("failure_count"),
                "alert": signals.get("alert"),
            },
            depth_level=2,
        )
        log.info(
            "Internet outages L2: 1 internet_disruption obs persisted for %s",
            country,
        )
        return {"internet_disruption_obs": 1}


# ── Helpers (module-level for testability) ──────────────────────


def _resolve_dates(kwargs: dict) -> tuple[str, str, str | None]:
    """Parse since/until dates with defaults. Returns (since, until, error)."""
    now = datetime.now(UTC)
    since = kwargs.get("since") or (now - timedelta(days=7)).strftime("%Y-%m-%d")
    until = kwargs.get("until") or now.strftime("%Y-%m-%d")

    for label, val in [("since", since), ("until", until)]:
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except ValueError:
            return "", "", f"Invalid {label} date '{val}'. Use YYYY-MM-DD."

    return since, until, None


def _parse_ooni_measurements(
    raw: list[dict],
) -> tuple[list[dict], dict[str, int]]:
    """Parse OONI measurement results into records + counts."""
    records = []
    counts = {"anomaly": 0, "confirmed": 0, "failure": 0, "total": 0}

    for r in raw:
        is_anomaly = bool(r.get("anomaly"))
        is_confirmed = bool(r.get("confirmed"))
        is_failure = bool(r.get("failure"))

        counts["total"] += 1
        if is_anomaly:
            counts["anomaly"] += 1
        if is_confirmed:
            counts["confirmed"] += 1
        if is_failure:
            counts["failure"] += 1

        records.append(
            {
                "measurement_uid": r.get("measurement_uid", ""),
                "probe_cc": r.get("probe_cc", ""),
                "probe_asn": r.get("probe_asn", ""),
                "test_name": r.get("test_name", ""),
                "input": r.get("input", ""),
                "anomaly": is_anomaly,
                "confirmed": is_confirmed,
                "failure": is_failure,
                "measurement_start_time": r.get("measurement_start_time", ""),
                "scores": r.get("scores", {}),
            }
        )

    return records, counts


def _censorship_signals(
    counts: dict[str, int],
    country: str,
) -> dict[str, Any]:
    """Derive censorship alert signals from measurement counts."""
    total = counts["total"]
    anomaly_rate = (counts["anomaly"] / total * 100) if total else 0

    signals: dict[str, Any] = {
        "total_measurements": total,
        "anomaly_count": counts["anomaly"],
        "confirmed_count": counts["confirmed"],
        "failure_count": counts["failure"],
        "anomaly_rate_pct": round(anomaly_rate, 2),
    }

    if anomaly_rate > 50:
        signals["alert"] = "CRITICAL: >50% anomaly rate — possible internet shutdown"
    elif anomaly_rate > 20:
        signals["alert"] = "WARNING: >20% anomaly rate — significant censorship/outage"
    elif counts["confirmed"] > 0:
        signals["alert"] = f"NOTICE: {counts['confirmed']} confirmed blocking events"

    return signals


def _format_censorship(
    records: list[dict],
    signals: dict,
    country: str,
) -> str:
    """Format censorship measurement summary."""
    cc = country or "ALL"
    parts = [f"Internet Censorship — OONI — {cc}"]
    parts.append(f"Measurements: {signals['total_measurements']}")
    parts.append(
        f"Anomaly: {signals['anomaly_count']} "
        f"({signals['anomaly_rate_pct']}%) | "
        f"Confirmed: {signals['confirmed_count']} | "
        f"Failure: {signals['failure_count']}"
    )
    if "alert" in signals:
        parts.append(f"⚠ {signals['alert']}")

    blocked = [r for r in records if r.get("confirmed")]
    if blocked:
        parts.append("\nConfirmed blocked:")
        for b in blocked[:10]:
            parts.append(f"  {b.get('input', 'N/A')} (ASN: {b.get('probe_asn', '?')})")

    anomalies = [r for r in records if r.get("anomaly") and not r.get("confirmed")]
    if anomalies:
        parts.append(f"\nAnomaly measurements: {len(anomalies)}")
        for a in anomalies[:5]:
            parts.append(
                f"  {(a.get('input') or 'N/A')[:60]} — "
                f"ASN: {a.get('probe_asn', '?')} "
                f"({a.get('measurement_start_time', '?')})"
            )

    return "\n".join(parts)


def _extract_aggregation(body: dict) -> dict[str, int]:
    """Extract aggregation counts from OONI response."""
    result = body.get("result", body)
    if isinstance(result, list):
        agg = result[0] if result else {}
    else:
        agg = result

    anomaly = agg.get("anomaly_count", 0)
    confirmed = agg.get("confirmed_count", 0)
    failure = agg.get("failure_count", 0)
    ok = agg.get("ok_count", 0)
    total = agg.get(
        "measurement_count",
        anomaly + confirmed + failure + ok,
    )
    return {
        "anomaly": anomaly,
        "confirmed": confirmed,
        "failure": failure,
        "ok": ok,
        "total": total,
    }


def _aggregation_signals(agg: dict[str, int]) -> dict[str, Any]:
    """Derive alert signals from aggregated counts."""
    total = agg.get("total", 0)
    anomaly_rate = (agg["anomaly"] / total * 100) if total else 0
    failure_rate = (agg["failure"] / total * 100) if total else 0

    signals: dict[str, Any] = {
        "anomaly_count": agg["anomaly"],
        "confirmed_count": agg["confirmed"],
        "failure_count": agg["failure"],
        "ok_count": agg["ok"],
        "total_measurements": total,
        "anomaly_rate_pct": round(anomaly_rate, 2),
        "failure_rate_pct": round(failure_rate, 2),
    }

    if anomaly_rate > 50:
        signals["alert"] = "CRITICAL: >50% anomaly rate"
    elif anomaly_rate > 20:
        signals["alert"] = "WARNING: >20% anomaly rate"
    elif failure_rate > 30:
        signals["alert"] = "WARNING: >30% failure rate — possible infrastructure issue"

    return signals


def _parse_ripe_probes(
    probes: list[dict],
    country: str,
) -> tuple[list[dict], dict[str, int], dict[int, int]]:
    """Parse RIPE Atlas probe data into records, status counts, ASN counts."""
    records = []
    status_counts: dict[str, int] = {
        "Connected": 0,
        "Disconnected": 0,
        "Abandoned": 0,
        "Never Connected": 0,
    }
    asns: dict[int, int] = {}

    for p in probes:
        status_info = p.get("status", {})
        if isinstance(status_info, dict):
            status_name = status_info.get("name", "Unknown")
            status_since = status_info.get("since", "")
        else:
            status_name = str(status_info)
            status_since = ""

        if status_name in status_counts:
            status_counts[status_name] += 1

        asn = p.get("asn_v4", 0)
        if asn:
            asns[asn] = asns.get(asn, 0) + 1

        tags = []
        for t in p.get("tags") or []:
            if isinstance(t, dict):
                tags.append(t.get("name", ""))

        records.append(
            {
                "probe_id": p.get("id"),
                "asn_v4": asn,
                "country": p.get("country_code", country),
                "status": status_name,
                "status_since": status_since,
                "address_v4": p.get("address_v4", ""),
                "is_anchor": p.get("is_anchor", False),
                "tags": tags,
            }
        )

    return records, status_counts, asns


def _network_health_signals(
    status_counts: dict[str, int],
    asns: dict[int, int],
    total_count: int,
    country: str,
) -> dict[str, Any]:
    """Derive health signals from probe status distribution."""
    connected = status_counts.get("Connected", 0)
    disconnected = status_counts.get("Disconnected", 0)
    abandoned = status_counts.get("Abandoned", 0)
    never_connected = status_counts.get("Never Connected", 0)

    active = connected + disconnected
    disconnect_rate = (disconnected / active * 100) if active else 0

    signals: dict[str, Any] = {
        "total_probes": sum(status_counts.values()),
        "total_in_country": total_count,
        "connected": connected,
        "disconnected": disconnected,
        "abandoned": abandoned,
        "never_connected": never_connected,
        "disconnect_rate_pct": round(disconnect_rate, 2),
        "unique_asns": len(asns),
    }

    if disconnect_rate > 50:
        signals["alert"] = "CRITICAL: >50% probes disconnected — possible national outage"
    elif disconnect_rate > 20:
        signals["alert"] = "WARNING: >20% probes disconnected — significant connectivity issues"
    elif disconnected > 10:
        signals["alert"] = f"NOTICE: {disconnected} probes disconnected in {country}"

    top_asns = sorted(asns.items(), key=lambda x: x[1], reverse=True)[:5]
    signals["top_asns"] = [{"asn": a, "probe_count": c} for a, c in top_asns]

    return signals


def _format_network_health(
    records: list[dict],
    signals: dict,
    total_count: int,
    country: str,
    asns: dict[int, int],
) -> str:
    """Format network health summary."""
    parts = [f"Network Health — RIPE Atlas — {country}"]
    parts.append(f"Total probes in country: {total_count}")
    parts.append(f"Sample: {len(records)} probes")
    parts.append(
        f"Connected: {signals['connected']} | "
        f"Disconnected: {signals['disconnected']} | "
        f"Abandoned: {signals['abandoned']}"
    )
    parts.append(f"Disconnect rate: {signals['disconnect_rate_pct']:.1f}%")
    parts.append(f"Unique ASNs: {signals['unique_asns']}")

    if "alert" in signals:
        parts.append(f"⚠ {signals['alert']}")

    top = sorted(asns.items(), key=lambda x: x[1], reverse=True)[:5]
    if top:
        parts.append("\nTop ASNs by probe count:")
        for a, c in top:
            parts.append(f"  AS{a}: {c} probes")

    return "\n".join(parts)
