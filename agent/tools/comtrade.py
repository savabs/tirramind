"""
Tool: UN Comtrade — Global Bilateral Trade Flow Monitor

API:  https://comtradeapi.un.org/public/v1/preview/  (free, no auth)

Every bilateral trade flow between every country, by HS commodity code.
When China rare earth exports to Japan drop 40% → supply chain crisis.
When Russian wheat exports stop → food price spike.
When semiconductor imports surge → stockpiling ahead of sanctions.

The free preview endpoint returns up to 10 records per request — enough for
directional signal, not for deep analytics.  If TIRRA_UN_COMTRADE_KEY is
set, the premium endpoint is used for higher limits.

Modes
-----
flows       Bilateral trade flows between reporter and partner countries.
            Filter by HS commodity code, flow direction (export/import).

commodity   Search trade data by HS commodity code across reporters.

partners    Top trading partners for a reporter country, by trade value.

Signal theory:
  - Sudden drop in bilateral exports = sanctions, trade war, supply disruption
  - Mirror trade asymmetry (A says exported X to B, B says imported Y) = smuggling
  - Surge in strategic commodity imports = stockpiling ahead of crisis
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PUBLIC_BASE = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
_PREMIUM_BASE = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
_UA = "TirraMind/0.1 (comtrade-tool)"
_TIMEOUT = 25
_CACHE_TTL = 7200  # 2 hours — trade data is slow-moving
_MAX_LIMIT = 100

VALID_MODES = frozenset({"flows", "commodity", "partners"})

# Key M49 country codes (ISO-3166 numeric, which Comtrade uses)
M49_CODES: dict[str, int] = {
    "USA": 842, "CHN": 156, "JPN": 392, "DEU": 276, "GBR": 826,
    "FRA": 250, "IND": 356, "KOR": 410, "CAN": 124, "AUS": 36,
    "BRA": 76, "RUS": 643, "MEX": 484, "IDN": 360, "SAU": 682,
    "TWN": 158, "NLD": 528, "CHE": 756, "TUR": 792, "SGP": 702,
    "ARE": 784, "THA": 764, "VNM": 704, "MYS": 458, "PHL": 608,
    "ZAF": 710, "NGA": 566, "EGY": 818, "ISR": 376, "NOR": 578,
    "SWE": 752, "POL": 616, "ITA": 380, "ESP": 724,
}

# Reverse lookup
_M49_TO_ISO: dict[int, str] = {v: k for k, v in M49_CODES.items()}

# Strategic HS commodity codes
STRATEGIC_COMMODITIES: dict[str, str] = {
    "2709": "Crude petroleum",
    "2711": "Natural gas",
    "2701": "Coal",
    "1001": "Wheat",
    "1005": "Maize (corn)",
    "1006": "Rice",
    "1201": "Soybeans",
    "2601": "Iron ore",
    "2603": "Copper ore",
    "2844": "Radioactive elements (uranium)",
    "8541": "Semiconductor devices",
    "8542": "Integrated circuits",
    "2846": "Rare earth compounds",
    "7108": "Gold",
    "2710": "Petroleum products (refined)",
    "3004": "Medicaments (packaged)",
    "8703": "Motor vehicles",
    "8802": "Aircraft",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_country(code_or_name: str) -> int | None:
    """Resolve a country ISO code or M49 number to M49 code."""
    code_or_name = code_or_name.strip().upper()
    if not code_or_name:
        return None
    # Try as ISO-3
    if code_or_name in M49_CODES:
        return M49_CODES[code_or_name]
    # Try as M49 number
    try:
        m49 = int(code_or_name)
        if m49 in _M49_TO_ISO or m49 == 0:
            return m49
    except ValueError:
        pass
    # Try as partial country name match in our known codes
    for iso, m49 in M49_CODES.items():
        if code_or_name in iso:
            return m49
    return None


def _fetch_json(
    url: str, client: httpx.Client, **params: Any
) -> dict | None:
    """Fetch URL and parse as JSON.  Returns dict or None."""
    try:
        r = client.get(url, params=params)
        if r.status_code != 200:
            log.warning("HTTP %d from %s", r.status_code, url)
            return None
        return r.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Error fetching %s: %s", url, exc)
        return None


def _parse_trade_records(data: dict) -> list[dict[str, Any]]:
    """Parse Comtrade response into clean trade records."""
    records = []
    for item in data.get("data", []):
        records.append({
            "period": item.get("period", ""),
            "reporter": item.get("reporterDesc", "Unknown"),
            "reporter_code": item.get("reporterCode", 0),
            "partner": item.get("partnerDesc", "Unknown"),
            "partner_code": item.get("partnerCode", 0),
            "flow": item.get("flowDesc", ""),
            "flow_code": item.get("flowCode", ""),
            "commodity_code": item.get("cmdCode", ""),
            "commodity": item.get("cmdDesc", ""),
            "trade_value_usd": item.get("primaryValue", 0),
            "quantity": item.get("qty", 0),
            "quantity_unit": item.get("qtUnit", ""),
        })
    return records


def _get_api_key() -> str | None:
    """Get Comtrade API key from environment."""
    import os
    key = os.environ.get("TIRRA_UN_COMTRADE_KEY", "").strip()
    return key if key else None


def _get_current_year() -> int:
    return datetime.now(timezone.utc).year


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ComtradeTool(Tool):
    """Monitor global bilateral trade flows via UN Comtrade."""

    name = "comtrade"
    description = (
        "Search UN Comtrade for bilateral trade flows between countries by "
        "HS commodity code. Detects trade disruptions, sanctions effects, "
        "supply chain restructuring, and strategic commodity stockpiling."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": (
                    "commodity: trade data for an HS commodity code. "
                    "flows: bilateral flows between reporter and partner. "
                    "partners: top trading partners for a country."
                ),
            },
            "reporter": {
                "type": "string",
                "description": "Reporter country (ISO-3 code: USA, CHN, DEU, etc.).",
            },
            "partner": {
                "type": "string",
                "description": "Partner country (ISO-3 code, or '0' for World).",
            },
            "commodity_code": {
                "type": "string",
                "description": "HS commodity code (e.g., '8542' for integrated circuits).",
            },
            "flow": {
                "type": "string",
                "enum": ["X", "M"],
                "description": "Trade flow direction: X=export, M=import.",
            },
            "period": {
                "type": "string",
                "description": "Year (e.g., '2023') or 'recent' for latest available.",
            },
        },
        "required": ["mode"],
    }

    def __init__(self, *, cache: DataCache | None = None) -> None:
        self._cache = cache

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = (kwargs.get("mode") or "").strip().lower()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: {', '.join(sorted(VALID_MODES))}",
            )

        if mode == "flows":
            return self._flows(**kwargs)
        if mode == "commodity":
            return self._commodity(**kwargs)
        return self._partners(**kwargs)

    # ── flows mode ───────────────────────────────────────────────────

    def _flows(self, **kwargs: Any) -> ToolResult:
        reporter_raw = (kwargs.get("reporter") or "").strip()
        partner_raw = (kwargs.get("partner") or "0").strip()
        flow = (kwargs.get("flow") or "X").strip().upper()
        commodity_code = (kwargs.get("commodity_code") or "").strip()
        period = (kwargs.get("period") or "recent").strip()

        if not reporter_raw:
            return ToolResult(
                success=False,
                output="Parameter 'reporter' is required for flows mode (e.g., 'USA', 'CHN').",
            )

        reporter_m49 = _resolve_country(reporter_raw)
        if reporter_m49 is None:
            return ToolResult(
                success=False,
                output=f"Unknown reporter country '{reporter_raw}'. Use ISO-3 codes: {', '.join(sorted(M49_CODES.keys())[:15])}...",
            )

        partner_m49 = _resolve_country(partner_raw) if partner_raw != "0" else 0
        if partner_m49 is None:
            partner_m49 = 0  # Default to world

        if period == "recent":
            period = str(_get_current_year() - 1)

        if flow not in ("X", "M"):
            flow = "X"

        cache_key = f"ct_flows_{reporter_m49}_{partner_m49}_{flow}_{commodity_code}_{period}"
        if self._cache:
            cached = self._cache.get("comtrade", cache_key)
            if cached is not None:
                return self._format_flows(cached, reporter_raw, partner_raw, flow, from_cache=True)

        params: dict[str, str] = {
            "reporterCode": str(reporter_m49),
            "partnerCode": str(partner_m49),
            "flowCode": flow,
            "period": period,
        }
        if commodity_code:
            params["cmdCode"] = commodity_code

        records = self._query_comtrade(params)

        if self._cache and records is not None:
            self._cache.set("comtrade", cache_key, records, ttl=_CACHE_TTL)

        if records is None:
            return ToolResult(success=False, output="Comtrade API unavailable.")

        return self._format_flows(records, reporter_raw, partner_raw, flow)

    def _format_flows(
        self,
        records: list[dict],
        reporter: str,
        partner: str,
        flow: str,
        *,
        from_cache: bool = False,
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        flow_name = "Exports" if flow == "X" else "Imports"
        lines = [
            f"UN Comtrade: {reporter} → {partner} {flow_name}{tag}",
            f"Records: {len(records)}",
            "",
        ]
        # Sort by trade value descending
        sorted_recs = sorted(records, key=lambda r: r.get("trade_value_usd", 0) or 0, reverse=True)
        for r in sorted_recs[:15]:
            val = r.get("trade_value_usd", 0) or 0
            val_str = f"${val:,.0f}" if val else "N/A"
            lines.append(
                f"  [{r.get('period', '?')}] {r.get('commodity', 'N/A')[:50]} "
                f"({r.get('commodity_code', '?')}) — {val_str}"
            )
        if len(records) > 15:
            lines.append(f"  ... and {len(records) - 15} more")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "flows",
                "reporter": reporter,
                "partner": partner,
                "flow": flow,
                "record_count": len(records),
                "records": sorted_recs,
            },
        )

    # ── commodity mode ───────────────────────────────────────────────

    def _commodity(self, **kwargs: Any) -> ToolResult:
        commodity_code = (kwargs.get("commodity_code") or "").strip()
        if not commodity_code:
            # Return list of strategic commodities
            lines = ["Strategic HS commodity codes:", ""]
            for code, desc in sorted(STRATEGIC_COMMODITIES.items()):
                lines.append(f"  {code}: {desc}")
            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={"mode": "commodity", "commodities": STRATEGIC_COMMODITIES},
            )

        flow = (kwargs.get("flow") or "X").strip().upper()
        period = (kwargs.get("period") or "recent").strip()
        if period == "recent":
            period = str(_get_current_year() - 1)
        if flow not in ("X", "M"):
            flow = "X"

        cache_key = f"ct_commodity_{commodity_code}_{flow}_{period}"
        if self._cache:
            cached = self._cache.get("comtrade", cache_key)
            if cached is not None:
                return self._format_commodity(cached, commodity_code, flow, from_cache=True)

        params: dict[str, str] = {
            "cmdCode": commodity_code,
            "flowCode": flow,
            "period": period,
        }

        records = self._query_comtrade(params)

        if self._cache and records is not None:
            self._cache.set("comtrade", cache_key, records, ttl=_CACHE_TTL)

        if records is None:
            return ToolResult(success=False, output="Comtrade API unavailable.")

        return self._format_commodity(records, commodity_code, flow)

    def _format_commodity(
        self,
        records: list[dict],
        commodity_code: str,
        flow: str,
        *,
        from_cache: bool = False,
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        flow_name = "Exports" if flow == "X" else "Imports"
        commodity_name = STRATEGIC_COMMODITIES.get(commodity_code, commodity_code)
        lines = [
            f"UN Comtrade: {commodity_name} ({commodity_code}) — Global {flow_name}{tag}",
            f"Records: {len(records)}",
            "",
        ]
        sorted_recs = sorted(records, key=lambda r: r.get("trade_value_usd", 0) or 0, reverse=True)
        for r in sorted_recs[:15]:
            val = r.get("trade_value_usd", 0) or 0
            val_str = f"${val:,.0f}" if val else "N/A"
            lines.append(
                f"  [{r.get('period', '?')}] {r.get('reporter', '?')} → "
                f"{r.get('partner', '?')} — {val_str}"
            )
        if len(records) > 15:
            lines.append(f"  ... and {len(records) - 15} more")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "commodity",
                "commodity_code": commodity_code,
                "commodity_name": commodity_name,
                "flow": flow,
                "record_count": len(records),
                "records": sorted_recs,
            },
        )

    # ── partners mode ─────────────────────────────────────────────

    def _partners(self, **kwargs: Any) -> ToolResult:
        reporter_raw = (kwargs.get("reporter") or "").strip()
        if not reporter_raw:
            return ToolResult(
                success=False,
                output="Parameter 'reporter' is required for partners mode.",
            )

        reporter_m49 = _resolve_country(reporter_raw)
        if reporter_m49 is None:
            return ToolResult(
                success=False,
                output=f"Unknown reporter country '{reporter_raw}'.",
            )

        flow = (kwargs.get("flow") or "X").strip().upper()
        period = (kwargs.get("period") or "recent").strip()
        if period == "recent":
            period = str(_get_current_year() - 1)
        if flow not in ("X", "M"):
            flow = "X"

        cache_key = f"ct_partners_{reporter_m49}_{flow}_{period}"
        if self._cache:
            cached = self._cache.get("comtrade", cache_key)
            if cached is not None:
                return self._format_partners(cached, reporter_raw, flow, from_cache=True)

        params: dict[str, str] = {
            "reporterCode": str(reporter_m49),
            "flowCode": flow,
            "period": period,
        }

        records = self._query_comtrade(params)

        if self._cache and records is not None:
            self._cache.set("comtrade", cache_key, records, ttl=_CACHE_TTL)

        if records is None:
            return ToolResult(success=False, output="Comtrade API unavailable.")

        return self._format_partners(records, reporter_raw, flow)

    def _format_partners(
        self,
        records: list[dict],
        reporter: str,
        flow: str,
        *,
        from_cache: bool = False,
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        flow_name = "Export" if flow == "X" else "Import"
        lines = [
            f"UN Comtrade: {reporter} Top {flow_name} Partners{tag}",
            f"Records: {len(records)}",
            "",
        ]
        sorted_recs = sorted(records, key=lambda r: r.get("trade_value_usd", 0) or 0, reverse=True)
        for r in sorted_recs[:15]:
            val = r.get("trade_value_usd", 0) or 0
            val_str = f"${val:,.0f}" if val else "N/A"
            lines.append(
                f"  {r.get('partner', '?')} — {val_str} "
                f"({r.get('commodity', 'TOTAL')[:40]})"
            )
        if len(records) > 15:
            lines.append(f"  ... and {len(records) - 15} more")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "partners",
                "reporter": reporter,
                "flow": flow,
                "record_count": len(records),
                "records": sorted_recs,
            },
        )

    # ── common query helper ──────────────────────────────────────

    def _query_comtrade(self, params: dict[str, str]) -> list[dict[str, Any]] | None:
        """Query UN Comtrade, using premium API if key is available."""
        api_key = _get_api_key()
        if api_key:
            url = _PREMIUM_BASE
            params["subscription-key"] = api_key
        else:
            url = _PUBLIC_BASE

        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
            follow_redirects=True,
        ) as client:
            data = _fetch_json(url, client, **params)

        if data is None:
            return None

        return _parse_trade_records(data)
