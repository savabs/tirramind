"""
Tool: Sovereign Debt — Government Bond Yields across US, EU, Japan, UK

US Treasury:  https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml
ECB IRS:      https://data-api.ecb.europa.eu/service/data/IRS/
Japan MOF:    https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/
UK DMO:       https://www.dmo.gov.uk/data/XmlDataReport

Bond markets are smarter than equity markets.  They price fiscal stress
months/years before equity investors notice.  Detroit munis screamed 18 months
before bankruptcy.  Greek 10Y yields blew out months before the equity crash.

Modes:
  us_yields  — US Treasury daily yield curve (1mo–30yr) for a given month.
  eu_yields  — ECB per-country government bond yields (monthly, 21 countries).
  jp_yields  — Japan MOF daily JGB yields (1Y–40Y).
  uk_gilts   — UK DMO gilt issuance history (auction yields + volumes).
  spreads    — Cross-country spread computation (IT-DE, GR-DE, etc.) as
               fiscal-stress signals, plus US 2s10s curve.

Signal theory:
  - Yield curve flattening/inversion → recession (2s10s, 3m10y)
  - IT-DE, GR-DE spread widening → eurozone fiscal stress
  - JGB breakout above BOJ policy band → global bond repricing
  - Simultaneous multi-country widening → systemic risk-off
  - UK gilt auction tail (yield above WI) → fiscal confidence breakdown
"""

from __future__ import annotations

import csv
import io
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

# --- Constants ---

_US_TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xml?data=daily_treasury_yield_curve"
    "&field_tdr_date_value_month={yyyymm}"
)
_ECB_IRS_URL = (
    "https://data-api.ecb.europa.eu/service/data/IRS/"
    "M.{cc}.L.L40.CI.0000.EUR.N.Z"
    "?startPeriod={start}&format=csvdata"
)
_JP_MOF_CURRENT_URL = (
    "https://www.mof.go.jp/english/policy/jgbs/reference/" "interest_rate/jgbcme.csv"
)
_UK_DMO_URL = "https://www.dmo.gov.uk/data/XmlDataReport?reportCode=D2.1E"

_UA = "TirraMind/0.1"
_TIMEOUT = 20
_CACHE_TTL_US = 1800  # 30 min — daily data, updates once per day
_CACHE_TTL_EU = 3600  # 1 hr — monthly data
_CACHE_TTL_JP = 1800  # 30 min — daily data
_CACHE_TTL_UK = 7200  # 2 hr — auction data, infrequent
_CACHE_TTL_SPREADS = 3600  # 1 hr — derived from EU data

VALID_MODES = {"us_yields", "eu_yields", "jp_yields", "uk_gilts", "spreads"}

# US Treasury XML maturity fields → short labels
_US_MATURITY_MAP = {
    "BC_1MONTH": "1m",
    "BC_2MONTH": "2m",
    "BC_3MONTH": "3m",
    "BC_4MONTH": "4m",
    "BC_6MONTH": "6m",
    "BC_1YEAR": "1y",
    "BC_2YEAR": "2y",
    "BC_3YEAR": "3y",
    "BC_5YEAR": "5y",
    "BC_7YEAR": "7y",
    "BC_10YEAR": "10y",
    "BC_20YEAR": "20y",
    "BC_30YEAR": "30y",
}

# ECB country codes that have working IRS data (actual sovereign issuers)
_EU_COUNTRIES_DEFAULT = [
    "DE",
    "FR",
    "IT",
    "ES",
    "GR",
    "PT",
    "NL",
    "BE",
    "AT",
    "IE",
    "FI",
]
_EU_COUNTRIES_ALL = {
    "AT",
    "BE",
    "BG",
    "CY",
    "DE",
    "EE",
    "ES",
    "FI",
    "FR",
    "GR",
    "HR",
    "IE",
    "IT",
    "LT",
    "LU",
    "LV",
    "MT",
    "NL",
    "PT",
    "SI",
    "SK",
}

# Japan MOF CSV maturity column headers → short labels
_JP_MATURITY_MAP = {
    "1Y": "1y",
    "2Y": "2y",
    "3Y": "3y",
    "4Y": "4y",
    "5Y": "5y",
    "6Y": "6y",
    "7Y": "7y",
    "8Y": "8y",
    "9Y": "9y",
    "10Y": "10y",
    "15Y": "15y",
    "20Y": "20y",
    "25Y": "25y",
    "30Y": "30y",
    "40Y": "40y",
}

_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


# --- Parsing helpers ---


def _safe_float(val: str | None) -> float | None:
    """Parse a numeric string; return None for missing/invalid values."""
    if not val or not val.strip() or val.strip() == "-":
        return None
    try:
        return float(val.strip())
    except (ValueError, TypeError):
        return None


def _parse_us_treasury_xml(text: str) -> list[dict[str, Any]]:
    """Parse US Treasury Atom XML feed into yield records."""
    records: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        log.warning("Failed to parse US Treasury XML: %s", exc)
        return records

    # Atom namespace + OData metadata/dataservices namespaces
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
        "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
    }

    for entry in root.findall(".//atom:entry", ns):
        props = entry.find(".//m:properties", ns)
        if props is None:
            continue

        date_el = props.find("d:NEW_DATE", ns)
        if date_el is None or not date_el.text:
            continue

        # Date format: "2026-03-27T00:00:00"
        date_str = date_el.text[:10]

        yields: dict[str, float | None] = {}
        for xml_field, label in _US_MATURITY_MAP.items():
            el = props.find(f"d:{xml_field}", ns)
            yields[label] = _safe_float(el.text if el is not None else None)

        # Compute curve spreads
        y2 = yields.get("2y")
        y10 = yields.get("10y")
        y3m = yields.get("3m")
        curve_2s10s = round(y10 - y2, 4) if y10 is not None and y2 is not None else None
        curve_3m10y = (
            round(y10 - y3m, 4) if y10 is not None and y3m is not None else None
        )

        records.append(
            {
                "date": date_str,
                "yields": {k: v for k, v in yields.items() if v is not None},
                "curve_2s10s": curve_2s10s,
                "curve_3m10y": curve_3m10y,
            }
        )

    return records


def _parse_ecb_csv(text: str) -> list[dict[str, Any]]:
    """Parse ECB SDMX CSV into [{period, yield_pct}] records."""
    records: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        period = row.get("TIME_PERIOD", "").strip()
        value = _safe_float(row.get("OBS_VALUE"))
        if period and value is not None:
            records.append({"period": period, "yield_pct": value})
    return records


def _parse_jp_mof_csv(text: str) -> list[dict[str, Any]]:
    """Parse Japan MOF JGB yield CSV into records."""
    records: list[dict[str, Any]] = []
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return records

    # Row 0 = title (e.g., "Interest Rate (March 2026),,,,,,,,,,,,,,,(Unit : %)")
    # Row 1 = headers: Date,1Y,2Y,...,40Y
    headers = [h.strip() for h in lines[1].split(",")]

    for line in lines[2:]:
        parts = line.split(",")
        if not parts or not parts[0].strip():
            continue

        date_raw = parts[0].strip()
        # Validate date format: YYYY/M/D or YYYY/MM/DD
        if not re.match(r"^\d{4}/\d{1,2}/\d{1,2}$", date_raw):
            continue

        # Normalize to YYYY-MM-DD
        try:
            dt = datetime.strptime(date_raw, "%Y/%m/%d")
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

        yields: dict[str, float | None] = {}
        for i, header in enumerate(headers[1:], start=1):
            label = _JP_MATURITY_MAP.get(header)
            if label and i < len(parts):
                yields[label] = _safe_float(parts[i])

        # Only include if we got at least some yields
        if any(v is not None for v in yields.values()):
            records.append(
                {
                    "date": date_str,
                    "yields": {k: v for k, v in yields.items() if v is not None},
                }
            )

    return records


def _parse_uk_dmo_xml(text: str) -> list[dict[str, Any]]:
    """Parse UK DMO Gilt Issuance History XML into records."""
    records: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        log.warning("Failed to parse UK DMO XML: %s", exc)
        return records

    for elem in root.iter("View_Gilt_Issuance_History"):
        name = elem.get("INSTRUMENT_NAME", "").strip()
        isin = elem.get("ISIN_CODE", "").strip()
        date_raw = elem.get("ACTUAL_DATE", "")
        issue_type = elem.get("ISSUANCE_TYPE", "").strip()
        nominal = _safe_float(elem.get("NOMINAL_ISSUED"))
        price = _safe_float(elem.get("ISSUE_CLEAN_PRICE"))
        yield_val = _safe_float(elem.get("ISSUE_YIELD"))

        if not date_raw:
            continue

        # Date format: "2013-08-21T00:00:00"
        date_str = date_raw[:10]

        records.append(
            {
                "date": date_str,
                "instrument": name,
                "isin": isin,
                "issuance_type": issue_type,
                "nominal_issued_m": round(nominal, 2) if nominal is not None else None,
                "clean_price": price,
                "yield_pct": yield_val,
            }
        )

    return records


# --- Tool class ---


class SovereignDebtTool(Tool):
    """Query government bond yield data across US, EU, Japan, and UK."""

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    @property
    def name(self) -> str:
        return "sovereign_debt"

    @property
    def description(self) -> str:
        return (
            "Fetch government bond yield data: US Treasury curve (daily, 1mo–30yr), "
            "European per-country yields (monthly, 21 EU countries), Japan JGB curve "
            "(daily, 1Y–40Y), UK gilt auction data, and cross-country spread computation. "
            "Bond markets price fiscal stress before equity markets react."
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
                        "Query mode: us_yields (daily Treasury curve), "
                        "eu_yields (per-country 10Y yields), "
                        "jp_yields (JGB daily curve), "
                        "uk_gilts (gilt auction data), "
                        "spreads (cross-country spread computation)."
                    ),
                },
                "month": {
                    "type": "string",
                    "description": (
                        "Target month as YYYY-MM (e.g. '2026-03'). "
                        "Defaults to current month. Used by us_yields, eu_yields, jp_yields."
                    ),
                },
                "countries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "EU country codes for eu_yields/spreads mode "
                        "(e.g. ['DE','IT','GR']). "
                        "Defaults to major issuers: DE,FR,IT,ES,GR,PT,NL,BE,AT,IE,FI."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max records to return (default 20, max 100). For uk_gilts, limits to most recent auctions.",
                },
            },
            "required": ["mode"],
        }

    # --- Fetch methods ---

    def _fetch_us_yields(self, month: str) -> ToolResult:
        """Fetch US Treasury daily yield curve for a given month."""
        yyyymm = month.replace("-", "")

        cache_key = f"sovereign_us_{yyyymm}"
        if self._cache:
            cached = self._cache.get("sovereign_debt", {"key": cache_key})
            if cached is not None:
                return ToolResult(
                    success=True,
                    output=f"US Treasury yields for {month} (cached)",
                    data=cached,
                )

        url = _US_TREASURY_URL.format(yyyymm=yyyymm)
        try:
            r = httpx.get(
                url,
                headers={"User-Agent": _UA},
                timeout=_TIMEOUT,
                follow_redirects=True,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                success=False,
                output=f"US Treasury API error: HTTP {exc.response.status_code}",
            )
        except httpx.RequestError as exc:
            return ToolResult(
                success=False, output=f"US Treasury request failed: {exc}"
            )

        records = _parse_us_treasury_xml(r.text)
        if not records:
            return ToolResult(
                success=False, output=f"No US Treasury data found for {month}"
            )

        result = {"month": month, "entries": len(records), "records": records}
        if self._cache:
            self._cache.put("sovereign_debt", {"key": cache_key}, result)

        latest = records[-1]
        curve_info = ""
        if latest.get("curve_2s10s") is not None:
            curve_info = f", 2s10s={latest['curve_2s10s']}%"

        return ToolResult(
            success=True,
            output=f"US Treasury yields for {month}: {len(records)} days{curve_info}",
            data=result,
        )

    def _fetch_eu_yields(self, countries: list[str], month: str) -> ToolResult:
        """Fetch ECB per-country government bond yields."""
        # Validate country codes
        valid_countries = [
            c.upper() for c in countries if c.upper() in _EU_COUNTRIES_ALL
        ]
        if not valid_countries:
            return ToolResult(
                success=False,
                output=f"No valid EU country codes. Valid: {sorted(_EU_COUNTRIES_ALL)}",
            )

        cache_key = f"sovereign_eu_{'_'.join(sorted(valid_countries))}_{month}"
        if self._cache:
            cached = self._cache.get("sovereign_debt", {"key": cache_key})
            if cached is not None:
                return ToolResult(
                    success=True,
                    output=f"EU yields for {', '.join(valid_countries)} (cached)",
                    data=cached,
                )

        # Start period: 6 months before requested month for trend context
        try:
            dt = datetime.strptime(month + "-01", "%Y-%m-%d")
            start_year = dt.year if dt.month > 6 else dt.year - 1
            start_month = dt.month - 6 if dt.month > 6 else dt.month + 6
            start_period = f"{start_year}-{start_month:02d}"
        except ValueError:
            start_period = month

        country_data: dict[str, Any] = {}
        errors: list[str] = []

        for cc in valid_countries:
            url = _ECB_IRS_URL.format(cc=cc, start=start_period)
            try:
                r = httpx.get(
                    url,
                    headers={"User-Agent": _UA, "Accept": "text/csv"},
                    timeout=_TIMEOUT,
                    follow_redirects=True,
                )
                if r.status_code == 404:
                    errors.append(f"{cc}: no data available")
                    continue
                r.raise_for_status()
            except httpx.RequestError as exc:
                errors.append(f"{cc}: request failed ({exc})")
                continue
            except httpx.HTTPStatusError as exc:
                errors.append(f"{cc}: HTTP {exc.response.status_code}")
                continue

            records = _parse_ecb_csv(r.text)
            if records:
                country_data[cc] = records

        if not country_data:
            return ToolResult(
                success=False,
                output=f"No EU yield data retrieved. Errors: {'; '.join(errors)}",
            )

        result = {
            "countries": country_data,
            "errors": errors if errors else None,
            "period_start": start_period,
        }

        if self._cache:
            self._cache.put("sovereign_debt", {"key": cache_key}, result)

        country_summary = ", ".join(
            f"{cc}={recs[-1]['yield_pct']:.2f}%"
            for cc, recs in sorted(country_data.items())
            if recs
        )
        return ToolResult(
            success=True,
            output=f"EU government bond yields ({len(country_data)} countries): {country_summary}",
            data=result,
        )

    def _fetch_jp_yields(self) -> ToolResult:
        """Fetch Japan MOF current-month JGB yields."""
        cache_key = "sovereign_jp_current"
        if self._cache:
            cached = self._cache.get("sovereign_debt", {"key": cache_key})
            if cached is not None:
                return ToolResult(
                    success=True, output="Japan JGB yields (cached)", data=cached
                )

        try:
            r = httpx.get(
                _JP_MOF_CURRENT_URL,
                headers={"User-Agent": _UA},
                timeout=_TIMEOUT,
                follow_redirects=True,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                success=False,
                output=f"Japan MOF API error: HTTP {exc.response.status_code}",
            )
        except httpx.RequestError as exc:
            return ToolResult(success=False, output=f"Japan MOF request failed: {exc}")

        records = _parse_jp_mof_csv(r.text)
        if not records:
            return ToolResult(success=False, output="No Japan JGB data found")

        result = {"entries": len(records), "records": records}
        if self._cache:
            self._cache.put("sovereign_debt", {"key": cache_key}, result)

        latest = records[-1]
        y10 = latest["yields"].get("10y")
        y10_str = f", 10Y={y10}%" if y10 is not None else ""
        return ToolResult(
            success=True,
            output=f"Japan JGB yields: {len(records)} days{y10_str}",
            data=result,
        )

    def _fetch_uk_gilts(self, limit: int) -> ToolResult:
        """Fetch UK DMO gilt issuance history."""
        cache_key = "sovereign_uk_gilts"
        if self._cache:
            cached = self._cache.get("sovereign_debt", {"key": cache_key})
            if cached is not None:
                # Apply limit to cached data
                limited = cached.copy()
                limited["records"] = limited["records"][:limit]
                return ToolResult(
                    success=True, output="UK gilt auctions (cached)", data=limited
                )

        try:
            r = httpx.get(
                _UK_DMO_URL,
                headers={"User-Agent": _UA},
                timeout=_TIMEOUT,
                follow_redirects=True,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                success=False,
                output=f"UK DMO API error: HTTP {exc.response.status_code}",
            )
        except httpx.RequestError as exc:
            return ToolResult(success=False, output=f"UK DMO request failed: {exc}")

        records = _parse_uk_dmo_xml(r.text)
        if not records:
            return ToolResult(success=False, output="No UK gilt issuance data found")

        # Sort by date descending (most recent first)
        records.sort(key=lambda x: x["date"], reverse=True)

        result = {"total_auctions": len(records), "records": records}
        if self._cache:
            self._cache.put("sovereign_debt", {"key": cache_key}, result)

        # Return limited subset
        limited = result.copy()
        limited["records"] = records[:limit]

        return ToolResult(
            success=True,
            output=f"UK gilt auctions: {limited['total_auctions']} total, showing {len(limited['records'])} most recent",
            data=limited,
        )

    def _compute_spreads(self, countries: list[str], month: str) -> ToolResult:
        """Compute cross-country bond yield spreads."""
        # Ensure DE is included as benchmark
        if "DE" not in [c.upper() for c in countries]:
            countries = ["DE"] + countries

        eu_result = self._fetch_eu_yields(countries, month)
        if not eu_result.success:
            return eu_result

        country_data = eu_result.data.get("countries", {})
        if "DE" not in country_data:
            return ToolResult(
                success=False,
                output="Cannot compute spreads: Germany (DE) baseline not available",
            )

        # Get latest DE yield as benchmark
        de_latest = country_data["DE"][-1]["yield_pct"]

        spreads: list[dict[str, Any]] = []
        for cc, records in sorted(country_data.items()):
            if cc == "DE" or not records:
                continue
            latest_yield = records[-1]["yield_pct"]
            spread = round(latest_yield - de_latest, 4)
            spreads.append(
                {
                    "country": cc,
                    "yield_pct": latest_yield,
                    "de_yield_pct": de_latest,
                    "spread_vs_de": spread,
                    "period": records[-1]["period"],
                }
            )

        # Sort by spread descending (highest stress first)
        spreads.sort(key=lambda x: x["spread_vs_de"], reverse=True)

        # Also fetch US 2s10s if available
        now = datetime.now(timezone.utc)
        us_month = month or f"{now.year}-{now.month:02d}"
        us_result = self._fetch_us_yields(us_month)
        us_curve: dict[str, Any] | None = None
        if us_result.success and us_result.data:
            us_records = us_result.data.get("records", [])
            if us_records:
                latest_us = us_records[-1]
                us_curve = {
                    "date": latest_us["date"],
                    "curve_2s10s": latest_us.get("curve_2s10s"),
                    "curve_3m10y": latest_us.get("curve_3m10y"),
                }

        result = {
            "de_benchmark_yield": de_latest,
            "spreads": spreads,
            "us_curve": us_curve,
        }

        spread_summary = ", ".join(
            f"{s['country']}={s['spread_vs_de']:+.2f}%" for s in spreads[:5]
        )
        return ToolResult(
            success=True,
            output=f"Sovereign spreads vs DE ({de_latest:.2f}%): {spread_summary}",
            data=result,
        )

    # --- Main dispatcher ---

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = kwargs.get("mode", "")
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}",
            )

        # Parse common params
        now = datetime.now(timezone.utc)
        month = (kwargs.get("month") or "").strip()
        if month:
            if not _MONTH_RE.match(month):
                return ToolResult(
                    success=False,
                    output=f"Invalid month format '{month}'. Use YYYY-MM (e.g. '2026-03').",
                )
        else:
            month = f"{now.year}-{now.month:02d}"

        countries_raw = kwargs.get("countries") or _EU_COUNTRIES_DEFAULT
        countries = [
            c.strip().upper() for c in countries_raw if isinstance(c, str) and c.strip()
        ]

        limit = min(max(int(kwargs.get("limit", 20)), 1), 100)

        if mode == "us_yields":
            return self._fetch_us_yields(month)
        elif mode == "eu_yields":
            return self._fetch_eu_yields(countries, month)
        elif mode == "jp_yields":
            return self._fetch_jp_yields()
        elif mode == "uk_gilts":
            return self._fetch_uk_gilts(limit)
        elif mode == "spreads":
            return self._compute_spreads(countries, month)

        return ToolResult(success=False, output=f"Unhandled mode: {mode}")
