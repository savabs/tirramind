"""
Tool: Macro Data — Multi-Source Economic Time Series

Fetches macroeconomic time series from multiple sources:
  fred       — FRED (Federal Reserve Economic Data). US-focused. Requires API key.
  ecb        — ECB Data API (SDMX JSON). Eurozone monetary data. No auth.
  world_bank — World Bank Indicators API. 200+ countries. No auth.

FRED:  https://fred.stlouisfed.org/docs/api/api_key.html
ECB:   https://data-api.ecb.europa.eu/
World Bank: https://api.worldbank.org/v2/

Signal theory:
  - Cross-source comparison reveals divergent monetary policy trajectories
  - ECB balance sheet vs Fed balance sheet = relative liquidity signal
  - Emerging market indicators (World Bank) = capital flow early warning
  - Interest rate differentials across central banks = FX carry signals
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FRED_SERIES_INFO = "https://api.stlouisfed.org/fred/series"
_ECB_BASE = "https://data-api.ecb.europa.eu/service/data"
_WB_BASE = "https://api.worldbank.org/v2"
_TIMEOUT = 20

VALID_SOURCES = {"fred", "ecb", "world_bank"}

# Common ECB series aliases → SDMX flow/key
_ECB_ALIASES: dict[str, str] = {
    "EURUSD": "EXR/D.USD.EUR.SP00.A",
    "EURGBP": "EXR/D.GBP.EUR.SP00.A",
    "EURJPY": "EXR/D.JPY.EUR.SP00.A",
    "EURCHF": "EXR/D.CHF.EUR.SP00.A",
    "ECB_RATE": "FM/B.U2.EUR.4F.KR.MFI.NWT",
    "ECB_DEPOSIT_RATE": "FM/B.U2.EUR.4F.KR.DFR.LEV",
    "ECB_BALANCE_SHEET": "ILM/W.U2.C.T000000.Z5.Z01",
    "HICP": "ICP/M.U2.N.000000.4.ANR",
}


class MacroDataTool(Tool):
    name = "macro_data"
    description = (
        "Fetch macroeconomic time series. "
        "Sources: fred (US — GDP, CPI, Fed Funds, balance sheet), "
        "ecb (Eurozone — exchange rates, interest rates, ECB balance sheet), "
        "world_bank (200+ countries — GDP, population, trade). "
        "Returns date + value pairs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "series_id": {
                "type": "string",
                "description": (
                    "Series identifier. "
                    "FRED: GDP, CPIAUCSL, DFF, UNRATE, WALCL, M2SL. "
                    "ECB: alias (EURUSD, ECB_RATE, ECB_BALANCE_SHEET, HICP) "
                    "or SDMX path (EXR/D.USD.EUR.SP00.A). "
                    "World Bank: indicator code (NY.GDP.MKTP.CD, SP.POP.TOTL). "
                    "Comma-separated for multiple series (FRED/ECB only)."
                ),
            },
            "source": {
                "type": "string",
                "enum": sorted(VALID_SOURCES),
                "description": (
                    "Data source: fred (default, requires API key), "
                    "ecb (no auth, Eurozone), world_bank (no auth, global)."
                ),
            },
            "country": {
                "type": "string",
                "description": (
                    "ISO 3166-1 alpha-2 or alpha-3 country code for World Bank. "
                    "Default: all countries. Examples: US, GB, DE, BRA, CHN."
                ),
            },
            "start_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format. Default: 5 years ago.",
                "default": "",
            },
            "end_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format. Default: today.",
                "default": "",
            },
        },
        "required": ["series_id"],
    }

    def __init__(self, fred_api_key: str = "", cache: DataCache | None = None) -> None:
        self._api_key = fred_api_key
        self._cache = cache

    def execute(
        self,
        *,
        series_id: str,
        source: str = "",
        country: str = "",
        start_date: str = "",
        end_date: str = "",
        **_: Any,
    ) -> ToolResult:
        source = (source or "fred").strip().lower()
        if source not in VALID_SOURCES:
            return ToolResult(
                success=False,
                output=f"Invalid source '{source}'. Must be one of: {sorted(VALID_SOURCES)}",
            )

        if source == "ecb":
            return self._execute_ecb(series_id, start_date, end_date)
        if source == "world_bank":
            return self._execute_world_bank(series_id, country, start_date, end_date)
        return self._execute_fred(series_id, start_date, end_date)

    # ── FRED ────────────────────────────────────────────────────

    def _execute_fred(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> ToolResult:
        """FRED dispatch — Federal Reserve Economic Data."""
        if not self._api_key:
            return ToolResult(
                success=False,
                output=(
                    "FRED API key not configured. Set TIRRA_FRED_API_KEY in your .env file. "
                    "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
                ),
            )

        series_list = [s.strip().upper() for s in series_id.split(",") if s.strip()]
        if not series_list:
            return ToolResult(success=False, output="No valid series IDs provided.")

        results: list[str] = []
        all_data: dict[str, list[dict[str, str]]] = {}

        for sid in series_list:
            cache_params = {
                "series_id": sid,
                "start_date": start_date,
                "end_date": end_date,
            }
            cached = self._cache.get("macro_data", cache_params) if self._cache else None

            if cached is not None:
                log.debug("Cache hit for %s", sid)
                results.append(f"[{sid}] (cached) {cached['summary']}")
                all_data[sid] = cached["data"]
                continue

            try:
                observations = self._fetch_series(sid, start_date, end_date)
            except Exception as exc:
                log.exception("FRED fetch failed for %s", sid)
                results.append(f"[{sid}] Error: {exc}")
                continue

            if not observations:
                results.append(f"[{sid}] No data returned.")
                continue

            # Filter out entries with missing values (FRED uses "." for missing)
            clean = [obs for obs in observations if obs["value"] != "."]
            if not clean:
                results.append(f"[{sid}] All values are missing for this range.")
                continue

            all_data[sid] = clean

            first = clean[0]
            last = clean[-1]
            summary = (
                f"{len(clean)} observations\n"
                f"  First: {first['date']} → {first['value']}\n"
                f"  Last:  {last['date']} → {last['value']}"
            )
            results.append(f"[{sid}] {summary}")

            if self._cache:
                self._cache.put("macro_data", cache_params, {"summary": summary, "data": clean})

        output = "\n\n".join(results) if results else "No data returned for any series."
        return ToolResult(success=bool(all_data), output=output, data=all_data)

    def _fetch_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, str]]:
        """Fetch observations for a single FRED series. Returns list of {date, value}."""
        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date

        with httpx.Client(timeout=15) as client:
            resp = client.get(_FRED_BASE, params=params)
            resp.raise_for_status()

        data = resp.json()
        if "observations" not in data:
            error_msg = data.get("error_message", "Unknown FRED API error")
            raise ValueError(error_msg)

        return [{"date": obs["date"], "value": obs["value"]} for obs in data["observations"]]

    # ── ECB ─────────────────────────────────────────────────────

    def _execute_ecb(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> ToolResult:
        """ECB dispatch — European Central Bank SDMX Data API."""
        series_list = [s.strip() for s in series_id.split(",") if s.strip()]
        if not series_list:
            return ToolResult(success=False, output="No valid series IDs provided.")

        results: list[str] = []
        all_data: dict[str, list[dict[str, str]]] = {}

        for sid in series_list:
            # Resolve alias → SDMX key, or use raw if it contains "/"
            sdmx_key = _ECB_ALIASES.get(sid.upper(), sid)
            if "/" not in sdmx_key:
                available = ", ".join(sorted(_ECB_ALIASES.keys()))
                results.append(
                    f"[{sid}] Unknown ECB alias. Use a known alias ({available}) "
                    f"or a full SDMX path like 'EXR/D.USD.EUR.SP00.A'."
                )
                continue

            cache_params = {"ecb": sdmx_key, "start": start_date, "end": end_date}
            cached = self._cache.get("macro_data", cache_params) if self._cache else None
            if cached is not None:
                log.debug("Cache hit for ECB %s", sdmx_key)
                results.append(f"[{sid}] (cached) {cached['summary']}")
                all_data[sid] = cached["data"]
                continue

            try:
                observations = self._fetch_ecb(sdmx_key, start_date, end_date)
            except Exception as exc:
                log.exception("ECB fetch failed for %s", sdmx_key)
                results.append(f"[{sid}] Error: {exc}")
                continue

            if not observations:
                results.append(f"[{sid}] No data returned from ECB.")
                continue

            all_data[sid] = observations

            first = observations[0]
            last = observations[-1]
            summary = (
                f"{len(observations)} observations\n"
                f"  First: {first['date']} → {first['value']}\n"
                f"  Last:  {last['date']} → {last['value']}"
            )
            results.append(f"[{sid}] {summary}")

            if self._cache:
                self._cache.put(
                    "macro_data",
                    cache_params,
                    {"summary": summary, "data": observations},
                )

        output = "\n\n".join(results) if results else "No data returned for any ECB series."
        return ToolResult(success=bool(all_data), output=output, data=all_data)

    def _fetch_ecb(
        self,
        sdmx_key: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, str]]:
        """Fetch observations from ECB SDMX JSON API. sdmx_key = 'flow/key'."""
        url = f"{_ECB_BASE}/{sdmx_key}"
        params: dict[str, str] = {"format": "jsondata"}
        if start_date:
            params["startPeriod"] = start_date[:10]
        if end_date:
            params["endPeriod"] = end_date[:10]

        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": "TirraMind/0.1"}) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()

        data = resp.json()
        return self._parse_ecb_sdmx_json(data)

    @staticmethod
    def _parse_ecb_sdmx_json(data: dict) -> list[dict[str, str]]:
        """Parse ECB SDMX JSON (jsondata format) into [{date, value}, ...]."""
        datasets = data.get("dataSets", [])
        if not datasets:
            return []

        structure = data.get("structure", {})
        obs_dims = structure.get("dimensions", {}).get("observation", [])
        if not obs_dims:
            return []

        # Time dimension values
        time_values = obs_dims[0].get("values", [])
        time_index = {str(i): v.get("id", v.get("name", "")) for i, v in enumerate(time_values)}

        # Collect observations from all series
        observations: list[dict[str, str]] = []
        series_map = datasets[0].get("series", {})
        for _series_key, series_data in series_map.items():
            obs = series_data.get("observations", {})
            for obs_idx, obs_val in obs.items():
                date_str = time_index.get(obs_idx, "")
                value = obs_val[0] if isinstance(obs_val, list) and obs_val else None
                if date_str and value is not None:
                    observations.append({"date": date_str, "value": str(value)})

        observations.sort(key=lambda o: o["date"])
        return observations

    # ── World Bank ──────────────────────────────────────────────

    def _execute_world_bank(
        self,
        series_id: str,
        country: str,
        start_date: str,
        end_date: str,
    ) -> ToolResult:
        """World Bank dispatch — World Bank Indicators API."""
        indicator = series_id.strip()
        if not indicator:
            return ToolResult(success=False, output="No indicator provided.")

        cache_params = {
            "wb": indicator,
            "country": country,
            "start": start_date,
            "end": end_date,
        }
        cached = self._cache.get("macro_data", cache_params) if self._cache else None
        if cached is not None:
            log.debug("Cache hit for World Bank %s", indicator)
            return ToolResult(
                success=True,
                output=f"[{indicator}] (cached) {cached['summary']}",
                data=cached["data"],
            )

        try:
            observations = self._fetch_world_bank(indicator, country, start_date, end_date)
        except Exception as exc:
            log.exception("World Bank fetch failed for %s", indicator)
            return ToolResult(success=False, output=f"World Bank API error: {exc}")

        if not observations:
            return ToolResult(
                success=False,
                output=f"No data returned from World Bank for '{indicator}'.",
            )

        # Group by country
        by_country: dict[str, list[dict[str, str]]] = {}
        for obs in observations:
            cc = obs.get("country_code", "??")
            by_country.setdefault(cc, []).append(obs)

        lines: list[str] = []
        for cc in sorted(by_country.keys()):
            cdata = by_country[cc]
            last = cdata[-1]
            lines.append(f"  {cc}: {len(cdata)} obs, latest {last['date']} → {last['value']}")

        summary = f"{len(observations)} observations across {len(by_country)} countries"
        output = f"[{indicator}] {summary}\n" + "\n".join(lines[:20])
        if len(lines) > 20:
            output += f"\n  ... and {len(lines) - 20} more countries"

        if self._cache:
            self._cache.put("macro_data", cache_params, {"summary": summary, "data": by_country})

        return ToolResult(success=True, output=output, data=by_country)

    def _fetch_world_bank(
        self,
        indicator: str,
        country: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, str]]:
        """Fetch indicator data from World Bank API. Returns [{country_code, country, date, value}]."""
        country_path = country.strip().upper() or "all"
        url = f"{_WB_BASE}/country/{country_path}/indicator/{indicator}"
        params: dict[str, str] = {"format": "json", "per_page": "300"}
        if start_date:
            params["date"] = f"{start_date[:4]}:{end_date[:4]}" if end_date else start_date[:4]
        elif end_date:
            params["date"] = end_date[:4]

        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": "TirraMind/0.1"}) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()

        data = resp.json()
        # World Bank returns [metadata, records] — records is index 1
        if not isinstance(data, list) or len(data) < 2:
            return []

        records = data[1]
        if not records:
            return []

        observations: list[dict[str, str]] = []
        for rec in records:
            value = rec.get("value")
            if value is None:
                continue
            observations.append(
                {
                    "country_code": rec.get("countryiso3code", rec.get("country", {}).get("id", "")),
                    "country": rec.get("country", {}).get("value", ""),
                    "date": str(rec.get("date", "")),
                    "value": str(value),
                }
            )

        observations.sort(key=lambda o: (o["country_code"], o["date"]))
        return observations
