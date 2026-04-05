"""
Tool: Global PMI — OECD Composite Leading Indicators (CLI/BCI/CCI)

Fetch and parse OECD leading indicators via the SDMX API.
Provides Composite Leading Indicators (CLI), Business Confidence
Indicators (BCI), and Consumer Confidence Indicators (CCI) for 40+
countries.

Data source: https://sdmx.oecd.org/ (free, no auth, OECD ToC).
Frequency: Monthly.
History: 1960s–present for major economies.

Signal theory:
  - CLI turning points precede GDP turning points by 6-9 months
  - CLI < 100 declining = contraction signal
  - Cross-country CLI divergence = relative growth momentum (FX, commodities)
  - Simultaneous G7 CLI decline = synchronized global slowdown (rare, high-impact)
  - BCI-CLI divergence = manufacturing stress before hard data
  - CLI rate of change (6m momentum) more actionable than level
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_SDMX_BASE = "https://sdmx.oecd.org/public/rest/data"

# Dataflow identifiers
_DATAFLOWS = {
    "cli": "OECD.SDD.STES,DSD_STES@DF_CLI",
    "bci": "OECD.SDD.STES,DSD_STES@DF_BCI",
    "cci": "OECD.SDD.STES,DSD_STES@DF_CCI",
}

# Dimension selection: country.freq.measure...adjustment...transformation
# CLI amplitude-adjusted: {country}.M.LI...AA...H
# BCI/CCI: {country}.M.LI...AA...H (same pattern)
_DIM_PATTERN = "{countries}.M.LI...AA...H"

_UA = "TirraMind/0.1 (research)"
_TIMEOUT = 30  # OECD can be slow
_PERIOD_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

VALID_MODES = frozenset(_DATAFLOWS)

# Default country set: G7 + major emerging markets + OECD aggregate
_DEFAULT_COUNTRIES = "USA,GBR,DEU,FRA,JPN,CHN,CAN,ITA"

# Recognized country codes (subset — OECD has more)
_KNOWN_CODES = frozenset(
    {
        "USA",
        "GBR",
        "DEU",
        "FRA",
        "JPN",
        "CHN",
        "KOR",
        "AUS",
        "CAN",
        "ITA",
        "ESP",
        "BRA",
        "IND",
        "MEX",
        "TUR",
        "ZAF",
        "IDN",
        "RUS",
        "NLD",
        "BEL",
        "AUT",
        "CHE",
        "SWE",
        "NOR",
        "DNK",
        "FIN",
        "POL",
        "CZE",
        "HUN",
        "GRC",
        "PRT",
        "IRL",
        "NZL",
        "ISR",
        "CHL",
        "COL",
        "CRI",
        "LVA",
        "LTU",
        "SVK",
        "SVN",
        "EST",
        "ISL",
        "LUX",
        # Aggregates
        "OECD",
        "G-7",
        "EA19",
        "G-20",
    }
)


def _parse_float(val: str | None) -> float | None:
    """Parse a float, returning None for empty/invalid."""
    if val is None:
        return None
    v = val.strip()
    if not v or v in ("", "NaN", "nan", "null"):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


class GlobalPmiTool(Tool):
    name = "global_pmi"
    description = (
        "Fetch OECD leading indicators for 40+ countries. "
        "Modes: cli (Composite Leading Indicators — GDP turning point predictor), "
        "bci (Business Confidence), cci (Consumer Confidence). "
        "Computes signals: 6-month momentum, regime classification "
        "(expanding/contracting), cross-country spreads. "
        "Monthly, free, no auth."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": (
                    "cli = Composite Leading Indicators (GDP turning point predictor); "
                    "bci = Business Confidence; cci = Consumer Confidence"
                ),
            },
            "countries": {
                "type": "string",
                "default": _DEFAULT_COUNTRIES,
                "description": (
                    "Comma-separated ISO 3-letter country codes. "
                    "Available: USA, GBR, DEU, FRA, JPN, CHN, KOR, AUS, CAN, ITA, "
                    "ESP, BRA, IND, MEX, OECD, G-7, EA19, G-20, etc. "
                    f"Default: {_DEFAULT_COUNTRIES}"
                ),
            },
            "start_period": {
                "type": "string",
                "default": "",
                "description": "Start period (YYYY-MM). Default: 24 months ago.",
            },
            "end_period": {
                "type": "string",
                "default": "",
                "description": "End period (YYYY-MM). Default: latest available.",
            },
            "include_signals": {
                "type": "boolean",
                "default": True,
                "description": "Compute momentum, regime, and spread signals.",
            },
        },
        "required": ["mode"],
    }

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    # ── Public execute ───────────────────────────────────────────────

    def execute(
        self,
        *,
        mode: str = "cli",
        countries: str = _DEFAULT_COUNTRIES,
        start_period: str = "",
        end_period: str = "",
        include_signals: bool = True,
        **_: Any,
    ) -> ToolResult:
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Choose from: {', '.join(sorted(VALID_MODES))}.",
            )

        # Validate periods
        for label, val in [("start_period", start_period), ("end_period", end_period)]:
            if val and not _PERIOD_RE.match(val):
                return ToolResult(
                    success=False,
                    output=f"Invalid {label} '{val}'. Use YYYY-MM format (e.g. 2024-01).",
                )
        if start_period and end_period and start_period > end_period:
            return ToolResult(
                success=False,
                output=f"start_period ({start_period}) is after end_period ({end_period}).",
            )

        # Parse and validate countries
        country_list = [c.strip().upper() for c in countries.split(",") if c.strip()]
        if not country_list:
            country_list = _DEFAULT_COUNTRIES.split(",")

        try:
            rows = self._fetch(mode, country_list, start_period, end_period)
        except Exception as exc:
            log.exception("OECD %s fetch failed", mode)
            return ToolResult(success=False, output=f"OECD API error: {exc}")

        if not rows:
            return ToolResult(
                success=True,
                output=f"No data returned for mode='{mode}', countries={countries}.",
                data={"mode": mode, "records": []},
            )

        # Organize by country
        by_country: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            cc = row.get("country", "?")
            by_country.setdefault(cc, []).append(row)

        # Sort each country's data by period
        for cc in by_country:
            by_country[cc].sort(key=lambda r: r.get("period", ""))

        # Build output
        output, data = self._format_output(mode, by_country, include_signals)
        return ToolResult(success=True, output=output, data=data)

    # ── Fetch ────────────────────────────────────────────────────────

    def _fetch(
        self,
        mode: str,
        countries: list[str],
        start_period: str,
        end_period: str,
    ) -> list[dict[str, Any]]:
        country_key = "+".join(countries)
        cache_key = {
            "source": f"oecd_{mode}",
            "countries": country_key,
            "start": start_period,
            "end": end_period,
        }
        if self._cache:
            cached = self._cache.get("global_pmi", cache_key)
            if cached is not None:
                log.debug("OECD %s: cache hit", mode)
                return cached

        dataflow = _DATAFLOWS[mode]
        dim_selection = _DIM_PATTERN.format(countries=country_key)
        url = f"{_SDMX_BASE}/{dataflow}/{dim_selection}"

        params: dict[str, str] = {
            "dimensionAtObservation": "AllDimensions",
            "format": "csvfilewithlabels",
        }
        if start_period:
            params["startPeriod"] = start_period
        if end_period:
            params["endPeriod"] = end_period

        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url, params=params, headers={"User-Agent": _UA})
            resp.raise_for_status()

        rows = self._parse_csv(resp.text)

        if self._cache and rows:
            self._cache.put("global_pmi", cache_key, rows)

        return rows

    # ── CSV Parsing ──────────────────────────────────────────────────

    def _parse_csv(self, text: str) -> list[dict[str, Any]]:
        """Parse OECD CSV-with-labels response into standardized rows."""
        if not text.strip():
            return []

        reader = csv.DictReader(io.StringIO(text))
        rows: list[dict[str, Any]] = []

        for record in reader:
            # OECD CSV columns vary but include: REF_AREA, TIME_PERIOD, OBS_VALUE
            country = record.get("REF_AREA") or record.get("Reference area", "")
            period = record.get("TIME_PERIOD") or record.get("Time period", "")
            value = _parse_float(
                record.get("OBS_VALUE") or record.get("Observation value")
            )

            if country and period and value is not None:
                rows.append(
                    {
                        "country": country,
                        "period": period,
                        "value": value,
                    }
                )

        return rows

    # ── Output ───────────────────────────────────────────────────────

    def _format_output(
        self,
        mode: str,
        by_country: dict[str, list[dict[str, Any]]],
        include_signals: bool,
    ) -> tuple[str, dict[str, Any]]:
        mode_label = {
            "cli": "Composite Leading Indicators",
            "bci": "Business Confidence",
            "cci": "Consumer Confidence",
        }
        lines = [f"## OECD {mode_label.get(mode, mode).upper()}\n"]
        all_signals: dict[str, dict[str, Any]] = {}

        for cc, entries in sorted(by_country.items()):
            latest = entries[-1]
            lines.append(f"  {cc}: {latest['value']:.2f} ({latest['period']})")

            if include_signals and len(entries) >= 2:
                signals = self._compute_signals(cc, entries)
                all_signals[cc] = signals

                regime = signals.get("regime", "?")
                mom = signals.get("momentum_6m")
                mom_str = f"{mom:+.2f}" if mom is not None else "N/A"
                lines.append(f"    Regime: {regime} | 6m momentum: {mom_str}")

        # Cross-country spreads
        spread_lines: list[str] = []
        if include_signals and len(by_country) >= 2:
            spreads = self._compute_spreads(by_country)
            for label, val in spreads.items():
                spread_lines.append(f"  {label}: {val:+.2f}")
                all_signals.setdefault("_spreads", {})[label] = val

        if spread_lines:
            lines.append("\n  Cross-country spreads (latest):")
            lines.extend(spread_lines)

        total_records = sum(len(v) for v in by_country.values())
        lines.append(f"\n  Countries: {len(by_country)} | Records: {total_records}")

        # Flatten records for data payload
        all_records = []
        for entries in by_country.values():
            all_records.extend(entries)

        return "\n".join(lines), {
            "mode": mode,
            "records": all_records,
            "by_country": {cc: entries for cc, entries in by_country.items()},
            "signals": all_signals,
        }

    def _compute_signals(
        self, country: str, entries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute momentum, regime, and direction for a single country."""
        signals: dict[str, Any] = {}
        latest = entries[-1]["value"]

        # Regime: > 100 = expansion, < 100 = contraction
        if latest > 100:
            direction = (
                "expanding"
                if len(entries) >= 2 and entries[-1]["value"] > entries[-2]["value"]
                else "peaking"
            )
            signals["regime"] = direction
        else:
            direction = (
                "contracting"
                if len(entries) >= 2 and entries[-1]["value"] < entries[-2]["value"]
                else "troughing"
            )
            signals["regime"] = direction

        # Month-over-month change
        if len(entries) >= 2:
            prev = entries[-2]["value"]
            signals["mom_change"] = round(latest - prev, 4)

        # 6-month momentum
        if len(entries) >= 7:
            six_ago = entries[-7]["value"]
            if six_ago != 0:
                signals["momentum_6m"] = round(
                    (latest - six_ago) / abs(six_ago) * 100, 2
                )

        signals["latest_value"] = latest
        signals["latest_period"] = entries[-1]["period"]
        return signals

    def _compute_spreads(
        self, by_country: dict[str, list[dict[str, Any]]]
    ) -> dict[str, float]:
        """Compute cross-country spreads using latest values."""
        latest_vals: dict[str, float] = {}
        for cc, entries in by_country.items():
            if entries:
                latest_vals[cc] = entries[-1]["value"]

        spreads: dict[str, float] = {}
        # Predefined interesting spreads
        spread_pairs = [
            ("USA", "CHN"),
            ("USA", "DEU"),
            ("USA", "JPN"),
            ("DEU", "ITA"),
            ("GBR", "DEU"),
        ]
        for a, b in spread_pairs:
            if a in latest_vals and b in latest_vals:
                spreads[f"{a}-{b}"] = round(latest_vals[a] - latest_vals[b], 2)

        return spreads
