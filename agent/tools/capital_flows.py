"""
Tool: Cross-Border Capital Flows — Global Capital Movement Monitor

FRED:  https://api.stlouisfed.org/docs/api/  (requires free API key)

When Japan + China + Saudi all sell US Treasuries simultaneously →
coordinated de-dollarization.  EM→DM flow reversals precede crises.
Reserve drawdown/accumulation patterns reveal central bank positioning.

The Treasury International Capital (TIC) data is the gold standard
for tracking foreign holdings of US securities.  FRED aggregates
the major TIC series with clean JSON access.

Modes
-----
holdings        Latest foreign holdings of US Treasury securities
                by major country.  Detects coordinated selling/buying.

flows           Net capital flows (purchases - sales) over time.
                Sudden reversals = crisis signal.

reserves        Foreign exchange reserves for major holders.
                Drawdown = BoP stress; accumulation = intervention.

Signal theory:
  - Coordinated selling by top-3 holders = de-dollarization event
  - Reserve drawdowns > 5% in single quarter = currency defense
  - EM flow reversals (positive → negative) precede EM crises
  - Divergence between reported holdings and estimated = stealth selling
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_UA = "TirraMind/0.1 (capital-flows-tool)"
_TIMEOUT = 20
_CACHE_TTL = 3600  # 1 hr — TIC data is monthly

VALID_MODES = frozenset({"holdings", "flows", "reserves"})

# Period → calendar days
_PERIOD_DAYS: dict[str, int] = {
    "6m": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "10y": 3650,
}

# ---------------------------------------------------------------------------
# FRED series for TIC holdings (major foreign holders of US Treasuries)
# Series: monthly, billions USD, seasonally adjusted where available
# ---------------------------------------------------------------------------

HOLDINGS_SERIES: dict[str, dict[str, str]] = {
    "japan": {
        "series_id": "FDHBFIN",
        "name": "Japan",
        "description": "Japan holdings of US Treasury securities",
    },
    "china": {
        "series_id": "FDHBFCH",
        "name": "China (Mainland)",
        "description": "China mainland holdings of US Treasury securities",
    },
    "uk": {
        "series_id": "FDHBFUK",
        "name": "United Kingdom",
        "description": "UK holdings of US Treasury securities",
    },
    "total": {
        "series_id": "FDHBFIN",  # Total foreign holdings — grand total
        "name": "Total Foreign",
        "description": "Total foreign holdings of US Treasury securities",
    },
}

# FRED series for net capital flows (TIC flows)
FLOW_SERIES: dict[str, dict[str, str]] = {
    "net_foreign_purchases": {
        "series_id": "NETFI",
        "name": "Net Foreign Investment",
        "description": "Net foreign purchases of US long-term securities",
    },
    "foreign_official": {
        "series_id": "BOPFOIA",
        "name": "Foreign Official Institutions",
        "description": "Foreign official institution net acquisitions of US assets",
    },
    "foreign_private": {
        "series_id": "BOPPRIA",
        "name": "Foreign Private",
        "description": "Foreign private net acquisitions of US assets",
    },
}

# FRED series for foreign exchange reserves
RESERVE_SERIES: dict[str, dict[str, str]] = {
    "total_reserves_ex_gold": {
        "series_id": "TRESEGUSM052N",
        "name": "Total Reserves excl Gold (US)",
        "description": "Total reserves excluding gold, current US$",
    },
    "china_reserves": {
        "series_id": "TRESEGCNM052N",
        "name": "China Foreign Reserves",
        "description": "China total reserves excluding gold",
    },
    "japan_reserves": {
        "series_id": "TRESEGJPM052N",
        "name": "Japan Foreign Reserves",
        "description": "Japan total reserves excluding gold",
    },
    "saudi_reserves": {
        "series_id": "TRESEGSAM052N",
        "name": "Saudi Arabia Foreign Reserves",
        "description": "Saudi Arabia total reserves excluding gold",
    },
    "india_reserves": {
        "series_id": "TRESEGINM052N",
        "name": "India Foreign Reserves",
        "description": "India total reserves excluding gold",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_fred(
    series_id: str,
    api_key: str,
    start_date: str,
    end_date: str,
) -> list[dict[str, str]]:
    """Fetch observations from FRED.  Returns list of {date, value}."""
    try:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
        }
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}) as client:
            resp = client.get(_FRED_BASE, params=params)
            resp.raise_for_status()
        data = resp.json()
        return [
            {"date": obs["date"], "value": obs["value"]}
            for obs in data.get("observations", [])
            if obs.get("value") not in (".", "", None)
        ]
    except Exception:
        log.exception("FRED fetch failed for %s", series_id)
        return []


def _latest(observations: list[dict[str, str]]) -> dict[str, str] | None:
    """Return most recent observation."""
    return observations[-1] if observations else None


def _pct_change(old: float, new: float) -> float | None:
    """Percentage change from old to new.  None if old == 0."""
    if old == 0:
        return None
    return ((new - old) / abs(old)) * 100


def _detect_coordinated(
    country_changes: dict[str, float | None],
    threshold: float = -2.0,
) -> dict[str, Any]:
    """Detect if multiple holders are selling simultaneously."""
    sellers = []
    buyers = []
    for name, chg in country_changes.items():
        if chg is None:
            continue
        if chg <= threshold:
            sellers.append({"name": name, "change_pct": chg})
        elif chg >= abs(threshold):
            buyers.append({"name": name, "change_pct": chg})
    return {
        "coordinated_selling": len(sellers) >= 2,
        "coordinated_buying": len(buyers) >= 2,
        "sellers": sellers,
        "buyers": buyers,
    }


def _reserve_stress(
    observations: list[dict[str, str]],
    lookback_months: int = 3,
) -> dict[str, Any]:
    """Detect reserve drawdown stress from a time series."""
    if len(observations) < 2:
        return {"stress": False, "drawdown_pct": None}

    latest_val = float(observations[-1]["value"])
    # Find value ~lookback_months ago
    idx = max(0, len(observations) - lookback_months - 1)
    old_val = float(observations[idx]["value"])

    chg = _pct_change(old_val, latest_val)
    return {
        "stress": chg is not None and chg < -5.0,
        "drawdown_pct": round(chg, 2) if chg is not None else None,
        "latest_value": latest_val,
        "comparison_value": old_val,
        "latest_date": observations[-1]["date"],
        "comparison_date": observations[idx]["date"],
    }


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class CapitalFlowsTool(Tool):
    """Monitor cross-border capital flows and foreign holdings."""

    name = "capital_flows"
    description = (
        "Track foreign holdings of US Treasuries, net capital flows, and "
        "foreign exchange reserves.  Detects coordinated selling/buying by "
        "major holders, EM flow reversals, and reserve stress."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": (
                    "flows: net foreign purchases of US securities over time. "
                    "holdings: foreign holdings of US Treasuries by country. "
                    "reserves: foreign exchange reserve levels for major holders."
                ),
            },
            "period": {
                "type": "string",
                "enum": sorted(_PERIOD_DAYS.keys()),
                "description": "Lookback period (default: 2y).",
            },
            "country": {
                "type": "string",
                "description": "Country filter (lowercase key: japan, china, uk, etc.).",
            },
        },
        "required": ["mode"],
    }

    def __init__(
        self,
        *,
        fred_api_key: str = "",
        cache: DataCache | None = None,
    ) -> None:
        self._api_key = fred_api_key
        self._cache = cache

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = (kwargs.get("mode") or "").strip().lower()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: {', '.join(sorted(VALID_MODES))}",
            )
        if not self._api_key:
            return ToolResult(
                success=False,
                output="FRED API key required. Set TIRRA_FRED_API_KEY.",
            )

        period_key = (kwargs.get("period") or "2y").strip()
        if period_key not in _PERIOD_DAYS:
            period_key = "2y"
        days = _PERIOD_DAYS[period_key]
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        if mode == "holdings":
            return self._holdings(start, end, **kwargs)
        if mode == "flows":
            return self._flows(start, end, **kwargs)
        return self._reserves(start, end, **kwargs)

    # ── holdings mode ────────────────────────────────────────────

    def _holdings(self, start: str, end: str, **kwargs: Any) -> ToolResult:
        country_filter = (kwargs.get("country") or "").strip().lower()

        series_to_fetch = HOLDINGS_SERIES
        if country_filter and country_filter in HOLDINGS_SERIES:
            series_to_fetch = {country_filter: HOLDINGS_SERIES[country_filter]}
        elif country_filter and country_filter not in HOLDINGS_SERIES:
            return ToolResult(
                success=False,
                output=(
                    f"Unknown country '{country_filter}'. "
                    f"Available: {', '.join(sorted(HOLDINGS_SERIES.keys()))}"
                ),
            )

        results: list[dict[str, Any]] = []
        country_changes: dict[str, float | None] = {}
        errors: list[str] = []

        for key, info in series_to_fetch.items():
            cache_key = {"series": info["series_id"], "start": start, "end": end}
            cached = (
                self._cache.get("capital_flows", cache_key) if self._cache else None
            )
            if cached is not None:
                obs = cached
            else:
                obs = _fetch_fred(info["series_id"], self._api_key, start, end)
                if self._cache and obs:
                    self._cache.put("capital_flows", cache_key, obs)

            if not obs:
                errors.append(f"{info['name']}: no data")
                continue

            latest = _latest(obs)
            latest_val = float(latest["value"]) if latest else 0

            # MoM change (compare last two observations)
            mom_chg = None
            if len(obs) >= 2:
                prev_val = float(obs[-2]["value"])
                mom_chg = _pct_change(prev_val, latest_val)

            country_changes[info["name"]] = mom_chg

            results.append(
                {
                    "country": info["name"],
                    "key": key,
                    "latest_value_billions": latest_val,
                    "latest_date": latest["date"] if latest else None,
                    "mom_change_pct": (
                        round(mom_chg, 2) if mom_chg is not None else None
                    ),
                    "observations": len(obs),
                }
            )

        # Check for coordinated moves
        coordination = _detect_coordinated(country_changes)

        # Format output
        lines = ["# Foreign Holdings of US Treasuries\n"]
        for r in results:
            flag = ""
            if r["mom_change_pct"] is not None and r["mom_change_pct"] < -3:
                flag = " **[SELLING]**"
            elif r["mom_change_pct"] is not None and r["mom_change_pct"] > 3:
                flag = " **[BUYING]**"
            val_str = f"${r['latest_value_billions']:,.1f}B"
            mom = (
                f"{r['mom_change_pct']:+.1f}%"
                if r["mom_change_pct"] is not None
                else "N/A"
            )
            lines.append(f"**{r['country']}**: {val_str} (MoM: {mom}){flag}")
            lines.append(f"  Date: {r['latest_date']}")

        if coordination["coordinated_selling"]:
            names = ", ".join(s["name"] for s in coordination["sellers"])
            lines.append(f"\n⚠ **COORDINATED SELLING** detected: {names}")
        if coordination["coordinated_buying"]:
            names = ", ".join(b["name"] for b in coordination["buyers"])
            lines.append(f"\n📈 **COORDINATED BUYING** detected: {names}")

        if errors:
            lines.append("\n**Notes:**")
            for e in errors:
                lines.append(f"  - {e}")

        return ToolResult(
            success=bool(results),
            output="\n".join(lines),
            data={
                "mode": "holdings",
                "holdings": results,
                "coordination": coordination,
                "errors": errors,
            },
        )

    # ── flows mode ───────────────────────────────────────────────

    def _flows(self, start: str, end: str, **kwargs: Any) -> ToolResult:
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for key, info in FLOW_SERIES.items():
            cache_key = {"series": info["series_id"], "start": start, "end": end}
            cached = (
                self._cache.get("capital_flows", cache_key) if self._cache else None
            )
            if cached is not None:
                obs = cached
            else:
                obs = _fetch_fred(info["series_id"], self._api_key, start, end)
                if self._cache and obs:
                    self._cache.put("capital_flows", cache_key, obs)

            if not obs:
                errors.append(f"{info['name']}: no data")
                continue

            latest = _latest(obs)
            latest_val = float(latest["value"]) if latest else 0

            # Detect flow reversal (sign change in last N observations)
            reversal = False
            if len(obs) >= 3:
                recent_vals = [float(o["value"]) for o in obs[-6:]]
                for i in range(1, len(recent_vals)):
                    if recent_vals[i - 1] > 0 and recent_vals[i] < 0:
                        reversal = True
                        break

            # Average over period
            all_vals = [float(o["value"]) for o in obs]
            avg_val = sum(all_vals) / len(all_vals) if all_vals else 0

            results.append(
                {
                    "series": info["name"],
                    "key": key,
                    "description": info["description"],
                    "latest_value": latest_val,
                    "latest_date": latest["date"] if latest else None,
                    "period_average": round(avg_val, 2),
                    "flow_reversal": reversal,
                    "observations": len(obs),
                }
            )

        # Format
        lines = ["# Net Capital Flows\n"]
        for r in results:
            direction = "inflow" if r["latest_value"] > 0 else "outflow"
            reversal_flag = " **[REVERSAL]**" if r["flow_reversal"] else ""
            lines.append(
                f"**{r['series']}**: ${r['latest_value']:,.1f}M ({direction}){reversal_flag}"
            )
            lines.append(
                f"  Period avg: ${r['period_average']:,.1f}M | Date: {r['latest_date']}"
            )

        if errors:
            lines.append("\n**Notes:**")
            for e in errors:
                lines.append(f"  - {e}")

        return ToolResult(
            success=bool(results),
            output="\n".join(lines),
            data={
                "mode": "flows",
                "flows": results,
                "errors": errors,
            },
        )

    # ── reserves mode ────────────────────────────────────────────

    def _reserves(self, start: str, end: str, **kwargs: Any) -> ToolResult:
        country_filter = (kwargs.get("country") or "").strip().lower()

        series_to_fetch = RESERVE_SERIES
        if country_filter:
            # Match by country name in key
            matched = {k: v for k, v in RESERVE_SERIES.items() if country_filter in k}
            if matched:
                series_to_fetch = matched

        results: list[dict[str, Any]] = []
        stress_alerts: list[dict[str, Any]] = []
        errors: list[str] = []

        for key, info in series_to_fetch.items():
            cache_key = {"series": info["series_id"], "start": start, "end": end}
            cached = (
                self._cache.get("capital_flows", cache_key) if self._cache else None
            )
            if cached is not None:
                obs = cached
            else:
                obs = _fetch_fred(info["series_id"], self._api_key, start, end)
                if self._cache and obs:
                    self._cache.put("capital_flows", cache_key, obs)

            if not obs:
                errors.append(f"{info['name']}: no data")
                continue

            latest = _latest(obs)
            latest_val = float(latest["value"]) if latest else 0

            stress = _reserve_stress(obs)

            entry = {
                "series": info["name"],
                "key": key,
                "latest_value": latest_val,
                "latest_date": latest["date"] if latest else None,
                "observations": len(obs),
                "stress": stress,
            }
            results.append(entry)

            if stress["stress"]:
                stress_alerts.append(
                    {
                        "country": info["name"],
                        "drawdown_pct": stress["drawdown_pct"],
                    }
                )

        # Format
        lines = ["# Foreign Exchange Reserves\n"]
        for r in results:
            stress_flag = " **[STRESS]**" if r["stress"]["stress"] else ""
            val_str = f"${r['latest_value']:,.0f}"
            lines.append(f"**{r['series']}**: {val_str}{stress_flag}")
            lines.append(f"  Date: {r['latest_date']}")
            if r["stress"]["drawdown_pct"] is not None:
                lines.append(f"  3M change: {r['stress']['drawdown_pct']:+.1f}%")

        if stress_alerts:
            names = ", ".join(s["country"] for s in stress_alerts)
            lines.append(f"\n⚠ **RESERVE STRESS** detected: {names}")

        if errors:
            lines.append("\n**Notes:**")
            for e in errors:
                lines.append(f"  - {e}")

        return ToolResult(
            success=bool(results),
            output="\n".join(lines),
            data={
                "mode": "reserves",
                "reserves": results,
                "stress_alerts": stress_alerts,
                "errors": errors,
            },
        )
