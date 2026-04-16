"""
Tool: Central Bank Balance Sheets — Cross-CB Liquidity Analytics

FRED:  https://api.stlouisfed.org/docs/api/  (requires free API key)
ECB:   https://data-api.ecb.europa.eu/        (free, no auth)

Central bank balance sheets drive global liquidity.  When the Fed, ECB, and BOJ
expand simultaneously, risk assets rally.  When they contract together, everything
correlates to 1 and sells off.  The individual numbers are public; the *cross-CB
relative positioning* is what nobody computes systematically outside top macro
funds.

Modes:
  balance_sheets     — Snapshot of all major CB balance sheets (native + USD).
  liquidity_index    — Global net liquidity = sum(CB assets) - RRP - TGA.
  policy_divergence  — Who's expanding vs contracting, rate differentials.
  rate_monitor       — Current policy rates, last change, days since change.

Data lag: Fed weekly (Wed), ECB weekly (Tue), BOJ/BOE/SNB/BOC/RBA monthly.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key as _entity_id_from_key
except ImportError:  # pragma: no cover — optional dependency
    _entity_id_from_key = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_ECB_BASE = "https://data-api.ecb.europa.eu/service/data"
_UA = "TirraMind/0.1"
_TIMEOUT = 20

VALID_MODES = frozenset(
    {
        "balance_sheets",
        "liquidity_index",
        "policy_divergence",
        "rate_monitor",
    }
)

# Period string → calendar days
_PERIOD_DAYS: dict[str, int] = {
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
}

# Cache TTLs (seconds)
_CACHE_TTL_BS = 3600  # 1 hr — balance sheet data (weekly/monthly source)
_CACHE_TTL_RATES = 1800  # 30 min — rate data can change on decision days
_CACHE_TTL_FX = 3600  # 1 hr — FX rates for normalization

# ---------------------------------------------------------------------------
# Central Bank Registry
# ---------------------------------------------------------------------------

# Each CB: balance sheet FRED series, currency, FX FRED series (to USD),
# rate series (FRED or ECB), display name.
# FX series: FRED series that gives units of foreign currency per USD
# (we invert for non-USD CBs to get USD value).

CB_REGISTRY: dict[str, dict[str, str]] = {
    "fed": {
        "name": "Federal Reserve",
        "bs_series": "WALCL",  # Total Assets, Millions USD, Weekly
        "currency": "USD",
        "fx_series": "",  # No conversion needed
        "rate_series": "DFF",  # Fed Funds Effective Rate
        "unit_scale": "1e6",  # FRED reports in millions
        "frequency": "weekly",
    },
    "ecb": {
        "name": "European Central Bank",
        "bs_series": "ECBASSETSW",  # Total Assets, Millions EUR, Weekly
        "currency": "EUR",
        "fx_series": "DEXUSEU",  # USD per EUR
        "rate_series": "_ECB_DFR",  # Special: fetched from ECB SDW
        "unit_scale": "1e6",
        "frequency": "weekly",
    },
    "boj": {
        "name": "Bank of Japan",
        "bs_series": "JPNASSETS",  # Total Assets, 100M Yen, Monthly
        "currency": "JPY",
        "fx_series": "DEXJPUS",  # Yen per USD (we need 1/this)
        "rate_series": "",  # BOJ rate ~0, no good FRED series
        "unit_scale": "1e8",  # FRED reports in 100M yen
        "frequency": "monthly",
    },
    "boe": {
        "name": "Bank of England",
        "bs_series": "WALCL",  # Placeholder — see _BOE_FALLBACK
        "currency": "GBP",
        "fx_series": "DEXUSUK",  # USD per GBP
        "rate_series": "",
        "unit_scale": "1e6",
        "frequency": "monthly",
        "_skip_bs": "true",  # BOE FRED series unreliable
    },
    "snb": {
        "name": "Swiss National Bank",
        "bs_series": "SNBASSETM",  # Total Assets, Millions CHF, Monthly
        "currency": "CHF",
        "fx_series": "DEXSZUS",  # CHF per USD (we need 1/this)
        "rate_series": "",
        "unit_scale": "1e6",
        "frequency": "monthly",
    },
    "boc": {
        "name": "Bank of Canada",
        "bs_series": "BCBASSETM",  # May not exist — graceful fallback
        "currency": "CAD",
        "fx_series": "DEXCAUS",  # CAD per USD (we need 1/this)
        "rate_series": "",
        "unit_scale": "1e6",
        "frequency": "monthly",
    },
    "rba": {
        "name": "Reserve Bank of Australia",
        "bs_series": "RBASSETSM",  # May not exist — graceful fallback
        "currency": "AUD",
        "fx_series": "DEXUSAL",  # USD per AUD
        "rate_series": "",
        "unit_scale": "1e6",
        "frequency": "monthly",
    },
}

# CBs that reliably have FRED balance sheet data
_CORE_CBS = ("fed", "ecb", "boj")

# Fed liquidity drain series
_FED_RRP_SERIES = "RRPONTSYD"  # Reverse Repo, Billions USD, Daily
_FED_TGA_SERIES = "WDTGAL"  # Treasury General Account, Millions USD, Weekly

# FX series where FRED gives "USD per foreign" (no inversion needed)
_USD_PER_FOREIGN = {"DEXUSEU", "DEXUSUK", "DEXUSAL"}
# FX series where FRED gives "foreign per USD" (need inversion)
_FOREIGN_PER_USD = {"DEXJPUS", "DEXSZUS", "DEXCAUS"}

# ECB SDW keys
_ECB_DFR_KEY = "FM/B.U2.EUR.4F.KR.DFR.LEV"  # Deposit Facility Rate
_ECB_BS_KEY = "ILM/W.U2.C.T000000.Z5.Z01"  # Total Assets (EUR millions)

# Deterministic CB → country mapping (Phase 27).
# Only explicit, verifiable relationships.
CB_TO_COUNTRY: dict[str, str] = {
    "fed": "US",
    "ecb": "EU",
    "boj": "JP",
    "boe": "GB",
    "snb": "CH",
    "boc": "CA",
    "rba": "AU",
}


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class CentralBankBalanceTool(Tool):
    """Cross-central-bank balance sheet analytics."""

    name = "central_bank_balance"
    description = (
        "Global central bank balance sheet analytics. "
        "Modes: balance_sheets (cross-CB snapshot in USD), "
        "liquidity_index (net global liquidity = CB assets - RRP - TGA), "
        "policy_divergence (expanding vs contracting, rate differentials), "
        "rate_monitor (current policy rates + last change detection). "
        "Covers: Fed, ECB, BOJ, BOE, SNB, BOC, RBA."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": (
                    "balance_sheets: snapshot of all CB balance sheets. "
                    "liquidity_index: net global liquidity time series. "
                    "policy_divergence: who's expanding vs contracting. "
                    "rate_monitor: current rates + last change."
                ),
            },
            "period": {
                "type": "string",
                "enum": sorted(_PERIOD_DAYS.keys()),
                "description": "Lookback period. Default: 1y.",
            },
            "banks": {
                "type": "string",
                "description": (
                    "Comma-separated CB codes to focus on. "
                    "Options: fed,ecb,boj,boe,snb,boc,rba. Default: all."
                ),
            },
        },
        "required": ["mode"],
    }

    def __init__(
        self,
        fred_api_key: str = "",
        cache: DataCache | None = None,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._api_key = fred_api_key
        self._cache = cache
        self._store = pipeline_store

    # ── Public entry point ─────────────────────────────────────

    def execute(
        self,
        *,
        mode: str = "",
        period: str = "1y",
        banks: str = "",
        **_: Any,
    ) -> ToolResult:
        mode = (mode or "").strip().lower()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}",
            )

        if not self._api_key:
            return ToolResult(
                success=False,
                output=(
                    "FRED API key not configured. Set TIRRA_FRED_API_KEY in .env. "
                    "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
                ),
            )

        period = (period or "1y").strip().lower()
        if period not in _PERIOD_DAYS:
            period = "1y"

        # Parse bank filter
        if banks:
            bank_list = [b.strip().lower() for b in banks.split(",") if b.strip()]
            bank_list = [b for b in bank_list if b in CB_REGISTRY]
            if not bank_list:
                return ToolResult(
                    success=False,
                    output=f"No valid banks. Options: {sorted(CB_REGISTRY.keys())}",
                )
        else:
            bank_list = list(CB_REGISTRY.keys())

        dispatch = {
            "balance_sheets": self._mode_balance_sheets,
            "liquidity_index": self._mode_liquidity_index,
            "policy_divergence": self._mode_policy_divergence,
            "rate_monitor": self._mode_rate_monitor,
        }
        result = dispatch[mode](period, bank_list)

        # L2: persist monetary-state observations onto country entities (Phase 27)
        if result.success and result.data:
            self._persist_entities(result.data, mode, bank_list)

        return result

    # ── Mode: balance_sheets ───────────────────────────────────

    def _mode_balance_sheets(
        self,
        period: str,
        bank_list: list[str],
    ) -> ToolResult:
        """Snapshot of all CB balance sheets with USD normalization."""
        now = datetime.now(timezone.utc)
        days = _PERIOD_DAYS[period]
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        # Fetch FX rates first
        fx_rates = self._fetch_fx_rates()

        rows: list[dict[str, Any]] = []
        errors: list[str] = []

        for cb_code in bank_list:
            cb = CB_REGISTRY[cb_code]
            if cb.get("_skip_bs") == "true":
                errors.append(f"{cb['name']}: FRED series unavailable, skipped")
                continue

            series = self._fetch_fred_observations(
                cb["bs_series"], start, end, f"bs_{cb_code}", _CACHE_TTL_BS
            )
            if not series:
                errors.append(f"{cb['name']}: no data returned for {cb['bs_series']}")
                continue

            # Latest value
            latest = series[-1]
            latest_val = float(latest["value"])
            scale = float(cb["unit_scale"])
            native_usd = latest_val * scale  # Convert to base currency units

            # USD conversion
            usd_val = self._to_usd(native_usd, cb_code, fx_rates)

            # Compute changes
            changes = self._compute_changes(series, scale)

            rows.append(
                {
                    "bank": cb["name"],
                    "code": cb_code,
                    "currency": cb["currency"],
                    "latest_date": latest["date"],
                    "native_trillions": round(native_usd / 1e12, 3),
                    "usd_trillions": round(usd_val / 1e12, 3) if usd_val else None,
                    "wow_pct": changes.get("wow"),
                    "mom_pct": changes.get("mom"),
                    "yoy_pct": changes.get("yoy"),
                }
            )

        # Format output
        lines = ["# Central Bank Balance Sheets\n"]
        for r in rows:
            direction = ""
            if r.get("mom_pct") is not None:
                direction = (
                    " ↑"
                    if r["mom_pct"] > 0.5
                    else (" ↓" if r["mom_pct"] < -0.5 else " →")
                )
            usd_str = f"${r['usd_trillions']}T" if r["usd_trillions"] else "N/A"
            lines.append(
                f"**{r['bank']}** ({r['currency']})\n"
                f"  Level: {r['native_trillions']}T {r['currency']} "
                f"({usd_str} USD){direction}\n"
                f"  Date: {r['latest_date']}\n"
                f"  Changes — WoW: {_fmt_pct(r.get('wow_pct'))} | "
                f"MoM: {_fmt_pct(r.get('mom_pct'))} | "
                f"YoY: {_fmt_pct(r.get('yoy_pct'))}\n"
            )

        if errors:
            lines.append("\n**Notes:**")
            for e in errors:
                lines.append(f"  - {e}")

        return ToolResult(
            success=bool(rows),
            output="\n".join(lines),
            data={"banks": rows, "errors": errors},
        )

    # ── Mode: liquidity_index ──────────────────────────────────

    def _mode_liquidity_index(
        self,
        period: str,
        bank_list: list[str],
    ) -> ToolResult:
        """Global net liquidity = sum(CB assets in USD) - RRP - TGA."""
        now = datetime.now(timezone.utc)
        days = _PERIOD_DAYS[period]
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        fx_rates = self._fetch_fx_rates()

        # Fetch CB balance sheet latest values
        cb_totals: dict[str, float] = {}
        cb_details: list[dict[str, Any]] = []
        errors: list[str] = []

        for cb_code in bank_list:
            cb = CB_REGISTRY[cb_code]
            if cb.get("_skip_bs") == "true":
                continue
            series = self._fetch_fred_observations(
                cb["bs_series"], start, end, f"bs_{cb_code}", _CACHE_TTL_BS
            )
            if not series:
                errors.append(f"{cb['name']}: no data")
                continue

            latest_val = float(series[-1]["value"]) * float(cb["unit_scale"])
            usd_val = self._to_usd(latest_val, cb_code, fx_rates)
            if usd_val:
                cb_totals[cb_code] = usd_val
                cb_details.append(
                    {
                        "bank": cb["name"],
                        "usd_trillions": round(usd_val / 1e12, 3),
                    }
                )

        gross = sum(cb_totals.values())

        # Fetch drain factors (Fed-specific)
        rrp_val = 0.0
        tga_val = 0.0

        rrp_series = self._fetch_fred_observations(
            _FED_RRP_SERIES, start, end, "drain_rrp", _CACHE_TTL_BS
        )
        if rrp_series:
            # RRPONTSYD is in billions USD
            rrp_val = float(rrp_series[-1]["value"]) * 1e9

        tga_series = self._fetch_fred_observations(
            _FED_TGA_SERIES, start, end, "drain_tga", _CACHE_TTL_BS
        )
        if tga_series:
            # WDTGAL is in millions USD
            tga_val = float(tga_series[-1]["value"]) * 1e6

        net = gross - rrp_val - tga_val

        # Build time series of gross liquidity (simplified: use available dates)
        # For a proper time series we'd need date-aligned data; for now show latest
        lines = [
            "# Global Liquidity Index\n",
            f"**Gross Liquidity:** ${gross / 1e12:.3f}T USD",
            f"**Fed Reverse Repo (drain):** -${rrp_val / 1e12:.3f}T USD",
            f"**Fed TGA (drain):** -${tga_val / 1e12:.3f}T USD",
            f"**Net Liquidity:** ${net / 1e12:.3f}T USD\n",
            "## Composition:",
        ]
        for d in cb_details:
            pct = (d["usd_trillions"] * 1e12 / gross * 100) if gross else 0
            lines.append(f"  {d['bank']}: ${d['usd_trillions']}T ({pct:.1f}%)")

        if errors:
            lines.append("\n**Notes:**")
            for e in errors:
                lines.append(f"  - {e}")

        return ToolResult(
            success=bool(cb_totals),
            output="\n".join(lines),
            data={
                "gross_usd": gross,
                "rrp_usd": rrp_val,
                "tga_usd": tga_val,
                "net_usd": net,
                "components": cb_details,
                "errors": errors,
            },
        )

    # ── Mode: policy_divergence ────────────────────────────────

    def _mode_policy_divergence(
        self,
        period: str,
        bank_list: list[str],
    ) -> ToolResult:
        """Who's expanding vs contracting + rate differentials."""
        now = datetime.now(timezone.utc)
        days = _PERIOD_DAYS[period]
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        # Also need a short lookback for recent growth
        start_3m = (now - timedelta(days=90)).strftime("%Y-%m-%d")
        start_12m = (now - timedelta(days=365)).strftime("%Y-%m-%d")

        assessments: list[dict[str, Any]] = []
        errors: list[str] = []

        for cb_code in bank_list:
            cb = CB_REGISTRY[cb_code]
            if cb.get("_skip_bs") == "true":
                continue

            # Fetch full period for growth rate calculation
            series = self._fetch_fred_observations(
                cb["bs_series"], start_12m, end, f"bs_{cb_code}", _CACHE_TTL_BS
            )
            if not series or len(series) < 2:
                errors.append(f"{cb['name']}: insufficient data")
                continue

            scale = float(cb["unit_scale"])
            latest_val = float(series[-1]["value"]) * scale

            # Find values at ~3m and ~12m ago for growth rates
            growth_3m = self._compute_growth_rate(series, 90, scale)
            growth_12m = self._compute_growth_rate(series, 365, scale)

            # Classify stance
            stance = "stable"
            if growth_12m is not None:
                if growth_12m > 2.0:
                    stance = "expanding"
                elif growth_12m < -2.0:
                    stance = "contracting"

            assessments.append(
                {
                    "bank": cb["name"],
                    "code": cb_code,
                    "growth_3m_ann": growth_3m,
                    "growth_12m": growth_12m,
                    "stance": stance,
                }
            )

        # Fetch policy rates
        rate_data: dict[str, float | None] = {}
        for cb_code in bank_list:
            rate = self._fetch_policy_rate(cb_code, start, end)
            if rate is not None:
                rate_data[cb_code] = rate

        # Detect divergence pairs
        divergences: list[str] = []
        expanding = [a for a in assessments if a["stance"] == "expanding"]
        contracting = [a for a in assessments if a["stance"] == "contracting"]
        for exp in expanding:
            for con in contracting:
                divergences.append(
                    f"{exp['bank']} EXPANDING vs {con['bank']} CONTRACTING"
                )

        # Synchronized?
        all_stances = {a["stance"] for a in assessments}
        sync_signal = ""
        if len(all_stances) == 1 and assessments:
            sync_signal = f"SYNCHRONIZED {assessments[0]['stance'].upper()}"

        # Format output
        lines = ["# Policy Divergence Analysis\n"]

        if sync_signal:
            lines.append(f"**⚠ {sync_signal}** — All major CBs in same direction\n")

        for a in assessments:
            arrow = {"expanding": "↑", "contracting": "↓", "stable": "→"}[a["stance"]]
            lines.append(
                f"**{a['bank']}** {arrow} {a['stance'].upper()}\n"
                f"  3M growth (ann.): {_fmt_pct(a['growth_3m_ann'])} | "
                f"12M growth: {_fmt_pct(a['growth_12m'])}"
            )
            if a["code"] in rate_data:
                lines.append(f"  Policy rate: {rate_data[a['code']]:.2f}%")
            lines.append("")

        if divergences:
            lines.append("## Divergence Pairs:")
            for d in divergences:
                lines.append(f"  - {d}")

        # Rate differentials
        if len(rate_data) >= 2:
            lines.append("\n## Rate Differentials:")
            codes = sorted(rate_data.keys())
            for i in range(len(codes)):
                for j in range(i + 1, len(codes)):
                    c1, c2 = codes[i], codes[j]
                    diff = rate_data[c1] - rate_data[c2]
                    n1 = CB_REGISTRY[c1]["name"]
                    n2 = CB_REGISTRY[c2]["name"]
                    lines.append(f"  {n1} - {n2}: {diff:+.2f}%")

        if errors:
            lines.append("\n**Notes:**")
            for e in errors:
                lines.append(f"  - {e}")

        return ToolResult(
            success=bool(assessments),
            output="\n".join(lines),
            data={
                "assessments": assessments,
                "rates": rate_data,
                "divergences": divergences,
                "synchronized": sync_signal,
                "errors": errors,
            },
        )

    # ── Mode: rate_monitor ─────────────────────────────────────

    def _mode_rate_monitor(
        self,
        period: str,
        bank_list: list[str],
    ) -> ToolResult:
        """Current policy rates + last change detection."""
        now = datetime.now(timezone.utc)
        days = _PERIOD_DAYS[period]
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for cb_code in bank_list:
            cb = CB_REGISTRY[cb_code]
            rate_series_id = cb.get("rate_series", "")
            if not rate_series_id:
                continue

            # Special handling for ECB (fetched from ECB SDW)
            if rate_series_id == "_ECB_DFR":
                rate_info = self._fetch_ecb_rate(start, end)
                if rate_info:
                    results.append(
                        {
                            "bank": cb["name"],
                            "code": cb_code,
                            **rate_info,
                        }
                    )
                else:
                    errors.append(f"{cb['name']}: ECB rate data unavailable")
                continue

            # FRED rate series
            series = self._fetch_fred_observations(
                rate_series_id, start, end, f"rate_{cb_code}", _CACHE_TTL_RATES
            )
            if not series:
                errors.append(f"{cb['name']}: no rate data for {rate_series_id}")
                continue

            # Current rate
            current_rate = float(series[-1]["value"])
            current_date = series[-1]["date"]

            # Detect last change
            last_change = self._detect_rate_change(series)

            results.append(
                {
                    "bank": cb["name"],
                    "code": cb_code,
                    "current_rate": current_rate,
                    "rate_date": current_date,
                    "last_change_date": last_change.get("date"),
                    "last_change_direction": last_change.get("direction"),
                    "last_change_bps": last_change.get("bps"),
                    "days_since_change": last_change.get("days_since"),
                }
            )

        # Format
        lines = ["# Policy Rate Monitor\n"]
        for r in results:
            recent_flag = ""
            if r.get("days_since_change") is not None and r["days_since_change"] < 30:
                recent_flag = " **[RECENT CHANGE]**"
            lines.append(
                f"**{r['bank']}**: {r['current_rate']:.2f}%{recent_flag}\n"
                f"  As of: {r['rate_date']}"
            )
            if r.get("last_change_date"):
                direction = r.get("last_change_direction", "")
                bps = r.get("last_change_bps", 0)
                days = r.get("days_since_change", "?")
                lines.append(
                    f"  Last change: {r['last_change_date']} "
                    f"({direction} {abs(bps):.0f}bp, {days} days ago)"
                )
            lines.append("")

        if errors:
            lines.append("**Notes:**")
            for e in errors:
                lines.append(f"  - {e}")

        return ToolResult(
            success=bool(results),
            output="\n".join(lines),
            data={"rates": results, "errors": errors},
        )

    # ── FRED helpers ───────────────────────────────────────────

    def _fetch_fred_observations(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
        cache_key: str,
        cache_ttl: int,
    ) -> list[dict[str, str]]:
        """Fetch FRED observations. Returns list of {date, value} dicts.

        Filters out missing values (FRED marks them with '.').
        """
        cache_params = {
            "series_id": series_id,
            "start": start_date,
            "end": end_date,
        }
        cached = self._cache.get("cb_balance", cache_params) if self._cache else None
        if cached is not None:
            return cached

        try:
            params = {
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
            }
            with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}) as client:
                resp = client.get(_FRED_BASE, params=params)
                resp.raise_for_status()

            data = resp.json()
            observations = data.get("observations", [])
            # Filter out missing values
            clean = [
                {"date": obs["date"], "value": obs["value"]}
                for obs in observations
                if obs.get("value") not in (".", "", None)
            ]

            if self._cache and clean:
                self._cache.put("cb_balance", cache_params, clean)

            return clean

        except Exception:
            log.exception("FRED fetch failed for %s", series_id)
            return []

    def _fetch_fx_rates(self) -> dict[str, float]:
        """Fetch latest FX rates from FRED. Returns {series_id: rate}."""
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        fx_series = set()
        for cb in CB_REGISTRY.values():
            if cb.get("fx_series"):
                fx_series.add(cb["fx_series"])

        rates: dict[str, float] = {}
        for series_id in fx_series:
            obs = self._fetch_fred_observations(
                series_id, start, end, f"fx_{series_id}", _CACHE_TTL_FX
            )
            if obs:
                rates[series_id] = float(obs[-1]["value"])

        return rates

    def _to_usd(
        self,
        amount: float,
        cb_code: str,
        fx_rates: dict[str, float],
    ) -> float | None:
        """Convert native currency amount to USD."""
        cb = CB_REGISTRY[cb_code]
        if cb["currency"] == "USD":
            return amount

        fx_series = cb.get("fx_series", "")
        if not fx_series or fx_series not in fx_rates:
            return None

        rate = fx_rates[fx_series]
        if rate == 0:
            return None

        if fx_series in _USD_PER_FOREIGN:
            # Rate is USD per unit of foreign currency → multiply
            return amount * rate
        elif fx_series in _FOREIGN_PER_USD:
            # Rate is foreign per USD → divide
            return amount / rate

        return None

    # ── Rate helpers ───────────────────────────────────────────

    def _fetch_policy_rate(
        self,
        cb_code: str,
        start: str,
        end: str,
    ) -> float | None:
        """Fetch the latest policy rate for a CB."""
        cb = CB_REGISTRY[cb_code]
        rate_series = cb.get("rate_series", "")
        if not rate_series:
            return None

        if rate_series == "_ECB_DFR":
            info = self._fetch_ecb_rate(start, end)
            return info.get("current_rate") if info else None

        series = self._fetch_fred_observations(
            rate_series, start, end, f"rate_{cb_code}", _CACHE_TTL_RATES
        )
        if series:
            return float(series[-1]["value"])
        return None

    def _fetch_ecb_rate(
        self,
        start: str,
        end: str,
    ) -> dict[str, Any] | None:
        """Fetch ECB deposit facility rate from ECB SDW."""
        cache_params = {"key": _ECB_DFR_KEY, "start": start, "end": end}
        cached = self._cache.get("cb_ecb_rate", cache_params) if self._cache else None
        if cached is not None:
            return cached

        try:
            url = f"{_ECB_BASE}/{_ECB_DFR_KEY}"
            params = {
                "format": "jsondata",
                "startPeriod": start,
                "endPeriod": end,
            }
            with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()

            data = resp.json()
            observations = _parse_ecb_observations(data)
            if not observations:
                return None

            latest = observations[-1]
            current_rate = latest["value"]

            # Detect last change
            last_change = self._detect_rate_change_from_obs(observations)

            result = {
                "current_rate": current_rate,
                "rate_date": latest["date"],
                "last_change_date": last_change.get("date"),
                "last_change_direction": last_change.get("direction"),
                "last_change_bps": last_change.get("bps"),
                "days_since_change": last_change.get("days_since"),
            }

            if self._cache:
                self._cache.put("cb_ecb_rate", cache_params, result)

            return result

        except Exception:
            log.exception("ECB rate fetch failed")
            return None

    def _detect_rate_change(
        self,
        series: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Find the most recent rate level change in a FRED series."""
        observations = [{"date": s["date"], "value": float(s["value"])} for s in series]
        return self._detect_rate_change_from_obs(observations)

    def _detect_rate_change_from_obs(
        self,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Find the most recent rate level change in observations."""
        if len(observations) < 2:
            return {}

        now = datetime.now(timezone.utc)
        # Walk backwards to find the last change
        for i in range(len(observations) - 1, 0, -1):
            curr = observations[i]["value"]
            prev = observations[i - 1]["value"]
            if abs(curr - prev) > 0.001:  # Rate changed
                change_bps = (curr - prev) * 100
                direction = "hike" if change_bps > 0 else "cut"
                change_date = observations[i]["date"]
                try:
                    dt = datetime.strptime(change_date, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
                    days_since = (now - dt).days
                except (ValueError, TypeError):
                    days_since = None

                return {
                    "date": change_date,
                    "direction": direction,
                    "bps": round(change_bps, 1),
                    "days_since": days_since,
                }

        return {}

    # ── L2 entity persistence (Phase 27) ──────────────────────

    def _persist_entities(
        self,
        data: dict[str, Any],
        mode: str,
        bank_list: list[str],
    ) -> dict[str, int]:
        """Persist monetary-state observations onto country entities.

        Observation families:
        - ``cb_balance_sheet``: balance sheet level/changes on country nodes.
        - ``cb_policy_rate``: policy rate state on country nodes.

        Skips silently if no PipelineStore or entity module is available.
        Returns counts: {balance_sheet_obs, rate_obs}.
        """
        if self._store is None or _entity_id_from_key is None:
            return {"balance_sheet_obs": 0, "rate_obs": 0}
        try:
            return self._persist_entities_inner(data, mode, bank_list)
        except Exception:
            log.exception("CB entity persistence failed (non-fatal)")
            return {"balance_sheet_obs": 0, "rate_obs": 0}

    def _persist_entities_inner(
        self,
        data: dict[str, Any],
        mode: str,
        bank_list: list[str],
    ) -> dict[str, int]:
        """Inner persistence logic separated for testability."""
        assert self._store is not None  # noqa: S101 — guarded by caller
        assert _entity_id_from_key is not None  # noqa: S101

        store = self._store
        counts = {"balance_sheet_obs": 0, "rate_obs": 0}
        now_ts = time.time()

        # ── Balance sheet observations ──
        # Available from balance_sheets, liquidity_index, policy_divergence modes
        if mode == "balance_sheets":
            for bank_row in data.get("banks", []):
                cb_code = bank_row.get("code", "")
                country_code = CB_TO_COUNTRY.get(cb_code)
                if not country_code:
                    continue

                country_eid = _entity_id_from_key("country", country_code)
                store.register_entity(
                    entity_type="country",
                    canonical_name=country_code,
                    entity_id=country_eid,
                )

                obs_value = {
                    "cb_code": cb_code,
                    "native_trillions": bank_row.get("native_trillions"),
                    "usd_trillions": bank_row.get("usd_trillions"),
                    "wow_pct": bank_row.get("wow_pct"),
                    "mom_pct": bank_row.get("mom_pct"),
                    "yoy_pct": bank_row.get("yoy_pct"),
                }
                store.store_entity_observation(
                    entity_id=country_eid,
                    source_tool="central_bank_balance",
                    observed_at=now_ts,
                    observation_type="cb_balance_sheet",
                    value=obs_value,
                    depth_level=2,
                )
                counts["balance_sheet_obs"] += 1

        elif mode == "policy_divergence":
            for assessment in data.get("assessments", []):
                cb_code = assessment.get("code", "")
                country_code = CB_TO_COUNTRY.get(cb_code)
                if not country_code:
                    continue

                country_eid = _entity_id_from_key("country", country_code)
                store.register_entity(
                    entity_type="country",
                    canonical_name=country_code,
                    entity_id=country_eid,
                )

                obs_value = {
                    "cb_code": cb_code,
                    "growth_3m_ann": assessment.get("growth_3m_ann"),
                    "growth_12m": assessment.get("growth_12m"),
                    "stance": assessment.get("stance"),
                }
                store.store_entity_observation(
                    entity_id=country_eid,
                    source_tool="central_bank_balance",
                    observed_at=now_ts,
                    observation_type="cb_balance_sheet",
                    value=obs_value,
                    depth_level=2,
                )
                counts["balance_sheet_obs"] += 1

        # ── Policy rate observations ──
        # Available from rate_monitor mode and from policy_divergence (rates dict)
        if mode == "rate_monitor":
            for rate_row in data.get("rates", []):
                cb_code = rate_row.get("code", "")
                country_code = CB_TO_COUNTRY.get(cb_code)
                if not country_code:
                    continue

                country_eid = _entity_id_from_key("country", country_code)
                store.register_entity(
                    entity_type="country",
                    canonical_name=country_code,
                    entity_id=country_eid,
                )

                obs_value = {
                    "cb_code": cb_code,
                    "current_rate": rate_row.get("current_rate"),
                    "last_change_date": rate_row.get("last_change_date"),
                    "last_change_direction": rate_row.get("last_change_direction"),
                    "last_change_bps": rate_row.get("last_change_bps"),
                    "days_since_change": rate_row.get("days_since_change"),
                }
                store.store_entity_observation(
                    entity_id=country_eid,
                    source_tool="central_bank_balance",
                    observed_at=now_ts,
                    observation_type="cb_policy_rate",
                    value=obs_value,
                    depth_level=2,
                )
                counts["rate_obs"] += 1

        elif mode == "policy_divergence":
            # policy_divergence also has rate data as a dict
            for cb_code, rate_val in data.get("rates", {}).items():
                country_code = CB_TO_COUNTRY.get(cb_code)
                if not country_code:
                    continue

                country_eid = _entity_id_from_key("country", country_code)
                store.register_entity(
                    entity_type="country",
                    canonical_name=country_code,
                    entity_id=country_eid,
                )

                obs_value = {
                    "cb_code": cb_code,
                    "current_rate": rate_val,
                }
                store.store_entity_observation(
                    entity_id=country_eid,
                    source_tool="central_bank_balance",
                    observed_at=now_ts,
                    observation_type="cb_policy_rate",
                    value=obs_value,
                    depth_level=2,
                )
                counts["rate_obs"] += 1

        if counts["balance_sheet_obs"] or counts["rate_obs"]:
            log.info(
                "CB L2 persistence: %d balance_sheet obs, %d rate obs",
                counts["balance_sheet_obs"],
                counts["rate_obs"],
            )
        return counts

    # ── Growth rate helpers ────────────────────────────────────

    @staticmethod
    def _compute_changes(
        series: list[dict[str, str]],
        scale: float,
    ) -> dict[str, float | None]:
        """Compute WoW, MoM, YoY percentage changes from a time series."""
        if not series:
            return {}

        latest_val = float(series[-1]["value"]) * scale
        result: dict[str, float | None] = {}

        # Find comparison points by approximate days back
        for label, target_days in [("wow", 7), ("mom", 30), ("yoy", 365)]:
            comp_val = _find_value_n_days_back(series, target_days, scale)
            if comp_val is not None and comp_val != 0:
                result[label] = round((latest_val - comp_val) / comp_val * 100, 2)
            else:
                result[label] = None

        return result

    @staticmethod
    def _compute_growth_rate(
        series: list[dict[str, str]],
        days_back: int,
        scale: float,
    ) -> float | None:
        """Compute annualized growth rate over given period."""
        if not series:
            return None

        latest_val = float(series[-1]["value"]) * scale
        comp_val = _find_value_n_days_back(series, days_back, scale)
        if comp_val is None or comp_val == 0:
            return None

        raw_pct = (latest_val - comp_val) / comp_val * 100
        # Annualize if period < 365 days
        if days_back < 365:
            annualized = raw_pct * (365 / days_back)
            return round(annualized, 2)
        return round(raw_pct, 2)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _find_value_n_days_back(
    series: list[dict[str, str]],
    target_days: int,
    scale: float,
) -> float | None:
    """Find the observation closest to N days before the last observation."""
    if len(series) < 2:
        return None

    try:
        last_date = datetime.strptime(series[-1]["date"], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None

    target_date = last_date - timedelta(days=target_days)
    best_obs = None
    best_diff = float("inf")

    for obs in series:
        try:
            obs_date = datetime.strptime(obs["date"], "%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        diff = abs((obs_date - target_date).days)
        # Don't use data that's too far from target (within 20% or 30 days)
        max_tolerance = max(target_days * 0.2, 30)
        if diff < best_diff and diff <= max_tolerance:
            best_diff = diff
            best_obs = obs

    if best_obs is None:
        return None
    return float(best_obs["value"]) * scale


def _parse_ecb_observations(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse ECB SDMX JSON response into list of {date, value} dicts."""
    try:
        datasets = data.get("dataSets", [])
        if not datasets:
            return []

        structure = data.get("structure", {})
        dimensions = structure.get("dimensions", {}).get("observation", [])
        if not dimensions:
            return []

        # Get time periods from dimension values
        time_dim = dimensions[0]
        time_values = time_dim.get("values", [])

        # Get the first (usually only) series
        series_data = datasets[0].get("series", {})
        if not series_data:
            return []

        first_key = next(iter(series_data))
        observations = series_data[first_key].get("observations", {})

        result = []
        for idx_str, obs_array in sorted(observations.items(), key=lambda x: int(x[0])):
            idx = int(idx_str)
            if idx < len(time_values) and obs_array:
                date_str = time_values[idx].get("id", "")
                value = obs_array[0]
                if value is not None:
                    result.append({"date": date_str, "value": float(value)})

        return result

    except (KeyError, IndexError, TypeError, ValueError):
        log.exception("Failed to parse ECB SDMX response")
        return []


def _fmt_pct(val: float | None) -> str:
    """Format a percentage value for display."""
    if val is None:
        return "N/A"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}%"
