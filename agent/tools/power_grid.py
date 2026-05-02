"""
Tool: Power Grid Demand (NYISO)

Fetch power grid data from NYISO's free public CSV files.
Four modes:
  1. demand    — Actual load by zone (5-min resolution)
  2. fuel_mix  — Generation by fuel type with proportions
  3. pricing   — Day-ahead and real-time LBMPs with spread
  4. forecast  — Load forecast vs actual with deviation %

Signals: demand-forecast deviation (economic activity proxy),
DA-RT price spread (congestion/stress), fuel mix shift (energy cost),
zone-level anomalies.

Data source: http://mis.nyiso.com/public/csv/ (zero cost, no auth).
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import date, datetime
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_USER_AGENT = "TirraMind/0.1 (research; https://github.com/tirramind)"
_BASE_URL = "http://mis.nyiso.com/public/csv"
_TIMEOUT = 20
_ZIP_TIMEOUT = 60

# All valid NYISO load zones
NYISO_ZONES = [
    "CAPITL",
    "CENTRL",
    "DUNWOD",
    "GENESE",
    "HUD VL",
    "LONGIL",
    "MHK VL",
    "MILLWD",
    "N.Y.C.",
    "NORTH",
    "WEST",
]

# Dataset keys mapping
_DATASETS = {
    "demand": "pal",
    "fuel_mix": "rtfuelmix",
    "da_pricing": "damlbmp",
    "rt_pricing": "realtime",
    "forecast": "isolf",
}


class PowerGridTool(Tool):
    """NYISO power grid demand, fuel mix, pricing, and forecast data."""

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    @property
    def name(self) -> str:
        return "power_grid"

    @property
    def description(self) -> str:
        return (
            "Fetch NYISO power grid data. Modes: "
            "'demand' for actual load by zone (5-min, MW), "
            "'fuel_mix' for generation by fuel type with proportions, "
            "'pricing' for day-ahead and real-time LBMPs ($/MWh) with DA-RT spread, "
            "'forecast' for load forecast vs actual with deviation %. "
            "Covers 11 New York zones. Free, no API key."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["demand", "fuel_mix", "pricing", "forecast"],
                    "description": (
                        "Data mode: 'demand' (actual load), 'fuel_mix' (generation by fuel), "
                        "'pricing' (DA/RT LBMPs), 'forecast' (forecast vs actual deviation)"
                    ),
                },
                "zone": {
                    "type": "string",
                    "description": ("NYISO zone filter (e.g. 'N.Y.C.', 'CAPITL'). Omit for all zones."),
                },
                "date": {
                    "type": "string",
                    "description": "Date YYYY-MM-DD. Default: today.",
                },
            },
            "required": ["mode"],
        }

    # ── Public execute ────────────────────────────────────────────

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = kwargs.get("mode", "")
        if mode not in ("demand", "fuel_mix", "pricing", "forecast"):
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: demand, fuel_mix, pricing, forecast",
            )

        zone = kwargs.get("zone")
        if zone is not None:
            zone = zone.strip().upper()
            # Handle common variations
            zone = self._normalize_zone(zone)
            if zone not in NYISO_ZONES:
                return ToolResult(
                    success=False,
                    output=f"Unknown zone '{zone}'. Valid: {', '.join(NYISO_ZONES)}",
                )

        date_str = kwargs.get("date")
        if date_str:
            try:
                parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
                if parsed > date.today():
                    return ToolResult(
                        success=False,
                        output=f"Date {date_str} is in the future.",
                    )
            except ValueError:
                return ToolResult(
                    success=False,
                    output=f"Invalid date format '{date_str}'. Use YYYY-MM-DD.",
                )
        else:
            date_str = date.today().strftime("%Y-%m-%d")

        try:
            if mode == "demand":
                return self._demand(date_str, zone)
            elif mode == "fuel_mix":
                return self._fuel_mix(date_str, zone)
            elif mode == "pricing":
                return self._pricing(date_str, zone)
            else:
                return self._forecast(date_str, zone)
        except httpx.TimeoutException:
            return ToolResult(success=False, output="NYISO request timed out.")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"HTTP error: {exc}")
        except Exception as exc:
            log.exception("PowerGridTool error")
            return ToolResult(success=False, output=f"Unexpected error: {exc}")

    # ── Mode implementations ──────────────────────────────────────

    def _demand(self, date_str: str, zone: str | None) -> ToolResult:
        rows = self._fetch_csv("pal", date_str)
        if rows is None:
            return ToolResult(
                success=False,
                output=f"No demand data available for {date_str}.",
            )
        if not rows:
            return ToolResult(
                success=False,
                output=f"Empty demand data for {date_str}.",
            )

        # Filter by zone
        filtered = self._filter_by_zone(rows, "Name", zone)
        if not filtered:
            return ToolResult(
                success=False,
                output=f"No data for zone '{zone}' on {date_str}.",
            )

        # Group by zone, compute summary stats
        zone_data: dict[str, list[float]] = {}
        for row in filtered:
            z = row.get("Name", "").strip()
            load = _safe_float(row.get("Load", row.get("Integrated Load", "")))
            if z and load is not None:
                zone_data.setdefault(z, []).append(load)

        if not zone_data:
            return ToolResult(
                success=False,
                output=f"Could not parse load values for {date_str}.",
            )

        summaries = []
        total_peak = 0.0
        for z in sorted(zone_data.keys()):
            loads = zone_data[z]
            peak = max(loads)
            trough = min(loads)
            avg = sum(loads) / len(loads)
            total_peak += peak
            summaries.append(
                {
                    "zone": z,
                    "peak_mw": round(peak, 1),
                    "trough_mw": round(trough, 1),
                    "avg_mw": round(avg, 1),
                    "readings": len(loads),
                }
            )

        lines = [f"NYISO Demand — {date_str}", f"Zones: {len(summaries)}, Peak system: {round(total_peak, 0)} MW", ""]
        for s in summaries:
            lines.append(
                f"  {s['zone']:10s}  peak={s['peak_mw']:,.0f} MW  "
                f"trough={s['trough_mw']:,.0f} MW  avg={s['avg_mw']:,.0f} MW  "
                f"({s['readings']} readings)"
            )

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"date": date_str, "zones": summaries, "total_peak_mw": round(total_peak, 0)},
        )

    def _fuel_mix(self, date_str: str, zone: str | None) -> ToolResult:
        rows = self._fetch_csv("rtfuelmix", date_str)
        if rows is None:
            return ToolResult(
                success=False,
                output=f"No fuel mix data available for {date_str}.",
            )
        if not rows:
            return ToolResult(
                success=False,
                output=f"Empty fuel mix data for {date_str}.",
            )

        # Use latest timestamp snapshot
        timestamps = set()
        for row in rows:
            ts = row.get("Time Stamp", "").strip()
            if ts:
                timestamps.add(ts)

        if not timestamps:
            return ToolResult(success=False, output="No timestamps in fuel mix data.")

        latest_ts = max(timestamps)
        latest_rows = [r for r in rows if r.get("Time Stamp", "").strip() == latest_ts]

        # Zone filter if applicable
        if zone:
            latest_rows = [r for r in latest_rows if r.get("Zone Name", "").strip().upper() == zone]

        # Aggregate by fuel type
        fuel_totals: dict[str, float] = {}
        for row in latest_rows:
            fuel = row.get("Fuel Category", row.get("Gen Type", "")).strip()
            gen = _safe_float(row.get("Gen MWh", row.get("Gen MW", "")))
            if fuel and gen is not None:
                fuel_totals[fuel] = fuel_totals.get(fuel, 0.0) + gen

        total_gen = sum(fuel_totals.values())
        if total_gen == 0:
            return ToolResult(success=False, output=f"Zero total generation for {date_str}.")

        fuels = []
        for fuel_type in sorted(fuel_totals.keys()):
            mw = fuel_totals[fuel_type]
            pct = (mw / total_gen) * 100
            fuels.append(
                {
                    "fuel_type": fuel_type,
                    "mw": round(mw, 1),
                    "pct": round(pct, 1),
                }
            )

        # Sort by MW descending
        fuels.sort(key=lambda x: x["mw"], reverse=True)

        lines = [
            f"NYISO Fuel Mix — {date_str} (snapshot: {latest_ts})",
            f"Total generation: {round(total_gen, 0):,.0f} MW",
            "",
        ]
        for f in fuels:
            bar = "█" * int(f["pct"] / 2)
            lines.append(f"  {f['fuel_type']:20s} {f['mw']:>8,.0f} MW  {f['pct']:5.1f}%  {bar}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "date": date_str,
                "timestamp": latest_ts,
                "total_mw": round(total_gen, 0),
                "fuels": fuels,
            },
        )

    def _pricing(self, date_str: str, zone: str | None) -> ToolResult:
        da_rows = self._fetch_csv("damlbmp_zone", date_str, directory="damlbmp")
        rt_rows = self._fetch_csv("realtime_zone", date_str, directory="realtime")

        if da_rows is None and rt_rows is None:
            return ToolResult(
                success=False,
                output=f"No pricing data available for {date_str}.",
            )

        # Parse DA LBMPs — take latest hour per zone
        da_by_zone: dict[str, dict] = {}
        if da_rows:
            for row in da_rows:
                z = row.get("Name", "").strip()
                if zone and z.upper() != zone:
                    continue
                lbmp = _safe_float(row.get("LBMP ($/MWHr)", row.get("LBMP", "")))
                cong = _safe_float(row.get("Marginal Cost Congestion ($/MWHr)", ""))
                loss = _safe_float(row.get("Marginal Cost Losses ($/MWHr)", ""))
                ts = row.get("Time Stamp", "").strip()
                if z and lbmp is not None:
                    if z not in da_by_zone or ts > da_by_zone[z].get("ts", ""):
                        da_by_zone[z] = {
                            "ts": ts,
                            "lbmp": lbmp,
                            "congestion": cong,
                            "losses": loss,
                        }

        # Parse RT LBMPs — take latest 5-min per zone
        rt_by_zone: dict[str, dict] = {}
        if rt_rows:
            for row in rt_rows:
                z = row.get("Name", "").strip()
                if zone and z.upper() != zone:
                    continue
                lbmp = _safe_float(row.get("LBMP ($/MWHr)", row.get("LBMP", "")))
                cong = _safe_float(row.get("Marginal Cost Congestion ($/MWHr)", ""))
                loss = _safe_float(row.get("Marginal Cost Losses ($/MWHr)", ""))
                ts = row.get("Time Stamp", "").strip()
                if z and lbmp is not None:
                    if z not in rt_by_zone or ts > rt_by_zone[z].get("ts", ""):
                        rt_by_zone[z] = {
                            "ts": ts,
                            "lbmp": lbmp,
                            "congestion": cong,
                            "losses": loss,
                        }

        # Merge DA and RT
        all_zones = sorted(set(list(da_by_zone.keys()) + list(rt_by_zone.keys())))
        pricing = []
        for z in all_zones:
            da = da_by_zone.get(z)
            rt = rt_by_zone.get(z)
            da_price = da["lbmp"] if da else None
            rt_price = rt["lbmp"] if rt else None
            spread = None
            if da_price is not None and rt_price is not None:
                spread = round(rt_price - da_price, 2)
            pricing.append(
                {
                    "zone": z,
                    "da_lbmp": round(da_price, 2) if da_price is not None else None,
                    "rt_lbmp": round(rt_price, 2) if rt_price is not None else None,
                    "spread": spread,
                    "da_congestion": round(da["congestion"], 2) if da and da["congestion"] is not None else None,
                    "rt_congestion": round(rt["congestion"], 2) if rt and rt["congestion"] is not None else None,
                }
            )

        if not pricing:
            return ToolResult(success=False, output=f"No pricing zones parsed for {date_str}.")

        # Flag stressed zones
        stressed = [p for p in pricing if p["spread"] is not None and abs(p["spread"]) > 5.0]

        lines = [f"NYISO LBMPs — {date_str}", ""]
        lines.append(f"  {'Zone':12s} {'DA $/MWh':>10s} {'RT $/MWh':>10s} {'Spread':>10s} {'Cong(DA)':>10s}")
        lines.append("  " + "-" * 56)
        for p in pricing:
            da_s = f"${p['da_lbmp']:.2f}" if p["da_lbmp"] is not None else "n/a"
            rt_s = f"${p['rt_lbmp']:.2f}" if p["rt_lbmp"] is not None else "n/a"
            sp_s = f"${p['spread']:+.2f}" if p["spread"] is not None else "n/a"
            cg_s = f"${p['da_congestion']:.2f}" if p["da_congestion"] is not None else "n/a"
            flag = " ⚠" if p in stressed else ""
            lines.append(f"  {p['zone']:12s} {da_s:>10s} {rt_s:>10s} {sp_s:>10s} {cg_s:>10s}{flag}")

        if stressed:
            lines.append(f"\n⚠ {len(stressed)} zone(s) with |DA-RT spread| > $5/MWh")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "date": date_str,
                "zones": pricing,
                "stressed_zones": [s["zone"] for s in stressed],
            },
        )

    def _forecast(self, date_str: str, zone: str | None) -> ToolResult:
        forecast_rows = self._fetch_csv("isolf", date_str)
        actual_rows = self._fetch_csv("pal", date_str)

        if forecast_rows is None:
            return ToolResult(
                success=False,
                output=f"No forecast data available for {date_str}.",
            )
        if not forecast_rows:
            return ToolResult(
                success=False,
                output=f"Empty forecast data for {date_str}.",
            )

        # isolf CSV is columnar: "Time Stamp","Capitl","Centrl",...,"NYISO"
        # Build forecast per zone per hour
        fc_by_zone_hour: dict[str, dict[str, float]] = {}
        for row in forecast_rows:
            ts = row.get("Time Stamp", "").strip().strip('"')
            if not ts:
                continue
            for col_name, val_str in row.items():
                col = col_name.strip().strip('"')
                if col in ("Time Stamp", "Time Zone", "File Seq", ""):
                    continue
                val = _safe_float(val_str)
                if val is None:
                    continue
                # Normalize zone name to match NYISO_ZONES
                norm_zone = col.upper()
                if zone and norm_zone != zone and norm_zone != "NYISO":
                    continue
                fc_by_zone_hour.setdefault(norm_zone, {})[ts] = val

        # Build hourly actuals (aggregate 5-min into hourly) if available
        act_by_zone_hour: dict[str, dict[str, float]] = {}
        if actual_rows:
            for row in actual_rows:
                z = row.get("Name", "").strip().upper()
                if zone and z != zone:
                    continue
                ts = row.get("Time Stamp", "").strip().strip('"')
                load = _safe_float(row.get("Load", row.get("Integrated Load", "")))
                if z and load is not None and ts:
                    # Truncate to hour for matching (forecast has no seconds)
                    hour_ts = self._truncate_to_hour(ts)
                    if hour_ts:
                        bucket = act_by_zone_hour.setdefault(z, {})
                        if hour_ts not in bucket:
                            bucket[hour_ts] = []
                        bucket[hour_ts].append(load)

            # Convert lists to averages
            for z in act_by_zone_hour:
                for h in list(act_by_zone_hour[z].keys()):
                    vals = act_by_zone_hour[z][h]
                    if isinstance(vals, list) and vals:
                        act_by_zone_hour[z][h] = sum(vals) / len(vals)

        # Compute deviations
        results = []
        for z in sorted(fc_by_zone_hour.keys()):
            if z in ("", "NYISO"):
                continue  # Skip the total for per-zone analysis
            fc_hours = fc_by_zone_hour[z]
            act_hours = act_by_zone_hour.get(z, {})
            deviations = []
            for ts, fc_val in sorted(fc_hours.items()):
                act_val = act_hours.get(ts)
                dev_pct = None
                if act_val is not None and isinstance(act_val, (int, float)) and fc_val != 0:
                    dev_pct = round(((act_val - fc_val) / fc_val) * 100, 1)
                deviations.append(
                    {
                        "hour": ts,
                        "forecast_mw": round(fc_val, 0),
                        "actual_mw": round(act_val, 0) if isinstance(act_val, (int, float)) else None,
                        "deviation_pct": dev_pct,
                    }
                )

            significant = [d for d in deviations if d["deviation_pct"] is not None and abs(d["deviation_pct"]) > 5.0]
            avg_dev = None
            dev_vals = [d["deviation_pct"] for d in deviations if d["deviation_pct"] is not None]
            if dev_vals:
                avg_dev = round(sum(dev_vals) / len(dev_vals), 1)

            results.append(
                {
                    "zone": z,
                    "hours": len(deviations),
                    "avg_deviation_pct": avg_dev,
                    "significant_deviations": len(significant),
                    "deviations": deviations,
                }
            )

        if not results:
            # Fall back to just showing forecast
            lines = [f"NYISO Load Forecast — {date_str}", ""]
            for z in sorted(fc_by_zone_hour.keys()):
                fc = fc_by_zone_hour[z]
                if fc:
                    latest_ts = max(fc.keys())
                    lines.append(f"  {z}: {fc[latest_ts]:,.0f} MW (latest: {latest_ts})")
            return ToolResult(success=True, output="\n".join(lines), data={"date": date_str, "forecast_only": True})

        lines = [f"NYISO Forecast vs Actual — {date_str}", ""]
        for r in results:
            dev_str = f"avg dev: {r['avg_deviation_pct']:+.1f}%" if r["avg_deviation_pct"] is not None else "no actuals"
            sig_str = f", {r['significant_deviations']} sig. deviations" if r["significant_deviations"] else ""
            lines.append(f"  {r['zone']:10s} {r['hours']} hrs  {dev_str}{sig_str}")

        # Highlight zones with persistent deviation
        persistent = [r for r in results if r["avg_deviation_pct"] is not None and abs(r["avg_deviation_pct"]) > 3.0]
        if persistent:
            lines.append(f"\n⚠ {len(persistent)} zone(s) with |avg deviation| > 3%:")
            for p in persistent:
                direction = "OVER-consuming" if p["avg_deviation_pct"] > 0 else "UNDER-consuming"
                lines.append(f"    {p['zone']}: {p['avg_deviation_pct']:+.1f}% ({direction} vs forecast)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "date": date_str,
                "zones": [{k: v for k, v in r.items() if k != "deviations"} for r in results],
                "persistent_deviation_zones": [p["zone"] for p in persistent],
            },
        )

    # ── CSV fetch / parse ─────────────────────────────────────────

    def _fetch_csv(self, dataset: str, date_str: str, directory: str | None = None) -> list[dict] | None:
        """Fetch and parse a NYISO CSV. Returns list of row dicts, or None if unavailable.

        Args:
            dataset: File name key (e.g. 'pal', 'damlbmp_zone')
            date_str: Date in YYYY-MM-DD format
            directory: URL directory if different from dataset (e.g. 'damlbmp' for 'damlbmp_zone')
        """
        dir_name = directory or dataset
        cache_params = {"dataset": dataset, "date": date_str}

        # Check cache
        if self._cache:
            cached = self._cache.get("nyiso", cache_params)
            if cached is not None:
                return cached

        # Try daily CSV first
        date_nodash = date_str.replace("-", "")
        url = f"{_BASE_URL}/{dir_name}/{date_nodash}{dataset}.csv"

        rows = self._http_get_csv(url)

        if rows is None:
            # Daily expired — try monthly archive
            rows = self._fetch_from_archive(dataset, date_str, directory=dir_name)

        if rows is not None and self._cache:
            self._cache.put("nyiso", cache_params, rows)

        return rows

    def _http_get_csv(self, url: str) -> list[dict] | None:
        """GET a CSV URL and parse to list of dicts. Returns None on 404/error."""
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(url, headers={"User-Agent": _USER_AGENT})
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return self._parse_csv(resp.text)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise

    def _fetch_from_archive(self, dataset: str, date_str: str, directory: str | None = None) -> list[dict] | None:
        """Fetch from monthly ZIP archive and extract the specific day's CSV."""
        dir_name = directory or dataset
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        month_key = dt.strftime("%Y%m")
        date_nodash = date_str.replace("-", "")
        target_filename = f"{date_nodash}{dataset}.csv"

        # Check if we cached this month's archive already
        archive_cache_params = {"dataset": dataset, "archive": month_key}
        archive_data: dict[str, list[dict]] | None = None

        if self._cache:
            archive_data = self._cache.get("nyiso_archive", archive_cache_params)

        if archive_data is None:
            zip_url = f"{_BASE_URL}/{dir_name}/{month_key}01{dataset}_csv.zip"
            try:
                with httpx.Client(timeout=_ZIP_TIMEOUT) as client:
                    resp = client.get(zip_url, headers={"User-Agent": _USER_AGENT})
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    return None
                raise

            # Parse all CSVs from the ZIP
            archive_data = {}
            try:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    for name in zf.namelist():
                        if name.endswith(".csv"):
                            with zf.open(name) as f:
                                text = f.read().decode("utf-8", errors="replace")
                                archive_data[name.lower()] = self._parse_csv(text)
            except (zipfile.BadZipFile, KeyError) as exc:
                log.warning("Bad archive ZIP for %s/%s: %s", dataset, month_key, exc)
                return None

            if self._cache and archive_data:
                self._cache.put("nyiso_archive", archive_cache_params, archive_data)

        # Find the target day's CSV in the archive
        target_lower = target_filename.lower()
        if target_lower in archive_data:
            return archive_data[target_lower]

        # Some ZIPs use different naming — search for the date
        for name, rows in archive_data.items():
            if date_nodash in name:
                return rows

        return None

    def _parse_csv(self, text: str) -> list[dict]:
        """Parse CSV text to list of dicts. Handles NYISO's format quirks."""
        if not text or not text.strip():
            return []
        reader = csv.DictReader(io.StringIO(text))
        rows = []
        for row in reader:
            # Skip completely empty rows
            if all(v is None or v.strip() == "" for v in row.values()):
                continue
            rows.append(dict(row))
        return rows

    # ── Helpers ────────────────────────────────────────────────────

    def _filter_by_zone(self, rows: list[dict], zone_col: str, zone: str | None) -> list[dict]:
        """Filter rows by zone. Returns all rows if zone is None."""
        if zone is None:
            return rows
        return [r for r in rows if r.get(zone_col, "").strip().upper() == zone]

    @staticmethod
    def _normalize_zone(zone: str) -> str:
        """Normalize common zone name variations."""
        mappings = {
            "NYC": "N.Y.C.",
            "NEW YORK CITY": "N.Y.C.",
            "CAPITAL": "CAPITL",
            "CENTRAL": "CENTRL",
            "DUNWOODY": "DUNWOD",
            "GENESEE": "GENESE",
            "HUDSON VALLEY": "HUD VL",
            "HUDSON": "HUD VL",
            "LONG ISLAND": "LONGIL",
            "MOHAWK VALLEY": "MHK VL",
            "MOHAWK": "MHK VL",
            "MILLWOOD": "MILLWD",
        }
        return mappings.get(zone, zone)

    @staticmethod
    def _truncate_to_hour(ts: str) -> str | None:
        """Truncate a NYISO timestamp to the hour for forecast matching.
        Returns format matching isolf: 'MM/DD/YYYY HH:00' (no seconds).
        """
        try:
            for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
                try:
                    dt = datetime.strptime(ts.strip(), fmt)
                    return dt.strftime("%m/%d/%Y %H:00")
                except ValueError:
                    continue
            return None
        except Exception:
            return None


def _safe_float(val: Any) -> float | None:
    """Parse a float from a string, returning None on failure."""
    if val is None:
        return None
    try:
        s = str(val).strip().replace(",", "")
        if not s or s in ("-", "N/A", "n/a", ""):
            return None
        return float(s)
    except (ValueError, TypeError):
        return None
