"""
Tool: CFTC Commitments of Traders

Fetch and parse the CFTC Disaggregated Futures-Only report.
Provides latest weekly positioning + historical yearly data.
Computes quant signals: managed money net, producer/merchant net,
concentration, weekly flows.

Data source: https://www.cftc.gov (zero cost, public domain).
Schedule: Tuesday snapshot, Friday 3:30 PM ET release.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import datetime
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

try:
    from agent.pipeline.entity import entity_id_from_key
    from agent.pipeline.store import PipelineStore
except ImportError:  # pragma: no cover — optional dependency
    PipelineStore = None  # type: ignore[assignment,misc]
    entity_id_from_key = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_USER_AGENT = "TirraMind/0.1 (research; https://github.com/tirramind)"
_WEEKLY_URL = "https://www.cftc.gov/dea/newcot/f_disagg.txt"
_HIST_URL_TPL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"

# ── Authoritative 191-column header from historical ZIP ──────────────
# Weekly flat file has no headers; we apply these in the same order.
_HEADERS: list[str] = [
    "Market_and_Exchange_Names",
    "As_of_Date_In_Form_YYMMDD",
    "Report_Date_as_YYYY-MM-DD",
    "CFTC_Contract_Market_Code",
    "CFTC_Market_Code",
    "CFTC_Region_Code",
    "CFTC_Commodity_Code",
    # ── Open Interest + Positions (_All = all contracts) ─────────────
    "Open_Interest_All",
    "Prod_Merc_Positions_Long_All",
    "Prod_Merc_Positions_Short_All",
    "Swap_Positions_Long_All",
    "Swap__Positions_Short_All",
    "Swap__Positions_Spread_All",
    "M_Money_Positions_Long_All",
    "M_Money_Positions_Short_All",
    "M_Money_Positions_Spread_All",
    "Other_Rept_Positions_Long_All",
    "Other_Rept_Positions_Short_All",
    "Other_Rept_Positions_Spread_All",
    "Tot_Rept_Positions_Long_All",
    "Tot_Rept_Positions_Short_All",
    "NonRept_Positions_Long_All",
    "NonRept_Positions_Short_All",
    # ── Old crop ─────────────────────────────────────────────────────
    "Open_Interest_Old",
    "Prod_Merc_Positions_Long_Old",
    "Prod_Merc_Positions_Short_Old",
    "Swap_Positions_Long_Old",
    "Swap__Positions_Short_Old",
    "Swap__Positions_Spread_Old",
    "M_Money_Positions_Long_Old",
    "M_Money_Positions_Short_Old",
    "M_Money_Positions_Spread_Old",
    "Other_Rept_Positions_Long_Old",
    "Other_Rept_Positions_Short_Old",
    "Other_Rept_Positions_Spread_Old",
    "Tot_Rept_Positions_Long_Old",
    "Tot_Rept_Positions_Short_Old",
    "NonRept_Positions_Long_Old",
    "NonRept_Positions_Short_Old",
    # ── Other ────────────────────────────────────────────────────────
    "Open_Interest_Other",
    "Prod_Merc_Positions_Long_Other",
    "Prod_Merc_Positions_Short_Other",
    "Swap_Positions_Long_Other",
    "Swap__Positions_Short_Other",
    "Swap__Positions_Spread_Other",
    "M_Money_Positions_Long_Other",
    "M_Money_Positions_Short_Other",
    "M_Money_Positions_Spread_Other",
    "Other_Rept_Positions_Long_Other",
    "Other_Rept_Positions_Short_Other",
    "Other_Rept_Positions_Spread_Other",
    "Tot_Rept_Positions_Long_Other",
    "Tot_Rept_Positions_Short_Other",
    "NonRept_Positions_Long_Other",
    "NonRept_Positions_Short_Other",
    # ── Weekly changes ───────────────────────────────────────────────
    "Change_in_Open_Interest_All",
    "Change_in_Prod_Merc_Long_All",
    "Change_in_Prod_Merc_Short_All",
    "Change_in_Swap_Long_All",
    "Change_in_Swap_Short_All",
    "Change_in_Swap_Spread_All",
    "Change_in_M_Money_Long_All",
    "Change_in_M_Money_Short_All",
    "Change_in_M_Money_Spread_All",
    "Change_in_Other_Rept_Long_All",
    "Change_in_Other_Rept_Short_All",
    "Change_in_Other_Rept_Spread_All",
    "Change_in_Tot_Rept_Long_All",
    "Change_in_Tot_Rept_Short_All",
    "Change_in_NonRept_Long_All",
    "Change_in_NonRept_Short_All",
    # ── Pct of OI (_All) ────────────────────────────────────────────
    "Pct_of_Open_Interest_All",
    "Pct_of_OI_Prod_Merc_Long_All",
    "Pct_of_OI_Prod_Merc_Short_All",
    "Pct_of_OI_Swap_Long_All",
    "Pct_of_OI_Swap_Short_All",
    "Pct_of_OI_Swap_Spread_All",
    "Pct_of_OI_M_Money_Long_All",
    "Pct_of_OI_M_Money_Short_All",
    "Pct_of_OI_M_Money_Spread_All",
    "Pct_of_OI_Other_Rept_Long_All",
    "Pct_of_OI_Other_Rept_Short_All",
    "Pct_of_OI_Other_Rept_Spread_All",
    "Pct_of_OI_Tot_Rept_Long_All",
    "Pct_of_OI_Tot_Rept_Short_All",
    "Pct_of_OI_NonRept_Long_All",
    "Pct_of_OI_NonRept_Short_All",
    # ── Pct of OI (Old) ─────────────────────────────────────────────
    "Pct_of_Open_Interest_Old",
    "Pct_of_OI_Prod_Merc_Long_Old",
    "Pct_of_OI_Prod_Merc_Short_Old",
    "Pct_of_OI_Swap_Long_Old",
    "Pct_of_OI_Swap_Short_Old",
    "Pct_of_OI_Swap_Spread_Old",
    "Pct_of_OI_M_Money_Long_Old",
    "Pct_of_OI_M_Money_Short_Old",
    "Pct_of_OI_M_Money_Spread_Old",
    "Pct_of_OI_Other_Rept_Long_Old",
    "Pct_of_OI_Other_Rept_Short_Old",
    "Pct_of_OI_Other_Rept_Spread_Old",
    "Pct_of_OI_Tot_Rept_Long_Old",
    "Pct_of_OI_Tot_Rept_Short_Old",
    "Pct_of_OI_NonRept_Long_Old",
    "Pct_of_OI_NonRept_Short_Old",
    # ── Pct of OI (Other) ───────────────────────────────────────────
    "Pct_of_Open_Interest_Other",
    "Pct_of_OI_Prod_Merc_Long_Other",
    "Pct_of_OI_Prod_Merc_Short_Other",
    "Pct_of_OI_Swap_Long_Other",
    "Pct_of_OI_Swap_Short_Other",
    "Pct_of_OI_Swap_Spread_Other",
    "Pct_of_OI_M_Money_Long_Other",
    "Pct_of_OI_M_Money_Short_Other",
    "Pct_of_OI_M_Money_Spread_Other",
    "Pct_of_OI_Other_Rept_Long_Other",
    "Pct_of_OI_Other_Rept_Short_Other",
    "Pct_of_OI_Other_Rept_Spread_Other",
    "Pct_of_OI_Tot_Rept_Long_Other",
    "Pct_of_OI_Tot_Rept_Short_Other",
    "Pct_of_OI_NonRept_Long_Other",
    "Pct_of_OI_NonRept_Short_Other",
    # ── Trader counts ────────────────────────────────────────────────
    "Traders_Tot_All",
    "Traders_Prod_Merc_Long_All",
    "Traders_Prod_Merc_Short_All",
    "Traders_Swap_Long_All",
    "Traders_Swap_Short_All",
    "Traders_Swap_Spread_All",
    "Traders_M_Money_Long_All",
    "Traders_M_Money_Short_All",
    "Traders_M_Money_Spread_All",
    "Traders_Other_Rept_Long_All",
    "Traders_Other_Rept_Short_All",
    "Traders_Other_Rept_Spread_All",
    "Traders_Tot_Rept_Long_All",
    "Traders_Tot_Rept_Short_All",
    "Traders_Tot_Old",
    "Traders_Prod_Merc_Long_Old",
    "Traders_Prod_Merc_Short_Old",
    "Traders_Swap_Long_Old",
    "Traders_Swap_Short_Old",
    "Traders_Swap_Spread_Old",
    "Traders_M_Money_Long_Old",
    "Traders_M_Money_Short_Old",
    "Traders_M_Money_Spread_Old",
    "Traders_Other_Rept_Long_Old",
    "Traders_Other_Rept_Short_Old",
    "Traders_Other_Rept_Spread_Old",
    "Traders_Tot_Rept_Long_Old",
    "Traders_Tot_Rept_Short_Old",
    "Traders_Tot_Other",
    "Traders_Prod_Merc_Long_Other",
    "Traders_Prod_Merc_Short_Other",
    "Traders_Swap_Long_Other",
    "Traders_Swap_Short_Other",
    "Traders_Swap_Spread_Other",
    "Traders_M_Money_Long_Other",
    "Traders_M_Money_Short_Other",
    "Traders_M_Money_Spread_Other",
    "Traders_Other_Rept_Long_Other",
    "Traders_Other_Rept_Short_Other",
    "Traders_Other_Rept_Spread_Other",
    "Traders_Tot_Rept_Long_Other",
    "Traders_Tot_Rept_Short_Other",
    # ── Concentration ratios ─────────────────────────────────────────
    "Conc_Gross_LE_4_TDR_Long_All",
    "Conc_Gross_LE_4_TDR_Short_All",
    "Conc_Gross_LE_8_TDR_Long_All",
    "Conc_Gross_LE_8_TDR_Short_All",
    "Conc_Net_LE_4_TDR_Long_All",
    "Conc_Net_LE_4_TDR_Short_All",
    "Conc_Net_LE_8_TDR_Long_All",
    "Conc_Net_LE_8_TDR_Short_All",
    "Conc_Gross_LE_4_TDR_Long_Old",
    "Conc_Gross_LE_4_TDR_Short_Old",
    "Conc_Gross_LE_8_TDR_Long_Old",
    "Conc_Gross_LE_8_TDR_Short_Old",
    "Conc_Net_LE_4_TDR_Long_Old",
    "Conc_Net_LE_4_TDR_Short_Old",
    "Conc_Net_LE_8_TDR_Long_Old",
    "Conc_Net_LE_8_TDR_Short_Old",
    "Conc_Gross_LE_4_TDR_Long_Other",
    "Conc_Gross_LE_4_TDR_Short_Other",
    "Conc_Gross_LE_8_TDR_Long_Other",
    "Conc_Gross_LE_8_TDR_Short_Other",
    "Conc_Net_LE_4_TDR_Long_Other",
    "Conc_Net_LE_4_TDR_Short_Other",
    "Conc_Net_LE_8_TDR_Long_Other",
    "Conc_Net_LE_8_TDR_Short_Other",
    # ── Metadata ─────────────────────────────────────────────────────
    "Contract_Units",
    "CFTC_Contract_Market_Code_Quotes",
    "CFTC_Market_Code_Quotes",
    "CFTC_Commodity_Code_Quotes",
    "CFTC_SubGroup_Code",
    "FutOnly_or_Combined",
]

assert len(_HEADERS) == 191, f"Header count mismatch: {len(_HEADERS)}"


def _safe_int(val: str) -> int | None:
    """Parse an integer, returning None for missing/invalid values."""
    v = val.strip()
    if not v or v == ".":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _safe_float(val: str) -> float | None:
    """Parse a float, returning None for missing/invalid values."""
    v = val.strip()
    if not v or v == ".":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _report_date_to_ts(date_str: str) -> float:
    """Convert 'YYYY-MM-DD' report date to Unix timestamp (midnight UTC).

    Returns 0.0 for missing/malformed dates.
    """
    if not date_str or not date_str.strip():
        return 0.0
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        # Use calendar.timegm for UTC (avoids local-timezone offset)
        import calendar

        return float(calendar.timegm(dt.timetuple()))
    except ValueError:
        return 0.0


# ── Columns we actually extract (indices into _HEADERS) ─────────────
# We don't need all 191 columns. Extract the signal-relevant ones.
_KEY_COLUMNS: dict[str, int] = {h: i for i, h in enumerate(_HEADERS)}


class CFTCTool(Tool):
    name = "cftc"
    description = (
        "Fetch CFTC Commitments of Traders (COT) positioning data. "
        "Shows managed money, producer/merchant, and swap dealer positions "
        "for futures contracts (commodities, currencies, energy, metals, financials). "
        "Computes net positioning, weekly flows, and concentration signals. "
        "Modes: 'latest' (current week) or 'historical' (full year). "
        "Filter by contract name (e.g. 'crude', 'gold') or CFTC code."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["latest", "historical"],
                "default": "latest",
                "description": "latest = current week; historical = full year",
            },
            "contract_filter": {
                "type": "string",
                "default": "",
                "description": (
                    "Case-insensitive substring to match contract names "
                    "(e.g. 'crude', 'wheat', 'gold', 'euro', 'bitcoin')"
                ),
            },
            "code_filter": {
                "type": "string",
                "default": "",
                "description": "Exact CFTC contract market code (e.g. '006765' for WTI crude)",
            },
            "top_n": {
                "type": "integer",
                "default": 20,
                "description": "Max contracts to return, sorted by open interest",
            },
            "year": {
                "type": "integer",
                "default": 0,
                "description": "Year for historical mode (default: current year)",
            },
        },
    }

    def __init__(
        self,
        cache: DataCache | None = None,
        *,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    # ── Public execute ───────────────────────────────────────────────

    def execute(
        self,
        *,
        mode: str = "latest",
        contract_filter: str = "",
        code_filter: str = "",
        top_n: int = 20,
        year: int = 0,
        **_: Any,
    ) -> ToolResult:
        if mode not in ("latest", "historical"):
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use 'latest' or 'historical'.",
            )

        try:
            if mode == "latest":
                csv_text = self._fetch_latest()
                rows = self._parse_rows(csv_text, has_headers=False)
            else:
                yr = year if year > 0 else datetime.now().year
                if yr < 2006:
                    return ToolResult(
                        success=False,
                        output="Disaggregated reports start in 2006.",
                    )
                csv_text = self._fetch_historical(yr)
                rows = self._parse_rows(csv_text, has_headers=True)
        except Exception as exc:
            log.exception("CFTC fetch/parse failed")
            return ToolResult(success=False, output=f"CFTC error: {exc}")

        if not rows:
            return ToolResult(success=False, output="No data returned from CFTC.")

        filtered = self._filter_contracts(rows, contract_filter, code_filter, top_n)
        if not filtered:
            return ToolResult(
                success=True,
                output=f"No contracts matched filter (filter='{contract_filter}', code='{code_filter}'). "
                f"Total rows parsed: {len(rows)}.",
                data={"contracts": [], "total_parsed": len(rows)},
            )

        enriched = self._compute_signals(filtered)

        # L2: persist entities + observations when PipelineStore available
        try:
            self._persist_entities(enriched)
        except Exception:
            log.exception("CFTC entity persistence failed (non-fatal)")

        output = self._format_output(enriched, len(rows))

        return ToolResult(
            success=True,
            output=output,
            data={"contracts": enriched, "total_parsed": len(rows)},
        )

    # ── Fetch methods ────────────────────────────────────────────────

    def _fetch_latest(self) -> str:
        cache_key = {"source": "cftc_weekly"}
        if self._cache:
            cached = self._cache.get("cftc", cache_key)
            if cached is not None:
                log.debug("CFTC weekly: cache hit")
                return cached

        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(_WEEKLY_URL, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()

        text = resp.text
        if self._cache:
            self._cache.put("cftc", cache_key, text)
        return text

    def _fetch_historical(self, year: int) -> str:
        cache_key = {"source": "cftc_historical", "year": year}
        if self._cache:
            cached = self._cache.get("cftc", cache_key)
            if cached is not None:
                log.debug("CFTC historical %d: cache hit", year)
                return cached

        url = _HIST_URL_TPL.format(year=year)
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = zf.namelist()
        if not names:
            raise RuntimeError(f"Empty ZIP from {url}")

        with zf.open(names[0]) as f:
            text = f.read().decode("utf-8", errors="replace")

        if self._cache:
            self._cache.put("cftc", cache_key, text)
        return text

    # ── Parsing ──────────────────────────────────────────────────────

    def _parse_rows(self, csv_text: str, *, has_headers: bool) -> list[dict[str, Any]]:
        lines = csv_text.strip().split("\n")
        if not lines:
            return []

        if has_headers:
            reader = csv.reader(io.StringIO(lines[0]))
            file_headers = [h.strip() for h in next(reader)]
            data_lines = lines[1:]
        else:
            file_headers = _HEADERS
            data_lines = lines

        rows: list[dict[str, Any]] = []
        for line in data_lines:
            if not line.strip():
                continue
            reader = csv.reader(io.StringIO(line))
            fields = next(reader)
            if len(fields) < 22:
                continue  # skip malformed rows

            row: dict[str, Any] = {}
            for i, val in enumerate(fields):
                if i >= len(file_headers):
                    break
                col = file_headers[i]
                v = val.strip()

                # String columns
                if col in (
                    "Market_and_Exchange_Names",
                    "Report_Date_as_YYYY-MM-DD",
                    "CFTC_Contract_Market_Code",
                    "CFTC_Market_Code",
                    "CFTC_Region_Code",
                    "CFTC_Commodity_Code",
                    "Contract_Units",
                    "CFTC_Contract_Market_Code_Quotes",
                    "CFTC_Market_Code_Quotes",
                    "CFTC_Commodity_Code_Quotes",
                    "CFTC_SubGroup_Code",
                    "FutOnly_or_Combined",
                    "As_of_Date_In_Form_YYMMDD",
                ):
                    row[col] = v
                elif "Pct_of" in col or "Conc_" in col:
                    row[col] = _safe_float(v)
                else:
                    row[col] = _safe_int(v)

            rows.append(row)

        return rows

    # ── Filtering ────────────────────────────────────────────────────

    def _filter_contracts(
        self,
        rows: list[dict[str, Any]],
        contract_filter: str,
        code_filter: str,
        top_n: int,
    ) -> list[dict[str, Any]]:
        result = rows

        if contract_filter:
            filt = contract_filter.lower()
            result = [r for r in result if filt in r.get("Market_and_Exchange_Names", "").lower()]

        if code_filter:
            code = code_filter.strip()
            result = [r for r in result if r.get("CFTC_Contract_Market_Code", "").strip() == code]

        # Sort by open interest descending
        result.sort(key=lambda r: r.get("Open_Interest_All") or 0, reverse=True)

        # Always keep mapped contracts (regardless of OI rank) so all 19 instrument
        # mappings are persisted even if they don't appear in the top-N by volume.
        from agent.tools.instrument_universe import cftc_code_to_ticker

        mapped_codes = set(cftc_code_to_ticker().keys())

        top = result[:top_n]
        top_codes = {r.get("CFTC_Contract_Market_Code", "").strip() for r in top}
        extras = [
            r
            for r in result[top_n:]
            if r.get("CFTC_Contract_Market_Code", "").strip() in mapped_codes
            and r.get("CFTC_Contract_Market_Code", "").strip() not in top_codes
        ]
        return top + extras

    # ── Signal computation ───────────────────────────────────────────

    def _compute_signals(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            mm_long = row.get("M_Money_Positions_Long_All") or 0
            mm_short = row.get("M_Money_Positions_Short_All") or 0
            pm_long = row.get("Prod_Merc_Positions_Long_All") or 0
            pm_short = row.get("Prod_Merc_Positions_Short_All") or 0
            swap_long = row.get("Swap_Positions_Long_All") or 0
            swap_short = row.get("Swap__Positions_Short_All") or 0
            oi = row.get("Open_Interest_All") or 0

            row["_mm_net"] = mm_long - mm_short
            row["_pm_net"] = pm_long - pm_short
            row["_swap_net"] = swap_long - swap_short
            row["_mm_net_pct_oi"] = round((mm_long - mm_short) / oi * 100, 2) if oi > 0 else 0.0

            # Weekly changes
            chg_mm_long = row.get("Change_in_M_Money_Long_All") or 0
            chg_mm_short = row.get("Change_in_M_Money_Short_All") or 0
            row["_mm_weekly_flow"] = chg_mm_long - chg_mm_short

            row["_oi_change"] = row.get("Change_in_Open_Interest_All") or 0

            # Concentration
            row["_conc_top4_long"] = row.get("Conc_Net_LE_4_TDR_Long_All")
            row["_conc_top4_short"] = row.get("Conc_Net_LE_4_TDR_Short_All")

        return rows

    # ── Entity persistence (L2) ──────────────────────────────────────

    def _persist_entities(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        """Persist CFTC positioning as L2 entity observations.

        For each CFTC row whose contract market code maps to a known
        instrument ticker (via ``cftc_code_to_ticker``), store a
        ``futures_positioning`` observation on the instrument entity
        and link it to the CFTC contract entity.

        Skips silently if no PipelineStore is configured.
        Returns counts of entities/observations/links created.
        """
        if self._store is None or entity_id_from_key is None:
            return {"observations": 0, "contracts": 0, "links": 0}
        if not rows:
            return {"observations": 0, "contracts": 0, "links": 0}

        try:
            return self._persist_entities_inner(rows)
        except Exception:
            log.exception("CFTC entity persistence failed (non-fatal)")
            return {"observations": 0, "contracts": 0, "links": 0}

    def _persist_entities_inner(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        """Inner persistence logic separated for testability."""
        from agent.tools.instrument_universe import _entity_id, cftc_code_to_ticker

        assert self._store is not None  # noqa: S101 — guarded by caller
        store = self._store
        code_map = cftc_code_to_ticker()

        counts = {"observations": 0, "contracts": 0, "links": 0}
        seen_contracts: set[str] = set()

        for row in rows:
            cftc_code = (row.get("CFTC_Contract_Market_Code") or "").strip()
            if not cftc_code:
                continue

            report_date = row.get("Report_Date_as_YYYY-MM-DD", "")
            market_name = row.get("Market_and_Exchange_Names", "")

            # Register the CFTC contract entity (deduped)
            if cftc_code not in seen_contracts:
                seen_contracts.add(cftc_code)
                contract_eid = entity_id_from_key("cftc_contract", cftc_code)
                store.register_entity(
                    entity_type="cftc_contract",
                    canonical_name=market_name.strip() or cftc_code,
                    entity_id=contract_eid,
                    metadata={
                        "cftc_code": cftc_code,
                        "source": "cftc",
                    },
                )
                counts["contracts"] += 1

            contract_eid = entity_id_from_key("cftc_contract", cftc_code)

            # Store positioning observation on the CFTC contract entity
            observed_at = _report_date_to_ts(report_date)
            obs_value = {
                "open_interest": row.get("Open_Interest_All"),
                "mm_net": row.get("_mm_net"),
                "pm_net": row.get("_pm_net"),
                "swap_net": row.get("_swap_net"),
                "mm_net_pct_oi": row.get("_mm_net_pct_oi"),
                "mm_weekly_flow": row.get("_mm_weekly_flow"),
                "oi_change": row.get("_oi_change"),
                "conc_top4_long": row.get("_conc_top4_long"),
                "conc_top4_short": row.get("_conc_top4_short"),
            }
            store.store_entity_observation(
                entity_id=contract_eid,
                source_tool="cftc",
                observed_at=observed_at,
                observation_type="futures_positioning",
                depth_level=2,
                value=obs_value,
            )
            counts["observations"] += 1

            # Link CFTC contract → instrument (if mapping exists)
            ticker = code_map.get(cftc_code)
            if ticker:
                inst_eid = _entity_id(ticker)
                link_id = store.link_entities(
                    entity_id_a=contract_eid,
                    entity_id_b=inst_eid,
                    link_type="cftc_tracks",
                    source="cftc",
                    confidence=1.0,
                    metadata={"cftc_code": cftc_code, "ticker": ticker},
                )
                if link_id:
                    counts["links"] += 1

        log.info(
            "CFTC L2: %d contracts, %d observations, %d instrument links",
            counts["contracts"],
            counts["observations"],
            counts["links"],
        )
        return counts

    # ── Formatting ───────────────────────────────────────────────────

    def _format_output(self, rows: list[dict[str, Any]], total_parsed: int) -> str:
        lines: list[str] = []
        lines.append(f"CFTC COT — {len(rows)} contracts (of {total_parsed} total)")

        report_dates = {r.get("Report_Date_as_YYYY-MM-DD", "?") for r in rows}
        lines.append(f"Report date(s): {', '.join(sorted(report_dates))}")
        lines.append("")

        for row in rows:
            name = row.get("Market_and_Exchange_Names", "?")
            oi = row.get("Open_Interest_All") or 0
            mm_net = row.get("_mm_net", 0)
            pm_net = row.get("_pm_net", 0)
            swap_net = row.get("_swap_net", 0)
            mm_pct = row.get("_mm_net_pct_oi", 0.0)
            mm_flow = row.get("_mm_weekly_flow", 0)
            oi_chg = row.get("_oi_change", 0)
            c4l = row.get("_conc_top4_long")
            c4s = row.get("_conc_top4_short")
            units = row.get("Contract_Units", "")

            lines.append(f"  {name}")
            lines.append(f"    OI: {oi:>12,}  ΔOI: {oi_chg:>+10,}  {units}")
            lines.append(f"    MM net: {mm_net:>+10,} ({mm_pct:>+.1f}% OI)  ΔMM: {mm_flow:>+10,}")
            lines.append(f"    PM net: {pm_net:>+10,}  Swap net: {swap_net:>+10,}")
            if c4l is not None or c4s is not None:
                c4l_s = f"{c4l:.1f}%" if c4l is not None else "n/a"
                c4s_s = f"{c4s:.1f}%" if c4s is not None else "n/a"
                lines.append(f"    Top-4 conc: long={c4l_s}  short={c4s_s}")
            lines.append("")

        return "\n".join(lines)
