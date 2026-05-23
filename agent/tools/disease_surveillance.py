"""
Tool: Disease & Pandemic Surveillance — CDC Wastewater + WHO DON + ECDC + NCBI

The most valuable L0 disease surveillance data available for free.

Sources:
  CDC NWSS Wastewater (Socrata)  — pathogen PCR concentrations in sewage
  WHO Disease Outbreak News       — global outbreak declarations (OData)
  ECDC Open Data                  — EU COVID cases/variants/hospitalizations
  NCBI E-utilities                — genomic sequence submission velocity

Modes:
  wastewater       — CDC NWSS: pathogen concentrations in US wastewater.
                     Physics that can't be faked or retracted.  Detects waves
                     2-3 weeks before hospital admissions.
  outbreaks        — WHO DON: global outbreak declarations and novel pathogen
                     alerts.  Escalation velocity = how worried WHO is.
  eu_surveillance  — ECDC: EU/EEA COVID cases, variants, hospitalizations.
  genomics         — NCBI: GenBank sequence submission velocity as an outbreak
                     proxy.  Accelerating submissions = labs actively
                     characterizing something new.

Signal theory:
  - Wastewater surge in 10+ US states = pandemic wave forming
  - Novel pathogen in sewage (H5 avian flu 0→positive) before any hospital case
  - WHO DON frequency increasing for same disease = global spread
  - ECDC variant share crossing 20% = replacement wave
  - NCBI submission velocity 2× 30-day baseline = outbreak expanding

Market relevance:
  Pandemics → pharma (vaccines, PPE), travel disruption (airlines, hotels),
  labor shortages (absenteeism), supply chain (factory shutdowns), healthcare
  sector revenue.  H5N1 in wastewater + poultry culling = food price spike.
"""

from __future__ import annotations

import logging
import re
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

# ── CDC NWSS dataset IDs ───────────────────────────────────────

_CDC_DATASETS: dict[str, str] = {
    "sars-cov-2": "j9g8-acpt",
    "influenza_a": "ymmh-divb",
    "rsv": "45cq-cw4i",
    "mpox": "xpxn-rzgz",
    "measles": "akvg-8vrb",
    "avian_h5": "mtpu-urpp",
}

_CDC_AGGREGATE_ID = "2ew6-ywp6"

_PATHOGEN_ALIASES: dict[str, str] = {
    "covid": "sars-cov-2",
    "covid-19": "sars-cov-2",
    "covid19": "sars-cov-2",
    "flu": "influenza_a",
    "flu_a": "influenza_a",
    "influenza": "influenza_a",
    "h5n1": "avian_h5",
    "h5": "avian_h5",
    "bird_flu": "avian_h5",
    "avian_influenza": "avian_h5",
    "monkeypox": "mpox",
}

# ── ECDC endpoints ──────────────────────────────────────────────

_ECDC_DATASETS: dict[str, str] = {
    "cases": "nationalcasedeath",
    "variants": "virusvariant",
    "hospital": "hospitalicuadmissionrates",
}

# ── US state codes for validation ───────────────────────────────

_US_STATES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
}


# ── Helpers ─────────────────────────────────────────────────────


def _resolve_pathogen(name: str) -> str | None:
    """Resolve a pathogen name/alias to a canonical key, or None."""
    key = name.strip().lower().replace(" ", "_")
    if key in _CDC_DATASETS:
        return key
    # Try with hyphens converted to underscores and vice versa
    key_under = key.replace("-", "_")
    if key_under in _CDC_DATASETS:
        return key_under
    key_hyphen = key.replace("_", "-")
    if key_hyphen in _CDC_DATASETS:
        return key_hyphen
    # Try aliases on all forms
    return _PATHOGEN_ALIASES.get(key) or _PATHOGEN_ALIASES.get(key_under) or _PATHOGEN_ALIASES.get(key_hyphen)


def _parse_who_title(title: str) -> dict[str, str]:
    """Extract disease and country from a WHO DON title.

    Titles follow patterns like:
      "Nipah virus infection - Bangladesh"
      "Mpox: recombinant virus ... – Global situation"
      "Avian Influenza A(H5N1) - United States of America"
    """
    result: dict[str, str] = {"disease": "", "country": ""}
    if not title:
        return result

    # Try "Disease - Country" pattern (dash or en-dash)
    m = re.match(r"^(.+?)\s*[-–—]\s*(.+)$", title)
    if m:
        result["disease"] = m.group(1).strip()
        result["country"] = m.group(2).strip()
    else:
        # Try "Disease: detail" pattern
        m2 = re.match(r"^(.+?):\s*(.+)$", title)
        if m2:
            result["disease"] = m2.group(1).strip()
            result["country"] = m2.group(2).strip()
        else:
            result["disease"] = title.strip()

    return result


def _safe_float(val: Any) -> float | None:
    """Safely convert a value to float, return None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> int | None:
    """Safely convert a value to int, return None on failure."""
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# ── Tool ────────────────────────────────────────────────────────


class DiseaseSurveillanceTool(Tool):
    name = "disease_surveillance"
    description = (
        "Monitor pathogen concentrations in US wastewater (CDC NWSS — physics, "
        "can't be faked), global outbreak declarations (WHO DON), EU surveillance "
        "(ECDC), and genomic sequence velocity (NCBI). Mode 'wastewater' = "
        "6 pathogens via sewage PCR. Mode 'outbreaks' = WHO global alerts. "
        "Mode 'eu_surveillance' = EU cases/variants/hospital. Mode 'genomics' = "
        "GenBank sequence submission velocity."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["wastewater", "outbreaks", "eu_surveillance", "genomics"],
                "default": "wastewater",
                "description": (
                    "wastewater = CDC NWSS wastewater pathogen concentrations (US). "
                    "outbreaks = WHO Disease Outbreak News (global). "
                    "eu_surveillance = ECDC EU/EEA data. "
                    "genomics = NCBI sequence submission velocity."
                ),
            },
            "pathogen": {
                "type": "string",
                "default": "",
                "description": (
                    "For wastewater/genomics: pathogen filter. "
                    "Options: sars-cov-2, influenza_a, rsv, mpox, measles, avian_h5. "
                    "Aliases: covid, flu, h5n1, bird_flu, monkeypox. "
                    "Empty = all pathogens (wastewater) or SARS-CoV-2 (genomics)."
                ),
            },
            "state": {
                "type": "string",
                "default": "",
                "description": ("For wastewater: US state code filter (e.g., 'CA', 'NY'). Empty = all states."),
            },
            "disease": {
                "type": "string",
                "default": "",
                "description": (
                    "For outbreaks: keyword filter on disease name in WHO DON title. "
                    "E.g., 'mpox', 'cholera', 'avian'. Case-insensitive."
                ),
            },
            "dataset": {
                "type": "string",
                "enum": ["cases", "variants", "hospital"],
                "default": "cases",
                "description": (
                    "For eu_surveillance: which ECDC dataset. "
                    "cases = national case/death counts. "
                    "variants = virus variant sequencing shares. "
                    "hospital = ICU/hospital admission rates."
                ),
            },
            "country": {
                "type": "string",
                "default": "",
                "description": ("For eu_surveillance: country code filter (e.g., 'DE', 'FR'). Empty = all EU/EEA."),
            },
            "organism": {
                "type": "string",
                "default": "",
                "description": (
                    "For genomics: organism name for NCBI query. "
                    "Default is SARS-CoV-2. E.g., 'measles', 'H5N1', 'Ebola'."
                ),
            },
            "days_back": {
                "type": "integer",
                "default": 30,
                "description": "How many days of history. Default 30, max 180.",
            },
            "limit": {
                "type": "integer",
                "default": 50,
                "description": "Max results. Default 50, max 1000.",
            },
        },
        "required": [],
    }

    def __init__(
        self,
        cache: DataCache | None = None,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    def execute(
        self,
        *,
        mode: str = "wastewater",
        pathogen: str = "",
        state: str = "",
        disease: str = "",
        dataset: str = "cases",
        country: str = "",
        organism: str = "",
        days_back: int = 30,
        limit: int = 50,
        _backfill: bool = False,
        **_: Any,
    ) -> ToolResult:
        mode = mode.lower().strip()
        if mode not in ("wastewater", "outbreaks", "eu_surveillance", "genomics"):
            return ToolResult(
                success=False,
                output=(f"Invalid mode '{mode}'. Use 'wastewater', 'outbreaks', 'eu_surveillance', or 'genomics'."),
            )

        if not _backfill:
            days_back = max(1, min(int(days_back), 180))
        limit = max(1, min(int(limit), 1000))

        if mode == "wastewater":
            result = self._execute_wastewater(
                pathogen=pathogen,
                state=state,
                days_back=days_back,
                limit=limit,
            )
        elif mode == "outbreaks":
            result = self._execute_outbreaks(disease=disease, limit=limit)
        elif mode == "eu_surveillance":
            result = self._execute_eu_surveillance(
                dataset=dataset,
                country=country,
                limit=limit,
            )
        else:
            # genomics — skip persistence (no country dimension)
            return self._execute_genomics(organism=organism)

        if result.success and result.data:
            self._persist_entities(result.data, mode)

        return result

    # ------------------------------------------------------------------
    # wastewater mode — CDC NWSS Socrata
    # ------------------------------------------------------------------

    def _execute_wastewater(
        self,
        *,
        pathogen: str,
        state: str,
        days_back: int,
        limit: int,
    ) -> ToolResult:
        # Resolve pathogen
        resolved = None
        if pathogen.strip():
            resolved = _resolve_pathogen(pathogen)
            if resolved is None:
                return ToolResult(
                    success=False,
                    output=(
                        f"Unknown pathogen '{pathogen}'. Available: "
                        + ", ".join(sorted(_CDC_DATASETS.keys()))
                        + ". Aliases: covid, flu, h5n1, bird_flu, monkeypox."
                    ),
                )

        # Validate state
        state_upper = state.strip().upper()
        if state_upper and state_upper not in _US_STATES:
            return ToolResult(
                success=False,
                output=f"Unknown US state code '{state_upper}'. Use 2-letter code (e.g., CA, NY, TX).",
            )

        # If specific pathogen requested, fetch raw concentration data
        if resolved:
            return self._fetch_cdc_pathogen(
                pathogen_key=resolved,
                state=state_upper,
                days_back=days_back,
                limit=limit,
            )

        # All pathogens: use aggregate metrics endpoint
        return self._fetch_cdc_aggregate(
            state=state_upper,
            days_back=days_back,
            limit=limit,
        )

    def _fetch_cdc_pathogen(
        self,
        *,
        pathogen_key: str,
        state: str,
        days_back: int,
        limit: int,
    ) -> ToolResult:
        dataset_id = _CDC_DATASETS[pathogen_key]
        cutoff = (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00")

        where_clauses = [f"sample_collect_date > '{cutoff}'"]
        if state:
            where_clauses.append(f"state_territory = '{state}'")

        params = {
            "$where": " AND ".join(where_clauses),
            "$limit": str(min(limit, 1000)),
        }

        cache_key = {
            "src": "cdc_pathogen",
            "id": dataset_id,
            "state": state,
            "cutoff": cutoff,
            "limit": limit,
        }
        data, error = self._fetch_socrata(dataset_id, params, cache_key, ttl=7200)
        if error:
            return ToolResult(success=False, output=error)

        if not data:
            msg = f"CDC NWSS: No {pathogen_key} wastewater data"
            if state:
                msg += f" for {state}"
            msg += f" in last {days_back} days."
            return ToolResult(success=True, output=msg, data={"records": [], "count": 0})

        # Summarize: group by state, compute stats
        by_state: dict[str, list[dict]] = {}
        for rec in data:
            st = rec.get("state_territory", "??")
            by_state.setdefault(st, []).append(rec)

        summaries = []
        for st, recs in sorted(by_state.items()):
            concentrations = [
                c for c in (_safe_float(r.get("pcr_target_avg_conc")) for r in recs) if c is not None and c > 0
            ]
            detections = sum(1 for r in recs if str(r.get("pcr_target_detect", "")).lower() == "yes")
            pop_served = max(
                (_safe_int(r.get("population_served")) or 0 for r in recs),
                default=0,
            )
            summaries.append(
                {
                    "state": st,
                    "samples": len(recs),
                    "detections": detections,
                    "detection_rate": round(detections / len(recs), 3) if recs else 0,
                    "mean_concentration": (
                        round(sum(concentrations) / len(concentrations), 2) if concentrations else None
                    ),
                    "max_concentration": (round(max(concentrations), 2) if concentrations else None),
                    "population_served": pop_served,
                }
            )

        # Sort by detection rate descending
        summaries.sort(key=lambda s: -s["detection_rate"])

        lines = [
            f"CDC NWSS Wastewater — {pathogen_key.upper()}",
            f"  Period: last {days_back} days | {len(data)} samples | {len(by_state)} states",
            "",
        ]
        for s in summaries[:25]:
            conc_str = f"  mean={s['mean_concentration']}" if s["mean_concentration"] else ""
            lines.append(
                f"  {s['state']:2s}  detect={s['detection_rate']:.0%} ({s['detections']}/{s['samples']}){conc_str}"
            )

        # Flag high-detection states
        hot_states = [s for s in summaries if s["detection_rate"] > 0.5]
        if len(hot_states) >= 5:
            lines.append("")
            lines.append(f"  ⚠ MULTI-STATE WAVE: {len(hot_states)} states with >50% detection rate")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "pathogen": pathogen_key,
                "summaries": summaries,
                "total_samples": len(data),
                "states_count": len(by_state),
                "hot_states": len(hot_states) if hot_states else 0,
            },
        )

    def _fetch_cdc_aggregate(
        self,
        *,
        state: str,
        days_back: int,
        limit: int,
    ) -> ToolResult:
        cutoff = (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00")

        where_clauses = [f"date_end > '{cutoff}'"]
        if state:
            where_clauses.append(f"wwtp_jurisdiction = '{state}'")

        params = {
            "$where": " AND ".join(where_clauses),
            "$limit": str(min(limit, 1000)),
        }

        cache_key = {
            "src": "cdc_aggregate",
            "state": state,
            "cutoff": cutoff,
            "limit": limit,
        }
        data, error = self._fetch_socrata(
            _CDC_AGGREGATE_ID,
            params,
            cache_key,
            ttl=7200,
        )
        if error:
            return ToolResult(success=False, output=error)

        if not data:
            msg = "CDC NWSS: No aggregate wastewater data"
            if state:
                msg += f" for {state}"
            msg += f" in last {days_back} days."
            return ToolResult(success=True, output=msg, data={"records": [], "count": 0})

        # Summarize by jurisdiction: look for surges
        by_jurisdiction: dict[str, list[dict]] = {}
        for rec in data:
            jur = rec.get("wwtp_jurisdiction", "??")
            by_jurisdiction.setdefault(jur, []).append(rec)

        surges = []
        for jur, recs in sorted(by_jurisdiction.items()):
            ptc_vals = [v for v in (_safe_float(r.get("ptc_15d")) for r in recs) if v is not None]
            detect_vals = [v for v in (_safe_float(r.get("detect_prop_15d")) for r in recs) if v is not None]
            pct_vals = [v for v in (_safe_float(r.get("percentile")) for r in recs) if v is not None]

            avg_ptc = round(sum(ptc_vals) / len(ptc_vals), 1) if ptc_vals else None
            avg_detect = round(sum(detect_vals) / len(detect_vals), 3) if detect_vals else None
            avg_pct = round(sum(pct_vals) / len(pct_vals), 1) if pct_vals else None

            entry = {
                "jurisdiction": jur,
                "sites": len(recs),
                "avg_ptc_15d": avg_ptc,
                "avg_detect_prop_15d": avg_detect,
                "avg_percentile": avg_pct,
            }
            surges.append(entry)
            if avg_ptc is not None and avg_ptc > 100:
                entry["alert"] = "SURGE"

        surges.sort(key=lambda x: -(x.get("avg_ptc_15d") or -999))

        lines = [
            "CDC NWSS Wastewater — Aggregate Trends",
            f"  Period: last {days_back} days | {len(data)} records | {len(by_jurisdiction)} jurisdictions",
            "",
        ]

        surge_count = sum(1 for s in surges if s.get("alert") == "SURGE")
        if surge_count:
            lines.append(f"  ⚠ {surge_count} jurisdictions with SURGE (>100% 15d change)")
            lines.append("")

        for s in surges[:25]:
            ptc_str = f"ptc_15d={s['avg_ptc_15d']:+.1f}%" if s["avg_ptc_15d"] is not None else "ptc=N/A"
            det_str = f"detect={s['avg_detect_prop_15d']:.0%}" if s["avg_detect_prop_15d"] is not None else ""
            alert_str = " ⚠SURGE" if s.get("alert") == "SURGE" else ""
            lines.append(f"  {s['jurisdiction']:20s}  {ptc_str}  {det_str}{alert_str}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "summaries": surges,
                "total_records": len(data),
                "jurisdictions_count": len(by_jurisdiction),
                "surge_count": surge_count,
            },
        )

    def _fetch_socrata(
        self,
        dataset_id: str,
        params: dict[str, str],
        cache_key: dict[str, Any],
        ttl: int,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch from CDC Socrata. Returns (data, error)."""
        if self._cache:
            cached = self._cache.get("disease_surveillance_cdc", cache_key)
            if cached is not None:
                return cached, None

        url = f"https://data.cdc.gov/resource/{dataset_id}.json"
        try:
            with httpx.Client(
                timeout=_TIMEOUT,
                headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 429:
                    return [], "CDC Socrata: Rate limited. Try again later."
                if resp.status_code >= 400:
                    return [], f"CDC Socrata error: HTTP {resp.status_code}"
                data = resp.json()
        except httpx.TimeoutException:
            return [], "CDC Socrata: Request timed out."
        except Exception as exc:
            log.exception("CDC Socrata fetch failed")
            return [], f"CDC Socrata fetch error: {exc}"

        if not isinstance(data, list):
            return [], "CDC Socrata: Unexpected response format."

        if self._cache and data:
            self._cache.put("disease_surveillance_cdc", cache_key, data, ttl=ttl)

        return data, None

    # ------------------------------------------------------------------
    # outbreaks mode — WHO Disease Outbreak News
    # ------------------------------------------------------------------

    def _execute_outbreaks(
        self,
        *,
        disease: str,
        limit: int,
    ) -> ToolResult:
        limit = min(limit, 100)  # WHO API max per page

        cache_key = {"src": "who_don", "disease": disease.lower(), "limit": limit}
        if self._cache:
            cached = self._cache.get("disease_surveillance_who", cache_key)
            if cached is not None:
                return self._format_who_results(cached, disease)

        url = "https://www.who.int/api/hubs/diseaseoutbreaknews"
        params: dict[str, str] = {
            "$top": str(limit),
            "$orderby": "PublicationDate desc",
        }
        if disease.strip():
            safe = disease.strip().replace("'", "''")
            params["$filter"] = f"contains(Title, '{safe}')"

        try:
            with httpx.Client(
                timeout=_TIMEOUT,
                headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 429:
                    return ToolResult(
                        success=False,
                        output="WHO DON: Rate limited.",
                    )
                if resp.status_code >= 400:
                    return ToolResult(
                        success=False,
                        output=f"WHO DON error: HTTP {resp.status_code}",
                    )
                raw = resp.json()
        except httpx.TimeoutException:
            return ToolResult(success=False, output="WHO DON: Request timed out.")
        except Exception as exc:
            log.exception("WHO DON fetch failed")
            return ToolResult(
                success=False,
                output=f"WHO DON fetch error: {exc}",
            )

        entries = raw.get("value", [])
        if not isinstance(entries, list):
            return ToolResult(
                success=False,
                output="WHO DON: Unexpected response format.",
            )

        if self._cache and entries:
            self._cache.put(
                "disease_surveillance_who",
                cache_key,
                entries,
                ttl=21600,
            )

        return self._format_who_results(entries, disease)

    def _format_who_results(self, entries: list[dict[str, Any]], disease: str) -> ToolResult:
        if not entries:
            msg = "WHO DON: No outbreak entries found"
            if disease:
                msg += f" matching '{disease}'"
            msg += "."
            return ToolResult(success=True, output=msg, data={"entries": [], "count": 0})

        results = []
        for entry in entries:
            title = entry.get("Title", "")
            parsed = _parse_who_title(title)
            results.append(
                {
                    "title": title,
                    "date": entry.get("PublicationDate", "")[:10],
                    "disease_parsed": parsed["disease"],
                    "country_parsed": parsed["country"],
                    "don_id": entry.get("DonId", ""),
                    "url": entry.get("UrlName", ""),
                }
            )

        # Disease frequency analysis
        disease_freq: dict[str, int] = {}
        for r in results:
            d = r["disease_parsed"].lower()
            if d:
                disease_freq[d] = disease_freq.get(d, 0) + 1

        lines = [
            f"WHO Disease Outbreak News — {len(results)} entries:",
            "",
        ]

        # Show top diseases by frequency
        if disease_freq:
            top_diseases = sorted(
                disease_freq.items(),
                key=lambda x: -x[1],
            )[:10]
            lines.append("  Disease frequency:")
            for d, count in top_diseases:
                lines.append(f"    {d}: {count} entries")
            lines.append("")

        for r in results[:25]:
            lines.append(f"  [{r['date']}] {r['title'][:100]}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "entries": results,
                "count": len(results),
                "disease_frequency": disease_freq,
            },
        )

    # ------------------------------------------------------------------
    # eu_surveillance mode — ECDC Open Data
    # ------------------------------------------------------------------

    def _execute_eu_surveillance(
        self,
        *,
        dataset: str,
        country: str,
        limit: int,
    ) -> ToolResult:
        dataset_lower = dataset.strip().lower()
        if dataset_lower not in _ECDC_DATASETS:
            return ToolResult(
                success=False,
                output=(f"Unknown ECDC dataset '{dataset}'. Use 'cases', 'variants', or 'hospital'."),
            )

        ecdc_path = _ECDC_DATASETS[dataset_lower]
        cache_key = {
            "src": "ecdc",
            "dataset": ecdc_path,
            "country": country.upper(),
            "limit": limit,
        }
        if self._cache:
            cached = self._cache.get("disease_surveillance_ecdc", cache_key)
            if cached is not None:
                return self._format_ecdc_results(cached, dataset_lower, country)

        url = f"https://opendata.ecdc.europa.eu/covid19/{ecdc_path}/json"
        try:
            with httpx.Client(
                timeout=30,  # ECDC returns full dataset — can be large
                headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(url)
                if resp.status_code == 429:
                    return ToolResult(
                        success=False,
                        output="ECDC: Rate limited.",
                    )
                if resp.status_code >= 400:
                    return ToolResult(
                        success=False,
                        output=f"ECDC error: HTTP {resp.status_code}",
                    )
                data = resp.json()
        except httpx.TimeoutException:
            return ToolResult(success=False, output="ECDC: Request timed out.")
        except Exception as exc:
            log.exception("ECDC fetch failed")
            return ToolResult(
                success=False,
                output=f"ECDC fetch error: {exc}",
            )

        if not isinstance(data, list):
            return ToolResult(
                success=False,
                output="ECDC: Unexpected response format.",
            )

        if self._cache and data:
            self._cache.put(
                "disease_surveillance_ecdc",
                cache_key,
                data,
                ttl=43200,
            )

        return self._format_ecdc_results(data, dataset_lower, country)

    def _format_ecdc_results(
        self,
        data: list[dict[str, Any]],
        dataset: str,
        country: str,
    ) -> ToolResult:
        # Filter by country if specified
        country_upper = country.strip().upper()
        if country_upper:
            data = [
                r
                for r in data
                if str(r.get("country_code", "")).upper() == country_upper
                or str(r.get("country", "")).upper() == country_upper
            ]

        if not data:
            msg = f"ECDC ({dataset}): No data"
            if country_upper:
                msg += f" for country '{country_upper}'"
            msg += "."
            return ToolResult(success=True, output=msg, data={"records": [], "count": 0})

        # Sort by year_week descending to get most recent first
        data.sort(
            key=lambda r: str(r.get("year_week", "")),
            reverse=True,
        )

        # Take most recent records
        recent = data[:200]

        if dataset == "cases":
            return self._format_ecdc_cases(recent, country_upper)
        if dataset == "variants":
            return self._format_ecdc_variants(recent, country_upper)
        return self._format_ecdc_hospital(recent, country_upper)

    def _format_ecdc_cases(
        self,
        data: list[dict],
        country: str,
    ) -> ToolResult:
        # Group by country, show latest week for each
        by_country: dict[str, dict] = {}
        for rec in data:
            cc = rec.get("country_code", rec.get("country", "??"))
            if cc not in by_country:
                by_country[cc] = rec

        entries = sorted(by_country.values(), key=lambda r: str(r.get("country", "")))

        lines = [
            f"ECDC COVID-19 Cases/Deaths — {len(entries)} countries" + (f" (filtered: {country})" if country else ""),
            "",
        ]
        for r in entries[:30]:
            cc = r.get("country_code", r.get("country", "??"))
            week = r.get("year_week", "?")
            indicator = r.get("indicator", "?")
            value = r.get("weekly_count", r.get("cumulative_count", "?"))
            lines.append(f"  {cc:5s}  week={week}  {indicator}={value}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "records": entries,
                "count": len(entries),
                "countries": list(by_country.keys()),
            },
        )

    def _format_ecdc_variants(
        self,
        data: list[dict],
        country: str,
    ) -> ToolResult:
        # Group by variant, show latest shares
        by_variant: dict[str, list[dict]] = {}
        for rec in data:
            var = rec.get("variant", rec.get("source", "unknown"))
            by_variant.setdefault(var, []).append(rec)

        lines = [
            f"ECDC COVID-19 Variants — {len(by_variant)} variants, {len(data)} records"
            + (f" (filtered: {country})" if country else ""),
            "",
        ]
        # Sort variants by number of recent records
        sorted_vars = sorted(by_variant.items(), key=lambda x: -len(x[1]))
        for var, recs in sorted_vars[:15]:
            shares = [v for v in (_safe_float(r.get("percent_variant")) for r in recs) if v is not None]
            avg_share = round(sum(shares) / len(shares), 1) if shares else None
            share_str = f"avg_share={avg_share}%" if avg_share is not None else ""
            lines.append(f"  {var[:40]:40s}  records={len(recs)}  {share_str}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "variants": {k: len(v) for k, v in by_variant.items()},
                "count": len(data),
            },
        )

    def _format_ecdc_hospital(
        self,
        data: list[dict],
        country: str,
    ) -> ToolResult:
        # Group by country, latest week
        by_country: dict[str, dict] = {}
        for rec in data:
            cc = rec.get("country", "??")
            if cc not in by_country:
                by_country[cc] = rec

        entries = sorted(by_country.values(), key=lambda r: str(r.get("country", "")))

        lines = [
            f"ECDC Hospital/ICU Admissions — {len(entries)} countries" + (f" (filtered: {country})" if country else ""),
            "",
        ]
        for r in entries[:30]:
            cc = r.get("country", "??")
            week = r.get("year_week", "?")
            indicator = r.get("indicator", "?")
            value = r.get("value", "?")
            lines.append(f"  {cc:20s}  week={week}  {indicator}={value}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "records": entries,
                "count": len(entries),
                "countries": list(by_country.keys()),
            },
        )

    # ------------------------------------------------------------------
    # genomics mode — NCBI E-utilities
    # ------------------------------------------------------------------

    def _execute_genomics(self, *, organism: str) -> ToolResult:
        org_name = organism.strip() or "SARS-CoV-2"

        now = datetime.now(UTC)
        current_year = now.year
        prior_year = current_year - 1

        cache_key = {"src": "ncbi", "organism": org_name.lower(), "year": current_year}
        if self._cache:
            cached = self._cache.get("disease_surveillance_ncbi", cache_key)
            if cached is not None:
                return self._format_genomics_results(cached)

        # Fetch current year count
        current_count, err1 = self._fetch_ncbi_count(org_name, current_year)
        if err1:
            return ToolResult(success=False, output=err1)

        # Fetch prior year count
        prior_count, err2 = self._fetch_ncbi_count(org_name, prior_year)
        if err2:
            return ToolResult(success=False, output=err2)

        result = {
            "organism": org_name,
            "current_year": current_year,
            "current_count": current_count,
            "prior_year": prior_year,
            "prior_count": prior_count,
        }

        # Compute velocity
        if prior_count and prior_count > 0:
            # Annualize: scale current partial year to full year rate
            day_of_year = now.timetuple().tm_yday
            annualized = current_count * (365.0 / max(day_of_year, 1))
            ratio = annualized / prior_count
            result["annualized_rate"] = round(annualized)
            result["yoy_ratio"] = round(ratio, 2)

            if ratio > 2.0:
                result["signal"] = "ACCELERATING"
            elif ratio > 1.2:
                result["signal"] = "ELEVATED"
            elif ratio < 0.5:
                result["signal"] = "DECLINING"
            else:
                result["signal"] = "STABLE"
        else:
            result["signal"] = "NO_BASELINE"

        if self._cache:
            self._cache.put(
                "disease_surveillance_ncbi",
                cache_key,
                result,
                ttl=86400,
            )

        return self._format_genomics_results(result)

    def _format_genomics_results(self, result: dict[str, Any]) -> ToolResult:
        lines = [
            f"NCBI GenBank Sequence Velocity — {result['organism']}",
            f"  {result['current_year']}: {result['current_count']:,} sequences submitted",
            f"  {result['prior_year']}: {result['prior_count']:,} sequences submitted",
        ]
        if "annualized_rate" in result:
            lines.append(f"  Annualized rate: ~{result['annualized_rate']:,} (YoY ratio: {result['yoy_ratio']}x)")
        signal = result.get("signal", "UNKNOWN")
        if signal == "ACCELERATING":
            lines.append("  ⚠ ACCELERATING — submission rate >2× prior year")
        elif signal == "ELEVATED":
            lines.append("  ↑ ELEVATED — submission rate above prior year")
        elif signal == "DECLINING":
            lines.append("  ↓ DECLINING — submission rate well below prior year")
        else:
            lines.append(f"  Signal: {signal}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data=result,
        )

    def _fetch_ncbi_count(
        self,
        organism: str,
        year: int,
    ) -> tuple[int, str | None]:
        """Fetch GenBank sequence count for organism in a given year."""
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        term = f'"{organism}"[Organism] AND {year}[PDAT]'
        params = {
            "db": "nucleotide",
            "term": term,
            "retmode": "json",
            "retmax": "0",  # We only need the count
        }

        try:
            with httpx.Client(
                timeout=_TIMEOUT,
                headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 429:
                    return 0, "NCBI: Rate limited (3 req/sec without API key)."
                if resp.status_code >= 400:
                    return 0, f"NCBI error: HTTP {resp.status_code}"
                data = resp.json()
        except httpx.TimeoutException:
            return 0, "NCBI: Request timed out."
        except Exception as exc:
            log.exception("NCBI fetch failed")
            return 0, f"NCBI fetch error: {exc}"

        esearch = data.get("esearchresult", {})
        count_str = esearch.get("count", "0")
        try:
            return int(count_str), None
        except (ValueError, TypeError):
            return 0, f"NCBI: Could not parse count from response: {count_str}"

    # ── L2 entity persistence (Phase 32) ──────────────────────

    def _persist_entities(
        self,
        data: dict[str, Any],
        mode: str,
    ) -> dict[str, int]:
        if self._store is None or _entity_id_from_key is None:
            return {"pathogen_level_obs": 0}
        try:
            return self._persist_entities_inner(data, mode)
        except Exception:
            log.exception("Disease surveillance entity persistence failed (non-fatal)")
            return {"pathogen_level_obs": 0}

    def _persist_entities_inner(
        self,
        data: dict[str, Any],
        mode: str,
    ) -> dict[str, int]:
        assert self._store is not None  # noqa: S101 -- guarded
        assert _entity_id_from_key is not None  # noqa: S101

        now = time.time()
        count = 0

        if mode == "wastewater":
            # US-only data — aggregate to country US
            country_eid = _entity_id_from_key("country", "US")
            self._store.register_entity("country", "US", country_eid)
            self._store.store_entity_observation(
                entity_id=country_eid,
                source_tool="disease_surveillance",
                observed_at=now,
                observation_type="pathogen_level",
                value={
                    "mode": "wastewater",
                    "pathogen": data.get("pathogen"),
                    "total_samples": data.get("total_samples"),
                    "states_count": data.get("states_count"),
                    "hot_states": data.get("hot_states"),
                    "surge_count": data.get("surge_count"),
                },
                depth_level=2,
            )
            count = 1

        elif mode == "outbreaks":
            # Extract unique countries from WHO DON entries
            entries = data.get("entries", [])
            countries_seen: set[str] = set()
            for entry in entries:
                cc = str(entry.get("country_parsed", "")).strip()
                if cc and len(cc) >= 2 and cc not in countries_seen:
                    countries_seen.add(cc)
            for cc in sorted(countries_seen):
                # Use first 2 chars as ISO-2 approximation for well-known countries
                cc_key = cc[:2].upper()
                if len(cc_key) < 2:
                    continue
                country_eid = _entity_id_from_key("country", cc_key)
                self._store.register_entity("country", cc_key, country_eid)
                self._store.store_entity_observation(
                    entity_id=country_eid,
                    source_tool="disease_surveillance",
                    observed_at=now,
                    observation_type="pathogen_level",
                    value={
                        "mode": "outbreaks",
                        "country_name": cc,
                        "entry_count": sum(1 for e in entries if e.get("country_parsed", "").strip() == cc),
                    },
                    depth_level=2,
                )
                count += 1

        elif mode == "eu_surveillance":
            # Extract unique country codes from ECDC records
            records = data.get("records", [])
            countries_seen_eu: set[str] = set()
            for rec in records:
                cc = str(rec.get("country_code", rec.get("country", ""))).strip().upper()
                if cc and len(cc) == 2 and cc not in countries_seen_eu:
                    countries_seen_eu.add(cc)
            for cc in sorted(countries_seen_eu):
                country_eid = _entity_id_from_key("country", cc)
                self._store.register_entity("country", cc, country_eid)
                self._store.store_entity_observation(
                    entity_id=country_eid,
                    source_tool="disease_surveillance",
                    observed_at=now,
                    observation_type="pathogen_level",
                    value={
                        "mode": "eu_surveillance",
                        "dataset": data.get("dataset"),
                        "record_count": sum(
                            1 for r in records if str(r.get("country_code", r.get("country", ""))).strip().upper() == cc
                        ),
                    },
                    depth_level=2,
                )
                count += 1

        log.info(
            "Disease surveillance L2: %d pathogen_level obs persisted (mode=%s)",
            count,
            mode,
        )
        return {"pathogen_level_obs": count}
