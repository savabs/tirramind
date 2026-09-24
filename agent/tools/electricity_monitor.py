"""
Tool: Electricity Monitor — US-Wide Grid Demand, Generation, Interchange

EIA API v2  https://api.eia.gov/v2/  (free, API key required)

Expands electricity observation beyond NYISO to ALL US balancing authorities
(RTOs/ISOs: PJM, CAISO, ERCOT, MISO, SPP, ISO-NE, etc.).

Modes
-----
demand        Hourly electricity demand by balancing authority.
              Peak/trough/avg MW, demand-forecast deviation.

generation    Generation by fuel type (coal, gas, nuclear, solar, wind, hydro).
              Fuel mix proportions, renewable vs fossil share.

interchange   Inter-regional power flows between balancing authorities.
              Net imports/exports, largest trading relationships.

Signal theory:
  - Demand anomalies by region → economic activity shifts (factory shutdowns,
    data center buildout, extreme weather).
  - Fuel mix shifts → energy cost structure, transition speed.
  - Cross-region interchange patterns → grid stress, congestion, surplus/deficit.
  - Demand-forecast deviation → unplanned load changes (industrial activity proxy).

Data source: EIA API v2 — electricity/rto/ endpoints (free with key).

Completeness contract
---------------------
Every fetch here is measured against what EIA says it published, never
against "the request did not raise".  Two independent gates:

  1. ``EIAWindow.complete`` — ``response.total`` vs the rows we actually
     received.  For a bounded (``start``/``end``) query ``total`` is the exact
     size of the requested window, so a short read is provably a truncation.
  2. Hour coverage — hourly periods recovered vs the hours the caller asked
     for.  This is what catches a query that is *shaped* wrong (an
     under-sized ``length``, a missing facet) rather than transport-broken,
     which is invisible to gate 1 on an unbounded query.  For an unbounded
     ("last N days") query the count is taken over the *trailing* N*24 hours
     ending at the newest reading, never over every hour the row budget
     happened to drag in — see :meth:`_coverage_problems`.

  3. Malformed rows — a record we meant to store but could not parse (no
     period, no ``value``).  Distinct from a row we deliberately filtered.

A shortfall on either gate still persists the rows we did get — throwing them
away helps nobody — but it returns ``success=False`` naming the shortfall, so
the DAG records an incomplete collection instead of a complete-looking one.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key as _entity_id_from_key
except ImportError:
    _entity_id_from_key = None

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UA = "TirraMind/0.1 (electricity-monitor)"
_TIMEOUT = 25
_EIA_BASE = "https://api.eia.gov/v2"

# EIA v2 caps a single page at 5000 rows and exposes `offset` for the rest.
_EIA_MAX_PAGE = 5000
# Safety stop so a bad `total` cannot turn one collection into a crawl.
_EIA_MAX_PAGES = 8

VALID_MODES = {"demand", "generation", "interchange"}

# Row cardinality per hour, per endpoint — measured against the live API on
# 2026-09-23, not guessed.  `length` is sized from these; a guessed multiplier
# is how this collector fetched 14 of 24 hours for 26 days while reporting
# success (LESSONS.md F-16 shape).
#
#   region-data      4 rows/hour/BA (D, DF, NG, TI) — but we facet to type=D,
#                    so one row per hour survives and `length` is hours+buffer.
#   fuel-type-data   8-9 rows/hour/BA (CISO adds GEO, ERCO adds BAT); EIA's
#                    full fuel vocabulary is larger than EIA_FUEL_TYPES, so
#                    sizing by len(EIA_FUEL_TYPES) under-fetches.
#   interchange-data 1 row/hour per trading partner; PJM has 7 in the `fromba`
#                    direction, other BAs have more.
_DEMAND_BUFFER_HOURS = 24
_GENERATION_ROWS_PER_HOUR = 16
_INTERCHANGE_ROWS_PER_HOUR = 48

# EIA dates day-ahead forecast rows ahead of now.  PipelineStore refuses an
# observed_at more than a day out, so they are filtered rather than stored —
# and they must not count as hours collected either, or a forecast row would
# stand in for a missing actual reading.
_FUTURE_GRACE_SECONDS = 12 * 3600

# Major US balancing authorities (non-exhaustive — EIA has many more)
KNOWN_REGIONS = {
    "BPAT": "Bonneville Power Administration",
    "CISO": "California ISO",
    "ERCO": "Electric Reliability Council of Texas",
    "ISNE": "ISO New England",
    "MISO": "Midcontinent ISO",
    "NYIS": "New York ISO",
    "PJM": "PJM Interconnection",
    "SC": "South Carolina",
    "SCEG": "Dominion Energy South Carolina",
    "SOCO": "Southern Company",
    "SPA": "Southwestern Power Administration",
    "SWPP": "Southwest Power Pool",
    "TVA": "Tennessee Valley Authority",
    "WACM": "Western Area Power - CO/MO",
}

EIA_FUEL_TYPES = {
    "COL": "Coal",
    "NG": "Natural Gas",
    "NUC": "Nuclear",
    "OIL": "Petroleum",
    "OTH": "Other",
    "SUN": "Solar",
    "WAT": "Hydro",
    "WND": "Wind",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class PeriodFormatError(ValueError):
    """A caller-supplied ``start``/``end`` bound could not be parsed.

    Raised rather than returned as ``None``: dropping an unparseable bound
    silently widens the window to "most recent ``length`` rows", so a backfill
    asked to refill 2026-09-01 would quietly refill today and report success.
    """


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _strict_float(val: Any) -> float | None:
    """Parse a numeric field, returning ``None`` when it is absent/unparseable.

    Deliberately *not* :func:`_safe_float`.  A missing ``value`` key and a
    genuine reading of ``0`` are different facts, and conflating them is how
    this collector reported "Peak: 0 MW" for every hour while looking healthy
    (the request was missing ``data[0]=value``, so EIA returned records with
    no ``value`` at all).  Persistence refuses to store a ``None``.
    """
    if val is None:
        return None
    if isinstance(val, str) and not val.strip():
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# EIA period granularities, longest first.
_PERIOD_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H", "%Y-%m-%d")


def _period_to_epoch(period: Any) -> float | None:
    """Parse an EIA ``period`` ("2026-09-23T14") into UTC epoch seconds."""
    if not isinstance(period, str):
        return None
    text = period.strip().rstrip("Z")
    if not text:
        return None
    for fmt in _PERIOD_FORMATS:
        try:
            naive = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return naive.replace(tzinfo=UTC).timestamp()
    return None


def _clean_period(value: Any, *, bound: str) -> str | None:
    """Validate a caller-supplied start/end bound for EIA's ``period`` filter.

    Returns an hourly bound ("YYYY-MM-DDTHH") or ``None`` when the caller
    supplied nothing.  A date-only bound is widened to the whole day
    (``T00`` for ``start``, ``T23`` for ``end``) so the window is always an
    unambiguous span of hours and ``response.total`` is directly comparable
    against the hours requested.

    Raises :class:`PeriodFormatError` on anything else.  See that class for
    why this is not a silent ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if not isinstance(value, str):
        raise PeriodFormatError(f"{bound} must be a string, got {type(value).__name__}")

    text = value.strip().rstrip("Z")
    try:
        datetime.strptime(text, "%Y-%m-%dT%H")
    except ValueError:
        pass
    else:
        return text

    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        raise PeriodFormatError(f"{bound}={value!r} is not 'YYYY-MM-DDTHH' or 'YYYY-MM-DD'") from None
    return f"{text}T00" if bound == "start" else f"{text}T23"


def _expected_hours(start: str | None, end: str | None, days: int) -> int | None:
    """How many hourly periods the caller asked for, or ``None`` if unknowable.

    A half-bounded window ("everything since X") has no fixed width, so there
    is nothing honest to compare coverage against and the gate is skipped.
    """
    if start and end:
        a = _period_to_epoch(start)
        b = _period_to_epoch(end)
        if a is None or b is None or b < a:
            return None
        return int((b - a) // 3600) + 1
    if start or end:
        return None
    return days * 24


def _unpack_cached(cached: Any) -> tuple[str, list[dict]] | None:
    """Split a cache entry into (rendered text, raw records), or ``None``.

    ``None`` means "this entry cannot be replayed" — a legacy plain-string
    entry, or a dict with no records.  Replaying such an entry would return
    success with zero rows written for the rest of the 6h TTL, which is the
    exact green-with-no-data shape this module exists to eliminate, so the
    caller treats ``None`` as a cache miss and re-fetches.
    """
    if not isinstance(cached, dict) or "text" not in cached:
        return None
    raw = cached.get("records")
    if not isinstance(raw, list):
        return None
    records = [r for r in raw if isinstance(r, dict)]
    if not records:
        return None
    return str(cached["text"]), records


@dataclass(frozen=True)
class EIAWindow:
    """One EIA query's rows alongside the row count EIA itself declared.

    ``total`` is what makes a truncated read distinguishable from a complete
    one.  Discarding it (the pre-2026-09-23 behaviour) is F-16: a short
    window and a full one look identical to every caller.
    """

    records: list[dict]
    total: int
    complete: bool

    def __len__(self) -> int:
        return len(self.records)


def _fetch_eia(
    endpoint: str,
    facets: dict[str, list[str]] | None,
    api_key: str,
    length: int = 5000,
    sort_col: str = "period",
    sort_dir: str = "desc",
    start: str | None = None,
    end: str | None = None,
) -> EIAWindow | None:
    """Fetch up to ``length`` rows from EIA API v2, paging with ``offset``.

    Returns an :class:`EIAWindow` or ``None`` on failure.  A failure on *any*
    page fails the whole fetch: returning the pages that happened to succeed
    is precisely the transient-5xx-eats-72%-of-the-window failure (F-16), and
    a partial success never gets retried.

    ``data[0]=value`` is mandatory: without it EIA answers HTTP 200 with the
    right rows and periods but *no* ``value`` key, which every caller here
    then read as 0 MW.  ``start``/``end`` ("YYYY-MM-DDTHH" for hourly series)
    bound the window so a collection gap can be refilled instead of only ever
    fetching the most recent ``length`` rows — and so ``response.total``
    becomes the exact expected count for that window.
    """
    url = f"{_EIA_BASE}/{endpoint}/data/"
    length = max(1, int(length))

    collected: list[dict] = []
    total = 0
    pages = 0

    while len(collected) < length and pages < _EIA_MAX_PAGES:
        page_size = min(_EIA_MAX_PAGE, length - len(collected))
        # A list of pairs, not a dict: EIA repeats ``facets[x][]`` once per
        # value and a dict can only hold the last one.
        params: list[tuple[str, Any]] = [
            ("api_key", api_key),
            ("data[0]", "value"),
            ("length", page_size),
            ("offset", len(collected)),
            ("sort[0][column]", sort_col),
            ("sort[0][direction]", sort_dir),
        ]
        if facets:
            for key, values in facets.items():
                for v in values:
                    params.append((f"facets[{key}][]", v))
        if start:
            params.append(("start", start))
        if end:
            params.append(("end", end))

        try:
            resp = httpx.get(
                url,
                params=params,
                headers={"User-Agent": _UA},
                timeout=_TIMEOUT,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                log.warning("EIA HTTP %d for %s (offset=%d)", resp.status_code, endpoint, len(collected))
                return None
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("EIA fetch error for %s (offset=%d): %s", endpoint, len(collected), exc)
            return None

        response = body.get("response") or {} if isinstance(body, dict) else {}
        data = response.get("data") or []
        if not isinstance(data, list):
            data = []
        total = _safe_int(response.get("total"), default=total)

        collected.extend(r for r in data if isinstance(r, dict))
        pages += 1

        if len(data) < page_size:
            break  # EIA had nothing more to give
        if total and len(collected) >= total:
            break

    # What this query could legitimately have returned.
    ceiling = min(total, length) if total else len(collected)
    return EIAWindow(records=collected, total=total, complete=len(collected) >= ceiling)


def _is_actual_demand(rec: dict) -> bool:
    """True only for EIA's ``D`` (actual demand) series.

    ``DF`` (day-ahead forecast), ``NG`` (net generation) and ``TI`` (total
    interchange) share the region-data endpoint and would corrupt both the
    reported peak/average and the stored demand history.  A record carrying
    no type at all is accepted: the request facets to ``type=D`` server-side,
    so anything typeless that comes back is already demand.  This mirrors
    :meth:`ElectricityMonitorTool._observation_from_record` exactly, so what
    is displayed and what is stored cannot diverge.
    """
    type_code = str(rec.get("type") or "").strip().upper()
    if type_code:
        return type_code == "D"
    return str(rec.get("type-name") or "").strip().lower() in ("", "demand")


def _observable_hours(records: list[dict]) -> set[float]:
    """Distinct hourly periods that are real observations, as epoch seconds.

    Future-dated periods are excluded, exactly as
    :meth:`ElectricityMonitorTool._observation_from_record` excludes them:
    EIA's day-ahead rows are not hours anybody collected, and counting them
    would let a forecast stand in for a missing actual reading — and would
    drag the "newest hour" of a trailing window into tomorrow.
    """
    horizon = time.time() + _FUTURE_GRACE_SECONDS
    hours: set[float] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        epoch = _period_to_epoch(rec.get("period"))
        if epoch is None or epoch > horizon:
            continue
        hours.add(epoch - (epoch % 3600))
    return hours


def _aggregate_hourly(records: list[dict]) -> dict:
    """Compute peak, trough, avg from hourly demand records."""
    values = [_safe_float(r.get("value")) for r in records]
    values = [v for v in values if v > 0]
    if not values:
        return {"peak_mw": 0, "trough_mw": 0, "avg_mw": 0, "hours": 0}
    return {
        "peak_mw": round(max(values)),
        "trough_mw": round(min(values)),
        "avg_mw": round(sum(values) / len(values)),
        "hours": len(values),
    }


def _fuel_mix_proportions(records: list[dict]) -> dict:
    """Compute generation proportions by fuel type."""
    by_fuel: dict[str, float] = {}
    for r in records:
        fuel = r.get("fueltype", r.get("type-name", "unknown"))
        val = _safe_float(r.get("value"))
        if val > 0:
            by_fuel[fuel] = by_fuel.get(fuel, 0) + val

    total = sum(by_fuel.values())
    if total == 0:
        return {}

    result: dict[str, Any] = {}
    for fuel, mw in sorted(by_fuel.items(), key=lambda x: -x[1]):
        pct = (mw / total) * 100
        label = EIA_FUEL_TYPES.get(fuel, fuel)
        result[label] = {
            "total_mwh": round(mw),
            "share_pct": round(pct, 1),
        }

    # Compute renewable vs fossil
    renewable_fuels = {"SUN", "WND", "WAT"}
    fossil_fuels = {"COL", "NG", "OIL"}
    renewable = sum(by_fuel.get(f, 0) for f in renewable_fuels)
    fossil = sum(by_fuel.get(f, 0) for f in fossil_fuels)
    result["_summary"] = {
        "renewable_pct": round((renewable / total) * 100, 1) if total else 0,
        "fossil_pct": round((fossil / total) * 100, 1) if total else 0,
        "total_mwh": round(total),
    }
    return result


class _Skip(str, Enum):
    """Why a fetched record produced no observation.

    The two are not interchangeable and must not share a tolerance:

    ``FILTERED``  we never meant to store it — a DF forecast row in a demand
                  window, a future-dated period.  Routine.
    ``MALFORMED`` we did mean to store it and could not parse it — no period,
                  or no ``value``.  Never routine: a missing ``value`` is the
                  exact shape of the ``data[0]=value`` defect, and dropping
                  those rows quietly loses hours while the run reports the
                  ones that survived and a green tick.
    """

    FILTERED = "filtered"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class PersistStats:
    """Outcome of one persistence pass.

    ``skipped``, ``malformed`` and ``rejected`` are three different facts and
    must not share a tolerance.  ``skipped`` is deliberate filtering (a DF
    forecast row in a demand window).  ``malformed`` is a row we meant to
    store but could not parse.  ``rejected`` is the store refusing a row we
    meant to write — always a bug, never routine.
    """

    written: int = 0
    skipped: int = 0
    malformed: int = 0
    rejected: int = 0


@dataclass
class _ModeOutcome:
    """What a mode handler recovered, and everything wrong with it."""

    success: bool
    output: str
    records: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    data: Any = None


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class ElectricityMonitorTool(Tool):
    """US-wide electricity demand, generation mix, and interchange flows."""

    name = "electricity_monitor"
    description = (
        "Monitor US-wide electricity via EIA API. Covers all balancing "
        "authorities (PJM, CAISO, ERCOT, MISO, ISO-NE, etc.). "
        "Modes: 'demand' for hourly load (MW) with peak/trough, "
        "'generation' for fuel mix (coal/gas/nuclear/solar/wind/hydro), "
        "'interchange' for inter-regional power flows. "
        "Requires TIRRA_EIA_API_KEY."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": "demand|generation|interchange",
            },
            "region": {
                "type": "string",
                "description": (
                    "Balancing authority code: PJM, CISO, ERCO, MISO, NYIS, ISNE, SWPP, SOCO, TVA, BPAT, etc."
                ),
            },
            "days": {
                "type": "integer",
                "description": "Number of days of data (1-7, default 1).",
            },
            "start": {
                "type": "string",
                "description": (
                    "Optional window start, 'YYYY-MM-DDTHH' (hourly) or 'YYYY-MM-DD'. Used to backfill a collection gap."
                ),
            },
            "end": {
                "type": "string",
                "description": "Optional window end, same format as 'start'.",
            },
        },
        "required": ["mode", "region"],
    }

    def __init__(
        self,
        *,
        cache: DataCache | None = None,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store
        self._api_key = self._get_api_key()

    @staticmethod
    def _get_api_key() -> str | None:
        key = os.environ.get("TIRRA_EIA_API_KEY", "").strip()
        return key if key else None

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
                output="EIA API key required. Set TIRRA_EIA_API_KEY.",
            )

        region = (kwargs.get("region") or "").strip().upper()
        if not region:
            return ToolResult(
                success=False,
                output="Parameter 'region' required. Use BA code: PJM, CISO, ERCO, etc.",
            )

        days = kwargs.get("days", 1)
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 1
        days = max(1, min(7, days))

        try:
            start = _clean_period(kwargs.get("start"), bound="start")
            end = _clean_period(kwargs.get("end"), bound="end")
        except PeriodFormatError as exc:
            # Silently dropping the bound would widen the window to "most
            # recent N rows" and still report success — a backfill quietly
            # refilling the wrong days.
            return ToolResult(success=False, output=f"Invalid window bound: {exc}")

        if mode == "demand":
            outcome = self._demand(region, days, start, end)
        elif mode == "generation":
            outcome = self._generation(region, days, start, end)
        else:
            outcome = self._interchange(region, days, start, end)

        if not outcome.success:
            return ToolResult(success=False, output=outcome.output)

        problems = list(outcome.problems)
        output = outcome.output

        if self._store is not None:
            stats = self._persist_entities(region, mode, outcome.records)
            output = f"{output}\n\nStored {stats.written} observation(s)."
            if outcome.records and stats.written == 0:
                # Fetched rows but stored none. Reporting success here is
                # exactly the green-checkmark/zero-row failure this collector
                # shipped for 26 days.
                problems.append(
                    f"PERSISTENCE FAILURE: fetched {len(outcome.records)} record(s) "
                    f"for {region} but wrote 0 observations"
                )
            # Independent of the guard above, and of each other: a run can be
            # both short and partly refused, and naming only the first fact
            # sends whoever reads the log looking in the wrong place.
            if stats.rejected:
                # Partial persistence. The old code counted only total
                # failure, so 1-of-10 rows landing reported success.
                problems.append(
                    f"PARTIAL PERSISTENCE: the store refused {stats.rejected} of "
                    f"{stats.written + stats.rejected} row(s) we meant to write"
                )
            if stats.malformed:
                # A row we fetched, meant to store, and could not parse — a
                # missing `value` is the data[0] defect itself. The surviving
                # rows would otherwise land under a green tick with a
                # confident-looking count.
                problems.append(
                    f"MALFORMED ROWS: {stats.malformed} of "
                    f"{stats.written + stats.malformed + stats.rejected} fetched row(s) "
                    f"had no usable period/value and could not be stored"
                )

        if problems:
            detail = "\n".join(f"  - {p}" for p in problems)
            return ToolResult(
                success=False,
                output=f"{output}\n\nINCOMPLETE COLLECTION:\n{detail}",
                data=outcome.data,
            )
        return ToolResult(success=True, output=output, data=outcome.data)

    # ------------------------------------------------------------------
    # Cache plumbing
    # ------------------------------------------------------------------

    def _cached_outcome(self, namespace: str, key: str) -> _ModeOutcome | None:
        if not self._cache:
            return None
        cached = self._cache.get(namespace, key)
        if cached is None:
            return None
        entry = _unpack_cached(cached)
        if entry is None:
            # Unreplayable (legacy string, or no records): treat as a miss so
            # the run writes rows instead of reporting a green zero.
            log.info("electricity_monitor: dropping unreplayable cache entry %s/%s", namespace, key)
            return None
        text, records = entry
        return _ModeOutcome(success=True, output=text, records=records)

    def _store_cache(
        self,
        namespace: str,
        key: str,
        text: str,
        records: list[dict],
        problems: list[str],
    ) -> None:
        """Cache only a result that is worth replaying.

        An empty result is never cached: it would replay a zero-row run for
        the whole 6h TTL without even asking EIA.  Neither is an incomplete
        one — a cache hit carries no ``problems``, so caching a shortfall
        launders it into a clean success on every run inside the TTL.
        """
        if not self._cache or not records or problems:
            return
        self._cache.put(namespace, key, {"text": text, "records": records})

    @staticmethod
    def _cache_key(region: str, days: int, start: str | None, end: str | None) -> str:
        return f"{region}_{days}_{start or ''}_{end or ''}"

    @staticmethod
    def _window_problems(label: str, window: EIAWindow) -> list[str]:
        if window.complete:
            return []
        return [
            f"{label}: EIA declared {window.total} row(s) for this query but only {len(window.records)} were retrieved"
        ]

    @staticmethod
    def _coverage(
        records: list[dict],
        expected_hours: int | None,
        *,
        trailing: bool,
    ) -> tuple[int, float | None]:
        """(hours recovered that count toward the request, newest hour).

        The single place hour coverage is computed, so the number printed in
        the summary and the number the gate fails on cannot diverge.  They
        did: the gate counted every hour the row budget dragged in, so a day
        missing six of its hours still printed "Coverage: 42 distinct hour(s)
        of 24 requested" — a reassuring number for an incomplete window.
        """
        hours = _observable_hours(records)
        if not hours:
            return 0, None
        newest = max(hours)
        if not expected_hours or not trailing:
            return len(hours), newest
        floor = newest - (expected_hours - 1) * 3600
        return sum(1 for h in hours if h >= floor), newest

    @classmethod
    def _coverage_line(
        cls,
        records: list[dict],
        expected_hours: int | None,
        *,
        trailing: bool,
    ) -> str:
        covered, newest = cls._coverage(records, expected_hours, trailing=trailing)
        if not expected_hours:
            return f"Coverage: {covered} distinct hour(s)"
        window = ""
        if trailing and newest is not None:
            stamp = datetime.fromtimestamp(newest, tz=UTC).strftime("%Y-%m-%dT%H")
            window = f" in the {expected_hours}h window ending {stamp}"
        return f"Coverage: {covered} of {expected_hours} requested hour(s){window}"

    @classmethod
    def _coverage_problems(
        cls,
        label: str,
        records: list[dict],
        expected_hours: int | None,
        *,
        trailing: bool,
    ) -> list[str]:
        """Hours recovered vs hours asked for.

        ``trailing`` is the unbounded "last N days" case, and it is the one
        that needs care.  The row budget deliberately over-fetches (see
        ``_DEMAND_BUFFER_HOURS``) to absorb EIA's publication lag, so a plain
        count of distinct hours can be satisfied entirely by *older* hours
        the buffer dragged in.  Measured that way, a day missing six of its
        hours still reported "Coverage: 42 distinct hour(s) of 24 requested"
        and success=True — the 14-of-24 failure moved out of the query and
        into the gate.  So count only inside the N*24-hour window ending at
        the newest reading.

        A bounded ``start``/``end`` query needs no such window: the request
        itself is the window, so any shortfall is already a real one.
        """
        if not expected_hours:
            return []
        covered, newest = cls._coverage(records, expected_hours, trailing=trailing)
        if newest is None:
            return [f"{label}: no usable hourly period recovered from {len(records)} row(s)"]
        if covered >= expected_hours:
            return []
        where = ""
        if trailing:
            stamp = datetime.fromtimestamp(newest, tz=UTC).strftime("%Y-%m-%dT%H")
            where = f" in the window ending {stamp}"
        return [
            f"{label}: only {covered} of {expected_hours} requested hour(s) recovered"
            f"{where} ({expected_hours - covered} hour(s) missing)"
        ]

    # ------------------------------------------------------------------
    # Mode: demand
    # ------------------------------------------------------------------

    def _demand(
        self,
        region: str,
        days: int,
        start: str | None = None,
        end: str | None = None,
    ) -> _ModeOutcome:
        cache_ns = "electricity_demand"
        cache_key = self._cache_key(region, days, start, end)
        hit = self._cached_outcome(cache_ns, cache_key)
        if hit is not None:
            return hit

        expected_hours = _expected_hours(start, end, days)
        trailing = not (start and end)
        # With the type facet every returned row is one hour of real demand,
        # so `length` is hours + buffer rather than a guessed multiplier. The
        # buffer absorbs EIA's publication lag at the leading edge.
        budget = (expected_hours or days * 24) + _DEMAND_BUFFER_HOURS

        window = _fetch_eia(
            "electricity/rto/region-data",
            # `type: D` is the fix for the 42% under-fetch: unfaceted, EIA
            # spends 58% of the row budget on future-dated DF forecast rows
            # before reaching a single hour of actual demand.
            {"respondent": [region], "type": ["D"]},
            self._api_key,
            length=budget,
            start=start,
            end=end,
        )
        if window is None:
            return _ModeOutcome(
                success=False,
                output=f"Failed to fetch demand data for {region}.",
            )

        demand_records = [r for r in window.records if _is_actual_demand(r)]

        if not demand_records:
            # A high-volume always-on public source returning nothing for a BA
            # code is a bug report, not a fact about the world — and it is
            # never cached.
            return _ModeOutcome(
                success=False,
                output=(
                    f"No demand data returned for {region} (last {days} day(s), "
                    f"EIA declared total={window.total}, {len(window.records)} row(s) "
                    f"received, 0 of them actual demand)."
                ),
            )

        problems = self._window_problems("demand", window)
        problems += self._coverage_problems("demand", demand_records, expected_hours, trailing=trailing)

        stats = _aggregate_hourly(demand_records)
        region_name = KNOWN_REGIONS.get(region, region)

        lines = [
            f"⚡ Electricity Demand — {region_name} ({region})",
            f"Period: last {days} day(s), {stats['hours']} hours of data",
            self._coverage_line(demand_records, expected_hours, trailing=trailing),
            f"Peak: {stats['peak_mw']:,} MW",
            f"Trough: {stats['trough_mw']:,} MW",
            f"Average: {stats['avg_mw']:,} MW",
        ]

        # Show latest few records
        latest = demand_records[:6]
        if latest:
            lines.append("")
            lines.append("Recent readings:")
            for r in latest:
                period = r.get("period", "?")
                val = _safe_float(r.get("value"))
                tname = r.get("type-name", "")
                lines.append(f"  {period}: {val:,.0f} MW ({tname})")

        result = "\n".join(lines)
        self._store_cache(cache_ns, cache_key, result, demand_records, problems)
        return _ModeOutcome(
            success=True,
            output=result,
            records=demand_records,
            problems=problems,
        )

    # ------------------------------------------------------------------
    # Mode: generation
    # ------------------------------------------------------------------

    def _generation(
        self,
        region: str,
        days: int,
        start: str | None = None,
        end: str | None = None,
    ) -> _ModeOutcome:
        cache_ns = "electricity_generation"
        cache_key = self._cache_key(region, days, start, end)
        hit = self._cached_outcome(cache_ns, cache_key)
        if hit is not None:
            return hit

        expected_hours = _expected_hours(start, end, days)
        trailing = not (start and end)
        # Sized by the endpoint's true per-hour cardinality. len(EIA_FUEL_TYPES)
        # is 8 and under-fetches: CISO publishes 9 fuel rows/hour (it adds GEO)
        # and ERCO reports BAT — EIA's fuel vocabulary is wider than ours.
        budget = (expected_hours or days * 24) * _GENERATION_ROWS_PER_HOUR

        window = _fetch_eia(
            "electricity/rto/fuel-type-data",
            {"respondent": [region]},
            self._api_key,
            length=budget,
            start=start,
            end=end,
        )
        if window is None:
            return _ModeOutcome(
                success=False,
                output=f"Failed to fetch generation data for {region}.",
            )

        records = window.records
        if not records:
            return _ModeOutcome(
                success=False,
                output=(
                    f"No generation data returned for {region} (last {days} day(s), EIA declared total={window.total})."
                ),
            )

        problems = self._window_problems("generation", window)
        problems += self._coverage_problems("generation", records, expected_hours, trailing=trailing)

        mix = _fuel_mix_proportions(records)
        region_name = KNOWN_REGIONS.get(region, region)
        summary = mix.pop("_summary", {})

        lines = [
            f"⚡ Generation Mix — {region_name} ({region})",
            f"Period: last {days} day(s)",
            self._coverage_line(records, expected_hours, trailing=trailing),
            f"Total generation: {summary.get('total_mwh', 0):,} MWh",
            f"Renewable share: {summary.get('renewable_pct', 0)}%",
            f"Fossil share: {summary.get('fossil_pct', 0)}%",
            "",
            "By fuel type:",
        ]
        for fuel, info in mix.items():
            lines.append(f"  {fuel}: {info['total_mwh']:,} MWh ({info['share_pct']}%)")

        result = "\n".join(lines)
        self._store_cache(cache_ns, cache_key, result, records, problems)
        return _ModeOutcome(
            success=True,
            output=result,
            records=records,
            problems=problems,
        )

    # ------------------------------------------------------------------
    # Mode: interchange
    # ------------------------------------------------------------------

    def _interchange(
        self,
        region: str,
        days: int,
        start: str | None = None,
        end: str | None = None,
    ) -> _ModeOutcome:
        cache_ns = "electricity_interchange"
        cache_key = self._cache_key(region, days, start, end)
        hit = self._cached_outcome(cache_ns, cache_key)
        if hit is not None:
            return hit

        expected_hours = _expected_hours(start, end, days)
        trailing = not (start and end)
        budget = (expected_hours or days * 24) * _INTERCHANGE_ROWS_PER_HOUR

        # Interchange has fromba and toba — get both directions
        window_from = _fetch_eia(
            "electricity/rto/interchange-data",
            {"fromba": [region]},
            self._api_key,
            length=budget,
            start=start,
            end=end,
        )
        window_to = _fetch_eia(
            "electricity/rto/interchange-data",
            {"toba": [region]},
            self._api_key,
            length=budget,
            start=start,
            end=end,
        )

        # `and` here (the old code) failed only when BOTH directions died. A
        # single 5xx left `imports` at literally 0 MWh, printed a fabricated
        # net position, and persisted half a window as authoritative history.
        if window_from is None or window_to is None:
            missing = "exports (fromba)" if window_from is None else "imports (toba)"
            if window_from is None and window_to is None:
                missing = "both directions"
            return _ModeOutcome(
                success=False,
                output=(
                    f"Failed to fetch interchange data for {region}: {missing} "
                    f"unavailable. Refusing to report a net position from one side."
                ),
            )

        exports: dict[str, float] = {}
        imports: dict[str, float] = {}

        for r in window_from.records:
            partner = r.get("toba", "?")
            val = _safe_float(r.get("value"))
            exports[partner] = exports.get(partner, 0) + val

        for r in window_to.records:
            partner = r.get("fromba", "?")
            val = _safe_float(r.get("value"))
            imports[partner] = imports.get(partner, 0) + val

        combined = list(window_from.records) + list(window_to.records)
        if not combined:
            return _ModeOutcome(
                success=False,
                output=(
                    f"No interchange data returned for {region} (last {days} day(s), "
                    f"EIA declared total={window_from.total} out / {window_to.total} in)."
                ),
            )

        problems = self._window_problems("interchange/exports", window_from)
        problems += self._window_problems("interchange/imports", window_to)
        problems += self._coverage_problems(
            "interchange/exports", window_from.records, expected_hours, trailing=trailing
        )
        problems += self._coverage_problems("interchange/imports", window_to.records, expected_hours, trailing=trailing)

        total_export = sum(exports.values())
        total_import = sum(imports.values())
        net = total_import - total_export
        region_name = KNOWN_REGIONS.get(region, region)

        lines = [
            f"⚡ Interchange Flows — {region_name} ({region})",
            f"Period: last {days} day(s)",
            self._coverage_line(combined, expected_hours, trailing=trailing),
            f"Total exports: {total_export:,.0f} MWh",
            f"Total imports: {total_import:,.0f} MWh",
            f"Net: {'import' if net > 0 else 'export'} {abs(net):,.0f} MWh",
        ]

        if exports:
            lines.append("")
            lines.append("Exports to:")
            for partner, mwh in sorted(exports.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"  → {partner}: {mwh:,.0f} MWh")

        if imports:
            lines.append("")
            lines.append("Imports from:")
            for partner, mwh in sorted(imports.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"  ← {partner}: {mwh:,.0f} MWh")

        result = "\n".join(lines)
        self._store_cache(cache_ns, cache_key, result, combined, problems)
        return _ModeOutcome(
            success=True,
            output=result,
            records=combined,
            problems=problems,
        )

    # ------------------------------------------------------------------
    # L2 entity persistence
    # ------------------------------------------------------------------

    def _persist_entities(self, region: str, mode: str, records: list[dict] | None) -> PersistStats:
        """Store one observation per fetched reading."""
        if self._store is None or _entity_id_from_key is None:
            return PersistStats()
        if not region or not records:
            return PersistStats()
        try:
            return self._persist_entities_inner(region, mode, records)
        except Exception:
            log.exception("Electricity monitor entity persistence failed")
            # written == 0 with records present is surfaced as a hard failure
            # by execute(); this is not swallowed.
            return PersistStats()

    def _persist_entities_inner(self, region: str, mode: str, records: list[dict]) -> PersistStats:
        assert self._store is not None
        assert _entity_id_from_key is not None

        if not region or not records:
            return PersistStats()

        eid = _entity_id_from_key("organization", region)
        self._store.register_entity("organization", region, eid)

        written = 0
        skipped = 0
        malformed = 0
        rejected = 0
        for rec in records:
            observation = self._observation_from_record(rec, region, mode)
            if observation is _Skip.FILTERED:
                skipped += 1
                continue
            if observation is _Skip.MALFORMED:
                # A row we meant to store and could not parse. Counting this
                # as a routine skip is how a window silently loses hours:
                # every surviving row still lands and the run still prints a
                # confident total.
                log.warning(
                    "electricity_monitor %s/%s: unparseable record period=%r value=%r",
                    region,
                    mode,
                    rec.get("period") if isinstance(rec, dict) else None,
                    rec.get("value") if isinstance(rec, dict) else None,
                )
                malformed += 1
                continue
            observed_at, obs_type, value = observation
            try:
                self._store.store_entity_observation(
                    entity_id=eid,
                    source_tool="electricity_monitor",
                    observed_at=observed_at,
                    observation_type=obs_type,
                    value=value,
                    depth_level=2,
                )
            except Exception as exc:
                # One bad row must not cost the other 23 hours of the day —
                # but it is counted and execute() fails the run on it.
                log.warning(
                    "electricity_monitor %s/%s: store refused period %r: %s",
                    region,
                    mode,
                    rec.get("period"),
                    exc,
                )
                rejected += 1
                continue
            written += 1

        if skipped or malformed or rejected:
            log.info(
                "electricity_monitor %s/%s: wrote %d, filtered %d, malformed %d, store-rejected %d",
                region,
                mode,
                written,
                skipped,
                malformed,
                rejected,
            )
        return PersistStats(written=written, skipped=skipped, malformed=malformed, rejected=rejected)

    def _observation_from_record(
        self,
        rec: dict,
        region: str,
        mode: str,
    ) -> tuple[float, str, dict[str, Any]] | _Skip:
        """Turn one EIA record into (observed_at, observation_type, value).

        Anything that must not be stored comes back as a :class:`_Skip`
        saying *why*, because the two reasons carry opposite meanings:

        ``FILTERED``  a day-ahead *forecast* period in the future (which
                      ``PipelineStore._validate_observed_at`` rejects anyway),
                      or a foreign series in a demand window.  Expected.
        ``MALFORMED`` no parseable period, or no ``value`` — the ``data[0]``
                      failure mode.  Storing it as 0 MW would fabricate data;
                      dropping it quietly would lose an hour.  So it is
                      counted, and :meth:`execute` fails the run on it.
        """
        if not isinstance(rec, dict):
            return _Skip.MALFORMED

        observed_at = _period_to_epoch(rec.get("period"))
        if observed_at is None:
            return _Skip.MALFORMED
        # EIA publishes day-ahead forecast periods dated ahead of now; the
        # store refuses observed_at > now + 1 day. This guard applies in every
        # mode — fuel-type-data and interchange-data carry no `type` column,
        # so the demand type filter below cannot stand in for it.
        if observed_at > time.time() + _FUTURE_GRACE_SECONDS:
            return _Skip.FILTERED

        mw = _strict_float(rec.get("value"))
        if mw is None:
            return _Skip.MALFORMED

        type_name = str(rec.get("type-name") or "").strip()

        if mode == "demand":
            # "D" is actual demand. "DF" (day-ahead forecast), "NG" and "TI"
            # are different series and would corrupt the demand history.
            if not _is_actual_demand(rec):
                return _Skip.FILTERED
            return (
                observed_at,
                "grid_demand",
                {"mw": mw, "type": type_name or "Demand", "region": region},
            )

        if mode == "generation":
            fuel_code = str(rec.get("fueltype") or "").strip().upper()
            return (
                observed_at,
                "grid_generation",
                {
                    "mwh": mw,
                    "fuel_code": fuel_code,
                    "fuel": EIA_FUEL_TYPES.get(fuel_code, type_name or fuel_code),
                    "region": region,
                },
            )

        from_ba = str(rec.get("fromba") or "").strip().upper()
        to_ba = str(rec.get("toba") or "").strip().upper()
        if not from_ba and not to_ba:
            # An interchange row naming neither end is not a flow we can
            # attribute — we meant to store it and cannot.
            return _Skip.MALFORMED
        return (
            observed_at,
            "grid_interchange",
            {"mwh": mw, "from": from_ba, "to": to_ba, "region": region},
        )
