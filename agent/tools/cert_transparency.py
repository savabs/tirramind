"""
Tool: Certificate Transparency — crt.sh CT Log Monitor

crt.sh: https://crt.sh/ (Sectigo's CT search engine, free, no auth)

Everyone checks certificate validity for security. Nobody auto-monitors
certificate issuance patterns as a corporate activity signal.

Modes:
  search     — Search CT logs for certificates issued to a domain.
  subdomains — Discover subdomains via wildcard CT certificate search.
               Reveals internal project names, staging environments,
               unreleased products.
  recent     — Recent certificate issuances for a domain (last N days).

What crt.sh actually does (measured 2026-09-23, not assumed):
  - It serves NO ``entry_timestamp`` any more. ``not_before`` is the issuance
    instant and is backfilled into that field at the fetch boundary.
  - Its query takes NO date parameter: the full history arrives on every call
    and ``days_back`` filters it client-side. A missed window is therefore
    recoverable by widening ``days_back`` — this source is backfillable, and
    anything in the repo still calling it "live-only" is out of date.
  - ``q=domain`` and ``q=%.domain`` return the same payload byte for byte
    (aqr.com 247/247, cftc.gov 398/398, sec.gov 923/923), so one query per
    domain is all this tool makes.
  - It fails often and transiently: 502s in 0.6s, 504s at 120s, then a 200.
    Hence the retry ladder, and hence a wall-clock budget per call so a bad
    hour cannot overrun the DAG node that drives 20 domains.
  - A 404 from it is a *backend failure*, not "no such domain" — an unknown
    domain answers ``200 []``. Re-measured on a second pass 2026-09-23:
    ``q=%.cftc.gov`` answered 404 and, 2.5s later, ``q=cftc.gov`` answered 200
    with 398 records. That is why 404 sits in ``_RETRY_STATUSES``.
  - Successful answers are slow, and the slowest exceed one call's budget:
    same pass, 200s at 10.0s (``%.aqr.com``), 22.6s (``cftc.gov``), 42.1s
    (``%.bis.org``) and 58.9s (``aqr.com``). A 58.9s success cannot fit a 55s
    budget, and at a 20-domain roster the budget cannot grow (20 x 55 + 47.5s
    of gaps already uses 1147.5s of the node's 1200s). Such a domain comes
    back ``success=False`` / "domain not reached" — a reached-nothing report,
    never a quiet day. Shrinking the roster, not widening the budget, is the
    lever if that starts costing real domains.
  - Some identities are simply stale in its index: ``%.sec.gov`` tops out at
    2024-10-12 while ``q=www.sec.gov`` reaches 2026-08-15. ``recent`` mode
    reports ``records_fetched`` and ``newest_not_before`` so that shows up as
    a fact rather than as an empty day.

Signal theory:
  - New subdomain = product launch, M&A integration, infrastructure expansion
  - Certificate surge = scaling event → growth signal
  - Unusual subdomains = stealth projects (ai.company.com before public launch)
  - Issuer switch = security posture change, cost optimization
  - Expiring certs + no renewal = potential outage risk
  - Cross-reference: new subdomains + job postings + silence = stealth launch
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key
except ImportError:  # pragma: no cover
    entity_id_from_key = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_CRTSH_URL = "https://crt.sh/"
_UA = "TirraMind/0.1"
#: Ceiling on a single HTTP attempt.  crt.sh is genuinely slow on large
#: domains (measured 2026-09-23: ``%.nasdaq.com`` answered 200 after 56s
#: having 504'd at 120s minutes earlier), so 30s guaranteed a timeout for
#: every domain bigger than a small one.  But an unbounded-per-attempt
#: budget is not the fix either: this tool runs 20 domains inside one DAG
#: node whose timeout is 1200s, and a node timeout marks the whole executor
#: pool degraded (``executor.py`` ``_degraded``), skipping every later node
#: in the DAG.  Hence a per-attempt cap *and* the wall-clock budget below.
_TIMEOUT = 50
_CACHE_TTL = 3600  # 1 hour — certs logged continuously

#: Wall-clock ceiling for one :meth:`CertTransparencyTool.execute` call —
#: every query it issues, every retry, every backoff sleep included.
#:
#: The caller that matters is ``daily_collection``'s ``fetch_cert_domains``
#: node: 20 domains, a 2.5s inter-domain gap (19 x 2.5 = 47.5s) and a 1200s
#: node timeout.  ``20 x 55 + 47.5 = 1147.5s`` fits; a per-attempt-only
#: budget did not (3 attempts x 120s x 2 queries was 732s for ONE domain).
#: Overrunning is not a slow node, it is a dead DAG layer, so a domain that
#: cannot be served inside its budget comes back ``success=False`` saying
#: "domain not reached", instead of silently eating the node's budget.
#: Pass ``time_budget=0`` to opt out (interactive/CLI use).
_DEFAULT_TIME_BUDGET = 55.0

#: An attempt with less than this much budget left is not made at all.  Live
#: 2026-09-23, a first attempt that times out at 50s leaves ~3s, and a 3s
#: request to a service whose fastest successful answer that hour was 5.0s is
#: not a retry, it is load on something already failing.  Reporting the domain
#: unreached is the honest result.
_MIN_ATTEMPT_SECONDS = 5.0

#: Backoff (seconds) before each subsequent attempt; its length is the attempt
#: count.  Five, not three, because crt.sh's failure mode is a *storm*, not a
#: blip: measured 2026-09-23, ``%.sec.gov`` answered 502, 502, ReadTimeout,
#: then 200; ``%.jpmorgan.com`` and ``%.goldmansachs.com`` burned three
#: attempts in 7.9s each because a 502 comes back in 0.6s.  Three cheap
#: attempts spend 8s of a 55s budget and then give the domain up for the day.
#: The sum of the sleeps (22s) stays well inside the budget, and the deadline
#: truncates the ladder for a domain whose failures are slow ones instead.
_RETRY_BACKOFF: tuple[float, ...] = (2.0, 4.0, 8.0, 8.0, 8.0)
#: Transient upstream statuses.  crt.sh answers 429/5xx under load and
#: recovers within seconds.  404 is in here too, against intuition: crt.sh
#: answers an *unknown* domain with ``200 []``, so a 404 from it is never
#: "no such domain", it is the backend having failed — observed 2026-09-23
#: on the identical ``%.nasdaq.com`` URL that returned 200 minutes later.
#: Retrying an absent domain therefore costs nothing real.
_RETRY_STATUSES = frozenset({404, 429, 500, 502, 503, 504})

VALID_MODES = {"search", "subdomains", "recent"}


class PersistOutcome(NamedTuple):
    """What actually happened at the store boundary for one domain's batch.

    ``written`` is rows handed to the store and accepted; ``rejected`` counts
    records that never got that far — the store's own guards refused them (a
    future-dated ``not_before``, for instance: legal CA pre-issuance, which
    crt.sh does serve) or they carried no parseable issuance instant;
    ``failed`` counts writes that blew up for any other reason (a locked
    database, a broken alias write); ``error`` carries the first such message.

    The point of the split: ``count > 0`` with ``written == 0`` is a hard
    failure, and it used to be indistinguishable from "nothing new to store".
    """

    written: int = 0
    rejected: int = 0
    failed: int = 0
    error: str | None = None
    #: False when no store is configured at all (CLI use, most unit tests) —
    #: then ``written == 0`` means "nobody asked us to write", not a loss.
    active: bool = False


def _observation_rejected_types() -> tuple[type[BaseException], ...]:
    """The store's write-boundary rejection type, if this process has a store.

    Imported lazily: ``agent.pipeline.store`` pulls in the model layer, and a
    Layer-1 tool must not drag that in just to be importable.  Returns an empty
    tuple when the store is unavailable, which makes the ``except`` clause that
    uses it match nothing — the generic handler below then does the work.
    """
    try:
        from agent.pipeline.store import ObservationRejected
    except Exception:  # pragma: no cover - store always importable in-repo
        return ()
    return (ObservationRejected,)


def _parse_timestamp(ts: str | None) -> datetime | None:
    """Parse ISO 8601 timestamp from crt.sh. Returns None on failure.

    crt.sh serves naive UTC ("2026-03-27T07:49:06.083", "2026-09-16T00:00:00"),
    but this function is also the persistence path's only timestamp parser now
    — ``_persist_entities_inner`` used ``datetime.fromisoformat`` before this
    changeset — and callers elsewhere in the repo hand it ``...Z``, ``+00:00``
    and bare-date forms that the two strptime patterns reject.  Returning None
    for those does not fail loudly, it drops the record: the same silent loss
    one layer down.  So every unambiguous ISO 8601 form is accepted, a bare
    date meaning that date's 00:00 UTC — the ISO reading of it, deterministic
    and identical on every re-fetch, so the store's idempotency key still
    collapses re-runs.  Genuine garbage is still refused; nothing here ever
    falls back to ingestion time, which is the bug this parser replaced.
    """
    if not ts:
        return None
    # crt.sh uses formats like "2026-03-27T07:49:06.083" or "2026-03-27T00:00:00"
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _backfill_entry_timestamp(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill in ``entry_timestamp`` from ``not_before`` when crt.sh omits it.

    crt.sh's JSON API stopped serving ``entry_timestamp`` entirely.  Verified
    live 2026-09-23: ``GET /?q=%25.aqr.com&output=json&deduplicate=Y`` returns
    247 records and the field is present on **0** of them — same for
    ``deduplicate=N``, for no dedup param, and for ``identity=``.  The field is
    gone from the API, not from one query shape.

    Every consumer in this module keys off ``entry_timestamp``: ``recent``
    mode's cutoff filter, ``search`` mode's sort, ``subdomains`` mode's
    latest-entry tracking, and the persisted ``observed_at``.  With the field
    absent, ``recent`` mode discarded 100% of records before persistence and
    the node still reported green — 26 days of CT data missed.

    That gap **is recoverable**, contrary to what this module and the DAG node
    used to claim.  crt.sh's query carries no date parameter at all
    (``GET /?q=%25.aqr.com&output=json&deduplicate=Y``): the full history
    arrives on every call and ``days_back`` filters it client-side.  Measured
    on one such response (247 records, 2026-09-23): 0 inside 7d, 2 inside 30d,
    11 inside 60d, 45 inside 365d — all from bytes already paid for.  So a
    missed window is refetched by raising ``days_back`` (it clamps at 365),
    for zero extra requests.  Anything in this repo that calls
    cert_transparency "live-only" or "not backfillable" is wrong; the
    reclassification lives in ``scripts/check_freshness.py`` and
    ``scripts/backfill.py``, which this tool does not own.

    ``not_before`` is the certificate's issuance instant: present on 100% of
    live records, semantically the right field for a "recent issuance" signal,
    and — critically — **stable across re-fetches**, so the store's
    idempotency key (``store.py`` OBSERVATION_UNIQUE_KEY) keeps collapsing
    re-runs instead of appending a fresh copy of every cert daily.  Patching
    here, at the fetch boundary, repairs all four consumers in one place.

    Mutates and returns *records* (the same list the cache stores).
    """
    for rec in records:
        if isinstance(rec, dict) and not rec.get("entry_timestamp"):
            rec["entry_timestamp"] = rec.get("not_before") or ""
    return records


def _shorten_issuer(issuer: str) -> str:
    """Extract the CN from an issuer DN, or truncate if too long."""
    if not issuer:
        return ""
    # Try to extract CN=...
    for part in issuer.split(","):
        part = part.strip()
        if part.upper().startswith("CN="):
            return part[3:].strip()
    # Fallback: truncate
    return issuer[:80] + ("…" if len(issuer) > 80 else "")


def _normalize_record(rec: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Normalize a crt.sh JSON record into a clean dict."""
    not_after = _parse_timestamp(rec.get("not_after"))
    not_before = _parse_timestamp(rec.get("not_before"))
    entry_ts = _parse_timestamp(rec.get("entry_timestamp"))

    is_expired = not_after < now if not_after else None
    days_remaining = None
    if not_after and not is_expired:
        days_remaining = (not_after - now).days

    return {
        "id": rec.get("id"),
        "common_name": rec.get("common_name", ""),
        "name_value": rec.get("name_value", ""),
        "issuer": _shorten_issuer(rec.get("issuer_name", "")),
        "issuer_full": rec.get("issuer_name", ""),
        # _persist_entities_inner reads "issuer_name"; without this key every
        # stored observation carried "issuer_name": "" (all 19 live rows do),
        # so the issuer-switch signal was never actually captured.
        "issuer_name": rec.get("issuer_name", ""),
        "not_before": rec.get("not_before", ""),
        "not_after": rec.get("not_after", ""),
        "entry_timestamp": rec.get("entry_timestamp", ""),
        "serial_number": rec.get("serial_number", ""),
        "is_expired": is_expired,
        "days_remaining": days_remaining,
    }


def _format_cert(cert: dict[str, Any], *, brief: bool = False) -> str:
    """Format a normalized cert record for text output."""
    name = cert["common_name"]
    issuer = cert["issuer"]
    entry = cert["entry_timestamp"][:19] if cert["entry_timestamp"] else "?"

    status = ""
    if cert["is_expired"] is True:
        status = " [EXPIRED]"
    elif cert["days_remaining"] is not None:
        if cert["days_remaining"] < 30:
            status = f" [EXPIRES IN {cert['days_remaining']}d]"

    parts = [f"  {entry}  {name}{status}"]
    if not brief:
        parts.append(f"    Issuer: {issuer}")
        parts.append(f"    Valid: {cert['not_before'][:10]} → {cert['not_after'][:10]}")
    return "\n".join(parts)


class CertTransparencyTool(Tool):
    name = "cert_transparency"
    description = (
        "Monitor Certificate Transparency logs via crt.sh. "
        "Mode 'search' finds certificates for a domain. "
        "Mode 'subdomains' discovers subdomains via wildcard cert search — "
        "reveals internal projects, staging envs, unreleased products. "
        "Mode 'recent' shows recently issued certificates. "
        "Free, no API key. Useful for corporate infrastructure reconnaissance."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["search", "subdomains", "recent"],
                "default": "search",
                "description": (
                    "search = find certs for a domain. "
                    "subdomains = discover subdomains via wildcard CT search. "
                    "recent = recently issued certs."
                ),
            },
            "domain": {
                "type": "string",
                "description": ("Domain to search for (e.g., 'stripe.com', 'api.openai.com'). Required for all modes."),
            },
            "exclude_expired": {
                "type": "boolean",
                "default": False,
                "description": "If true, exclude expired certificates from results.",
            },
            "days_back": {
                "type": "integer",
                "default": 30,
                "description": (
                    "For 'recent' mode: how many days back. Default 30, max 365. "
                    "crt.sh serves the full history on every call and this filters it "
                    "client-side, so a wider window costs no extra request — a missed "
                    "window is recovered by raising this, not lost."
                ),
            },
            "time_budget": {
                "type": "number",
                "description": (
                    "Wall-clock seconds for the whole call (queries, retries and backoff "
                    f"together). Default {_DEFAULT_TIME_BUDGET:.0f}s, so a roster of 20 domains "
                    "fits inside the DAG node's 1200s timeout. 0 = unbounded."
                ),
            },
            "limit": {
                "type": "integer",
                "default": 50,
                "description": "Max results. Default 50, max 200.",
            },
        },
        "required": ["domain"],
    }

    def __init__(
        self,
        cache: DataCache | None = None,
        *,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    def execute(
        self,
        *,
        mode: str = "search",
        domain: str = "",
        exclude_expired: bool = False,
        days_back: int = 30,
        limit: int = 50,
        time_budget: float | None = None,
        **_: Any,
    ) -> ToolResult:
        """Run one crt.sh query set for one domain.

        ``time_budget`` bounds the whole call in wall-clock seconds — queries,
        retries and backoff sleeps together — so a caller that loops over a
        roster can predict its own runtime.  ``None`` uses
        :data:`_DEFAULT_TIME_BUDGET`; ``0`` (or negative) means unbounded.
        """
        mode = mode.lower().strip()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: {', '.join(sorted(VALID_MODES))}.",
            )

        domain = domain.strip().lower()
        if not domain:
            return ToolResult(
                success=False,
                output="A 'domain' parameter is required (e.g., 'stripe.com').",
            )

        days_back = max(1, min(days_back, 365))
        limit = max(1, min(limit, 200))

        budget = _DEFAULT_TIME_BUDGET if time_budget is None else float(time_budget)
        deadline = time.monotonic() + budget if budget > 0 else None

        if mode == "search":
            return self._execute_search(
                domain=domain,
                exclude_expired=exclude_expired,
                limit=limit,
                deadline=deadline,
            )

        if mode == "subdomains":
            return self._execute_subdomains(
                domain=domain,
                exclude_expired=exclude_expired,
                limit=limit,
                deadline=deadline,
            )

        # recent
        return self._execute_recent(
            domain=domain,
            days_back=days_back,
            limit=limit,
            deadline=deadline,
        )

    # ------------------------------------------------------------------
    # search mode
    # ------------------------------------------------------------------

    def _execute_search(
        self,
        *,
        domain: str,
        exclude_expired: bool,
        limit: int,
        deadline: float | None = None,
    ) -> ToolResult:
        records, error = self._fetch_crtsh(query=domain, exclude_expired=exclude_expired, deadline=deadline)
        if error:
            return ToolResult(success=False, output=error)

        now = datetime.now(UTC)
        certs = [_normalize_record(r, now) for r in records]
        certs.sort(key=lambda c: c["entry_timestamp"], reverse=True)
        certs = certs[:limit]

        if not certs:
            return ToolResult(
                success=True,
                output=f"CT search: no certificates found for '{domain}'.",
                data={"domain": domain, "certs": [], "count": 0},
            )

        # Count expired vs active
        active = sum(1 for c in certs if c["is_expired"] is False)
        expired = sum(1 for c in certs if c["is_expired"] is True)

        lines = [
            f"CT Certificates for '{domain}': {len(certs)} results ({active} active, {expired} expired):",
            "",
        ]
        for cert in certs:
            lines.append(_format_cert(cert))
            lines.append("")

        persisted = self._persist_entities(domain, certs)
        data: dict[str, Any] = {
            "domain": domain,
            "certs": certs,
            "count": len(certs),
            "records_fetched": len(records),
            "rows_persisted": persisted.written,
            "certs_rejected": persisted.rejected,
            "persist_error": persisted.error,
            "active": active,
            "expired": expired,
        }
        failure = self._persist_failure(domain, certs, persisted)
        if failure is not None:
            return ToolResult(success=False, output=failure, data=data)

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data=data,
        )

    # ------------------------------------------------------------------
    # subdomains mode
    # ------------------------------------------------------------------

    def _execute_subdomains(
        self,
        *,
        domain: str,
        exclude_expired: bool,
        limit: int,
        deadline: float | None = None,
    ) -> ToolResult:
        # One query, not two.  See :meth:`_execute_recent` for the measurement:
        # crt.sh answers ``q=domain`` and ``q=%.domain`` with byte-identical
        # payloads, so the second fetch only doubled the wall time and the
        # rate-limit pressure of a service that 502s under load.
        query = f"%.{domain}"
        records, error = self._fetch_crtsh(query=query, exclude_expired=exclude_expired, deadline=deadline)
        if error:
            return ToolResult(success=False, output=error)

        # Extract and count unique common_names (subdomains)
        subdomain_counts: dict[str, int] = {}
        subdomain_latest: dict[str, str] = {}
        for rec in records:
            cn = rec.get("common_name", "").strip().lower()
            if not cn:
                continue
            subdomain_counts[cn] = subdomain_counts.get(cn, 0) + 1
            entry = rec.get("entry_timestamp", "")
            if entry > subdomain_latest.get(cn, ""):
                subdomain_latest[cn] = entry

        if not subdomain_counts:
            return ToolResult(
                success=True,
                output=f"CT subdomains: no subdomains found for '{domain}'.",
                data={
                    "domain": domain,
                    "subdomains": [],
                    "count": 0,
                },
            )

        # Sort by cert count descending, then alphabetically
        sorted_subs = sorted(subdomain_counts.items(), key=lambda x: (-x[1], x[0]))
        sorted_subs = sorted_subs[:limit]

        subdomains = [
            {
                "subdomain": name,
                "cert_count": count,
                "latest_entry": subdomain_latest.get(name, ""),
            }
            for name, count in sorted_subs
        ]

        # Separate wildcards from concrete subdomains
        wildcards = [s for s in subdomains if s["subdomain"].startswith("*")]
        concrete = [s for s in subdomains if not s["subdomain"].startswith("*")]

        lines = [
            f"CT Subdomains for '{domain}': {len(subdomains)} unique names "
            f"({len(concrete)} concrete, {len(wildcards)} wildcard):",
            "",
        ]
        if concrete:
            for sub in concrete[:limit]:
                latest = sub["latest_entry"][:10] if sub["latest_entry"] else "?"
                lines.append(f"  {sub['subdomain']:50s}  {sub['cert_count']:>4d} certs  latest: {latest}")
        if wildcards:
            lines.append("")
            lines.append("  Wildcard entries:")
            for sub in wildcards:
                lines.append(f"    {sub['subdomain']}  ({sub['cert_count']} certs)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "domain": domain,
                "subdomains": subdomains,
                "concrete": concrete,
                "wildcards": wildcards,
                "count": len(subdomains),
            },
        )

    # ------------------------------------------------------------------
    # recent mode
    # ------------------------------------------------------------------

    def _execute_recent(
        self,
        *,
        domain: str,
        days_back: int,
        limit: int,
        deadline: float | None = None,
    ) -> ToolResult:
        # ONE query per domain.  There used to be a second fetch of the bare
        # domain here, on the assumption that ``%.domain`` misses the apex.
        # It does not: crt.sh treats ``q=`` as a suffix match over identities,
        # and measured 2026-09-23 the two queries return the same payload,
        # byte for byte — aqr.com 247/247 records (78,228b each) with 0 ids
        # unique to either side, cftc.gov 398/398 (156,529b), sec.gov 923/923
        # (558,095b).  The second fetch bought nothing and cost half of every
        # domain's time budget against a service that answers 502 under load
        # (4 of 8 queries in that same probe).
        records, error = self._fetch_crtsh(query=f"%.{domain}", exclude_expired=False, deadline=deadline)
        if error:
            return ToolResult(success=False, output=error)
        records_fetched = len(records)

        now = datetime.now(UTC)
        cutoff = now - timedelta(days=days_back)

        # Filter to recent entries
        recent: list[dict[str, Any]] = []
        newest: datetime | None = None
        for rec in records:
            entry_ts = _parse_timestamp(rec.get("entry_timestamp"))
            if entry_ts is None:
                continue
            if newest is None or entry_ts > newest:
                newest = entry_ts
            if entry_ts >= cutoff:
                recent.append(_normalize_record(rec, now))
        # The newest issuance crt.sh served for THIS identity, window or not.
        # A domain whose ceiling predates the window is a permanently silent
        # domain, not a quiet day, and the difference is invisible from
        # ``count`` alone: ``%.sec.gov`` returns 923 records whose newest
        # not_before is 2024-10-12, while ``q=www.sec.gov`` — the same server,
        # a narrower identity — returns 207 with a newest of 2026-08-15
        # (measured 2026-09-23).  Whoever owns the roster needs to see that;
        # this collector will not guess at replacement identities on its own.
        newest_not_before = newest.isoformat() if newest else None

        # Deduplicate by (common_name, serial_number)
        seen: set[tuple[str, str]] = set()
        deduped: list[dict[str, Any]] = []
        for cert in recent:
            key = (cert["common_name"], cert["serial_number"])
            if key not in seen:
                seen.add(key)
                deduped.append(cert)

        deduped.sort(key=lambda c: c["entry_timestamp"], reverse=True)
        # How many certs the window actually held, BEFORE ``limit`` truncates.
        # Persisting 50 of 137 and reporting "50" is a quiet partial write of
        # exactly the F-16 shape; the caller can only see it if the number it
        # did not get is in the payload too.
        certs_in_window = len(deduped)
        if certs_in_window > limit:
            log.warning(
                "cert_transparency: %s had %d certs in the last %dd but limit=%d — %d not persisted",
                domain,
                certs_in_window,
                days_back,
                limit,
                certs_in_window - limit,
            )
        deduped = deduped[:limit]

        if not deduped:
            # ``records_fetched`` is the difference between "crt.sh has nothing
            # for this domain" and "crt.sh served thousands of certs, none of
            # them inside the window" — the second is a roster or window
            # problem, and without this number it reads as a quiet day.
            return ToolResult(
                success=True,
                output=(
                    f"CT recent: no certificates issued for '{domain}' in last {days_back}d "
                    f"({records_fetched} records fetched, 0 inside the window; "
                    f"newest issuance seen: {newest_not_before or 'none'})."
                ),
                data={
                    "domain": domain,
                    "certs": [],
                    "count": 0,
                    "records_fetched": records_fetched,
                    "certs_in_window": 0,
                    "newest_not_before": newest_not_before,
                    "rows_persisted": 0,
                    "certs_rejected": 0,
                    "persist_error": None,
                    "days_back": days_back,
                },
            )

        # Count unique subdomains in recent certs
        recent_subs = set(c["common_name"] for c in deduped)

        truncated = f" — TRUNCATED, {certs_in_window} were in the window" if certs_in_window > len(deduped) else ""
        lines = [
            f"CT Recent for '{domain}': {len(deduped)} certs issued "
            f"in last {days_back}d ({len(recent_subs)} unique names){truncated}:",
            "",
        ]
        for cert in deduped:
            lines.append(_format_cert(cert))
            lines.append("")

        persisted = self._persist_entities(domain, deduped)
        data: dict[str, Any] = {
            "domain": domain,
            "certs": deduped,
            "count": len(deduped),
            "records_fetched": records_fetched,
            "certs_in_window": certs_in_window,
            "newest_not_before": newest_not_before,
            "rows_persisted": persisted.written,
            "certs_rejected": persisted.rejected,
            "persist_error": persisted.error,
            "unique_names": len(recent_subs),
            "days_back": days_back,
        }
        failure = self._persist_failure(domain, deduped, persisted)
        if failure is not None:
            return ToolResult(success=False, output=failure, data=data)

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data=data,
        )

    # ------------------------------------------------------------------
    # Entity persistence (L2)
    # ------------------------------------------------------------------

    @staticmethod
    def _persist_failure(
        domain: str,
        certs: list[dict[str, Any]],
        persisted: PersistOutcome,
    ) -> str | None:
        """The message for a batch that was fetched but did not land, else None.

        ``count > 0`` and ``rows_persisted == 0`` against a configured store is
        not a quiet day and not a skip — every record of this fetch was lost.
        The tool says so with ``success=False`` so the caller's own
        error-handling sees it; a payload that reports success here is exactly
        the shape (F-16) this collector keeps reproducing.
        """
        if not persisted.active or not certs or persisted.written > 0:
            return None
        detail = persisted.error or "no exception was raised"
        return (
            f"cert_transparency: {len(certs)} certificate(s) for '{domain}' were fetched but "
            f"0 rows reached the store ({persisted.rejected} rejected by the store's guards, "
            f"{persisted.failed} failed to write). Last error: {detail}"
        )

    def _persist_entities(self, domain: str, certs: list[dict[str, Any]]) -> PersistOutcome:
        """Register domain entity and store L2 cert observations.

        Returns a :class:`PersistOutcome` rather than a bare count, so a caller
        can tell "nothing to write" from "writing broke".
        """
        if self._store is None or entity_id_from_key is None:
            return PersistOutcome(active=False)
        if not domain:
            return PersistOutcome(active=False)
        try:
            return self._persist_entities_inner(domain, certs)
        except Exception as exc:
            # Setup (entity registration, alias, company link) failed, so no
            # cert in this batch was even attempted.
            log.exception("cert_transparency: entity setup failed for %s", domain)
            return PersistOutcome(
                written=0,
                rejected=0,
                failed=len(certs),
                error=f"{type(exc).__name__}: {exc}",
                active=True,
            )

    def _persist_entities_inner(self, domain: str, certs: list[dict[str, Any]]) -> PersistOutcome:
        assert self._store is not None  # noqa: S101
        store = self._store

        domain_eid = entity_id_from_key("domain", domain)
        store.register_entity(
            entity_type="domain",
            canonical_name=domain,
            entity_id=domain_eid,
        )
        store.add_entity_alias(domain_eid, "domain_name", domain)

        # Attempt domain → company link (Phase 36)
        self._link_domain_to_company(store, domain, domain_eid)

        written = 0
        rejected = 0
        failed = 0
        first_error: str | None = None
        for cert in certs:
            # Never fall back to time.time(): an ingestion-time observed_at
            # defeats the store's idempotency key, so every daily run would
            # append a fresh copy of every cert it has ever seen.  A cert with
            # no usable issuance instant is skipped, loudly, instead.
            ts_dt = _parse_timestamp(cert.get("entry_timestamp") or "") or _parse_timestamp(
                cert.get("not_before") or ""
            )
            if ts_dt is None:
                # Counted, not just logged. A skip nobody counts is a row
                # that left no trace of having existed.
                rejected += 1
                if first_error is None:
                    first_error = (
                        f"unparseable issuance timestamp: {cert.get('entry_timestamp') or cert.get('not_before')!r}"
                    )
                log.warning(
                    "cert_transparency: skipping cert for %s with no parseable entry_timestamp/not_before (id=%s)",
                    domain,
                    cert.get("id"),
                )
                continue

            # One bad record skips one record.  This used to be a single
            # try/except around the whole loop, so the first cert the store
            # refused (a future-dated not_before is enough, and the sort puts
            # the newest cert first) abandoned the entire domain's batch while
            # the tool still reported success=True with a non-zero count.
            try:
                store.store_entity_observation(
                    entity_id=domain_eid,
                    source_tool="cert_transparency",
                    observed_at=ts_dt.timestamp(),
                    observation_type="cert_issued",
                    depth_level=2,
                    value={
                        "is_expired": cert.get("is_expired", False),
                        "common_name": cert.get("common_name", ""),
                        # _normalize_record emits issuer_full; it used to emit
                        # no "issuer_name" at all, so every stored row read "".
                        "issuer_name": cert.get("issuer_name") or cert.get("issuer_full", ""),
                        # Without these two, distinct certs sharing a
                        # common_name and a same-second issuance collapse under
                        # the store's unique key and the second is suppressed.
                        "serial_number": cert.get("serial_number", ""),
                        "crt_id": cert.get("id"),
                    },
                )
            except _observation_rejected_types() as exc:
                # The store's write-boundary guard refused this record (out of
                # the [1990, now+1d] range, most plausibly a legal CA
                # pre-issuance).  That is one bad record, not a bad batch —
                # and it is never silently repaired: the guard stays as it is.
                rejected += 1
                if first_error is None:
                    first_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "cert_transparency: store rejected cert for %s (id=%s): %s",
                    domain,
                    cert.get("id"),
                    exc,
                )
            except Exception as exc:  # sqlite lock, disk error, schema drift…
                failed += 1
                if first_error is None:
                    first_error = f"{type(exc).__name__}: {exc}"
                log.exception("cert_transparency: failed to store cert for %s (id=%s)", domain, cert.get("id"))
            else:
                written += 1
        return PersistOutcome(
            written=written,
            rejected=rejected,
            failed=failed,
            error=first_error,
            active=True,
        )

    @staticmethod
    def _link_domain_to_company(store: Any, domain: str, domain_eid: str) -> None:
        """Attempt to link a domain entity to a company entity (Phase 36).

        Extracts the base name from the domain (e.g. ``stripe`` from
        ``api.stripe.com``) and looks it up in the instrument-universe
        company keyword map.
        """
        from agent.tools.instrument_universe import build_domain_company_map

        parts = domain.rsplit(".", 2)
        base = parts[-2] if len(parts) >= 2 else parts[0]
        base = base.lower()
        if not base:
            return
        company_map = build_domain_company_map()
        match = company_map.get(base)
        if match is None:
            return
        _canon, company_eid = match
        store.link_entities(
            entity_id_a=domain_eid,
            entity_id_b=company_eid,
            link_type="domain_owned_by",
            source="cert_transparency",
            confidence=0.8,
        )

    # ------------------------------------------------------------------
    # crt.sh fetch
    # ------------------------------------------------------------------

    def _fetch_crtsh(
        self,
        *,
        query: str,
        exclude_expired: bool = False,
        deadline: float | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch from crt.sh JSON API. Returns (records, error).

        Three attempts with 2s/4s/8s backoff on the transient failures crt.sh
        actually produces (timeout, 429, 404, 5xx). Measured 2026-09-23: fired
        back-to-back, 1 of 4 domains succeeded; with a gap and a retry, 4 of 4.

        *deadline* is a :func:`time.monotonic` instant this call must not run
        past — it caps each attempt's HTTP timeout at the time that is actually
        left and abandons the retry ladder rather than overrunning.  The caller
        gets a "budget exhausted" error naming how many attempts it got, which
        is a reached-nothing report, not a "no certificates" one.

        Every returned record passes through :func:`_backfill_entry_timestamp`
        — including one served from cache, which may predate this fix.
        """
        cache_key = {"query": query, "exclude_expired": exclude_expired}
        if self._cache:
            cached = self._cache.get("cert_transparency", cache_key)
            if cached is not None:
                return _backfill_entry_timestamp(cached), None

        params: dict[str, str] = {
            "q": query,
            "output": "json",
            "deduplicate": "Y",
        }
        if exclude_expired:
            params["exclude"] = "expired"

        resp = None
        last_error = "crt.sh fetch failed."
        for attempt, backoff in enumerate(_RETRY_BACKOFF, start=1):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining < _MIN_ATTEMPT_SECONDS:
                return [], (
                    f"crt.sh time budget exhausted for query '{query}' — "
                    f"domain not reached. ({attempt - 1} attempt(s), last: {last_error})"
                )
            timeout = _TIMEOUT if remaining is None else min(_TIMEOUT, remaining)
            retryable = True
            try:
                resp = httpx.get(
                    _CRTSH_URL,
                    params=params,
                    timeout=timeout,
                    follow_redirects=True,
                    headers={"User-Agent": _UA},
                )
                resp.raise_for_status()
                break
            except httpx.TimeoutException:
                last_error = f"crt.sh timed out after {timeout:.0f}s for query '{query}'."
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                retryable = code in _RETRY_STATUSES
                if code == 503:
                    last_error = f"crt.sh returned 503 (overloaded) for '{query}'."
                else:
                    last_error = f"crt.sh HTTP {code} for query '{query}'."
            except httpx.ConnectError:
                last_error = f"crt.sh connection failed for query '{query}'."
            resp = None
            if not retryable or attempt == len(_RETRY_BACKOFF):
                return [], f"{last_error} ({attempt} attempt(s))"
            if deadline is not None and (deadline - time.monotonic()) <= backoff:
                return [], (
                    f"{last_error} ({attempt} attempt(s); time budget exhausted "
                    f"before the {backoff:.0f}s backoff — domain not reached)"
                )
            log.info(
                "crt.sh attempt %d/%d failed (%s); retrying in %.0fs", attempt, len(_RETRY_BACKOFF), last_error, backoff
            )
            time.sleep(backoff)

        if resp is None:  # pragma: no cover - defensive
            return [], last_error

        try:
            data = resp.json()
        except (ValueError, Exception):
            return [], "crt.sh returned invalid JSON."

        if not isinstance(data, list):
            return [], "crt.sh returned unexpected response format."

        # crt.sh no longer sends entry_timestamp. Repair the record here, once,
        # so every downstream consumer (and the cache) sees a usable field.
        data = _backfill_entry_timestamp(data)

        if self._cache:
            self._cache.put("cert_transparency", cache_key, data)

        return data, None
