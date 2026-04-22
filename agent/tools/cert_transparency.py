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
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

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
_TIMEOUT = 30  # crt.sh can be slow
_CACHE_TTL = 3600  # 1 hour — certs logged continuously

VALID_MODES = {"search", "subdomains", "recent"}


def _parse_timestamp(ts: str | None) -> datetime | None:
    """Parse ISO 8601 timestamp from crt.sh. Returns None on failure."""
    if not ts:
        return None
    # crt.sh uses formats like "2026-03-27T07:49:06.083" or "2026-03-27T00:00:00"
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


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
                "description": (
                    "Domain to search for (e.g., 'stripe.com', 'api.openai.com'). "
                    "Required for all modes."
                ),
            },
            "exclude_expired": {
                "type": "boolean",
                "default": False,
                "description": "If true, exclude expired certificates from results.",
            },
            "days_back": {
                "type": "integer",
                "default": 30,
                "description": "For 'recent' mode: how many days back. Default 30, max 365.",
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
        **_: Any,
    ) -> ToolResult:
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

        if mode == "search":
            return self._execute_search(
                domain=domain,
                exclude_expired=exclude_expired,
                limit=limit,
            )

        if mode == "subdomains":
            return self._execute_subdomains(
                domain=domain,
                exclude_expired=exclude_expired,
                limit=limit,
            )

        # recent
        return self._execute_recent(
            domain=domain,
            days_back=days_back,
            limit=limit,
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
    ) -> ToolResult:
        records, error = self._fetch_crtsh(
            query=domain, exclude_expired=exclude_expired
        )
        if error:
            return ToolResult(success=False, output=error)

        now = datetime.now(timezone.utc)
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
            f"CT Certificates for '{domain}': {len(certs)} results "
            f"({active} active, {expired} expired):",
            "",
        ]
        for cert in certs:
            lines.append(_format_cert(cert))
            lines.append("")

        self._persist_entities(domain, certs)

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "domain": domain,
                "certs": certs,
                "count": len(certs),
                "active": active,
                "expired": expired,
            },
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
    ) -> ToolResult:
        # Use wildcard query to find all subdomains
        query = f"%.{domain}"
        records, error = self._fetch_crtsh(query=query, exclude_expired=exclude_expired)
        if error:
            return ToolResult(success=False, output=error)

        # Also fetch the base domain itself
        base_records, _ = self._fetch_crtsh(
            query=domain, exclude_expired=exclude_expired
        )
        if base_records:
            records.extend(base_records)

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
                data={"domain": domain, "subdomains": [], "count": 0},
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
                lines.append(
                    f"  {sub['subdomain']:50s}  {sub['cert_count']:>4d} certs  "
                    f"latest: {latest}"
                )
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
    ) -> ToolResult:
        # Fetch with wildcard to catch subdomains too
        records, error = self._fetch_crtsh(query=f"%.{domain}", exclude_expired=False)
        if error:
            return ToolResult(success=False, output=error)

        # Also fetch base domain
        base_records, _ = self._fetch_crtsh(query=domain, exclude_expired=False)
        if base_records:
            records.extend(base_records)

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days_back)

        # Filter to recent entries
        recent: list[dict[str, Any]] = []
        for rec in records:
            entry_ts = _parse_timestamp(rec.get("entry_timestamp"))
            if entry_ts and entry_ts >= cutoff:
                recent.append(_normalize_record(rec, now))

        # Deduplicate by (common_name, serial_number)
        seen: set[tuple[str, str]] = set()
        deduped: list[dict[str, Any]] = []
        for cert in recent:
            key = (cert["common_name"], cert["serial_number"])
            if key not in seen:
                seen.add(key)
                deduped.append(cert)

        deduped.sort(key=lambda c: c["entry_timestamp"], reverse=True)
        deduped = deduped[:limit]

        if not deduped:
            return ToolResult(
                success=True,
                output=f"CT recent: no certificates issued for '{domain}' "
                f"in last {days_back}d.",
                data={
                    "domain": domain,
                    "certs": [],
                    "count": 0,
                    "days_back": days_back,
                },
            )

        # Count unique subdomains in recent certs
        recent_subs = set(c["common_name"] for c in deduped)

        lines = [
            f"CT Recent for '{domain}': {len(deduped)} certs issued "
            f"in last {days_back}d ({len(recent_subs)} unique names):",
            "",
        ]
        for cert in deduped:
            lines.append(_format_cert(cert))
            lines.append("")

        self._persist_entities(domain, deduped)

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "domain": domain,
                "certs": deduped,
                "count": len(deduped),
                "unique_names": len(recent_subs),
                "days_back": days_back,
            },
        )

    # ------------------------------------------------------------------
    # Entity persistence (L2)
    # ------------------------------------------------------------------

    def _persist_entities(self, domain: str, certs: list[dict[str, Any]]) -> None:
        """Register domain entity and store L2 cert observations."""
        if self._store is None or entity_id_from_key is None:
            return
        if not domain:
            return
        try:
            self._persist_entities_inner(domain, certs)
        except Exception:
            log.exception("Entity persistence failed (non-fatal)")

    def _persist_entities_inner(self, domain: str, certs: list[dict[str, Any]]) -> None:
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

        for cert in certs:
            ts_str = cert.get("entry_timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
            except (ValueError, AttributeError):
                ts = time.time()

            store.store_entity_observation(
                entity_id=domain_eid,
                source_tool="cert_transparency",
                observed_at=ts,
                observation_type="cert_issued",
                depth_level=2,
                value={
                    "is_expired": cert.get("is_expired", False),
                    "common_name": cert.get("common_name", ""),
                    "issuer_name": cert.get("issuer_name", ""),
                },
            )

    @staticmethod
    def _link_domain_to_company(
        store: Any, domain: str, domain_eid: str
    ) -> None:
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
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch from crt.sh JSON API. Returns (records, error)."""
        cache_key = {"query": query, "exclude_expired": exclude_expired}
        if self._cache:
            cached = self._cache.get("cert_transparency", cache_key)
            if cached is not None:
                return cached, None

        params: dict[str, str] = {
            "q": query,
            "output": "json",
            "deduplicate": "Y",
        }
        if exclude_expired:
            params["exclude"] = "expired"

        try:
            resp = httpx.get(
                _CRTSH_URL,
                params=params,
                timeout=_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _UA},
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            return (
                [],
                f"crt.sh timed out for query '{query}'. Try a more specific domain.",
            )
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 503:
                return [], (
                    f"crt.sh returned 503 (overloaded) for '{query}'. "
                    "Try again shortly or use a more specific domain."
                )
            return [], f"crt.sh HTTP {code} for query '{query}'."
        except httpx.ConnectError:
            return [], "crt.sh connection failed."

        try:
            data = resp.json()
        except (ValueError, Exception):
            return [], "crt.sh returned invalid JSON."

        if not isinstance(data, list):
            return [], "crt.sh returned unexpected response format."

        if self._cache:
            self._cache.put("cert_transparency", cache_key, data, ttl=_CACHE_TTL)

        return data, None
