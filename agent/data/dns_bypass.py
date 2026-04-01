"""
TirraMind — DNS Bypass for ISP-Blocked Domains

Some ISPs (particularly in India) poison DNS responses for domains like
polymarket.com, redirecting them to a local block page IP. This module
detects the poisoning and patches ``socket.getaddrinfo`` to resolve
blocked domains via DNS-over-HTTPS (Cloudflare) instead.

Usage — call once at startup or before first request::

    from agent.data.dns_bypass import ensure_polymarket_dns

    ensure_polymarket_dns()          # auto-patches if needed
    # Now httpx / requests / urllib all resolve correctly

The patch is idempotent: calling it multiple times is safe.  It only
modifies resolution for domains that are confirmed blocked — all other
DNS resolution is untouched.
"""

from __future__ import annotations

import logging
import socket
from typing import Any

log = logging.getLogger(__name__)

# Domains that may be ISP-blocked and need bypass
_POLYMARKET_DOMAINS = (
    "data-api.polymarket.com",
    "gamma-api.polymarket.com",
    "polymarket.com",
    "clob.polymarket.com",
    "strapi-matic.polymarket.com",
)

_DOH_URL = "https://cloudflare-dns.com/dns-query"

# Module-level state — guards idempotency
_patched = False
_override_map: dict[str, str] = {}  # domain → real IPv4


def _resolve_via_doh(domain: str) -> str | None:
    """Resolve *domain* using Cloudflare DNS-over-HTTPS (JSON wire format).

    Returns an IPv4 address string or ``None`` on failure.
    """
    try:
        import httpx  # deferred — avoids circular import at module load

        resp = httpx.get(
            _DOH_URL,
            params={"name": domain, "type": "A"},
            headers={"Accept": "application/dns-json"},
            timeout=10.0,
        )
        data = resp.json()
        for answer in data.get("Answer", []):
            if answer.get("type") == 1:  # A record
                return answer["data"]
    except Exception as exc:  # noqa: BLE001
        log.debug("DoH resolution failed for %s: %s", domain, exc)
    return None


def _is_dns_poisoned(domain: str) -> bool:
    """Return True if OS-level DNS for *domain* disagrees with DoH."""
    try:
        local_ips = {
            addr[4][0]
            for addr in socket.getaddrinfo(
                domain, 443, socket.AF_INET, socket.SOCK_STREAM
            )
        }
    except socket.gaierror:
        return True  # can't resolve at all → need bypass

    real_ip = _resolve_via_doh(domain)
    if real_ip is None:
        return False  # can't verify → assume not poisoned

    return real_ip not in local_ips


_orig_getaddrinfo = socket.getaddrinfo


def _patched_getaddrinfo(
    host: str,
    port: Any,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
) -> list:
    """Drop-in replacement that routes blocked domains to real IPs."""
    if host in _override_map:
        real_ip = _override_map[host]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (real_ip, port))]
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


def ensure_polymarket_dns() -> None:
    """Detect and fix DNS poisoning for Polymarket domains.

    Safe to call multiple times (idempotent).  Only patches
    ``socket.getaddrinfo`` if at least one domain is confirmed blocked.
    """
    global _patched  # noqa: PLW0603

    if _patched:
        return

    # Probe the first two domains (the ones actually used by our tools)
    probe_domains = _POLYMARKET_DOMAINS[:2]
    any_poisoned = False

    for domain in probe_domains:
        if _is_dns_poisoned(domain):
            any_poisoned = True
            break

    if not any_poisoned:
        log.debug("Polymarket DNS looks clean — no bypass needed")
        _patched = True
        return

    log.info("ISP DNS poisoning detected for Polymarket — resolving via DoH")

    # Resolve all domains via DoH and populate override map
    for domain in _POLYMARKET_DOMAINS:
        real_ip = _resolve_via_doh(domain)
        if real_ip:
            _override_map[domain] = real_ip
            log.info("  %s → %s (DoH)", domain, real_ip)
        else:
            log.warning("  %s: DoH resolution failed, skipping", domain)

    if _override_map:
        socket.getaddrinfo = _patched_getaddrinfo  # type: ignore[assignment]
        log.info("DNS bypass active for %d domains", len(_override_map))
    else:
        log.warning("DoH resolved zero domains — DNS bypass NOT activated")

    _patched = True
