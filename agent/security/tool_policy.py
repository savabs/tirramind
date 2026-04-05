"""
TirraMind Agent — Tool Security Policy

Provides:
  - ToolRisk: classification of tools by danger level
  - TrustLevel: provenance tag for data flowing through the system
  - is_safe_url(): SSRF protection for HTTP-fetching tools
  - is_safe_path(): path traversal / sandbox enforcement for file tools
  - ToolPolicyGuard: gate that blocks dangerous tool calls in autonomous mode
  - get_tools_for_phase(): restrict tool visibility by orchestrator phase
"""

from __future__ import annotations

import ipaddress
import logging
import re
from enum import Enum
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)


# =====================================================================
# Enums
# =====================================================================


class ToolRisk(str, Enum):
    """How dangerous a tool is if manipulated by untrusted content."""

    READ_ONLY = "read_only"  # web_search, market_data, list_directory, ...
    DATA_FETCH = "data_fetch"  # web_browse (fetches arbitrary URLs)
    STATE_CHANGING = "state_changing"  # write_file, pipeline_query
    DANGEROUS = "dangerous"  # run_shell, execute_python


class TrustLevel(str, Enum):
    """Provenance label for content flowing through the system."""

    SYSTEM = "system"  # hardcoded prompts, config
    USER = "user"  # direct human input
    TOOL_TRUSTED = "tool_trusted"  # structured API responses (FRED, SEC, etc.)
    TOOL_UNTRUSTED = "tool_untrusted"  # web search/browse — arbitrary internet content
    MEMORY = "memory"  # retrieved from persisted memory (may be tainted)


# =====================================================================
# Tool Risk Registry
# =====================================================================

# Explicit classification for every known tool.
# Unknown tools default to STATE_CHANGING (safe-by-default).
TOOL_RISK_REGISTRY: dict[str, ToolRisk] = {
    # Read-only: return structured data, no side effects
    "market_data": ToolRisk.READ_ONLY,
    "macro_data": ToolRisk.READ_ONLY,
    "polymarket": ToolRisk.READ_ONLY,
    "polymarket_whales": ToolRisk.READ_ONLY,
    "insider_filings": ToolRisk.READ_ONLY,
    "gdelt": ToolRisk.READ_ONLY,
    "cftc": ToolRisk.READ_ONLY,
    "whale_alert": ToolRisk.READ_ONLY,
    "form144": ToolRisk.READ_ONLY,
    "finra_short_volume": ToolRisk.READ_ONLY,
    "power_grid": ToolRisk.READ_ONLY,
    "wikipedia_pageviews": ToolRisk.READ_ONLY,
    "ais_vessel": ToolRisk.READ_ONLY,
    "regulatory_gazette": ToolRisk.READ_ONLY,
    "weather_alerts": ToolRisk.READ_ONLY,
    "earthquake_proximity": ToolRisk.READ_ONLY,
    "transport_throughput": ToolRisk.READ_ONLY,
    "defi_flows": ToolRisk.READ_ONLY,
    "gov_contracts": ToolRisk.READ_ONLY,
    "academic_preprints": ToolRisk.READ_ONLY,
    "sanctions_monitor": ToolRisk.READ_ONLY,
    "cert_transparency": ToolRisk.READ_ONLY,
    "bankruptcy_court": ToolRisk.READ_ONLY,
    "dns_monitor": ToolRisk.READ_ONLY,
    "sovereign_debt": ToolRisk.READ_ONLY,
    "central_bank_balance": ToolRisk.READ_ONLY,
    "foia_requests": ToolRisk.READ_ONLY,
    "creditor_filings": ToolRisk.READ_ONLY,
    "comtrade": ToolRisk.READ_ONLY,
    "job_postings": ToolRisk.READ_ONLY,
    "building_permits": ToolRisk.READ_ONLY,
    "capital_flows": ToolRisk.READ_ONLY,
    "patent_filings": ToolRisk.READ_ONLY,
    "lobbying": ToolRisk.READ_ONLY,
    "satellite_activity": ToolRisk.READ_ONLY,
    "electricity_monitor": ToolRisk.READ_ONLY,
    "interconnection_queue": ToolRisk.READ_ONLY,
    "disease_surveillance": ToolRisk.READ_ONLY,
    "food_security": ToolRisk.READ_ONLY,
    "political_risk": ToolRisk.READ_ONLY,
    "internet_outages": ToolRisk.READ_ONLY,
    "labor_disruptions": ToolRisk.READ_ONLY,
    "migration_flows": ToolRisk.READ_ONLY,
    "energy_supply": ToolRisk.READ_ONLY,
    "treasury_receipts": ToolRisk.READ_ONLY,
    "drug_regulatory": ToolRisk.READ_ONLY,
    "global_pmi": ToolRisk.READ_ONLY,
    "consumer_sentiment": ToolRisk.READ_ONLY,
    "supply_chain_monitor": ToolRisk.READ_ONLY,
    "internet_infrastructure": ToolRisk.READ_ONLY,
    "liquidity_regime": ToolRisk.READ_ONLY,
    "backtest": ToolRisk.READ_ONLY,
    "list_directory": ToolRisk.READ_ONLY,
    "read_file": ToolRisk.READ_ONLY,
    "pipeline_query": ToolRisk.READ_ONLY,
    # Data fetch: fetches arbitrary user-specified URLs
    "web_search": ToolRisk.DATA_FETCH,
    "web_browse": ToolRisk.DATA_FETCH,
    # State changing: modifies local state
    "write_file": ToolRisk.STATE_CHANGING,
    # Dangerous: arbitrary code / command execution
    "run_shell": ToolRisk.DANGEROUS,
    "execute_python": ToolRisk.DANGEROUS,
}

# Tools whose output should be treated as untrusted internet content.
_UNTRUSTED_OUTPUT_TOOLS = frozenset({"web_search", "web_browse"})

# Tools whose output is structured API data (higher trust, but still external).
_TRUSTED_API_TOOLS = (
    frozenset(TOOL_RISK_REGISTRY.keys())
    - _UNTRUSTED_OUTPUT_TOOLS
    - {
        "run_shell",
        "execute_python",
        "write_file",
        "read_file",
        "list_directory",
    }
)


def get_tool_risk(tool_name: str) -> ToolRisk:
    """Look up risk level for a tool. Unknown tools default to STATE_CHANGING."""
    return TOOL_RISK_REGISTRY.get(tool_name, ToolRisk.STATE_CHANGING)


def get_trust_level_for_tool(tool_name: str) -> TrustLevel:
    """Determine what trust level to assign to a tool's output."""
    if tool_name in _UNTRUSTED_OUTPUT_TOOLS:
        return TrustLevel.TOOL_UNTRUSTED
    if tool_name in _TRUSTED_API_TOOLS:
        return TrustLevel.TOOL_TRUSTED
    # Shell/python/file results — trust depends on what generated the call,
    # but conservatively mark as untrusted.
    return TrustLevel.TOOL_UNTRUSTED


# =====================================================================
# SSRF Protection
# =====================================================================

# Private / reserved IP ranges that must never be fetched.
_BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),  # Carrier-grade NAT
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / cloud metadata
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),  # TEST-NET-1
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),  # Benchmarking
    ipaddress.ip_network("198.51.100.0/24"),  # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),  # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),  # Multicast
    ipaddress.ip_network("240.0.0.0/4"),  # Reserved
    ipaddress.ip_network("255.255.255.255/32"),
    # IPv6
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),  # Unique local
    ipaddress.ip_network("fe80::/10"),  # Link-local
    ipaddress.ip_network("ff00::/8"),  # Multicast
]

_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata.internal",
    }
)

# Patterns for encoded/obfuscated IPs (decimal, hex, octal).
_DECIMAL_IP_RE = re.compile(r"^(\d{8,10})$")  # e.g. 2130706433 = 127.0.0.1
_HEX_IP_RE = re.compile(r"^0x[0-9a-fA-F]+$")


def _is_blocked_ip(host: str) -> bool:
    """Check if a hostname is a private/reserved IP address."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # Not an IP literal — check for encoded forms.
        if _DECIMAL_IP_RE.match(host):
            try:
                addr = ipaddress.ip_address(int(host))
            except (ValueError, OverflowError):
                return False
        elif _HEX_IP_RE.match(host):
            try:
                addr = ipaddress.ip_address(int(host, 16))
            except (ValueError, OverflowError):
                return False
        else:
            return False

    return any(addr in net for net in _BLOCKED_IP_NETWORKS)


def is_safe_url(url: str) -> tuple[bool, str]:
    """Validate a URL is safe to fetch (no SSRF).

    Returns (is_safe, reason).
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Malformed URL"

    # Scheme check
    if parsed.scheme not in ("http", "https"):
        return False, f"Blocked scheme: {parsed.scheme}"

    hostname = (parsed.hostname or "").lower().strip().rstrip(".")

    if not hostname:
        return False, "Empty hostname"

    # Block known dangerous hostnames
    if hostname in _BLOCKED_HOSTNAMES:
        return False, f"Blocked hostname: {hostname}"

    # Block IP literals pointing to private ranges
    if _is_blocked_ip(hostname):
        return False, f"Blocked private/reserved IP: {hostname}"

    # Block ports commonly used for internal services
    port = parsed.port
    if port is not None and port not in (80, 443, 8080, 8443):
        # Allow common web ports, block everything else (e.g. 6379 Redis, 5432 Postgres)
        if port < 1024 and port not in (80, 443):
            return False, f"Blocked low port: {port}"

    # Block credentials in URL
    if parsed.username or parsed.password:
        return False, "Credentials in URL not allowed"

    return True, ""


# =====================================================================
# Path Sandboxing
# =====================================================================

# Default allowed directories for file operations.
# Configurable — the orchestrator can expand this.
_DEFAULT_ALLOWED_ROOTS: list[str] = []


def set_allowed_roots(roots: list[str]) -> None:
    """Configure allowed root directories for file operations."""
    global _DEFAULT_ALLOWED_ROOTS
    from pathlib import Path

    _DEFAULT_ALLOWED_ROOTS = [str(Path(r).resolve()) for r in roots]


def is_safe_path(path: str, allowed_roots: list[str] | None = None) -> tuple[bool, str]:
    """Check if a file path is within allowed directories.

    Returns (is_safe, reason).
    """
    from pathlib import Path

    roots = allowed_roots if allowed_roots is not None else _DEFAULT_ALLOWED_ROOTS

    # If no roots configured, allow everything (backward compat, log warning).
    if not roots:
        log.warning(
            "No allowed_roots configured for path sandboxing — allowing all paths"
        )
        return True, ""

    try:
        resolved = str(Path(path).expanduser().resolve())
    except Exception:
        return False, f"Cannot resolve path: {path}"

    for root in roots:
        if resolved.startswith(root):
            return True, ""

    return False, f"Path {resolved} is outside allowed directories"


# =====================================================================
# Policy Guard
# =====================================================================


class ToolPolicyGuard:
    """Enforces security policy on tool execution.

    Rules:
    1. DANGEROUS tools are blocked in autonomous mode unless explicitly allowed.
    2. STATE_CHANGING tools log a warning in autonomous mode.
    3. DATA_FETCH tools validate URLs against SSRF rules.
    4. All tools return results with appropriate trust labels.
    """

    def __init__(
        self,
        autonomous_mode: bool = False,
        allowed_dangerous: frozenset[str] | None = None,
        allowed_roots: list[str] | None = None,
    ) -> None:
        self._autonomous = autonomous_mode
        self._allowed_dangerous = allowed_dangerous or frozenset()
        self._allowed_roots = allowed_roots

    @property
    def autonomous_mode(self) -> bool:
        return self._autonomous

    @autonomous_mode.setter
    def autonomous_mode(self, value: bool) -> None:
        self._autonomous = value

    def check_execution(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check whether a tool call is allowed.

        Returns (allowed, denial_reason).
        """
        risk = get_tool_risk(tool_name)

        # DANGEROUS tools blocked in autonomous mode unless whitelisted
        if risk == ToolRisk.DANGEROUS:
            if self._autonomous and tool_name not in self._allowed_dangerous:
                reason = (
                    f"BLOCKED: Tool '{tool_name}' (risk={risk.value}) is not allowed "
                    f"in autonomous mode. Allowed dangerous tools: "
                    f"{sorted(self._allowed_dangerous) if self._allowed_dangerous else 'none'}"
                )
                log.warning(reason)
                return False, reason

        # STATE_CHANGING tools: warn in autonomous mode
        if risk == ToolRisk.STATE_CHANGING and self._autonomous:
            log.info("State-changing tool '%s' executing in autonomous mode", tool_name)

        # DATA_FETCH: check URL if tool has a url/query parameter
        if risk == ToolRisk.DATA_FETCH:
            url = kwargs.get("url", "")
            if url:
                safe, reason = is_safe_url(url)
                if not safe:
                    msg = f"BLOCKED: SSRF protection — {reason} (tool={tool_name}, url={url})"
                    log.warning(msg)
                    return False, msg

        # File tools: check path sandboxing
        if tool_name in ("write_file", "read_file"):
            path = kwargs.get("path", "")
            if path:
                safe, reason = is_safe_path(path, self._allowed_roots)
                if not safe:
                    msg = f"BLOCKED: Path sandbox — {reason} (tool={tool_name})"
                    log.warning(msg)
                    return False, msg

        return True, ""


# =====================================================================
# Phase-based Tool Filtering
# =====================================================================


class OrchestratorPhase(str, Enum):
    RESEARCH = "research"
    PLANNING = "planning"
    EXECUTION = "execution"
    SYNTHESIS = "synthesis"


# Which risk levels are allowed in each phase.
_PHASE_ALLOWED_RISKS: dict[OrchestratorPhase, frozenset[ToolRisk]] = {
    OrchestratorPhase.RESEARCH: frozenset({ToolRisk.READ_ONLY, ToolRisk.DATA_FETCH}),
    OrchestratorPhase.PLANNING: frozenset({ToolRisk.READ_ONLY}),
    OrchestratorPhase.EXECUTION: frozenset(
        {
            ToolRisk.READ_ONLY,
            ToolRisk.DATA_FETCH,
            ToolRisk.STATE_CHANGING,
            ToolRisk.DANGEROUS,
        }
    ),
    OrchestratorPhase.SYNTHESIS: frozenset({ToolRisk.READ_ONLY}),
}


def get_tools_for_phase(
    all_tool_names: list[str],
    phase: OrchestratorPhase,
) -> list[str]:
    """Filter tool names to only those allowed in the given phase."""
    allowed_risks = _PHASE_ALLOWED_RISKS.get(phase, frozenset())
    return [name for name in all_tool_names if get_tool_risk(name) in allowed_risks]
