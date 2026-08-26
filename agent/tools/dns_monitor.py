"""
Tool: DNS Change Monitor — Google/Cloudflare DoH-based DNS surveillance

Google DoH: https://dns.google/resolve  (free, no auth)
Cloudflare DoH: https://cloudflare-dns.com/dns-query  (free, no auth)

Nobody systematically monitors DNS record changes as corporate activity signals.
CT logs (cert_transparency tool) reveal *new subdomains*. DNS monitoring reveals
*what changed behind them* — provider migrations, email infra shifts, SaaS adoption
tokens, and TTL drops that predict imminent infrastructure changes.

Modes:
  resolve      — Query all record types for a domain. Structured output with
                 provider identification and TTL analysis.
  diff         — Compare current DNS state against last cached snapshot.
                 Returns added/removed/changed records.
  bulk_resolve — Resolve multiple domains in a single call. Watchlist scanning.

Signal theory:
  - A record → new IP in cloud provider range = infrastructure migration
  - MX change → email provider switch (Google→O365 = Microsoft deal?)
  - MX disappears → company shutting down email = possible shutdown/acquisition
  - TXT verification tokens → SaaS adoption (Google, Microsoft, Atlassian, etc.)
  - NS change → DNS provider migration, often precedes broader infra changes
  - TTL drop (86400→300) → imminent change planned. T-1 predictive signal.
  - Two companies → same NS/IP block → M&A infrastructure consolidation
"""

from __future__ import annotations

import logging
import re
import time
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

# --- Constants ---

_GOOGLE_DOH = "https://dns.google/resolve"
_CLOUDFLARE_DOH = "https://cloudflare-dns.com/dns-query"
_TIMEOUT = 10  # seconds per DNS query
_CACHE_TTL_RESOLVE = 3600  # 1 hour for resolve results
_CACHE_TTL_SNAPSHOT = 604800  # 7 days for diff snapshots
_RATE_LIMIT_INTERVAL = 0.05  # 50ms between queries (20 req/sec)
_MAX_BULK_DOMAINS = 20

VALID_MODES = {"resolve", "diff", "bulk_resolve"}
DEFAULT_RECORD_TYPES = ["A", "AAAA", "MX", "NS", "TXT", "CNAME"]
ALL_RECORD_TYPES = {"A", "AAAA", "MX", "NS", "TXT", "CNAME", "SOA", "SRV", "CAA", "PTR"}

# DNS status codes
_DNS_STATUS = {
    0: "NOERROR",
    1: "FORMERR",
    2: "SERVFAIL",
    3: "NXDOMAIN",
    5: "REFUSED",
}

# Domain validation: labels separated by dots, TLD >= 2 chars
_DOMAIN_RE = re.compile(
    r"^(?!-)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,62}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,62}[a-zA-Z0-9])?)*"
    r"\.[a-zA-Z]{2,63}$"
)

# --- Known provider IP ranges (prefixes for fast matching) ---
# These are simplified prefix checks. For production accuracy, download full
# ranges from cloud providers, but for signal generation this is sufficient.

_CLOUD_PROVIDERS: list[tuple[str, str]] = [
    # AWS (common prefixes)
    ("3.", "AWS"),
    ("13.", "AWS"),
    ("18.", "AWS"),
    ("34.", "AWS"),
    ("35.", "AWS"),
    ("44.", "AWS"),
    ("46.51.", "AWS"),
    ("50.", "AWS"),
    ("52.", "AWS"),
    ("54.", "AWS"),
    ("63.", "AWS"),
    ("99.", "AWS"),
    ("100.", "AWS"),
    ("107.20.", "AWS"),
    ("107.21.", "AWS"),
    ("107.22.", "AWS"),
    ("107.23.", "AWS"),
    ("174.129.", "AWS"),
    ("176.34.", "AWS"),
    ("184.72.", "AWS"),
    ("184.73.", "AWS"),
    ("204.236.", "AWS"),
    # GCP
    ("34.64.", "GCP"),
    ("34.65.", "GCP"),
    ("34.66.", "GCP"),
    ("34.67.", "GCP"),
    ("34.68.", "GCP"),
    ("34.69.", "GCP"),
    ("34.70.", "GCP"),
    ("34.71.", "GCP"),
    ("34.72.", "GCP"),
    ("34.80.", "GCP"),
    ("34.96.", "GCP"),
    ("34.102.", "GCP"),
    ("34.104.", "GCP"),
    ("34.110.", "GCP"),
    ("34.120.", "GCP"),
    ("34.128.", "GCP"),
    ("34.149.", "GCP"),
    ("34.160.", "GCP"),
    ("35.184.", "GCP"),
    ("35.186.", "GCP"),
    ("35.188.", "GCP"),
    ("35.190.", "GCP"),
    ("35.192.", "GCP"),
    ("35.193.", "GCP"),
    ("35.194.", "GCP"),
    ("35.196.", "GCP"),
    ("35.197.", "GCP"),
    ("35.198.", "GCP"),
    ("35.199.", "GCP"),
    ("35.200.", "GCP"),
    ("35.201.", "GCP"),
    ("35.202.", "GCP"),
    ("35.203.", "GCP"),
    ("35.204.", "GCP"),
    ("35.205.", "GCP"),
    ("35.206.", "GCP"),
    ("35.207.", "GCP"),
    ("35.208.", "GCP"),
    ("35.209.", "GCP"),
    ("35.210.", "GCP"),
    ("35.211.", "GCP"),
    ("35.212.", "GCP"),
    ("35.213.", "GCP"),
    ("35.214.", "GCP"),
    ("35.215.", "GCP"),
    ("35.216.", "GCP"),
    ("35.217.", "GCP"),
    ("35.220.", "GCP"),
    ("35.228.", "GCP"),
    ("35.230.", "GCP"),
    ("35.232.", "GCP"),
    ("35.234.", "GCP"),
    ("35.236.", "GCP"),
    ("35.238.", "GCP"),
    ("35.240.", "GCP"),
    ("35.242.", "GCP"),
    ("35.244.", "GCP"),
    ("35.246.", "GCP"),
    # Azure
    ("13.64.", "Azure"),
    ("13.65.", "Azure"),
    ("13.66.", "Azure"),
    ("13.67.", "Azure"),
    ("13.68.", "Azure"),
    ("13.69.", "Azure"),
    ("13.70.", "Azure"),
    ("13.71.", "Azure"),
    ("13.72.", "Azure"),
    ("13.73.", "Azure"),
    ("13.74.", "Azure"),
    ("13.75.", "Azure"),
    ("13.76.", "Azure"),
    ("13.77.", "Azure"),
    ("13.78.", "Azure"),
    ("13.79.", "Azure"),
    ("13.80.", "Azure"),
    ("13.81.", "Azure"),
    ("13.82.", "Azure"),
    ("13.83.", "Azure"),
    ("13.84.", "Azure"),
    ("13.85.", "Azure"),
    ("13.86.", "Azure"),
    ("13.87.", "Azure"),
    ("13.88.", "Azure"),
    ("13.89.", "Azure"),
    ("13.90.", "Azure"),
    ("13.91.", "Azure"),
    ("13.92.", "Azure"),
    ("13.93.", "Azure"),
    ("13.94.", "Azure"),
    ("13.95.", "Azure"),
    ("20.", "Azure"),
    ("40.64.", "Azure"),
    ("40.65.", "Azure"),
    ("40.66.", "Azure"),
    ("40.67.", "Azure"),
    ("40.68.", "Azure"),
    ("40.69.", "Azure"),
    ("40.70.", "Azure"),
    ("40.71.", "Azure"),
    ("40.72.", "Azure"),
    ("40.73.", "Azure"),
    ("40.74.", "Azure"),
    ("40.75.", "Azure"),
    ("40.76.", "Azure"),
    ("40.77.", "Azure"),
    ("40.78.", "Azure"),
    ("40.79.", "Azure"),
    ("40.80.", "Azure"),
    ("40.112.", "Azure"),
    ("40.113.", "Azure"),
    ("40.114.", "Azure"),
    ("40.115.", "Azure"),
    ("40.116.", "Azure"),
    ("40.117.", "Azure"),
    ("40.118.", "Azure"),
    ("40.119.", "Azure"),
    ("40.120.", "Azure"),
    ("40.121.", "Azure"),
    ("40.122.", "Azure"),
    ("40.123.", "Azure"),
    ("40.124.", "Azure"),
    ("40.125.", "Azure"),
    ("40.126.", "Azure"),
    ("40.127.", "Azure"),
    ("51.104.", "Azure"),
    ("51.105.", "Azure"),
    ("52.136.", "Azure"),
    ("52.137.", "Azure"),
    ("52.138.", "Azure"),
    ("52.139.", "Azure"),
    ("52.140.", "Azure"),
    ("52.141.", "Azure"),
    ("52.142.", "Azure"),
    ("52.143.", "Azure"),
    ("52.146.", "Azure"),
    ("52.147.", "Azure"),
    ("52.148.", "Azure"),
    ("52.149.", "Azure"),
    ("52.150.", "Azure"),
    ("52.151.", "Azure"),
    ("52.152.", "Azure"),
    ("52.153.", "Azure"),
    ("52.154.", "Azure"),
    ("52.155.", "Azure"),
    ("52.156.", "Azure"),
    ("52.157.", "Azure"),
    ("104.40.", "Azure"),
    ("104.41.", "Azure"),
    ("104.42.", "Azure"),
    ("104.43.", "Azure"),
    ("104.44.", "Azure"),
    ("104.45.", "Azure"),
    ("104.46.", "Azure"),
    ("104.47.", "Azure"),
    ("104.208.", "Azure"),
    ("104.209.", "Azure"),
    ("104.210.", "Azure"),
    ("104.211.", "Azure"),
    ("104.214.", "Azure"),
    ("104.215.", "Azure"),
    # Cloudflare
    ("104.16.", "Cloudflare"),
    ("104.17.", "Cloudflare"),
    ("104.18.", "Cloudflare"),
    ("104.19.", "Cloudflare"),
    ("104.20.", "Cloudflare"),
    ("104.21.", "Cloudflare"),
    ("104.22.", "Cloudflare"),
    ("104.23.", "Cloudflare"),
    ("104.24.", "Cloudflare"),
    ("104.25.", "Cloudflare"),
    ("104.26.", "Cloudflare"),
    ("104.27.", "Cloudflare"),
    ("104.28.", "Cloudflare"),
    ("172.64.", "Cloudflare"),
    ("172.65.", "Cloudflare"),
    ("172.66.", "Cloudflare"),
    ("172.67.", "Cloudflare"),
    ("172.68.", "Cloudflare"),
    ("172.69.", "Cloudflare"),
    ("172.70.", "Cloudflare"),
    ("172.71.", "Cloudflare"),
    ("188.114.", "Cloudflare"),
    ("190.93.", "Cloudflare"),
    ("197.234.", "Cloudflare"),
    ("198.41.", "Cloudflare"),
    # Fastly
    ("151.101.", "Fastly"),
    ("199.232.", "Fastly"),
    # Akamai
    ("23.0.", "Akamai"),
    ("23.1.", "Akamai"),
    ("23.2.", "Akamai"),
    ("23.3.", "Akamai"),
    ("23.4.", "Akamai"),
    ("23.5.", "Akamai"),
    ("23.6.", "Akamai"),
    ("23.7.", "Akamai"),
    ("23.8.", "Akamai"),
    ("23.9.", "Akamai"),
    ("23.10.", "Akamai"),
    ("23.11.", "Akamai"),
    ("23.12.", "Akamai"),
    ("23.13.", "Akamai"),
    ("23.14.", "Akamai"),
    ("23.15.", "Akamai"),
    ("23.32.", "Akamai"),
    ("23.33.", "Akamai"),
    ("23.34.", "Akamai"),
    ("23.35.", "Akamai"),
    ("23.36.", "Akamai"),
    ("23.37.", "Akamai"),
    ("23.38.", "Akamai"),
    ("23.39.", "Akamai"),
    ("23.40.", "Akamai"),
    ("23.41.", "Akamai"),
    ("23.42.", "Akamai"),
    ("23.43.", "Akamai"),
    ("23.44.", "Akamai"),
    ("23.45.", "Akamai"),
    ("23.46.", "Akamai"),
    ("23.47.", "Akamai"),
    ("23.48.", "Akamai"),
    ("23.49.", "Akamai"),
    ("23.50.", "Akamai"),
    ("23.51.", "Akamai"),
    ("23.52.", "Akamai"),
    ("23.53.", "Akamai"),
    ("23.54.", "Akamai"),
    ("23.55.", "Akamai"),
    ("23.56.", "Akamai"),
    ("23.57.", "Akamai"),
    ("23.58.", "Akamai"),
    ("23.59.", "Akamai"),
    ("23.60.", "Akamai"),
    ("23.61.", "Akamai"),
    ("23.62.", "Akamai"),
    ("23.63.", "Akamai"),
    ("23.64.", "Akamai"),
    ("23.65.", "Akamai"),
    ("23.66.", "Akamai"),
    ("23.67.", "Akamai"),
    ("23.193.", "Akamai"),
    ("23.194.", "Akamai"),
    ("23.195.", "Akamai"),
    ("23.196.", "Akamai"),
    ("23.197.", "Akamai"),
    ("23.198.", "Akamai"),
    ("23.199.", "Akamai"),
    ("23.200.", "Akamai"),
    ("23.201.", "Akamai"),
    ("23.202.", "Akamai"),
    ("23.203.", "Akamai"),
    ("23.204.", "Akamai"),
    ("23.205.", "Akamai"),
    ("23.206.", "Akamai"),
    ("23.207.", "Akamai"),
    ("23.208.", "Akamai"),
    ("23.209.", "Akamai"),
    ("23.210.", "Akamai"),
    ("23.211.", "Akamai"),
    ("23.212.", "Akamai"),
    ("23.213.", "Akamai"),
    ("23.214.", "Akamai"),
    ("23.215.", "Akamai"),
    ("23.216.", "Akamai"),
    ("23.217.", "Akamai"),
    ("23.218.", "Akamai"),
    ("23.219.", "Akamai"),
    ("23.220.", "Akamai"),
    ("23.221.", "Akamai"),
    ("23.222.", "Akamai"),
    ("23.223.", "Akamai"),
]

# MX provider detection
_MX_PROVIDERS: list[tuple[str, str]] = [
    ("google.com", "Google Workspace"),
    ("googlemail.com", "Google Workspace"),
    ("outlook.com", "Microsoft 365"),
    ("protection.outlook.com", "Microsoft 365"),
    ("pphosted.com", "Proofpoint"),
    ("mimecast.com", "Mimecast"),
    ("mailgun.org", "Mailgun"),
    ("sendgrid.net", "SendGrid"),
    ("zoho.com", "Zoho Mail"),
    ("yahoodns.net", "Yahoo Mail"),
    ("messagelabs.com", "Broadcom/Symantec"),
    ("barracudanetworks.com", "Barracuda"),
]

# TXT verification token patterns
_TXT_TOKENS: list[tuple[str, str]] = [
    ("google-site-verification=", "Google"),
    ("MS=", "Microsoft"),
    ("atlassian-domain-verification=", "Atlassian"),
    ("facebook-domain-verification=", "Facebook/Meta"),
    ("adobe-idp-site-verification=", "Adobe"),
    ("docusign=", "DocuSign"),
    ("hubspot-developer-verification=", "HubSpot"),
    ("apple-domain-verification=", "Apple"),
    ("amazonses:", "Amazon SES"),
    ("stripe-verification=", "Stripe"),
    ("_globalsign-domain-verification=", "GlobalSign"),
    ("logmein-verification-code=", "LogMeIn/GoTo"),
    ("cisco-ci-domain-verification=", "Cisco Webex"),
    ("zoom-domain-verification=", "Zoom"),
    ("slack-domain-verification=", "Slack"),
]


def _validate_domain(domain: str) -> str | None:
    """Validate and normalize a domain. Returns error string or None if valid."""
    if not domain:
        return "Domain is required."
    if len(domain) > 253:
        return f"Domain too long ({len(domain)} chars, max 253)."
    if not _DOMAIN_RE.match(domain):
        return f"Invalid domain format: '{domain}'. Expected format: 'example.com'."
    return None


def _identify_cloud_provider(ip: str) -> str | None:
    """Identify cloud/CDN provider from an IP address using prefix matching.
    Prefers longest (most specific) prefix match."""
    best_match: str | None = None
    best_len = 0
    for prefix, provider in _CLOUD_PROVIDERS:
        if ip.startswith(prefix) and len(prefix) > best_len:
            best_match = provider
            best_len = len(prefix)
    return best_match


def _identify_mx_provider(mx_value: str) -> str | None:
    """Identify email provider from MX record value."""
    mx_lower = mx_value.lower()
    for pattern, provider in _MX_PROVIDERS:
        if pattern in mx_lower:
            return provider
    return None


def _identify_txt_tokens(txt_value: str) -> list[str]:
    """Identify SaaS verification tokens in a TXT record."""
    tokens = []
    for pattern, service in _TXT_TOKENS:
        if pattern.lower() in txt_value.lower():
            tokens.append(service)
    return tokens


def _identify_ns_provider(ns_value: str) -> str | None:
    """Identify DNS provider from NS record."""
    ns_lower = ns_value.lower().rstrip(".")
    providers = [
        ("cloudflare.com", "Cloudflare"),
        ("awsdns", "AWS Route53"),
        ("azure-dns", "Azure DNS"),
        ("googledomains.com", "Google Domains"),
        ("google.com", "Google Cloud DNS"),
        ("dynect.net", "Oracle Dyn"),
        ("nsone.net", "NS1"),
        ("ultradns", "UltraDNS"),
        ("domaincontrol.com", "GoDaddy"),
        ("registrar-servers.com", "Namecheap"),
        ("digitalocean.com", "DigitalOcean"),
        ("linode.com", "Linode/Akamai"),
        ("hetzner.com", "Hetzner"),
        ("ovh.net", "OVH"),
    ]
    for pattern, provider in providers:
        if pattern in ns_lower:
            return provider
    return None


def _parse_doh_response(data: dict[str, Any], record_type: str) -> list[dict[str, Any]]:
    """Parse a Google/Cloudflare DoH JSON response into records."""
    status = data.get("Status", -1)
    answers = data.get("Answer", [])

    records = []
    for ans in answers:
        ans_type = ans.get("type")
        # DNS type numbers: A=1, AAAA=28, CNAME=5, MX=15, NS=2, TXT=16, SOA=6
        type_map = {
            1: "A",
            28: "AAAA",
            5: "CNAME",
            15: "MX",
            2: "NS",
            16: "TXT",
            6: "SOA",
        }
        resolved_type = type_map.get(ans_type, str(ans_type))

        # Only include records matching our requested type
        if resolved_type != record_type:
            continue

        value = ans.get("data", "")
        # TXT records come wrapped in quotes from DoH
        if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]

        records.append(
            {
                "type": record_type,
                "value": value,
                "ttl": ans.get("TTL", 0),
            }
        )

    return records


def _query_doh(
    domain: str,
    record_type: str,
    *,
    provider: str = "google",
) -> tuple[list[dict[str, Any]], int, str | None]:
    """Query DNS-over-HTTPS. Returns (records, status_code, error)."""
    if provider == "google":
        url = _GOOGLE_DOH
        params = {"name": domain, "type": record_type}
        headers = {"Accept": "application/json"}
    else:
        url = _CLOUDFLARE_DOH
        params = {"name": domain, "type": record_type}
        headers = {"Accept": "application/dns-json"}

    try:
        resp = httpx.get(
            url,
            params=params,
            headers=headers,
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.TimeoutException:
        return [], -1, f"DoH timeout ({provider}) for {domain}/{record_type}"
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        return [], -1, f"DoH HTTP {code} ({provider}) for {domain}/{record_type}"
    except httpx.ConnectError:
        return [], -1, f"DoH connection failed ({provider})"

    try:
        data = resp.json()
    except (ValueError, Exception):
        return [], -1, f"DoH returned invalid JSON ({provider})"

    if not isinstance(data, dict):
        return [], -1, f"DoH returned unexpected format ({provider})"

    status = data.get("Status", -1)
    records = _parse_doh_response(data, record_type)
    return records, status, None


def _resolve_type(
    domain: str,
    record_type: str,
) -> tuple[list[dict[str, Any]], int]:
    """Resolve a single record type with Google→Cloudflare failover.
    Returns (records, dns_status_code).
    """
    records, status, error = _query_doh(domain, record_type, provider="google")
    if error:
        log.debug(
            "Google DoH failed for %s/%s: %s — trying Cloudflare",
            domain,
            record_type,
            error,
        )
        records, status, error2 = _query_doh(domain, record_type, provider="cloudflare")
        if error2:
            log.warning(
                "Both DoH providers failed for %s/%s: %s / %s",
                domain,
                record_type,
                error,
                error2,
            )
            return [], -1
    return records, status


def _resolve_all_types(
    domain: str,
    record_types: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Resolve all requested record types for a domain. Rate-limited."""
    results: dict[str, list[dict[str, Any]]] = {}
    for rtype in record_types:
        records, status = _resolve_type(domain, rtype)
        if records:
            results[rtype] = records
        elif status == 3:  # NXDOMAIN
            # Domain doesn't exist — no point querying more types
            break
        time.sleep(_RATE_LIMIT_INTERVAL)
    return results


def _analyze_records(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Extract intelligence from DNS records."""
    analysis: dict[str, Any] = {
        "cloud_providers": [],
        "mx_provider": None,
        "ns_provider": None,
        "saas_tokens": [],
        "min_ttl": None,
        "low_ttl_warning": False,
        "record_count": sum(len(v) for v in records.values()),
    }

    # Analyze A/AAAA records for cloud providers
    providers_seen: set[str] = set()
    min_ttl = float("inf")
    for rtype in ("A", "AAAA"):
        for rec in records.get(rtype, []):
            provider = _identify_cloud_provider(rec["value"])
            if provider:
                providers_seen.add(provider)
            if rec["ttl"] > 0:
                min_ttl = min(min_ttl, rec["ttl"])

    # Track min TTL across all record types
    for rtype, recs in records.items():
        for rec in recs:
            if rec["ttl"] > 0:
                min_ttl = min(min_ttl, rec["ttl"])

    analysis["cloud_providers"] = sorted(providers_seen)
    analysis["min_ttl"] = int(min_ttl) if min_ttl != float("inf") else None
    analysis["low_ttl_warning"] = min_ttl < 600 if min_ttl != float("inf") else False

    # MX analysis
    for rec in records.get("MX", []):
        provider = _identify_mx_provider(rec["value"])
        if provider:
            analysis["mx_provider"] = provider
            break

    # NS analysis
    for rec in records.get("NS", []):
        provider = _identify_ns_provider(rec["value"])
        if provider:
            analysis["ns_provider"] = provider
            break

    # TXT token analysis
    tokens: list[str] = []
    for rec in records.get("TXT", []):
        tokens.extend(_identify_txt_tokens(rec["value"]))
    analysis["saas_tokens"] = sorted(set(tokens))

    return analysis


def _compute_diff(
    old_records: dict[str, list[dict[str, Any]]],
    new_records: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Compute differences between two DNS snapshots."""
    changes: list[dict[str, Any]] = []

    all_types = set(list(old_records.keys()) + list(new_records.keys()))
    for rtype in sorted(all_types):
        old_vals = {r["value"]: r for r in old_records.get(rtype, [])}
        new_vals = {r["value"]: r for r in new_records.get(rtype, [])}

        # Added records
        for val in sorted(set(new_vals.keys()) - set(old_vals.keys())):
            changes.append(
                {
                    "type": rtype,
                    "action": "added",
                    "value": val,
                    "ttl": new_vals[val]["ttl"],
                }
            )

        # Removed records
        for val in sorted(set(old_vals.keys()) - set(new_vals.keys())):
            changes.append(
                {
                    "type": rtype,
                    "action": "removed",
                    "value": val,
                    "old_ttl": old_vals[val]["ttl"],
                }
            )

        # TTL changes on existing records
        for val in sorted(set(old_vals.keys()) & set(new_vals.keys())):
            old_ttl = old_vals[val]["ttl"]
            new_ttl = new_vals[val]["ttl"]
            if old_ttl != new_ttl:
                changes.append(
                    {
                        "type": rtype,
                        "action": "ttl_changed",
                        "value": val,
                        "old_ttl": old_ttl,
                        "new_ttl": new_ttl,
                    }
                )

    return changes


def _format_records(records: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Format DNS records for text output."""
    lines: list[str] = []
    for rtype in DEFAULT_RECORD_TYPES + ["SOA", "SRV", "CAA"]:
        recs = records.get(rtype, [])
        if not recs:
            continue
        for rec in recs:
            lines.append(f"  {rtype:6s}  TTL={rec['ttl']:<6d}  {rec['value']}")
    return lines


def _format_analysis(analysis: dict[str, Any]) -> list[str]:
    """Format analysis results for text output."""
    lines: list[str] = []
    if analysis["cloud_providers"]:
        lines.append(f"  Cloud: {', '.join(analysis['cloud_providers'])}")
    if analysis["mx_provider"]:
        lines.append(f"  Email: {analysis['mx_provider']}")
    if analysis["ns_provider"]:
        lines.append(f"  DNS Provider: {analysis['ns_provider']}")
    if analysis["saas_tokens"]:
        lines.append(f"  SaaS: {', '.join(analysis['saas_tokens'])}")
    if analysis["low_ttl_warning"]:
        lines.append(f"  ⚠ LOW TTL ({analysis['min_ttl']}s) — imminent change likely")
    elif analysis["min_ttl"] is not None:
        lines.append(f"  Min TTL: {analysis['min_ttl']}s")
    return lines


def _format_changes(changes: list[dict[str, Any]]) -> list[str]:
    """Format diff changes for text output."""
    lines: list[str] = []
    for ch in changes:
        action = ch["action"]
        rtype = ch["type"]
        if action == "added":
            lines.append(f"  + {rtype:6s}  {ch['value']}  (TTL={ch['ttl']})")
        elif action == "removed":
            lines.append(f"  - {rtype:6s}  {ch['value']}  (was TTL={ch['old_ttl']})")
        elif action == "ttl_changed":
            direction = "↓" if ch["new_ttl"] < ch["old_ttl"] else "↑"
            lines.append(f"  ~ {rtype:6s}  {ch['value']}  TTL {ch['old_ttl']}→{ch['new_ttl']} {direction}")
    return lines


class DnsMonitorTool(Tool):
    name = "dns_monitor"
    description = (
        "Monitor DNS record changes for domains via DoH (DNS-over-HTTPS). "
        "Mode 'resolve' queries A/AAAA/MX/NS/TXT/CNAME records with provider identification. "
        "Mode 'diff' compares current DNS against cached snapshot to detect changes. "
        "Mode 'bulk_resolve' resolves multiple domains in one call. "
        "Detects cloud migrations, email provider switches, SaaS adoption tokens, "
        "and TTL drops that signal imminent infrastructure changes. "
        "Free, no API key."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["resolve", "diff", "bulk_resolve"],
                "default": "resolve",
                "description": (
                    "resolve = query DNS records for a domain. "
                    "diff = compare current vs cached snapshot. "
                    "bulk_resolve = resolve multiple domains."
                ),
            },
            "domain": {
                "type": "string",
                "description": ("Domain to query (e.g., 'stripe.com'). Required for resolve and diff modes."),
            },
            "domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": ("List of domains for bulk_resolve mode. Max 20."),
            },
            "record_types": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["A", "AAAA", "MX", "NS", "TXT", "CNAME"],
                "description": ("DNS record types to query. Default: A, AAAA, MX, NS, TXT, CNAME."),
            },
        },
        "required": ["mode"],
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
        mode: str = "resolve",
        domain: str = "",
        domains: list[str] | None = None,
        record_types: list[str] | None = None,
        **_: Any,
    ) -> ToolResult:
        mode = mode.lower().strip()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: {', '.join(sorted(VALID_MODES))}.",
            )

        # Normalize record types
        if record_types:
            rtypes = [rt.upper().strip() for rt in record_types if rt.strip()]
            invalid = [rt for rt in rtypes if rt not in ALL_RECORD_TYPES]
            if invalid:
                return ToolResult(
                    success=False,
                    output=(
                        f"Invalid record type(s): {', '.join(invalid)}. Valid: {', '.join(sorted(ALL_RECORD_TYPES))}."
                    ),
                )
        else:
            rtypes = list(DEFAULT_RECORD_TYPES)

        if mode == "resolve":
            return self._execute_resolve(domain=domain, record_types=rtypes)
        if mode == "diff":
            return self._execute_diff(domain=domain, record_types=rtypes)
        return self._execute_bulk_resolve(domains=domains or [], record_types=rtypes)

    # ------------------------------------------------------------------
    # resolve mode
    # ------------------------------------------------------------------

    def _execute_resolve(self, *, domain: str, record_types: list[str]) -> ToolResult:
        domain = domain.strip().lower()
        err = _validate_domain(domain)
        if err:
            return ToolResult(success=False, output=err)

        # Check cache
        cache_key = {"mode": "resolve", "domain": domain, "types": record_types}
        if self._cache:
            cached = self._cache.get("dns_monitor", cache_key)
            if cached is not None:
                return ToolResult(
                    success=True,
                    output=cached["output"],
                    data=cached["data"],
                )

        records = _resolve_all_types(domain, record_types)

        if not records:
            return ToolResult(
                success=True,
                output=f"DNS resolve: no records found for '{domain}'. Domain may not exist (NXDOMAIN).",
                data={
                    "domain": domain,
                    "records": {},
                    "analysis": {},
                    "record_count": 0,
                },
            )

        analysis = _analyze_records(records)

        lines = [
            f"DNS Records for '{domain}' ({analysis['record_count']} records):",
            "",
        ]
        lines.extend(_format_records(records))
        lines.append("")
        lines.append("Intelligence:")
        analysis_lines = _format_analysis(analysis)
        if analysis_lines:
            lines.extend(analysis_lines)
        else:
            lines.append("  No notable signals detected.")

        output = "\n".join(lines)
        data = {
            "domain": domain,
            "records": records,
            "analysis": analysis,
            "record_count": analysis["record_count"],
        }

        self._persist_entities(domain, analysis)

        if self._cache:
            self._cache.put(
                "dns_monitor",
                cache_key,
                {"output": output, "data": data},
            )

        return ToolResult(success=True, output=output, data=data)

    # ------------------------------------------------------------------
    # Entity persistence (L2)
    # ------------------------------------------------------------------

    def _persist_entities(self, domain: str, analysis: dict[str, Any]) -> None:
        """Register domain entity and store L2 DNS observations."""
        if self._store is None or entity_id_from_key is None:
            return
        if not domain:
            return
        try:
            self._persist_entities_inner(domain, analysis)
        except Exception:
            log.exception("Entity persistence failed (non-fatal)")

    def _persist_entities_inner(self, domain: str, analysis: dict[str, Any]) -> None:
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

        store.store_entity_observation(
            entity_id=domain_eid,
            source_tool="dns_monitor",
            observed_at=time.time(),
            observation_type="dns_change",
            depth_level=2,
            value={
                "cloud_providers": analysis.get("cloud_providers", []),
                "mx_provider": analysis.get("mx_provider"),
                "ns_provider": analysis.get("ns_provider"),
                "min_ttl": analysis.get("min_ttl"),
                "low_ttl_warning": analysis.get("low_ttl_warning", False),
                "record_count": analysis.get("record_count", 0),
            },
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
            source="dns_monitor",
            confidence=0.8,
        )

    # ------------------------------------------------------------------
    # diff mode
    # ------------------------------------------------------------------

    def _execute_diff(self, *, domain: str, record_types: list[str]) -> ToolResult:
        domain = domain.strip().lower()
        err = _validate_domain(domain)
        if err:
            return ToolResult(success=False, output=err)

        # Fetch current records
        current_records = _resolve_all_types(domain, record_types)

        snapshot_key = {"mode": "snapshot", "domain": domain}

        # Load previous snapshot
        previous: dict[str, list[dict[str, Any]]] | None = None
        if self._cache:
            previous = self._cache.get("dns_monitor", snapshot_key)

        # Store current as new snapshot (long TTL)
        if self._cache:
            self._cache.put(
                "dns_monitor",
                snapshot_key,
                current_records,
            )

        if previous is None:
            # First scan — establish baseline
            analysis = _analyze_records(current_records) if current_records else {}
            lines = [
                f"DNS diff for '{domain}': baseline established ({sum(len(v) for v in current_records.values())} records).",
                "Run again later to detect changes.",
            ]
            if current_records:
                lines.append("")
                lines.append("Current records:")
                lines.extend(_format_records(current_records))
                if analysis:
                    lines.append("")
                    lines.append("Intelligence:")
                    lines.extend(_format_analysis(analysis))

            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={
                    "domain": domain,
                    "baseline_established": True,
                    "records": current_records,
                    "changes": [],
                    "analysis": analysis,
                },
            )

        # Compute diff
        changes = _compute_diff(previous, current_records)
        analysis = _analyze_records(current_records) if current_records else {}

        if not changes:
            return ToolResult(
                success=True,
                output=f"DNS diff for '{domain}': no changes detected since last scan.",
                data={
                    "domain": domain,
                    "baseline_established": False,
                    "records": current_records,
                    "changes": [],
                    "analysis": analysis,
                },
            )

        lines = [
            f"DNS diff for '{domain}': {len(changes)} change(s) detected:",
            "",
        ]
        lines.extend(_format_changes(changes))
        lines.append("")
        lines.append("Current intelligence:")
        lines.extend(_format_analysis(analysis))

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "domain": domain,
                "baseline_established": False,
                "records": current_records,
                "changes": changes,
                "analysis": analysis,
            },
        )

    # ------------------------------------------------------------------
    # bulk_resolve mode
    # ------------------------------------------------------------------

    def _execute_bulk_resolve(self, *, domains: list[str], record_types: list[str]) -> ToolResult:
        if not domains:
            return ToolResult(
                success=False,
                output="'domains' list is required for bulk_resolve mode.",
            )

        if len(domains) > _MAX_BULK_DOMAINS:
            return ToolResult(
                success=False,
                output=f"Too many domains ({len(domains)}). Maximum {_MAX_BULK_DOMAINS}.",
            )

        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for raw_domain in domains:
            d = raw_domain.strip().lower()
            err = _validate_domain(d)
            if err:
                errors.append(f"{raw_domain}: {err}")
                continue

            records = _resolve_all_types(d, record_types)
            analysis = _analyze_records(records) if records else {}
            results.append(
                {
                    "domain": d,
                    "records": records,
                    "analysis": analysis,
                    "record_count": sum(len(v) for v in records.values()),
                }
            )

        total_records = sum(r["record_count"] for r in results)
        lines = [
            f"DNS bulk resolve: {len(results)} domains, {total_records} total records"
            + (f", {len(errors)} errors" if errors else "")
            + ":",
            "",
        ]

        for r in results:
            d = r["domain"]
            count = r["record_count"]
            a = r["analysis"]
            providers = ", ".join(a.get("cloud_providers", [])) or "unknown"
            mx = a.get("mx_provider", "—")
            ttl_warn = " ⚠ LOW TTL" if a.get("low_ttl_warning") else ""
            lines.append(f"  {d:40s}  {count:>3d} records  cloud={providers}  mx={mx}{ttl_warn}")

        if errors:
            lines.append("")
            lines.append("Errors:")
            for e in errors:
                lines.append(f"  {e}")

        for r in results:
            self._persist_entities(r["domain"], r.get("analysis", {}))

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "results": results,
                "errors": errors,
                "domain_count": len(results),
                "total_records": total_records,
            },
        )
