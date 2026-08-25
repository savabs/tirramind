---
title: "Feature: 7b-B DNS Change Monitoring"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/dns
---

# Feature: 7b-B DNS Change Monitoring

## Current Architecture
- Layer 1 (Surveillance Surface) → `agent/tools/`
- Tool base class: `Tool` with `name`, `description`, `parameters`, `execute(**kwargs)` → `ToolResult`
- Reuse patterns from `cert_transparency.py`: httpx client, DataCache, ToolResult construction
- 33 tools currently registered, 22 bandit arms

## Observations
- No existing DNS tool in `agent/tools/`
- cert_transparency detects new subdomains via certificate issuance; DNS detects what infrastructure is behind them
- Complementary signals: CT = "new subdomain exists", DNS = "what changed and where it points"

## Data Sources (Tier 1 — free, no auth)

| Source | Endpoint | Format |
|--------|----------|--------|
| **Google DoH** | `https://dns.google/resolve?name={domain}&type={type}` | JSON |
| **Cloudflare DoH** | `https://cloudflare-dns.com/dns-query?name={domain}&type={type}` (requires `Accept: application/dns-json`) | JSON |

Both free, no auth, generous rate limits (~1000/min).

## DNS Record Types — Signal Value

| Type | Signal |
|------|--------|
| A/AAAA | Cloud provider migration (AWS/GCP/Azure IP ranges are public) |
| MX | Email provider switch (Google Workspace, O365), MX disappearing = shutdown |
| NS | DNS provider migration, precedes broader infra changes |
| TXT | SaaS adoption verification tokens (Google, Microsoft, Atlassian, Salesforce) |
| CNAME | SaaS tool adoption (Shopify, Statuspage, GitBook) |

**Key unique signal:** TTL drop (86400→300) = imminent change planned. This is a T-1 predictive signal.

## Proposed Modes
1. `resolve` — Query all record types for a domain, return structured records
2. `diff` — Compare current vs cached snapshot, return changes
3. `bulk_resolve` — Multi-domain batch with internal rate limiting

Note: `reverse_ip` dropped — too complex for v1, limited free API support.

## Risks
- CDN IP rotation creates noise (mitigate: identify known CDN ranges)
- GeoDNS returns different results from different vantage points (document limitation)
- Google/Cloudflare may rate-limit bulk queries (mitigate: internal rate limiter, provider failover)
- DNS changes alone are weak signal; strongest when fused with CT, SEC filings, news

## Dependencies
- `httpx` — already in project
- No new packages needed (using DoH JSON API, not dnspython)

---

## Related

- [[7b-B_dns_changes_spec|Spec: 7B-B Dns Changes]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
