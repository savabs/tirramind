---
title: "Feature: Certificate Transparency Monitor (7b-P)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/cert-transparency
---

# Feature: Certificate Transparency Monitor (7b-P)

## Current Architecture
- Layer 1 (Surveillance Surface): `agent/tools/`
- Same pattern as sanctions_monitor.py, earthquake_proximity.py

## Data Source: crt.sh

### API Endpoint
- **URL**: `https://crt.sh/?q=QUERY&output=json`
- **Auth**: None (free, public)
- **Rate limiting**: Undocumented but server returns 503 under heavy load
- **Timeout**: Slow for large domains (google.com times out). 20-30s timeout recommended.

### Query Syntax
- `domain.com` — exact domain match
- `%.domain.com` — wildcard: all subdomains of domain.com
- Supports `deduplicate=Y` to remove duplicate log entries (124→85 for api.stripe.com)
- Supports `exclude=expired` to show only active certificates (124→7 for api.stripe.com)

### Response Fields (JSON array)
- `id`: Certificate ID (large int)
- `issuer_ca_id`: Issuer CA identifier
- `issuer_name`: Full issuer DN (e.g., "C=US, O=\"DigiCert, Inc.\", CN=...")
- `common_name`: Certificate common name (the domain)
- `name_value`: SAN/name value (often same as common_name)
- `entry_timestamp`: When cert was logged to CT (ISO 8601, e.g., "2026-03-27T07:49:06.083")
- `not_before`: Certificate validity start (ISO 8601)
- `not_after`: Certificate validity end (ISO 8601)
- `serial_number`: Hex serial number
- `result_count`: Number of CT log entries for this cert

### Observations
- Atom/RSS feed exists (`/atom?q=`) but times out — unreliable
- 503 errors common for popular domains (openai.com, palantir.com hit 503)
- Wildcard `%` search returns subdomains — powerful for infrastructure discovery
- Palantir.com wildcard: 2845 records, 234 unique subdomains (finance, health, torch, paas...)
- Response size can be large (1MB+ for wildcard on big domains)

## Signal Theory
- **New subdomain discovery** = product launch (staging.newproduct.company.com), M&A prep, expansion
- **Certificate surge** = infrastructure scaling → growth signal
- **Unusual subdomains** = project codenames, unreleased products (e.g., *.ai.company.com before AI launch)
- **Issuer changes** = security posture shifts, cost cutting (premium CA → Let's Encrypt), compliance
- **Expiring soon** = operational risk if not renewed (outage risk)
- **Cross-reference**: New subdomain pattern + job postings + press silence = stealth product launch

## Risks
- 503 on popular domains — retry with backoff, or use smaller queries
- Large response for wildcard queries — enforce limit, deduplicate
- crt.sh is a community project — no SLA, occasional downtime
- Timeout on very large result sets — 30s timeout, catch gracefully

## Architecture
- Single source: crt.sh JSON API
- 3 modes:
  - `search` — Search for certs by domain (exact or wildcard)
  - `subdomains` — Discover subdomains via wildcard cert search (unique common_names)
  - `recent` — Recent cert issuances for a domain (sorted by entry_timestamp desc)
- Always use `deduplicate=Y` to reduce noise
- Cache with 1h TTL (certs logged continuously, shorter TTL than sanctions)
- Limit results (default 50, max 200)

---

## Related

- [[7b-P_cert_transparency_spec|Spec: 7B-P Cert Transparency]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
