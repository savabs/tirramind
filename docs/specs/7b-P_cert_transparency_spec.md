---
title: "Spec: cert_transparency"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/cert-transparency
---

# Spec: cert_transparency

## Goal
Monitor Certificate Transparency logs via crt.sh to discover infrastructure changes, subdomain enumeration, and certificate issuance patterns as corporate activity signals.

## Files Affected
1. **CREATE** `agent/tools/cert_transparency.py` — Tool implementation
2. **MODIFY** `agent/cli.py` — Register tool
3. **MODIFY** `agent/learning/bandit.py` — Add bandit arm
4. **CREATE** `tests/test_cert_transparency_edge.py` — Edge case tests

## Implementation Steps

### 3.1: Create cert_transparency.py skeleton
- Module docstring with signal theory
- Constants: URL, timeout, user-agent, cache TTL
- CertTransparencyTool class with name, description, parameters, execute()
- 3 modes: search, subdomains, recent

### 3.2: Implement crt.sh fetch
- `_fetch_crtsh(query, exclude_expired, deduplicate) -> tuple[list[dict], str | None]`
- GET `https://crt.sh/?q=QUERY&output=json&deduplicate=Y`
- Optional `exclude=expired`
- 30s timeout, catch TimeoutException + HTTPStatusError + ConnectError
- Cache parsed results with 1h TTL

### 3.3: Implement search mode
- Input: domain (required), exclude_expired (bool), limit
- Fetch certs for exact domain
- Normalize records: id, common_name, issuer_name (shortened), not_before, not_after, entry_timestamp, serial_number
- Sort by entry_timestamp desc
- Annotate with is_expired (not_after < now), days_remaining (for active)
- Return top `limit` results

### 3.4: Implement subdomains mode
- Input: domain (required), exclude_expired (bool), limit
- Fetch with wildcard `%.domain` query
- Extract unique common_names, deduplicate
- Filter out wildcard names (*.domain)
- Count certs per subdomain
- Sort by cert count desc (most active subdomains first)
- Return unique subdomains with metadata

### 3.5: Implement recent mode
- Input: domain (required), days_back (default 30, max 365), limit
- Fetch certs, filter by entry_timestamp within days_back
- Sort by entry_timestamp desc
- Return recent issuances with full details

### 3.6: Register in cli.py + add bandit arm
- Import CertTransparencyTool
- `registry.register(CertTransparencyTool(cache=cache))`
- Bandit arm `infrastructure_recon` with tools: cert_transparency, web_search

### 3.7: Write edge case tests
- Input validation: invalid mode, missing domain, empty domain
- Search: exact match, no results, expired filtering, limit
- Subdomains: wildcard results, dedup common_names, cert count
- Recent: date filtering, no recent certs, boundary dates
- Parsing: missing fields, null values, unusual issuer names
- Error handling: timeout, 503, connection error, malformed JSON
- Cache: hit/miss/TTL
- Output formatting

## Edge Cases
- Domain with no certs → empty result, success=True
- 503 from crt.sh → retry once, then error
- Very large result set → enforce limit after dedup
- Unicode in domain names (IDN) → pass through to crt.sh
- Common_name with wildcards (*.domain.com) in subdomain mode → handle gracefully
- Malformed JSON response → catch JSONDecodeError
- Entry timestamps with varying precision → normalize

## Testing Plan
- All tests use mocked HTTP responses
- Mock crt.sh JSON with representative cert records
- Test each mode independently
- Verify error paths produce ToolResult not exceptions
- Verify caching behavior

---

## Related

- [[7b-P_cert_transparency|Research: 7B-P Cert Transparency]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
