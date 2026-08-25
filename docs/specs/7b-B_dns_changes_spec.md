---
title: "Spec: 7b-B DNS Change Monitor"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/dns
---

# Spec: 7b-B DNS Change Monitor

## Goal
Monitor DNS record changes for domains via Google/Cloudflare DoH APIs to detect infrastructure shifts, provider migrations, SaaS adoption, and corporate activity signals. Three modes: resolve, diff, bulk_resolve.

## Files Affected
- `agent/tools/dns_monitor.py` — NEW (main tool, ~400 lines)
- `agent/cli.py` — add import + register
- `agent/learning/bandit.py` — already has `infrastructure_recon` arm which covers this
- `tests/test_dns_monitor_edge.py` — NEW (edge case tests)
- 10+ existing test files — update tool count assertions 33→34

## Implementation Steps

### 7b-B.1: Core DNS resolution via Google DoH
- `_resolve_domain(domain, record_types)` → queries `dns.google/resolve` for each type
- Parse JSON response: `Answer[].data`, `Answer[].TTL`, `Status`
- Domain validation: `^[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}$`, max 253 chars
- Handle: NXDOMAIN (status=3), SERVFAIL (status=2), timeout, malformed JSON
- Cloudflare as fallback when Google fails
- Internal rate limiting (no more than 20 req/sec)

### 7b-B.2: `resolve` mode
- Query A, AAAA, MX, NS, TXT, CNAME for given domain (configurable)
- Return structured records with TTL, organized by type
- Identify cloud provider from A record using known IP ranges (AWS, GCP, Cloudflare)
- Identify MX provider (Google Workspace, Microsoft 365, etc.)
- Flag low TTL (< 600s) as "imminent change" signal
- Cache results with configurable TTL (default 1hr)

### 7b-B.3: `diff` mode
- Query current records, compare against cached snapshot
- Return list of changes: `{type, action: added|removed|changed, old_value, new_value}`
- If no cached snapshot exists, store current as baseline, return "baseline established"
- Snapshot cache key: `dns_snapshot:{domain}`, TTL: 7 days (long-lived for diffing)
- Detect: new records, removed records, value changes, TTL changes

### 7b-B.4: `bulk_resolve` mode
- Accept list of domains (max 20)
- Sequential resolution with rate limiting
- Return per-domain results
- On individual domain failure, continue to next (don't abort batch)

### 7b-B.5: Tool registration + bandit integration
- Register in `cli.py` as tool #34
- `infrastructure_recon` arm already exists and covers this — update its tools list + examples
- Update count assertions in all test files: 33→34

### 7b-B.6: Edge case test suite
- Domain validation (empty, too long, invalid chars, IDN, IP address)
- Mode routing (valid modes, invalid mode, missing mode)
- Google DoH response parsing (all status codes, empty Answer, malformed JSON)
- Cloudflare failover (Google fails → Cloudflare succeeds)
- Diff logic (no baseline, no changes, additions, removals, value changes, TTL changes)
- Bulk mode (empty list, too many domains, partial failures)
- Rate limiting behavior
- Cache integration
- Tool schema validation
- Registry integration (count = 34 tools, 22 arms)

## Edge Cases
- Domain with no DNS records (NXDOMAIN)
- Domain with only wildcard records
- TXT records with very long values (SPF chains)
- MX records with priority values
- Multiple A records (round-robin / load balancing)
- Unicode/IDN domains → punycode conversion
- Rate limit exceeded → graceful degradation
- Both Google and Cloudflare down → clear error message

## Testing Plan
- Mock all HTTP calls (no real DNS queries in tests)
- Test each mode independently
- Test failover logic
- Test diff algorithm with synthetic data
- Test domain validation exhaustively
- Verify tool schema against Tool base class requirements

---

## Related

- [[7b-B_dns_changes|Research: 7B-B Dns Changes]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
