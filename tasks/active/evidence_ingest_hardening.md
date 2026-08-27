---
title: Task — Evidence Ingest Hardening
tags:
  - doc/task
  - topic/security
  - topic/api
  - status/active
---

# Task — Evidence Ingest Hardening

Research: `docs/research/evidence_ingest_path_traversal.md`
Spec: `docs/specs/evidence_ingest_hardening_spec.md`

Started 2026-08-27.

**Context in one line:** `security-auditor`'s 2026-08-27 stress test found an
arbitrary server-side file read via the `/evidence/ingest` `path` parameter,
exfiltrable through the paid entity-graph read routes — and an auth gate that
fails open when the ingest token is empty, which the production deploy
template ships by default.

Production was verified NOT world-open at the time of discovery
(`TIRRA_INGEST_TOKEN` length 64, `TIRRA_REQUIRE_AUTH=1`), but that is an
accident of the manual deploy, not something the code or template enforces.

## Steps

- [x] Step 1 — fail the ingest token gate closed under `TIRRA_REQUIRE_AUTH`,
      constant-time compare — `_Handler._ingest_authorized`
- [x] Step 2 — confine `path` ingest to opt-in `TIRRA_INGEST_DIR`, realpath
      containment, reject when unset — `_Handler._resolve_ingest_path`
- [x] Step 3 — document both vars in `deploy/env.production.example`
- [x] Step 4 — targeted regression tests, including the exploit reproduction

## Outcome

✓ DONE — the arbitrary-file-read chain is closed at the write side.

Six regression tests in `tests/test_brief_server.py`. Verified they genuinely
catch the vulnerability rather than asserting current behaviour: with
`agent/brief_server.py` stashed, 4 of the 6 fail, and
`test_env_file_cannot_be_ingested_and_read_back` returns **200** — i.e. the
fabricated env file really was read off disk by the unpatched server. With
the fix applied it is 400 and the canary never reaches the store.

`tests/test_brief_server.py` + `tests/test_evidence.py`: 37 passed.

Not covered here — `security-auditor` findings 3–6 remain open:
query-string API keys landing in Caddy logs, the telemetry denylist that
should be an allowlist, unbounded threads / unguarded `int()` on `?limit=`,
and the 1-hour webhook replay window with no event-id dedup.
