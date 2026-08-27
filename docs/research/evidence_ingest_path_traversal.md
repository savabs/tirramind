---
title: "Research: Arbitrary file read via /evidence/ingest path parameter"
tags:
  - doc/research
  - topic/security
  - topic/api
  - status/active
date: 2026-08-27
---

# Research: Arbitrary file read via `/evidence/ingest` path parameter

Found by `security-auditor` during the 2026-08-27 stress test. This is a
composition break: two components that are each individually defensible
combine into full secret disclosure.

## The chain

**Half one — unvalidated write-side path.**
`POST /evidence/ingest` accepts a JSON body `{doc_id, text|path, doc_type}`.
When `text` is absent, `path` flows unvalidated from the request into the
filesystem in `agent/evidence/ingest.py::ingest_to_store`:

```python
ext = ingestor.ingest_text(
    doc_id=doc_id,
    text=Path(path).read_text(encoding="utf-8"),
    ...
)
```

There is no allowlist, no base-directory confinement, and no rejection of
absolute paths or `..`. The server reads **any file readable by the `tirra`
user** — which includes `/opt/tirramind/.env.production` (chmod 600
`tirra:tirra`), because the API process runs as that same user.

**Half two — read-back through the paid routes.**
The ingested bytes are stored as mention `sentence` and link `evidence`
fields, and returned verbatim by the tier-gated read routes
(`/evidence/graph`, `/evidence/graph/export`, `/evidence/analytics`).

With `doc_type: "csv"` the whole file is flattened into a single "sentence"
(there is no `.` to split on in an env file), so multi-line `.env` content
comes back intact and untruncated.

`security-auditor` demonstrated this locally against a sandbox store with
fabricated secrets: `store.graph_export()` returned the `evidence` field
containing `TIRRA_PADDLE_WEBHOOK_SECRET=...` and `TIRRA_PADDLE_API_KEY=...`
verbatim.

**Impact.** Ingest token + any Entity-Graph-tier key → disclosure of the live
Paddle server API key and webhook signing secret. That is billing-account
compromise plus the ability to forge `subscription.activated` webhooks and
mint free subscriptions indefinitely.

## Why nobody owned it

The ingest-token owner assumes "if you hold the admin token, you are
trusted." The tier-gating owner assumes "read routes only ever expose the
evidence graph." Neither owns the question *does `path` point somewhere it
is allowed to point*. The gap sits between two correct components.

## The amplifier: the token gate fails OPEN

`agent/brief_server.py:321-327`:

```python
if path == "/evidence/ingest":
    token = self.headers.get("X-Ingest-Token", "")
    admin_token = os.getenv("TIRRA_INGEST_TOKEN", "").strip()
    if admin_token and token.strip() != admin_token:
        self._send(403, ...)
        return
    self._serve_evidence_ingest(body)
```

When `admin_token` is empty the `and` short-circuits and the check is skipped
entirely — ingest is world-open.

Two things make this worse than it looks:

1. This gate is **independent of `TIRRA_REQUIRE_AUTH`**. The fail-closed
   contract added to `_authorized_for` (lines 129-136) protects every GET
   route but does not cover this POST. A deploy that correctly sets
   `TIRRA_REQUIRE_AUTH=1` and leaves the ingest token blank is still
   world-open on the write half of the chain above.
2. `deploy/env.production.example:44` ships `TIRRA_INGEST_TOKEN=` **empty**.
   Any redeploy from the template reintroduces the internet-wide variant.

## Production state at time of writing (VERIFIED 2026-08-27)

```
TIRRA_INGEST_TOKEN: set, length 64
TIRRA_REQUIRE_AUTH=1
```

The live box is **not** currently world-open. The token was populated by hand
during the 2026-08-27 deploy session. That is luck, not design — nothing in
the code or the template enforces it.

## Callers of `path` mode

Verified by grep across `*.py`, `*.sh`, `*.md`: **no programmatic caller uses
the `path` ingest mode.** Every reference is documentation or agent role
files. The mode exists for manual operator use only.

This means path ingest can be made opt-in and secure-by-default without
breaking any automated caller.

## Design space considered

| Option | Verdict |
|---|---|
| Remove `path` mode from the HTTP handler entirely | Safest, but removes a capability the operator may still want locally |
| Confine to a base dir, always enabled | Still exposes the whole base dir if misconfigured to `/` |
| **Confine to an opt-in `TIRRA_INGEST_DIR`; reject path mode when unset** | **Chosen.** Secure by default (mode is off unless deliberately configured), preserves operator capability, and the confinement check is `realpath`-based so `..` and symlinks cannot escape |

## Related

- [[data_platform_telemetry_leak]] — the same class of gap (serving layer
  exposing something the storage layer never intended), fixed with a denylist
  that `security-auditor` recommends inverting to an allowlist.
