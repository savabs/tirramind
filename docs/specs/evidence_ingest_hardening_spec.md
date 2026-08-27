---
title: "Spec: Harden /evidence/ingest against arbitrary file read"
tags:
  - doc/spec
  - topic/security
  - topic/api
  - status/active
date: 2026-08-27
---

# Spec: Harden `/evidence/ingest` against arbitrary file read

Research: [[evidence_ingest_path_traversal]]

Closes `security-auditor` findings 1 (CRITICAL) and 2 (HIGH) from the
2026-08-27 stress test.

## Goal

Make it impossible for a `/evidence/ingest` caller to read a file the
operator did not explicitly authorise, and make the ingest auth gate fail
closed rather than open.

## Non-goals

- Findings 3–6 (query-string keys, telemetry allowlist, rate limiting,
  webhook replay window). Separate specs; keeping this commit atomic.
- Changing the read-back routes. The whitelist projection there is correct;
  the defect is on the write side.

## Step 1 — Fail the ingest token gate closed

`agent/brief_server.py`, the `/evidence/ingest` branch of `do_POST`.

Current behaviour: empty `TIRRA_INGEST_TOKEN` ⇒ auth check skipped entirely.

Required behaviour, mirroring `_authorized_for`'s existing contract:

| `TIRRA_INGEST_TOKEN` | `TIRRA_REQUIRE_AUTH` | Result |
|---|---|---|
| set | any | constant-time compare; mismatch ⇒ 403 |
| empty | set (truthy) | **403, deny all**, log an error naming the cause |
| empty | unset | allow (dev mode preserved) |

Use `hmac.compare_digest` for the comparison, not `!=`.

Verifiable: unit test asserts 403 when the token is empty and
`TIRRA_REQUIRE_AUTH=1`.

## Step 2 — Confine path ingest to an opt-in allowlisted directory

`agent/brief_server.py::_serve_evidence_ingest`.

Introduce `TIRRA_INGEST_DIR`:

- **Unset ⇒ `path` mode is rejected with 400.** This is the secure default.
  `text` mode is unaffected and remains the normal way to ingest.
- **Set ⇒** resolve the request path with `os.path.realpath` and require the
  result to be inside `realpath(TIRRA_INGEST_DIR)`. Anything else ⇒ 400.

`realpath` on both sides is what defeats `..` traversal *and* symlinks that
point outside the base directory. A string-prefix check on the raw path would
not.

The rejection must not echo the requested path back to the caller — that
would turn the 400 into a filesystem-existence oracle.

Verifiable: unit tests for absolute path outside the dir, `..` traversal, a
symlink escape, and the allowed in-dir case.

## Step 3 — Fix the deploy template

`deploy/env.production.example` currently ships `TIRRA_INGEST_TOKEN=` empty,
which is what makes Step 1's fail-open reachable on a fresh deploy. Document
both variables and make the danger explicit in a comment.

Verifiable: grep the template for both names.

## Step 4 — Targeted regression tests

Per CLAUDE.md §4, a specific reproduction, not a smoke test. Add to
`tests/test_brief_server.py`:

1. empty token + `TIRRA_REQUIRE_AUTH=1` ⇒ 403 (Step 1)
2. path mode with `TIRRA_INGEST_DIR` unset ⇒ 400 (Step 2 default)
3. `../../etc/passwd`-style traversal with the dir set ⇒ 400
4. symlink inside the dir pointing outside ⇒ 400
5. legitimate file inside the dir ⇒ 200
6. the original exploit: attempt to ingest an env-shaped file from outside
   the dir ⇒ 400, and assert its contents never appear in the store

Test 6 is the one that actually encodes the vulnerability. Without it a
future refactor can silently reintroduce the chain.

## Rollback

Single commit, no schema or checkpoint impact, no migration. `git revert`
restores prior behaviour. Setting `TIRRA_INGEST_DIR` to the previous
effective root would restore the old capability without reverting the code.
