"""Brief server — serve the delivered Intelligence Brief over HTTP.

A minimal, dependency-light consumer surface: serves the persisted
`.tirra_delivery/intelligence_brief.json` (and markdown) that
`scripts/deliver_brief.py` writes, so an external system can fetch
"today's brief" programmatically.

Endpoints:
    GET /brief              → latest brief JSON
    GET /brief.json         → alias
    GET /brief.md           → latest brief as Markdown
    GET /status             → delivery status (count, latest, out dir)
    GET /api/v1/sources     → Data Platform tier: catalog of queryable sources
    GET /api/v1/data        → Data Platform tier: query pipeline data by source
    GET /api/v1/dag/runs    → Scheduler tier: DAG run history
    GET /api/v1/usage       → any tier: caller's own usage summary
    POST /api/v1/rotate-key → any active subscriber: self-service key rotation
                              (auth: current X-Brief-Key header; see
                              _serve_rotate_key's docstring for the contract)
    POST /api/v1/contact    → open: persists a contact-form submission
    GET /api/v1/admin/contact-messages → operator-only (X-Ingest-Token):
                              read back POST /api/v1/contact submissions
    GET /evidence/*         → Entity Graph tier: document-evidence graph + analytics
    GET /api/v1/entity-graph/*  → Entity Graph tier: REAL production graph,
                                  scoped to entities + entity_links (see
                                  docs/research/entity_graph_tier_mismatch.md)

Usage:
    .venv/bin/python -m agent.delivery.brief_server --port 8777
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent.delivery.brief_deliverer import BriefDeliverer

_DEFAULT_OUT = ".tirra_delivery"

# Product tiers that unlock each infrastructure surface. A static TIRRA_SUB_KEYS
# key or an active subscriber whose tier is in the set gets access; "scheduler"
# and "data" subscribers are treated as a superset (they paid for more surface).
_ENTITY_GRAPH_TIERS = {"entity", "data", "scheduler"}
_DATA_PLATFORM_TIERS = {"data", "scheduler"}
_SCHEDULER_TIERS = {"scheduler"}

# Hard cap on /api/v1/data's `limit` param — a metered endpoint must never
# let a single request pull an unbounded number of rows.
_MAX_DATA_LIMIT = 1000

# agent/pipeline/store.py's `pipeline_data` table is shared by two very
# different kinds of writer: L1 tools storing real external data (source=
# "cftc", "gdelt", ...) and internal DAG-stage operators storing their own
# execution telemetry under the node/operator name as `source` (found live,
# 2026-08-27: GET /api/v1/data?source=train_gnn returned
# {"trained": false, "loss_ewc": 579753920.0, ...} — a paying Data Platform
# customer reading the model's own untrained-state defect through the API
# they're paying for).
#
# C3 (2026-08-27 audit): a hand-maintained DENYLIST here is exactly how that
# leak happened — a brand-new internal stage is customer-queryable the
# instant someone adds it, with zero code change, unless every future author
# remembers to add its name to a list in an unrelated file. Inverted to an
# ALLOWLIST instead: `_external_source_allowlist()` below is *derived*, not
# guessed, from agent/pipeline/dags' own DAG registry — the same source of
# truth the executor itself uses (agent/pipeline/executor.py:
# `source=node.table_name or node.id`). A node only gets a `table_name` when
# its author intends its output to be a named, queryable dataset; nodes
# without one (train_gnn, score_entities, generate_features, run_detection,
# scan_adversarial, sac_inference, train_rl_policy, emit_portfolio,
# update_beliefs, load_models — every current internal-telemetry name) fall
# back to storing under their own node id, and are excluded here *by
# construction*, not by enumeration. A new internal stage added tomorrow
# without a `table_name` is unreachable through this API from the moment it
# ships, with no list to remember to update.
#
# The one place this can't be fully mechanical: a handful of sources
# (whale_tracking.py's `pm_trades` / `pm_resolutions` / `pm_wallet_scores`)
# are written by a manual `PipelineStore.store_data(source=...)` call inside
# a `store_result=False` FunctionOperator body rather than the executor's
# automatic per-node store, so there is no `Node.table_name` to introspect —
# every node in that DAG has `table_name=None` by construction. These are
# real, documented product data (docs/specs/polymarket_whale_spec.md names
# all three explicitly as what this surface serves), not a re-guess — unioned
# in below by name rather than silently missing from the catalog.
#
# Deliberately built from ONLY `agent.pipeline.dags.daily_collection` rather
# than `agent.pipeline.dags.get_default_dags()` (every builder module).
# Checked every other builder module individually (2026-08-27): every one of
# them — adversarial_scan, convergence_detection, entity_scoring,
# feature_generation, gnn_inference, inference, rl_training, whale_tracking,
# world_model_update — contributes ZERO `table_name` entries (their nodes are
# internal analysis/telemetry stages, `table_name=None` by construction, or
# — whale_tracking — manually stored under the names unioned in above). So
# importing any of them buys the allowlist nothing, while costing something
# real: `adversarial_scan` (via agent.adversarial → agent.fusion) transitively
# imports PyTorch, multiple seconds of cold-import latency the very
# module docstring at the top of this file promises callers this is a
# "minimal, dependency-light consumer surface" — and `whale_tracking` runs a
# pre-existing module-level live network probe on import
# (`agent.data.dns_bypass.ensure_polymarket_dns()`, a DNS-poisoning check for
# Polymarket domains). Both were caught live running this fix's own test
# suite (one as an occasional ~5s stall, the other as a stray DoH warning).
# `daily_collection.py` alone is the real, complete source of every
# `table_name`-declared dataset (48 of them) and has no such side effects.
_MANUAL_STORE_EXTERNAL_SOURCES = frozenset({"pm_trades", "pm_resolutions", "pm_wallet_scores"})

# ── 2026-08-27 triage of the 66-vs-51 gap (P0 #3) ─────────────────────────
#
# An independent security triage did a full SET DIFFERENCE (not the raw
# 66-51 arithmetic) between `SELECT DISTINCT source FROM pipeline_data` and
# this allowlist and found 23 sources present in local data but absent here.
# Verdict: ZERO of the 23 should be added. They split into three classes,
# reproduced here (not for gating — the allowlist above stays purely
# STRUCTURAL/derived, per the C3 comment — but so the next reader who greps
# for one of these 23 names finds the answer instead of re-deriving it, and
# so the regression test below has a name for "known, already classified"):
#
#   Class A — real external DAG nodes, but `table_name=None` by design
#   because their pipeline_data row is a RUN SUMMARY (counts/ticker lists),
#   not the data itself. Their actual data already lives elsewhere: in
#   `entity_observations` (deliberately excluded, see the comment above
#   `_serve_entity_graph_entities`) or, for the yield curve, under the
#   already-allowlisted `sovereign_debt` source. If a future reader sees one
#   of these 400 on /api/v1/data and is tempted to "fix" it by adding a
#   table_name, that would ship our collector's own success/failure counts
#   to a paying customer — don't.
_CLASS_A_RUN_SUMMARY_ONLY_SOURCES = frozenset(
    {
        "fetch_instruments",
        "fetch_dividends",
        "fetch_options_chains",
        "fetch_us_yield_curve",
        "fetch_cert_domains",
    }
)
#   Class B — dead legacy aliases: a single 2026-04-19 row each, superseded
#   months ago by the canonical allowlisted table_name their DAG node was
#   given afterward (fetch_cftc -> cftc, fetch_gdelt -> gdelt,
#   fetch_polymarket -> polymarket, fetch_power_demand/fetch_power_fuel ->
#   power_grid, fetch_finra_scan -> finra_short_volume). Zero writes since.
_CLASS_B_DEAD_LEGACY_ALIASES = frozenset(
    {
        "fetch_cftc",
        "fetch_gdelt",
        "fetch_polymarket",
        "fetch_power_demand",
        "fetch_power_fuel",
        "fetch_finra_scan",
    }
)
#   Class C — internal telemetry stages storing under their own node id
#   (no table_name by construction) — exactly the "internal-telemetry name"
#   class the C3 comment above already describes; listed here only so the
#   test below has a positive assertion instead of an open-ended one.
_CLASS_C_INTERNAL_TELEMETRY_SOURCES = frozenset(
    {
        "train_gnn",
        "score_entities",
        "generate_features",
        "emit_portfolio",
        "gnn_inference",
        "load_models",
        "run_detection",
        "sac_inference",
        "scan_adversarial",
        "train_rl_policy",
        "update_beliefs",
        "component_perf_gnn_epochs",
    }
)
# Union of all three — "known, already reasoned about, deliberately not
# customer-queryable." Anything found in a local DB that is in NEITHER this
# set NOR the allowlist is new and unclassified: fail closed (see the
# regression test in tests/test_brief_server_hardening.py) rather than
# guessing which bucket it belongs in — that guess is exactly how the
# original leak (P0 #3) happened.
_KNOWN_NON_EXTERNAL_SOURCES = (
    _CLASS_A_RUN_SUMMARY_ONLY_SOURCES | _CLASS_B_DEAD_LEGACY_ALIASES | _CLASS_C_INTERNAL_TELEMETRY_SOURCES
)


_EXTERNAL_SOURCE_ALLOWLIST_LOCK = threading.Lock()


def _external_source_allowlist() -> frozenset[str]:
    """The real, queryable-source catalog: every `daily_collection` DAG
    node's declared `table_name`, plus the documented manual-store sources
    above.

    Computed once per process and cached — building it is pure Python data
    traversal (see the module-selection note above: no tool registry,
    network, or heavy ML dependency is touched building it). Guarded by a
    lock purely so concurrent first-callers (ThreadingHTTPServer — one
    thread per request) wait for the one (fast) build instead of each
    running the import+build themselves.
    """
    global _EXTERNAL_SOURCE_ALLOWLIST_CACHE
    if _EXTERNAL_SOURCE_ALLOWLIST_CACHE is not None:
        return _EXTERNAL_SOURCE_ALLOWLIST_CACHE

    with _EXTERNAL_SOURCE_ALLOWLIST_LOCK:
        if _EXTERNAL_SOURCE_ALLOWLIST_CACHE is not None:
            return _EXTERNAL_SOURCE_ALLOWLIST_CACHE
        _EXTERNAL_SOURCE_ALLOWLIST_CACHE = _build_external_source_allowlist()
    return _EXTERNAL_SOURCE_ALLOWLIST_CACHE


def _build_external_source_allowlist() -> frozenset[str]:
    from agent.pipeline.dags.daily_collection import build_daily_collection_dag

    names: set[str] = set(_MANUAL_STORE_EXTERNAL_SOURCES)
    for node in build_daily_collection_dag().nodes.values():
        if node.table_name:
            names.add(node.table_name)
    return frozenset(names)


_EXTERNAL_SOURCE_ALLOWLIST_CACHE: frozenset[str] | None = None


def _truthy(value: str | None) -> bool:
    """Permissive truthy parse for env flags ('1', 'true', 'yes', 'on')."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_int(value: str | None, default: int) -> int:
    """Parse a query-param int, falling back to *default* on bad input.

    Never raises — a malformed `?limit=abc` must 400/clamp, not 500 with a
    stack trace.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: str | None, default: float | None) -> float | None:
    """Parse a query-param float, falling back to *default* on bad input.

    Same C4 class as `_safe_int`: `?since=notanumber` must not 500.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _valid_key(request_key: str | None) -> bool:
    """True if the request carries a valid subscriber key (any tier)."""
    return _authorized_for(request_key, allowed_tiers=None)


def _authorized_for(request_key: str | None, allowed_tiers: set[str] | None) -> bool:
    """True if the request key grants access to a product gated by `allowed_tiers`.

    `allowed_tiers=None` means "any active subscriber, regardless of tier"
    (used for the base brief). A static TIRRA_SUB_KEYS key always grants
    access to every tier — it's the admin/dev bypass.

    Open (dev mode) only when neither static keys nor Paddle webhooks are
    configured — never based on a stale shared store file.
    """
    configured = os.getenv("TIRRA_SUB_KEYS", "").strip()
    paddle_secret = os.getenv("TIRRA_PADDLE_WEBHOOK_SECRET", "").strip()

    # Fail CLOSED in production, always.
    #
    # The dev-mode-open branch below is convenient locally and catastrophic in
    # production: with an env file whose auth vars are present but empty — which
    # is exactly how deploy/env.production.example ships — an anonymous caller
    # is authorised for every paid tier. A deploy that loses or truncates its
    # env file silently converts the entire paid API into a free one, with no
    # error and nothing in the logs.
    #
    # TIRRA_REQUIRE_AUTH=1 removes that failure mode: no credentials configured
    # means nobody gets in, rather than everybody.
    if _truthy(os.getenv("TIRRA_REQUIRE_AUTH")):
        if not configured and not paddle_secret:
            logging.getLogger(__name__).error(
                "TIRRA_REQUIRE_AUTH is set but neither TIRRA_SUB_KEYS nor "
                "TIRRA_PADDLE_WEBHOOK_SECRET is configured — denying all "
                "requests. Configure credentials or unset TIRRA_REQUIRE_AUTH."
            )
            return False
    elif not configured and not paddle_secret:
        # Dev mode: nothing configured → serve open.
        return True

    if not request_key:
        return False
    key = request_key.strip()

    if configured and key in {k.strip() for k in configured.split(",")}:
        return True

    if paddle_secret:
        from agent.payments.handler import SubscriberStore

        store = SubscriberStore()
        if not store.is_active_key(key):
            return False
        if allowed_tiers is None:
            return True
        return store.tier_of_key(key) in allowed_tiers

    return False


def _log_usage(key: str | None, endpoint: str) -> None:
    """Best-effort usage metering. Never raises — a metering failure must
    never break the request it's trying to measure."""
    if not key:
        return
    try:
        from agent.payments.handler import SubscriberStore
        from agent.payments.usage import UsageStore

        tier = SubscriberStore().tier_of_key(key.strip())
        UsageStore().log(key_id=key.strip(), endpoint=endpoint, tier=tier)
    except Exception as exc:  # metering must never break the request
        import logging

        logging.getLogger(__name__).warning("[usage] failed to log endpoint=%s: %s", endpoint, exc)


# ── C4: bounded concurrency ───────────────────────────────────────────────
#
# ThreadingHTTPServer spawns one thread per connection with no cap — a
# single customer opening many connections can pile up unbounded threads all
# hitting the same SQLite-backed stores at once (agent/pipeline/store.py,
# agent/payments/*). This does not make the server single-threaded; it just
# bounds how many requests are *actively being handled* at any instant. A
# request beyond the cap blocks (queues) waiting for a slot rather than
# spawning a 51st concurrent SQLite writer. Configurable so a real deployment
# can size it to its box; not a rewrite of the server.
_REQUEST_SEMAPHORE = threading.BoundedSemaphore(_safe_int(os.getenv("TIRRA_MAX_CONCURRENT_REQUESTS"), 20))


class _RateLimiter:
    """In-process, in-memory sliding-window rate limiter keyed by an
    arbitrary string (an IP, a txn_id, ...).

    Correct ONLY because the deployment is a single process (ThreadingHTTPServer,
    no horizontal fleet) — see the /api/v1/claim threat-model note below. If
    that ever changes this needs to move to a shared store (e.g. the same
    SQLite DB), or a fleet-wide bypass becomes possible.
    """

    def __init__(self, max_calls: int, window_s: float) -> None:
        self.max_calls = max_calls
        self.window_s = window_s
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Returns (allowed, retry_after_s). Consumes one call on success."""
        now = time.time() if now is None else now
        cutoff = now - self.window_s
        with self._lock:
            hits = self._hits.get(key, [])
            hits = [h for h in hits if h > cutoff]
            if len(hits) >= self.max_calls:
                self._hits[key] = hits
                retry_after = max(0.0, hits[0] + self.window_s - now)
                return False, retry_after
            hits.append(now)
            self._hits[key] = hits
            return True, 0.0


# ── GET /api/v1/claim — threat model: transaction id enumeration ─────────
#
# A leaked/logged/screenshotted `txn_...` id is the realistic risk (not
# brute force — see docs pulled into the design doc: txn ids are ULID-based,
# 2^80 random bits). Two independent, generous-for-the-legit-poll-loop caps:
# Sized against welcome.html's ACTUAL poll cadence, not a guess: it fires at
# t=0,2,5,10,18,28,38,48,58,...,118s — 15 calls for ONE legitimate purchase.
# The previous caps (8/txn, 20/IP) were below that, so a normal customer whose
# webhook landed ~60s late got a 429 mid-poll and a false "setup failed"
# screen. Headroom here covers the full poll loop plus a page refresh; the
# real defence against a leaked txn id is single-use claiming, not this.
_CLAIM_TXN_LIMITER = _RateLimiter(max_calls=24, window_s=600)  # per txn_id / 10 min
_CLAIM_IP_LIMITER = _RateLimiter(max_calls=60, window_s=3600)  # per source IP / hour
_TXN_ID_RE = re.compile(r"^txn_[A-Za-z0-9_-]{6,64}$")

# ── POST /api/v1/contact ──────────────────────────────────────────────────
_CONTACT_IP_LIMITER = _RateLimiter(max_calls=5, window_s=3600)  # per source IP / hour
_MAX_CONTACT_BODY_BYTES = 8 * 1024  # the form is 4 short fields — 8KB is generous
_MAX_CONTACT_FIELD_LEN = 300
_MAX_CONTACT_MESSAGE_LEN = 8000
_DEFAULT_CONTACT_LOG = ".tirra_opportunities/contact_messages.jsonl"

# ── GET /api/v1/admin/contact-messages ────────────────────────────────────
# Operator-only read of the POST /api/v1/contact log — see _serve_contact's
# docstring for why messages land in a JSONL file (no mail service exists).
_MAX_CONTACT_READ_LIMIT = 500

# ── POST /api/v1/rotate-key ───────────────────────────────────────────────
#
# welcome.html shows the API key exactly ONCE and there is no mailbox (MX
# records absent — see ground truth); a customer who loses their key has no
# recovery path without this route. agent/payments/handler.py already ships
# the store-layer support (`SubscriberStore.rotate_key_for_api_key`) and its
# own docstring specifies the exact contract this route implements — this
# handler is deliberately a thin translation layer over it, same pattern as
# every other route in this file that defers to agent/payments/*.
#
# Rotating a key is destructive (the old key dies immediately, with no
# undo — there is no revocation list, `_by_api_key` only ever matches the
# CURRENT `api_key` field) and authenticated (must present the current valid
# key — there is no "rotate by subscription_id" path exposed here, only
# self-service). Rate-limited on TWO axes for two different reasons:
#   - per presented key: bounds how many times one credential can be spun
#     even by its legitimate holder in a short window (a buggy client stuck
#     in a retry loop must not be able to invalidate its own key every
#     request forever);
#   - per source IP: bounds how many *distinct* keys one caller can probe
#     rotation against, independent of whether any of them are valid.
# Format-validated BEFORE either limiter or the store lookup runs (like
# `_TXN_ID_RE` for /api/v1/claim) so a flood of garbage strings can't grow
# the in-memory rate-limiter dict unbounded — see _RateLimiter's per-key
# `dict` note.
_ROTATE_KEY_LIMITER = _RateLimiter(max_calls=5, window_s=3600)  # per presented key / hour
_ROTATE_IP_LIMITER = _RateLimiter(max_calls=20, window_s=3600)  # per source IP / hour
_API_KEY_RE = re.compile(r"^tirra_[A-Za-z0-9_-]{10,100}$")


class _Handler(BaseHTTPRequestHandler):
    server_version = "AWOSBrief/0.1"  # type: ignore[assignment]

    deliverer: BriefDeliverer  # class attr set by serve()

    # C6: the base handler's `version_string()` returns
    # f"{server_version} {sys_version}", and `sys_version` defaults to
    # "Python/<full interpreter version>" — every response (including the
    # Server header and log lines) was leaking the exact Python patch version
    # of the box. Overridden below to just the app version.
    def version_string(self) -> str:  # noqa: N802
        return self.server_version

    # ── HTTP verb handlers ───────────────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802
        with _REQUEST_SEMAPHORE:
            self._do_GET()

    def _do_GET(self) -> None:  # noqa: N802
        from urllib.parse import parse_qs, urlsplit

        parts = urlsplit(self.path)
        path = parts.path
        query = parse_qs(parts.query)

        # Unauthenticated by design (see agent/payments/claim.py's ownership-
        # split docstring) — dispatched before key extraction so it is never
        # affected by TIRRA_REJECT_QUERY_KEYS or key-parsing at all.
        if path == "/api/v1/claim":
            self._serve_claim(query)
            return

        # Admin-only, gated by the same `X-Ingest-Token` check as
        # POST /evidence/ingest rather than the subscriber-key system below —
        # this is an operator surface, not a paid tier, so it is dispatched
        # before subscriber-key extraction (same reasoning as /api/v1/claim
        # above: it must never be affected by TIRRA_REJECT_QUERY_KEYS or
        # subscriber key parsing).
        if path == "/api/v1/admin/contact-messages":
            if not self._ingest_authorized():
                self._send(403, "application/json", json.dumps({"ok": False, "error": "invalid ingest token"}))
                return
            self._serve_admin_contact_messages(query)
            return

        key, key_error = self._extract_key(query)
        if key_error is not None:
            self._send(400, "application/json", json.dumps(key_error))
            return

        if path == "/buy":
            self._serve_buy(query)
            return

        if path in ("/", "/landing", "/index.html"):
            self._serve_landing()
            return

        if path in (
            "/evidence/graph",
            "/evidence/stats",
            "/evidence/analytics",
            "/evidence/graph/export",
            "/evidence/graph/centrality",
        ):
            if not _authorized_for(key, _ENTITY_GRAPH_TIERS):
                self._send(403, "text/plain", "subscribe (Entity Graph tier) required — see /buy\n")
                return
            _log_usage(key, path)
            if path == "/evidence/graph":
                self._serve_evidence_graph(query)
            elif path == "/evidence/stats":
                self._serve_evidence_stats()
            elif path == "/evidence/analytics":
                self._serve_evidence_analytics(query)
            elif path == "/evidence/graph/export":
                self._serve_evidence_export()
            else:
                self._serve_evidence_centrality(query)
            return

        if path in (
            "/api/v1/entity-graph/entities",
            "/api/v1/entity-graph/entity",
            "/api/v1/entity-graph/links",
        ):
            if not _authorized_for(key, _ENTITY_GRAPH_TIERS):
                self._send(403, "text/plain", "subscribe (Entity Graph tier) required — see /buy\n")
                return
            _log_usage(key, path)
            if path == "/api/v1/entity-graph/entities":
                self._serve_entity_graph_entities(query)
            elif path == "/api/v1/entity-graph/entity":
                self._serve_entity_graph_entity(query)
            else:
                self._serve_entity_graph_links(query)
            return

        if path == "/api/v1/sources":
            if not _authorized_for(key, _DATA_PLATFORM_TIERS):
                self._send(403, "text/plain", "subscribe (Data Platform tier) required — see /buy\n")
                return
            _log_usage(key, path)
            self._serve_sources()
            return

        if path == "/api/v1/data":
            if not _authorized_for(key, _DATA_PLATFORM_TIERS):
                self._send(403, "text/plain", "subscribe (Data Platform tier) required — see /buy\n")
                return
            _log_usage(key, path)
            self._serve_data_api(query)
            return

        if path == "/api/v1/dag/runs":
            if not _authorized_for(key, _SCHEDULER_TIERS):
                self._send(403, "text/plain", "subscribe (Scheduler tier) required — see /buy\n")
                return
            _log_usage(key, path)
            self._serve_dag_runs(query)
            return

        if path == "/api/v1/usage":
            if not _valid_key(key):
                self._send(403, "text/plain", "subscribe required — see /buy\n")
                return
            self._serve_usage(key, query)
            return

        if path in ("/brief", "/brief.json", "/brief.md"):
            if not _valid_key(key):
                self._send(403, "text/plain", "subscribe required — see /buy\n")
                return
            _log_usage(key, path)
            if path == "/brief.md":
                self._serve_md()
            else:
                self._serve_json()
        elif path == "/status":
            self._serve_status()
        else:
            self._send(404, "text/plain", "not found\n")

    # C2 (2026-08-27 audit): ?key=... in the query string lands in Caddy's
    # access log (full URI logged) in cleartext, and can leak via the
    # Referer header on any outbound link/asset request the response body
    # triggers. The fix keeps ?key= working (tests/test_brief_server.py,
    # owned by another agent, uses it extensively) but:
    #   - always logs a deprecation warning when a key arrives via the query
    #     string, so it shows up in server logs pushing callers to migrate;
    #   - adds TIRRA_REJECT_QUERY_KEYS (default OFF) — when truthy, a
    #     query-string key is hard-rejected with 400 rather than silently
    #     accepted, so production can flip it on immediately without waiting
    #     for every customer to migrate to the X-Brief-Key header first.
    # Precedence when a query key is rejected: the header is NOT silently
    # substituted in its place if a query key was ALSO present — that would
    # make the reject-flag's behavior depend on which key happened to be
    # valid, instead of being an unconditional protocol rule.
    def _extract_key(self, query: dict) -> tuple[str | None, dict | None]:
        """Returns (key, error_body). error_body is not None iff the request
        must be rejected outright (400) before any route/auth logic runs."""
        query_key = (query.get("key") or [None])[0]
        header_key = self.headers.get("X-Brief-Key")
        if query_key:
            logging.getLogger(__name__).warning(
                "[deprecated] API key passed via ?key= query string (leaks into "
                "access logs / Referer headers) — use the X-Brief-Key header instead"
            )
            if _truthy(os.getenv("TIRRA_REJECT_QUERY_KEYS")):
                return None, {
                    "ok": False,
                    "error": "query-string API keys are disabled — pass the key via the X-Brief-Key header instead",
                }
            return query_key, None
        return header_key, None

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _serve_landing(self) -> None:
        landing = Path(os.getenv("TIRRA_LANDING_HTML", "products/brief_subscription/index.html"))
        try:
            body = landing.read_text(encoding="utf-8")
        except OSError:
            self._send(200, "text/plain", "Opportunity Brief — see /brief or /buy\n")
            return
        self._send(200, "text/html; charset=utf-8", body)

    def _serve_buy(self, query) -> None:
        tier = (query.get("tier") or [None])[0]
        url = None
        if tier:
            url = os.getenv(f"TIRRA_BUY_URL_{tier.strip().upper()}")
        url = url or os.getenv("TIRRA_BUY_URL")
        if not url:
            self._send(
                200,
                "text/plain",
                "buy link not configured — set TIRRA_BUY_URL"
                + (f" or TIRRA_BUY_URL_{tier.upper()}" if tier else "")
                + "\n",
            )
            return
        body = f'<meta http-equiv="refresh" content="0;url={url}">Subscribing…'
        self._send(200, "text/html; charset=utf-8", body)

    # ── GET /api/v1/claim — the one unauthenticated route (no key exists yet
    # at this point in the flow) ─────────────────────────────────────────────
    #
    # Routing/HTTP/status-codes/CORS/rate-limiting only — this method must not
    # reimplement any Paddle-verification or claim/idempotency-state logic;
    # that lives in agent.payments.claim (ClaimResult / claim_transaction /
    # ClaimStore), same layering _authorized_for already uses for
    # SubscriberStore/UsageStore. See agent/payments/claim.py's own docstring
    # for the ownership split.
    def _serve_claim(self, query) -> None:
        origin = os.getenv("TIRRA_WEB_ORIGIN", "https://tirramind.com")
        cors = {"Access-Control-Allow-Origin": origin}

        txn = (query.get("txn") or [None])[0]
        if not txn or not _TXN_ID_RE.match(txn):
            self._send(
                400,
                "application/json",
                json.dumps({"ok": False, "status": "bad_request", "message": "missing or malformed txn parameter"}),
                extra_headers=cors,
            )
            return

        allowed, retry_after = _CLAIM_TXN_LIMITER.allow(f"txn:{txn}")
        if allowed:
            ip = self.client_address[0] if self.client_address else "unknown"
            allowed, retry_after = _CLAIM_IP_LIMITER.allow(f"ip:{ip}")
        if not allowed:
            retry = max(1, int(retry_after) + 1)
            self._send(
                429,
                "application/json",
                json.dumps({"ok": False, "status": "rate_limited", "retry_after_s": retry}),
                extra_headers={**cors, "Retry-After": str(retry)},
            )
            return

        try:
            from agent.payments.claim import ClaimStore, claim_transaction
            from agent.payments.client import PaddleClient
            from agent.payments.config import PaddleConfig
            from agent.payments.handler import SubscriberStore

            cfg = PaddleConfig.from_env()
            result = claim_transaction(
                txn,
                paddle_client=PaddleClient(cfg),
                subscriber_store=SubscriberStore(),
                claim_store=ClaimStore(),
            )
        except Exception as exc:  # config error, unexpected exception from the payments layer
            logging.getLogger(__name__).warning("[claim] unexpected error txn=%s: %s", txn, exc)
            self._send(
                502,
                "application/json",
                json.dumps(
                    {
                        "ok": False,
                        "status": "upstream_error",
                        "message": "could not verify your transaction — try again shortly",
                    }
                ),
                extra_headers=cors,
            )
            return

        self._respond_claim_result(result, extra_headers=cors)

    def _respond_claim_result(self, result, extra_headers: dict[str, str]) -> None:
        status = result.status
        if status == "unknown_transaction":
            self._send(
                404,
                "application/json",
                json.dumps({"ok": False, "status": status, "message": "no transaction found for this id"}),
                extra_headers=extra_headers,
            )
        elif status == "not_completed":
            self._send(
                422,
                "application/json",
                json.dumps(
                    {
                        "ok": False,
                        "status": status,
                        "transaction_status": result.transaction_status,
                        "message": "this transaction has not completed successfully",
                    }
                ),
                extra_headers=extra_headers,
            )
        elif status == "pending":
            self._send(
                202,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "status": status,
                        "retry_after_s": 3,
                        "message": "payment received — provisioning your key",
                    }
                ),
                extra_headers={**extra_headers, "Retry-After": "3"},
            )
        elif status == "subscriber_inactive":
            self._send(
                409,
                "application/json",
                json.dumps({"ok": False, "status": status, "message": "your subscription is not currently active"}),
                extra_headers=extra_headers,
            )
        elif status == "claimed":
            self._send(
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "status": status,
                        "api_key": result.api_key,
                        "tier": result.tier,
                        "subscription_id": result.subscription_id,
                    }
                ),
                extra_headers=extra_headers,
            )
        elif status == "already_claimed":
            self._send(
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "status": status,
                        "subscription_id": result.subscription_id,
                        "message": "this key has already been delivered — contact support to rotate it if lost",
                    }
                ),
                extra_headers=extra_headers,
            )
        else:  # "upstream_error" or any status this route doesn't recognize — never a bare 500
            self._send(
                502,
                "application/json",
                json.dumps(
                    {
                        "ok": False,
                        "status": "upstream_error",
                        "message": "could not verify your transaction — try again shortly",
                    }
                ),
                extra_headers=extra_headers,
            )

    # ── POST /api/v1/contact — the contact form's only destination; it was
    # dead (posting to a route that didn't exist) before this. No email
    # service exists (see ground truth: MX records absent, support@ bounces),
    # so this deliberately just persists — see _serve_contact's docstring for
    # exactly where. ────────────────────────────────────────────────────────
    def _serve_contact(self, length: int) -> None:
        ip = self.client_address[0] if self.client_address else "unknown"
        allowed, retry_after = _CONTACT_IP_LIMITER.allow(f"ip:{ip}")
        if not allowed:
            retry = max(1, int(retry_after) + 1)
            # Don't consume the socket's unread body — this connection is done.
            self.close_connection = True
            self._send(
                429,
                "application/json",
                json.dumps({"ok": False, "error": "rate limited", "retry_after_s": retry}),
                extra_headers={"Retry-After": str(retry)},
            )
            return

        if length > _MAX_CONTACT_BODY_BYTES:
            # Refuse before allocating memory for / persisting an oversized
            # body — the disk-fill guard the task calls for. Body left
            # unread on purpose; close so the next request on this
            # connection isn't desynced by the unread bytes.
            self.close_connection = True
            self._send(413, "application/json", json.dumps({"ok": False, "error": "request body too large"}))
            return

        body = self.rfile.read(length) if length > 0 else b""

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send(400, "application/json", json.dumps({"ok": False, "error": "bad json"}))
            return

        if not isinstance(payload, dict):
            self._send(400, "application/json", json.dumps({"ok": False, "error": "expected a JSON object"}))
            return

        name = str(payload.get("name") or "").strip()
        email = str(payload.get("email") or "").strip()
        subject = str(payload.get("subject") or "").strip()
        message = str(payload.get("message") or "").strip()

        if not (name and email and subject and message):
            self._send(
                400,
                "application/json",
                json.dumps({"ok": False, "error": "name, email, subject and message are all required"}),
            )
            return

        if (
            len(name) > _MAX_CONTACT_FIELD_LEN
            or len(email) > _MAX_CONTACT_FIELD_LEN
            or len(subject) > _MAX_CONTACT_FIELD_LEN
            or len(message) > _MAX_CONTACT_MESSAGE_LEN
        ):
            self._send(
                400,
                "application/json",
                json.dumps({"ok": False, "error": "one or more fields exceed the maximum allowed length"}),
            )
            return

        # Lands here: a plain JSONL file, one line per submission. No DB
        # table, no dependency, no email (there is no mail service to send
        # through — see ground truth). Overridable via TIRRA_CONTACT_LOG for
        # tests / ops; default path lives alongside the other small
        # JSON-file-pattern stores (SubscriberStore, ClaimStore).
        log_path = Path(os.getenv("TIRRA_CONTACT_LOG", _DEFAULT_CONTACT_LOG))
        record = {
            "received_at": time.time(),
            "name": name,
            "email": email,
            "subject": subject,
            "message": message,
            "source_ip": ip,
        }
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logging.getLogger(__name__).error("[contact] failed to persist message: %s", exc)
            self._send(
                500,
                "application/json",
                json.dumps({"ok": False, "error": "could not store your message — please try again shortly"}),
            )
            return

        self._send(200, "application/json", json.dumps({"ok": True, "message": "received"}))

    # ── GET /api/v1/admin/contact-messages — operator-only read path ────────
    #
    # HOW THE OWNER READS CONTACT MESSAGES (documented once, here — no other
    # doc references this): the operator has two options, both reading the
    # exact same file (default `.tirra_opportunities/contact_messages.jsonl`,
    # overridable via TIRRA_CONTACT_LOG):
    #   1. This route: `curl -H "X-Ingest-Token: $TIRRA_INGEST_TOKEN"
    #      https://<host>/api/v1/admin/contact-messages` — returns JSON,
    #      newest-first, paginated via `?limit=&offset=`.
    #   2. Directly: `tail -f .tirra_opportunities/contact_messages.jsonl`
    #      on the box, or any JSONL-aware tool — one JSON object per line,
    #      append-only.
    # Deliberately reuses `_ingest_authorized()` (the same X-Ingest-Token gate
    # POST /evidence/ingest already uses) instead of inventing a second
    # admin-auth mechanism: one gate to reason about, not two. NOT wired into
    # any customer-facing tier or UI — an unset TIRRA_INGEST_TOKEN in
    # production denies this route the same way it denies ingest (see
    # `_ingest_authorized`'s fail-closed contract).
    def _serve_admin_contact_messages(self, query) -> None:
        log_path = Path(os.getenv("TIRRA_CONTACT_LOG", _DEFAULT_CONTACT_LOG))
        requested_limit = _safe_int((query.get("limit") or [None])[0], 50)
        limit = max(1, min(requested_limit, _MAX_CONTACT_READ_LIMIT))
        offset = max(0, _safe_int((query.get("offset") or [None])[0], 0))

        if not log_path.exists():
            self._send(200, "application/json", json.dumps({"ok": True, "messages": [], "count": 0, "total": 0}))
            return

        try:
            raw_lines = log_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logging.getLogger(__name__).error("[admin/contact] failed to read %s: %s", log_path, exc)
            self._send(
                500,
                "application/json",
                json.dumps({"ok": False, "error": "could not read contact log"}),
            )
            return

        records = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A single corrupt/partial line (e.g. a torn write) must not
                # take down the whole read.
                continue

        records.reverse()  # newest first
        total = len(records)
        page = records[offset : offset + limit]
        self._send(
            200,
            "application/json",
            json.dumps({"ok": True, "messages": page, "count": len(page), "total": total}),
        )

    # ── POST /api/v1/rotate-key — self-service key rotation ──────────────────
    #
    # See the constants block above (`_ROTATE_KEY_LIMITER` etc.) for the
    # threat-model rationale. This method is intentionally a thin translation
    # layer over `SubscriberStore.rotate_key_for_api_key` — see that method's
    # docstring in agent/payments/handler.py for the exact contract this
    # implements:
    #
    #   POST /api/v1/rotate-key
    #   Header: X-Brief-Key: <current tirra_... key>     (query-string keys
    #                                                      are NEVER accepted
    #                                                      here, unlike other
    #                                                      routes' ?key=
    #                                                      fallback — this is
    #                                                      a destructive
    #                                                      action)
    #   200 {"ok": true, "api_key": "tirra_<new>"}         — old key is
    #                                                        already dead by
    #                                                        the time this
    #                                                        returns
    #   401 {"ok": false, "error": "missing X-Brief-Key header"}
    #   400 {"ok": false, "error": "malformed key"}
    #   403 {"ok": false, "error": "invalid or inactive key"}
    #   429 {"ok": false, "error": "rate limited", "retry_after_s": N}
    #
    # The new key is returned exactly once — same "no other recovery channel"
    # situation as GET /api/v1/claim's "claimed" response (no mailbox exists;
    # see ground truth). The caller MUST persist it immediately.
    def _serve_rotate_key(self) -> None:
        key = self.headers.get("X-Brief-Key", "").strip()
        if not key:
            self._send(401, "application/json", json.dumps({"ok": False, "error": "missing X-Brief-Key header"}))
            return

        if not _API_KEY_RE.match(key):
            # Rejected before either rate limiter or the store is touched —
            # see the constants-block note on unbounded rate-limiter growth.
            self._send(400, "application/json", json.dumps({"ok": False, "error": "malformed key"}))
            return

        ip = self.client_address[0] if self.client_address else "unknown"
        allowed, retry_after = _ROTATE_KEY_LIMITER.allow(f"key:{key}")
        if allowed:
            allowed, retry_after = _ROTATE_IP_LIMITER.allow(f"ip:{ip}")
        if not allowed:
            retry = max(1, int(retry_after) + 1)
            self._send(
                429,
                "application/json",
                json.dumps({"ok": False, "error": "rate limited", "retry_after_s": retry}),
                extra_headers={"Retry-After": str(retry)},
            )
            return

        from agent.payments.handler import SubscriberStore

        new_key = SubscriberStore().rotate_key_for_api_key(key)
        if new_key is None:
            self._send(403, "application/json", json.dumps({"ok": False, "error": "invalid or inactive key"}))
            return
        self._send(200, "application/json", json.dumps({"ok": True, "api_key": new_key}))

    # ── Webhook (Paddle subscription lifecycle) ───────────────────────────────
    def do_POST(self) -> None:  # noqa: N802
        with _REQUEST_SEMAPHORE:
            self._do_POST()

    def _do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]

        # C4: Content-Length is caller-controlled input too — guard it the
        # same way query-param ints are guarded (a malformed header must not
        # crash the worker thread with an uncaught ValueError).
        length = max(0, _safe_int(self.headers.get("Content-Length"), 0))

        if path == "/api/v1/contact":
            self._serve_contact(length)
            return

        if path == "/api/v1/rotate-key":
            if length > 0:
                # No request body is expected (auth is header-only); drain
                # it so an unread body doesn't desync a keep-alive connection.
                self.rfile.read(length)
            self._serve_rotate_key()
            return

        # Read the RAW body (signature verification needs the exact bytes).
        body = self.rfile.read(length) if length > 0 else b""

        if path == "/evidence/ingest":
            if not self._ingest_authorized():
                self._send(403, "application/json", json.dumps({"ok": False, "error": "invalid ingest token"}))
                return
            self._serve_evidence_ingest(body)
            return

        if path != "/webhook":
            self._send(404, "text/plain", "not found\n")
            return

        signature = self.headers.get("Paddle-Signature", "")

        try:
            from agent.payments.config import PaddleConfig
            from agent.payments.handler import PaddleWebhookHandler

            cfg = PaddleConfig.from_env()
            handler = PaddleWebhookHandler(secret=cfg.webhook_secret)
            result = handler.handle(body=body, signature_header=signature)
        except Exception as exc:  # verification failure or config error
            import logging

            logging.getLogger(__name__).warning("[paddle] webhook rejected: %s", exc)
            self._send(400, "application/json", json.dumps({"ok": False, "error": str(exc)}))
            return

        # Don't echo the minted api_key back to Paddle's webhook ack — it has
        # no use for it, and there's no reason to put a customer secret on
        # that wire. Retrieve it via SubscriberStore for support/ops tooling.
        ack = {k: v for k, v in result.items() if k != "api_key"}
        self._send(200, "application/json", json.dumps({**ack, "ok": True}))

    # ── Evidence Graph (document-evidence extraction store) ─────────────────
    #
    # PARTIALLY RESOLVED (flagged 2026-08-26, see
    # docs/research/entity_graph_tier_mismatch.md): as of this pass, a
    # scoped slice of the REAL production graph is now served too — see
    # `/api/v1/entity-graph/*` below (`_serve_entity_graph_entities` etc.),
    # reading `entities`/`entity_links` straight from `agent/pipeline/store.py`
    # (the same tables `agent/models/gnn/graph_builder.py` trains on).
    #
    # These `/evidence/*` routes are a DIFFERENT, smaller thing: `agent/
    # evidence/`, a standalone document store built in an unrelated
    # session — regex-based entity extraction (~155 distinct entity strings
    # live right now) over manually-POSTed documents (5 of them live right
    # now). Both are gated under `_ENTITY_GRAPH_TIERS` (the tier now
    # genuinely includes both a document-evidence feature and real-graph
    # read access), and each carries its own honest `dataset_scope` block so
    # neither is mistaken for the other. `entity_observations` (365K+ rows of
    # raw pipeline signal values) remains deliberately unexposed everywhere —
    # see the comment above `_serve_entity_graph_entities`. Do not remove
    # either `dataset_scope` block without updating the pricing copy to
    # match whatever is actually served at the time.
    def _evidence_store(self):
        from agent.evidence import EvidenceGraphStore

        return EvidenceGraphStore()

    @staticmethod
    def _evidence_dataset_scope(store) -> dict:
        """Explicit, non-silent disclosure of what this dataset actually is.

        See the KNOWN MISMATCH note above `_evidence_store`. Computed fresh
        per request (cheap COUNT queries) so the disclosure never drifts
        stale relative to what the endpoint actually returned.
        """
        stats = store.stats()
        return {
            "dataset": "document_evidence_sample",
            "documents_ingested": stats["documents"],
            "mentions": stats["mentions"],
            "links": stats["links"],
            "note": (
                "This is a small, manually-ingested document-evidence sample "
                "(regex-based extraction), NOT the production entity/relationship "
                "graph described in the Entity Graph tier's marketing. Confidence "
                "scores here are co-occurrence heuristics, not model output. See "
                "docs/research/entity_graph_tier_mismatch.md."
            ),
        }

    def _single_body(self, body: bytes) -> dict:
        try:
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send(400, "application/json", json.dumps({"ok": False, "error": f"bad json: {exc}"}))
            return {}

    def _ingest_authorized(self) -> bool:
        """Gate POST /evidence/ingest. Fails CLOSED, unlike the original.

        The prior form was ``if admin_token and token != admin_token`` — an
        empty TIRRA_INGEST_TOKEN short-circuited the check and left ingest
        world-open. That gate was independent of TIRRA_REQUIRE_AUTH, so a
        deploy that correctly set TIRRA_REQUIRE_AUTH=1 and left the ingest
        token blank (as deploy/env.production.example ships it) still exposed
        the write half of the arbitrary-file-read chain to the internet.

        See docs/research/evidence_ingest_path_traversal.md.
        """
        admin_token = os.getenv("TIRRA_INGEST_TOKEN", "").strip()
        presented = self.headers.get("X-Ingest-Token", "").strip()
        if admin_token:
            return hmac.compare_digest(presented, admin_token)
        if _truthy(os.getenv("TIRRA_REQUIRE_AUTH")):
            logging.getLogger(__name__).error(
                "TIRRA_REQUIRE_AUTH is set but TIRRA_INGEST_TOKEN is empty — "
                "denying all ingest. Configure TIRRA_INGEST_TOKEN."
            )
            return False
        # Dev mode: nothing configured and auth not required → serve open,
        # matching _authorized_for's contract.
        return True

    @staticmethod
    def _resolve_ingest_path(raw: str) -> str | None:
        """Resolve a caller-supplied ingest path, or None if not allowed.

        Path ingest is opt-in: with TIRRA_INGEST_DIR unset the mode is
        refused outright, which is the secure default. When set, the resolved
        realpath must sit inside the resolved base directory — realpath on
        both sides is what defeats ``..`` traversal AND symlinks pointing out
        of the base dir, which a string-prefix check would not.
        """
        base = os.getenv("TIRRA_INGEST_DIR", "").strip()
        if not base:
            return None
        try:
            base_real = os.path.realpath(base)
            target = os.path.realpath(raw)
        except (OSError, ValueError):
            return None
        if target != base_real and not target.startswith(base_real + os.sep):
            return None
        return target

    def _serve_evidence_ingest(self, body: bytes) -> None:
        req = self._single_body(body)
        if not req:
            return
        doc_id = req.get("doc_id") or f"doc_{int(__import__('time').time())}"
        text = req.get("text")
        path = req.get("path")
        doc_type = req.get("doc_type", "text")
        from agent.evidence import EvidenceIngestor, ingest_to_store

        store = self._evidence_store()
        ing = EvidenceIngestor()
        if text is not None:
            ok = ingest_to_store(
                store,
                ing,
                doc_id=doc_id,
                text=text,
                source=req.get("source", ""),
                title=req.get("title", ""),
                doc_type=doc_type,
            )
        elif path:
            safe_path = self._resolve_ingest_path(str(path))
            if safe_path is None:
                # Deliberately does not echo the requested path back: that
                # would turn this 400 into a filesystem-existence oracle.
                self._send(
                    400,
                    "application/json",
                    json.dumps({"ok": False, "error": "path ingest not permitted"}),
                )
                return
            ok = ingest_to_store(
                store,
                ing,
                doc_id=doc_id,
                path=safe_path,
                source=req.get("source", ""),
                title=req.get("title", ""),
                doc_type=doc_type,
            )
        else:
            self._send(400, "application/json", json.dumps({"ok": False, "error": "provide 'text' or 'path'"}))
            return
        self._send(
            200, "application/json", json.dumps({"ok": True, "doc_id": doc_id, "new": ok, "stats": store.stats()})
        )

    def _serve_evidence_graph(self, query) -> None:
        q = (query.get("q") or [None])[0]
        if not q:
            self._send(400, "application/json", json.dumps({"ok": False, "error": "q required"}))
            return
        store = self._evidence_store()
        import urllib.parse

        key = urllib.parse.unquote(q).strip().lower()
        self._send(
            200,
            "application/json",
            json.dumps(
                {
                    "ok": True,
                    **store.search_entity(key),
                    "related": store.related(key),
                    "dataset_scope": self._evidence_dataset_scope(store),
                }
            ),
        )

    def _serve_evidence_stats(self) -> None:
        store = self._evidence_store()
        self._send(
            200,
            "application/json",
            json.dumps({"ok": True, "stats": store.stats(), "dataset_scope": self._evidence_dataset_scope(store)}),
        )

    def _serve_evidence_analytics(self, query) -> None:
        store = self._evidence_store()
        import urllib.parse

        q = (query.get("q") or [None])[0]
        co = store.co_occurrences(urllib.parse.unquote(q).strip().lower()) if q else []
        pairs = store.cross_doc_pairs(min_docs=_safe_int((query.get("min_docs") or [None])[0], 2))
        self._send(
            200,
            "application/json",
            json.dumps(
                {
                    "ok": True,
                    "co_occurrences": co,
                    "cross_doc_pairs": pairs,
                    "dataset_scope": self._evidence_dataset_scope(store),
                }
            ),
        )

    def _serve_evidence_export(self) -> None:
        store = self._evidence_store()
        self._send(
            200,
            "application/json",
            json.dumps(
                {"ok": True, "graph": store.graph_export(), "dataset_scope": self._evidence_dataset_scope(store)}
            ),
        )

    def _serve_evidence_centrality(self, query) -> None:
        from agent.evidence import degree_centrality, neighbors

        store = self._evidence_store()
        import urllib.parse

        q = (query.get("q") or [None])[0]
        top = _safe_int((query.get("top") or [None])[0], 10)
        if q:
            n = neighbors(store, urllib.parse.unquote(q).strip().lower())
        else:
            n = None
        self._send(
            200,
            "application/json",
            json.dumps(
                {
                    "ok": True,
                    "top_by_degree": degree_centrality(store, top_n=top),
                    "neighbors": n,
                    "dataset_scope": self._evidence_dataset_scope(store),
                }
            ),
        )

    # ── Entity Graph tier: REAL production entity graph (scoped) ────────────
    #
    # Added per docs/research/entity_graph_tier_mismatch.md, resolving part of
    # the mismatch flagged there: these routes serve the ACTUAL graph
    # agent/models/gnn/graph_builder.py trains on (agent/pipeline/store.py —
    # currently 5,628 entities / 16,870 links), not the small document-
    # evidence demo above. Deliberately scoped to two tables only:
    #
    #   - `entities`     (entity_id, entity_type, canonical_name, created_at)
    #   - `entity_links` (entity_id_a/b, link_type, confidence, source,
    #                     created_at)
    #
    # Deliberately EXCLUDED, and not a future TODO for this pass:
    #   - `entity_observations` (365K+ rows) — raw pipeline signal/feature
    #     values keyed by entity. That's the system's proprietary alpha
    #     input, not a "graph," and whether any of it should ever be
    #     customer-facing is a product/security decision, not a routing one.
    #   - each row's `metadata_json` — populated ad hoc by 20+ independent
    #     `agent/tools/*` call sites (tx hashes, CIKs, exchange names, ...)
    #     never audited as a set for tier-safety. Stripped unconditionally
    #     via `_project_entity`/`_project_link` below; add specific fields
    #     back only after someone actually reviews what each source puts
    #     there.
    #
    # This is additive — `/evidence/*` above is untouched and still serves
    # the document-evidence store under its own honest `dataset_scope`.
    _ENTITY_FIELDS = ("entity_id", "entity_type", "canonical_name", "created_at")
    _ENTITY_LINK_FIELDS = (
        "link_id",
        "entity_id_a",
        "entity_id_b",
        "link_type",
        "confidence",
        "source",
        "created_at",
    )

    @staticmethod
    def _real_graph_dataset_scope() -> dict:
        return {
            "dataset": "production_entity_graph",
            "scope": "entities + entity_links only",
            "excludes": [
                "entity_observations (raw pipeline signal/feature values)",
                "per-row metadata_json (unreviewed free-form tool fields)",
            ],
            "note": (
                "This is the real graph agent/models/gnn/graph_builder.py trains on "
                "(agent/pipeline/store.py), scoped down to entities and typed "
                "relationship links for this tier. See "
                "docs/research/entity_graph_tier_mismatch.md."
            ),
        }

    @classmethod
    def _project_entity(cls, entity: dict) -> dict:
        return {k: entity[k] for k in cls._ENTITY_FIELDS if k in entity}

    @classmethod
    def _project_link(cls, link: dict) -> dict:
        return {k: link[k] for k in cls._ENTITY_LINK_FIELDS if k in link}

    def _serve_entity_graph_entities(self, query) -> None:
        from agent.pipeline.store import PipelineStore

        entity_type = (query.get("type") or [None])[0]
        limit = max(1, min(_safe_int((query.get("limit") or [None])[0], 100), _MAX_DATA_LIMIT))
        offset = max(0, _safe_int((query.get("offset") or [None])[0], 0))

        store = PipelineStore()
        rows = store.query_all_entities(entity_type=entity_type, limit=limit, offset=offset)
        total = store.count_all_entities(entity_type=entity_type)
        self._send(
            200,
            "application/json",
            json.dumps(
                {
                    "ok": True,
                    "entities": [self._project_entity(e) for e in rows],
                    "count": len(rows),
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "dataset_scope": self._real_graph_dataset_scope(),
                }
            ),
        )

    def _serve_entity_graph_entity(self, query) -> None:
        entity_id = (query.get("id") or [None])[0]
        if not entity_id:
            self._send(400, "application/json", json.dumps({"ok": False, "error": "id required"}))
            return

        from agent.pipeline.store import PipelineStore

        store = PipelineStore()
        entity = store.get_entity(entity_id)
        if entity is None:
            self._send(
                404,
                "application/json",
                json.dumps({"ok": False, "error": f"no entity {entity_id!r} — see /api/v1/entity-graph/entities"}),
            )
            return

        limit = max(1, min(_safe_int((query.get("limit") or [None])[0], 100), _MAX_DATA_LIMIT))
        links = store.query_entity_links(entity_id, direction="both", limit=limit)
        self._send(
            200,
            "application/json",
            json.dumps(
                {
                    "ok": True,
                    "entity": self._project_entity(entity),
                    "links": [self._project_link(link) for link in links],
                    "dataset_scope": self._real_graph_dataset_scope(),
                }
            ),
        )

    def _serve_entity_graph_links(self, query) -> None:
        link_type = (query.get("link_type") or [None])[0]
        min_confidence = 0.0
        raw_min_conf = (query.get("min_confidence") or [None])[0]
        if raw_min_conf is not None:
            try:
                min_confidence = float(raw_min_conf)
            except ValueError:
                min_confidence = 0.0
        limit = max(1, min(_safe_int((query.get("limit") or [None])[0], 100), _MAX_DATA_LIMIT))
        offset = max(0, _safe_int((query.get("offset") or [None])[0], 0))

        from agent.pipeline.store import PipelineStore

        store = PipelineStore()
        rows = store.query_all_entity_links(
            link_type=link_type, min_confidence=min_confidence, limit=limit, offset=offset
        )
        total = store.count_all_entity_links(link_type=link_type, min_confidence=min_confidence)
        self._send(
            200,
            "application/json",
            json.dumps(
                {
                    "ok": True,
                    "links": [self._project_link(link) for link in rows],
                    "count": len(rows),
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "dataset_scope": self._real_graph_dataset_scope(),
                }
            ),
        )

    # ── Data Platform tier: query already-collected pipeline data ───────────
    def _serve_sources(self) -> None:
        from agent.pipeline.store import PipelineStore

        allowlist = _external_source_allowlist()
        sources = [s for s in PipelineStore().list_sources() if s["source"] in allowlist]
        self._send(200, "application/json", json.dumps({"ok": True, "sources": sources}))

    def _serve_data_api(self, query) -> None:
        source = (query.get("source") or [None])[0]
        if not source:
            self._send(400, "application/json", json.dumps({"ok": False, "error": "source required"}))
            return
        if source not in _external_source_allowlist():
            self._send(
                400,
                "application/json",
                json.dumps(
                    {
                        "ok": False,
                        "error": f"unknown source {source!r} — see /api/v1/sources for valid values",
                    }
                ),
            )
            return
        from agent.pipeline.store import PipelineStore

        store = PipelineStore()

        since = (query.get("since") or [None])[0]
        until = (query.get("until") or [None])[0]
        requested_limit = _safe_int((query.get("limit") or [None])[0], 100)
        limit = max(1, min(requested_limit, _MAX_DATA_LIMIT))
        rows = store.query_data(
            source,
            since=_safe_float(since, None),
            until=_safe_float(until, None),
            limit=limit,
        )
        self._send(200, "application/json", json.dumps({"ok": True, "source": source, "rows": rows}))

    # ── Scheduler tier: read-only visibility into DAG runs ───────────────────
    def _serve_dag_runs(self, query) -> None:
        from agent.pipeline.store import PipelineStore

        dag_name = (query.get("dag_name") or [None])[0]
        requested_limit = _safe_int((query.get("limit") or [None])[0], 20)
        limit = max(1, min(requested_limit, _MAX_DATA_LIMIT))
        runs = PipelineStore().get_runs(dag_name=dag_name, limit=limit)
        self._send(200, "application/json", json.dumps({"ok": True, "runs": runs}))

    # ── Any tier: self-serve usage summary for the caller's own key ─────────
    def _serve_usage(self, key: str | None, query) -> None:
        from agent.payments.usage import UsageStore

        since = (query.get("since") or [None])[0]
        summary = UsageStore().summary(
            (key or "").strip(),
            since=_safe_float(since, None),
        )
        self._send(200, "application/json", json.dumps({"ok": True, **summary}))

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _serve_json(self) -> None:
        latest = self.deliverer.latest()
        if latest is None:
            self._send(404, "text/plain", "no brief delivered yet\n")
            return
        try:
            data = Path(latest.json_path).read_text(encoding="utf-8")
        except OSError:
            self._send(404, "text/plain", "brief file missing\n")
            return
        self._send(200, "application/json", data)

    def _serve_md(self) -> None:
        latest = self.deliverer.latest()
        if latest is None:
            self._send(404, "text/plain", "no brief delivered yet\n")
            return
        try:
            data = Path(latest.md_path).read_text(encoding="utf-8")
        except OSError:
            self._send(404, "text/plain", "brief markdown missing\n")
            return
        self._send(200, "text/markdown; charset=utf-8", data)

    def _serve_status(self) -> None:
        self._send(200, "application/json", json.dumps(self.deliverer.status(), indent=2))

    def _send(self, code: int, ctype: str, body: str, extra_headers: dict[str, str] | None = None) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # C6: configurable via TIRRA_CORS_ORIGIN, defaulting to today's "*" so
        # nothing breaks for callers relying on the current wide-open value.
        # extra_headers (e.g. /api/v1/claim's TIRRA_WEB_ORIGIN-scoped value)
        # take precedence over this default rather than being sent twice.
        headers = {"Access-Control-Allow-Origin": os.getenv("TIRRA_CORS_ORIGIN", "*")}
        if extra_headers:
            headers.update(extra_headers)
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    # ── Logging to stderr to keep stdout clean ───────────────────────────────
    def log_message(self, fmt: str, *args) -> None:  # noqa: N802
        sys.stderr.write(f"[brief-server] {fmt % args}\n")


def serve(out_dir: str = _DEFAULT_OUT, port: int = 8777, host: str = "127.0.0.1") -> None:
    """Run the brief server (blocking)."""
    deliverer = BriefDeliverer(out_dir=out_dir)

    # Pre-warm the Data Platform source allowlist (C3) at boot rather than on
    # a customer's first /api/v1/sources or /api/v1/data request — cheap
    # (pure DAG-node traversal, deliberately avoids the one DAG module with an
    # import-time network side effect — see the comment above
    # `_external_source_allowlist`), but best-effort regardless: a failure
    # here must never block server startup; the allowlist is built lazily on
    # first use if this doesn't run.
    try:
        _external_source_allowlist()
    except Exception as exc:  # never block startup on this
        sys.stderr.write(f"[brief-server] source allowlist pre-warm failed (will build lazily): {exc}\n")

    # Bind the configured deliverer onto the handler class (class attr lookup
    # works for locals only via assignment after definition).
    class BriefHandler(_Handler):
        pass

    BriefHandler.deliverer = deliverer

    httpd = ThreadingHTTPServer((host, port), BriefHandler)
    sys.stderr.write(f"[brief-server] serving {out_dir} at http://{host}:{port}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[brief-server] stopped\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Serve the delivered Intelligence Brief over HTTP")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--out", type=str, default=_DEFAULT_OUT)
    args = parser.parse_args()
    serve(out_dir=args.out, port=args.port, host=args.host)
