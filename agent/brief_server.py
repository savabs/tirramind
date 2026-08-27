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
import sys
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
# they're paying for). Every name below is a DAG operator/stage id, not an
# external data source, confirmed by cross-referencing agent/pipeline/dags/.
# Excluded from BOTH the source catalog and direct /api/v1/data queries —
# querying one of these by name now 400s exactly like an unknown source,
# rather than silently working. New DAG stages must be added here; this is
# a denylist specifically because the failure mode (a new internal stage
# silently becoming customer-queryable) is worse than a stage briefly not
# being addable to a future admin/debug surface.
_INTERNAL_TELEMETRY_SOURCES = {
    "train_gnn",
    "gnn_inference",
    "score_entities",
    "generate_features",
    "run_detection",
    "scan_adversarial",
    "sac_inference",
    "train_rl_policy",
    "emit_portfolio",
    "update_beliefs",
    "load_models",
    "component_perf_gnn_epochs",
}


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


class _Handler(BaseHTTPRequestHandler):
    server_version = "AWOSBrief/0.1"  # type: ignore[assignment]

    deliverer: BriefDeliverer  # class attr set by serve()

    # ── HTTP verb handlers ───────────────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import parse_qs, urlsplit

        parts = urlsplit(self.path)
        path = parts.path
        query = parse_qs(parts.query)
        key = (query.get("key") or [None])[0] or self.headers.get("X-Brief-Key")

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

    # ── Webhook (Paddle subscription lifecycle) ───────────────────────────────
    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]

        # Read the RAW body (signature verification needs the exact bytes).
        length = int(self.headers.get("Content-Length", 0) or 0)
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
        pairs = store.cross_doc_pairs(min_docs=int((query.get("min_docs") or ["2"])[0]))
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
        top = int((query.get("top") or ["10"])[0])
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

        sources = [s for s in PipelineStore().list_sources() if s["source"] not in _INTERNAL_TELEMETRY_SOURCES]
        self._send(200, "application/json", json.dumps({"ok": True, "sources": sources}))

    def _serve_data_api(self, query) -> None:
        source = (query.get("source") or [None])[0]
        if not source:
            self._send(400, "application/json", json.dumps({"ok": False, "error": "source required"}))
            return
        if source in _INTERNAL_TELEMETRY_SOURCES:
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

        known = {s["source"] for s in store.list_sources()} - _INTERNAL_TELEMETRY_SOURCES
        if source not in known:
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

        since = (query.get("since") or [None])[0]
        until = (query.get("until") or [None])[0]
        requested_limit = int((query.get("limit") or ["100"])[0])
        limit = max(1, min(requested_limit, _MAX_DATA_LIMIT))
        rows = store.query_data(
            source,
            since=float(since) if since else None,
            until=float(until) if until else None,
            limit=limit,
        )
        self._send(200, "application/json", json.dumps({"ok": True, "source": source, "rows": rows}))

    # ── Scheduler tier: read-only visibility into DAG runs ───────────────────
    def _serve_dag_runs(self, query) -> None:
        from agent.pipeline.store import PipelineStore

        dag_name = (query.get("dag_name") or [None])[0]
        requested_limit = int((query.get("limit") or ["20"])[0])
        limit = max(1, min(requested_limit, _MAX_DATA_LIMIT))
        runs = PipelineStore().get_runs(dag_name=dag_name, limit=limit)
        self._send(200, "application/json", json.dumps({"ok": True, "runs": runs}))

    # ── Any tier: self-serve usage summary for the caller's own key ─────────
    def _serve_usage(self, key: str | None, query) -> None:
        from agent.payments.usage import UsageStore

        since = (query.get("since") or [None])[0]
        summary = UsageStore().summary(
            (key or "").strip(),
            since=float(since) if since else None,
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

    def _send(self, code: int, ctype: str, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    # ── Logging to stderr to keep stdout clean ───────────────────────────────
    def log_message(self, fmt: str, *args) -> None:  # noqa: N802
        sys.stderr.write(f"[brief-server] {fmt % args}\n")


def serve(out_dir: str = _DEFAULT_OUT, port: int = 8777, host: str = "127.0.0.1") -> None:
    """Run the brief server (blocking)."""
    deliverer = BriefDeliverer(out_dir=out_dir)

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
