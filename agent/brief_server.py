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
    GET /evidence/*         → Entity Graph tier: evidence graph + analytics

Usage:
    .venv/bin/python -m agent.delivery.brief_server --port 8777
"""

from __future__ import annotations

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


def _truthy(value: str | None) -> bool:
    """Permissive truthy parse for env flags ('1', 'true', 'yes', 'on')."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


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
            token = self.headers.get("X-Ingest-Token", "")
            admin_token = os.getenv("TIRRA_INGEST_TOKEN", "").strip()
            if admin_token and token.strip() != admin_token:
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

    # ── Evidence Graph ───────────────────────────────────────────────────────
    #
    # KNOWN MISMATCH (flagged 2026-08-26, unresolved — see
    # docs/research/entity_graph_tier_mismatch.md): the "Entity Graph" tier
    # (products/brief_subscription/pricing.html) is marketed as the real,
    # learned entity/relationship graph — agent/models/gnn/graph_builder.py
    # reading agent/pipeline/store.py, currently 5,628 entities / 16,870
    # links / 365K+ observations. These `/evidence/*` routes do NOT serve
    # that graph. They serve `agent/evidence/`: a separate, much smaller
    # document store built in an unrelated session — regex-based entity
    # extraction (~155 distinct entity strings live right now) over
    # manually-POSTed documents (5 of them live right now). The two were
    # never reconciled, and gating both under `_ENTITY_GRAPH_TIERS` sells
    # access to the wrong dataset.
    #
    # Wiring the *real* graph here is more than a routing fix: `entities` /
    # `entity_links` are plausibly safe to expose, but `entity_observations`
    # carries raw pipeline signal values — deciding what's safe to expose to
    # a paid tier is a product/security call, not something to decide
    # silently in this file. Until that's spec'd and reviewed, every
    # response below carries an explicit `dataset_scope` block so the API
    # contract itself is honest even though the marketing copy is not yet
    # fixed. Do not remove `dataset_scope` without either (a) replacing
    # these routes with real pipeline-graph data under the same review, or
    # (b) correcting the tier's marketing/pricing to match what's actually
    # sold here.
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
            ok = ingest_to_store(
                store,
                ing,
                doc_id=doc_id,
                path=path,
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

    # ── Data Platform tier: query already-collected pipeline data ───────────
    def _serve_sources(self) -> None:
        from agent.pipeline.store import PipelineStore

        sources = PipelineStore().list_sources()
        self._send(200, "application/json", json.dumps({"ok": True, "sources": sources}))

    def _serve_data_api(self, query) -> None:
        source = (query.get("source") or [None])[0]
        if not source:
            self._send(400, "application/json", json.dumps({"ok": False, "error": "source required"}))
            return
        from agent.pipeline.store import PipelineStore

        store = PipelineStore()

        known = {s["source"] for s in store.list_sources()}
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
