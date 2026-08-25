"""Brief server — serve the delivered Intelligence Brief over HTTP.

A minimal, dependency-light consumer surface: serves the persisted
`.tirra_delivery/intelligence_brief.json` (and markdown) that
`scripts/deliver_brief.py` writes, so an external system can fetch
"today's brief" programmatically.

Endpoints:
    GET /brief            → latest brief JSON
    GET /brief.json       → alias
    GET /brief.md         → latest brief as Markdown
    GET /status           → delivery status (count, latest, out dir)

Usage:
    .venv/bin/python -m agent.delivery.brief_server --port 8777
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent.delivery.brief_deliverer import BriefDeliverer

_DEFAULT_OUT = ".tirra_delivery"


def _valid_key(request_key: str | None) -> bool:
    """True if the request carries a valid subscriber key.

    A key is valid if it's in the static TIRRA_SUB_KEYS list OR it matches an
    active Paddle subscriber. The brief is served OPEN (dev mode) only when
    neither static keys nor Paddle webhooks are configured — never based on a
    stale shared store file.
    """
    configured = os.getenv("TIRRA_SUB_KEYS", "").strip()
    paddle_secret = os.getenv("TIRRA_PADDLE_WEBHOOK_SECRET", "").strip()

    # Dev mode: nothing configured → serve open.
    if not configured and not paddle_secret:
        return True

    if not request_key:
        return False
    key = request_key.strip()

    if configured and key in {k.strip() for k in configured.split(",")}:
        return True

    if paddle_secret:
        from agent.payments.handler import SubscriberStore
        if SubscriberStore().is_active(key):
            return True

    return False


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
            self._serve_buy()
            return

        if path in ("/", "/landing", "/index.html"):
            self._serve_landing()
            return

        if path in ("/brief", "/brief.json", "/brief.md"):
            if not _valid_key(key):
                self._send(403, "text/plain", "subscribe required — see /buy\n")
                return
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
            self._send(200, "text/plain",
                       "Opportunity Brief — see /brief or /buy\n")
            return
        self._send(200, "text/html; charset=utf-8", body)

    def _serve_buy(self) -> None:
        url = os.getenv("TIRRA_BUY_URL")
        if not url:
            self._send(200, "text/plain",
                       "buy link not configured — set TIRRA_BUY_URL\n")
            return
        body = f'<meta http-equiv="refresh" content="0;url={url}">Subscribing…'
        self._send(200, "text/html; charset=utf-8", body)

    # ── Webhook (Paddle subscription lifecycle) ───────────────────────────────
    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path != "/webhook":
            self._send(404, "text/plain", "not found\n")
            return

        # Read the RAW body (signature verification needs the exact bytes).
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length > 0 else b""
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

        self._send(200, "application/json", json.dumps({**result, "ok": True}))

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
