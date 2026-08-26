"""Paddle webhook signature verification (v2, HMAC-SHA256).

Paddle signs every webhook delivery with the notification destination's
endpoint_secret_key (a "pdl_ntfset_..." string, not hex). Verification per
https://developer.paddle.com/webhooks/signature-verification:

    headers:  Paddle-Signature = ts=<unix_seconds>;h1=<hex sig>
    message:  <timestamp>:<raw request body>
    verify   h1 == hex(HMAC-SHA256(key=endpoint_secret_key, message))

This prevents forged webhooks. The endpoint_secret_key is used directly as
the HMAC key (its raw utf-8 bytes) — it is not hex-decoded.

For a live endpoint, verification is mandatory (fail-closed). In sandbox/dev
without a secret configured, verification is skipped (fail-open) so local
testing works — but never in live.
"""

from __future__ import annotations

import hashlib
import hmac
import time


class WebhookVerificationError(ValueError):
    """Raised when a webhook signature is missing, malformed, or invalid."""


def _parse_signature_header(header: str) -> dict[str, str]:
    """Parse `ts=...;h1=...` into {ts, h1}. Empty if absent."""
    out: dict[str, str] = {}
    for part in header.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    return out


def verify_webhook_signature(
    *,
    body: bytes,
    signature_header: str,
    secret: str,
    max_timestamp_age_s: int = 3600,
    now: float | None = None,
) -> bool:
    """Verify a Paddle webhook signature. Raises on invalid; returns True on valid.

    Args:
        body: the raw request body (exactly as received, bytes).
        signature_header: the `Paddle-Signature` header value.
        secret: the endpoint_secret_key (e.g. "pdl_ntfset_...").
        max_timestamp_age_s: reject signatures older than this (replay guard).
        now: override clock (for tests).

    Raises:
        WebhookVerificationError on any failure (missing/malformed/stale/invalid).
    """
    if not secret:
        raise WebhookVerificationError("no endpoint secret configured")

    parts = _parse_signature_header(signature_header or "")
    ts = parts.get("ts")
    h1 = parts.get("h1")
    if not ts or not h1:
        raise WebhookVerificationError("missing ts or h1 in signature header")

    try:
        ts_i = int(ts)
    except ValueError as exc:
        raise WebhookVerificationError("invalid ts in signature header") from exc

    # Replay protection: reject stale deliveries.
    now_i = int(time.time() if now is None else now)
    if abs(now_i - ts_i) > max_timestamp_age_s:
        raise WebhookVerificationError("webhook signature timestamp is stale")

    # Message to sign: <timestamp>:<raw body>
    message = f"{ts}:{body.decode('utf-8', errors='replace')}".encode()

    expected = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, h1):
        raise WebhookVerificationError("invalid webhook signature")

    return True


def is_verified(secret: str) -> bool:
    """True if a webhook secret is configured (verification will run)."""
    return bool(secret and secret.strip())


__all__ = ["verify_webhook_signature", "WebhookVerificationError", "is_verified"]
