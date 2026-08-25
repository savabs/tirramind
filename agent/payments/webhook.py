"""Paddle webhook signature verification (v2, Ed25519).

Paddle signs every webhook delivery with the notification destination's
endpoint_secret_key. Verification:

    headers:  Paddle-Signature = ts=<unix_seconds>;h1=<hex sig>
    message:  ts:<timestamp>; <raw request body>
    verify   h1 with Ed25519(public_key = hex(endpoint_secret_key))

This prevents forged webhooks. The endpoint_secret_key is a 64-hex-char string
used as the Ed25519 public key; h1 is a 128-hex-char signature.

For a live endpoint, verification is mandatory (fail-closed). In sandbox/dev
without a secret configured, verification is skipped (fail-open) so local
testing works — but never in live.
"""

from __future__ import annotations

import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


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
        secret: the endpoint_secret_key (64 hex chars).
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

    # Message to sign: ts:<timestamp>; <body>
    message = f"{ts}:{body.decode('utf-8', errors='replace')}".encode()

    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(secret))
        public_key.verify(bytes.fromhex(h1), message)
    except (InvalidSignature, ValueError) as exc:
        raise WebhookVerificationError("invalid webhook signature") from exc

    return True


def is_verified(secret: str) -> bool:
    """True if a webhook secret is configured (verification will run)."""
    return bool(secret and secret.strip())


__all__ = ["verify_webhook_signature", "WebhookVerificationError", "is_verified"]
