"""Paddle payments integration for TirraMind's tiered products.

Layered, migration-ready:
  - config.py       : env-driven PaddleConfig (sandbox ↔ live via TIRRA_PADDLE_MODE)
  - client.py       : server-side Paddle Billing API client
  - webhook.py      : HMAC-SHA256 signature verification (mandatory in live).
                      NOTE: this was Ed25519 until 2026-08-26 and would have
                      rejected every real Paddle webhook — do not revert it.
  - event_ledger.py : bounded, disk-persisted processed-event_id ledger — the
                      actual replay defense (the signature timestamp check
                      alone only bounds how long a captured webhook stays
                      replayable, not whether it can be replayed within that
                      window).
  - handler.py      : subscription lifecycle → subscriber access store (opaque
                      API keys, per-tier access, key rotation/revocation)
  - claim.py        : server-side half of key delivery — verifies a checkout
                      transaction_id against Paddle before ever returning a
                      subscriber's key (used by GET /api/v1/claim).
  - delivery.py     : email-delivery groundwork for the API key (stdlib
                      smtplib only). Inert until TIRRA_SMTP_* env vars exist;
                      logs loudly and never raises when unconfigured.
  - usage.py        : per-subscriber metered API call log

Nothing here hard-codes secrets or sandbox/live IDs; everything is env-driven
so the sandbox → live migration is a config change, not a code change.
"""

from agent.payments.claim import ClaimResult, ClaimStore, claim_transaction
from agent.payments.client import PaddleClient
from agent.payments.config import PaddleConfig, PaddleConfigError
from agent.payments.delivery import DeliveryResult, attempt_deliver_key_email, is_configured, send_api_key_email
from agent.payments.event_ledger import ProcessedEventLedger
from agent.payments.handler import PaddleWebhookHandler, SubscriberStore
from agent.payments.usage import UsageStore
from agent.payments.webhook import (
    WebhookVerificationError,
    is_verified,
    verify_webhook_signature,
)

__all__ = [
    "PaddleConfig",
    "PaddleConfigError",
    "PaddleClient",
    "verify_webhook_signature",
    "WebhookVerificationError",
    "is_verified",
    "ProcessedEventLedger",
    "PaddleWebhookHandler",
    "SubscriberStore",
    "UsageStore",
    "ClaimStore",
    "ClaimResult",
    "claim_transaction",
    "DeliveryResult",
    "is_configured",
    "send_api_key_email",
    "attempt_deliver_key_email",
]
