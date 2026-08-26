"""Paddle payments integration for TirraMind's tiered products.

Layered, migration-ready:
  - config.py   : env-driven PaddleConfig (sandbox ↔ live via TIRRA_PADDLE_MODE)
  - client.py   : server-side Paddle Billing API client
  - webhook.py  : Ed25519 signature verification (mandatory in live)
  - handler.py  : subscription lifecycle → subscriber access store (opaque
                  API keys, per-tier access)
  - usage.py    : per-subscriber metered API call log

Nothing here hard-codes secrets or sandbox/live IDs; everything is env-driven
so the sandbox → live migration is a config change, not a code change.
"""

from agent.payments.client import PaddleClient
from agent.payments.config import PaddleConfig, PaddleConfigError
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
    "PaddleWebhookHandler",
    "SubscriberStore",
    "UsageStore",
]
