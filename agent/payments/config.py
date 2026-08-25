"""PaddleConfig — environment-driven Paddle settings.

Central, env-only configuration so sandbox ↔ live switching is a config change,
never a code change. No secrets are hard-coded; everything is read from env.

Environment variables (see .env.example):

    TIRRA_PADDLE_MODE          sandbox | live   (default sandbox)
    TIRRA_PADDLE_API_KEY       server-side API key (sandbox or live, per mode)
    TIRRA_PADDLE_CLIENT_TOKEN  client-side token (test_ / live_ per mode)
    TIRRA_PADDLE_WEBHOOK_SECRET  endpoint_secret_key from the notification
                               destination (sandbox or live, per mode)
    TIRRA_PADDLE_PRICE_ID      the checkout price (pri_...)
    TIRRA_PADDLE_RETAIN_ID     (optional) the Retain customer id for pwCustomer

Derived automatically from mode:
    API base    sandbox-api.paddle.com   | api.paddle.com
    checkout    checkout.paddle.com      | checkout.paddle.com
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class PaddleConfigError(ValueError):
    """Raised when Paddle config is missing or invalid."""


@dataclass(frozen=True)
class PaddleConfig:
    mode: str
    api_key: str
    client_token: str
    webhook_secret: str
    price_id: str
    retain_id: str | None

    # ── Derived endpoints ──────────────────────────────────────────────────
    @property
    def api_base(self) -> str:
        return "https://api.paddle.com" if self.mode == "live" else "https://sandbox-api.paddle.com"

    @property
    def checkout_base(self) -> str:
        return "https://checkout.paddle.com"

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    # ── Build from env ─────────────────────────────────────────────────────
    @classmethod
    def from_env(cls, env: dict | None = None) -> PaddleConfig:
        e = os.environ if env is None else env
        mode = e.get("TIRRA_PADDLE_MODE", "sandbox").strip().lower()
        if mode not in ("sandbox", "live"):
            raise PaddleConfigError(f"TIRRA_PADDLE_MODE must be 'sandbox' or 'live', got {mode!r}")

        api_key = e.get("TIRRA_PADDLE_API_KEY", "").strip()
        client_token = e.get("TIRRA_PADDLE_CLIENT_TOKEN", "").strip()
        webhook_secret = e.get("TIRRA_PADDLE_WEBHOOK_SECRET", "").strip()
        price_id = e.get("TIRRA_PADDLE_PRICE_ID", "").strip()
        retain_id = e.get("TIRRA_PADDLE_RETAIN_ID", "").strip() or None

        # A webhook secret is required for live signature verification. For
        # sandbox/local dev we can run without it (verification skipped), but
        # flag it so it's never silently skipped in live.
        if mode == "live" and not webhook_secret:
            raise PaddleConfigError("TIRRA_PADDLE_WEBHOOK_SECRET is required in live mode")

        return cls(
            mode=mode,
            api_key=api_key,
            client_token=client_token,
            webhook_secret=webhook_secret,
            price_id=price_id,
            retain_id=retain_id,
        )

    def to_public_dict(self) -> dict:
        """Non-secret subset safe to expose to the frontend (client token only)."""
        return {
            "mode": self.mode,
            "client_token": self.client_token,
            "price_id": self.price_id,
            "retain_id": self.retain_id,
            "checkout_base": self.checkout_base,
        }


__all__ = ["PaddleConfig", "PaddleConfigError"]
