"""Paddle server-side API client.

Thin, typed wrapper around the Paddle Billing API for the operations the
product needs. Env-driven base URL (sandbox vs live) via PaddleConfig.

Reference: https://developer.paddle.com/api-reference/billing
"""

from __future__ import annotations

from typing import Any

import httpx

from agent.payments.config import PaddleConfig


class PaddleClient:
    """Minimal Paddle Billing API client (read + simple helpers)."""

    def __init__(self, config: PaddleConfig) -> None:
        self.config = config
        self._headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

    # ── Prices / products ───────────────────────────────────────────────────
    def get_price(self, price_id: str) -> dict[str, Any]:
        """Fetch a price by ID (e.g. to confirm amount in a checkout)."""
        url = f"{self.config.api_base}/prices/{price_id}"
        r = httpx.get(url, headers=self._headers, timeout=30)
        r.raise_for_status()
        return r.json().get("data", {})

    def list_prices(self, product_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List prices, optionally filtered by product."""
        params: dict[str, Any] = {"limit": limit}
        if product_id:
            params["product_id"] = product_id
        url = f"{self.config.api_base}/prices"
        r = httpx.get(url, headers=self._headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("data", [])

    def list_products(self, limit: int = 50) -> list[dict[str, Any]]:
        """List products in the catalog (for audit / mapping)."""
        url = f"{self.config.api_base}/products"
        r = httpx.get(url, headers=self._headers, params={"limit": limit}, timeout=30)
        r.raise_for_status()
        return r.json().get("data", [])

    # ── Subscriptions ──────────────────────────────────────────────────────
    def get_subscription(self, subscription_id: str) -> dict[str, Any]:
        """Fetch a subscription by ID (status, customer, items)."""
        url = f"{self.config.api_base}/subscriptions/{subscription_id}"
        r = httpx.get(url, headers=self._headers, timeout=30)
        r.raise_for_status()
        return r.json().get("data", {})

    # ── Customers ──────────────────────────────────────────────────────────
    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Fetch a customer by ID (for Retain pwCustomer)."""
        url = f"{self.config.api_base}/customers/{customer_id}"
        r = httpx.get(url, headers=self._headers, timeout=30)
        r.raise_for_status()
        return r.json().get("data", {})

    # ── IP allowlist source of truth ────────────────────────────────────────
    @staticmethod
    def fetch_webhook_ips() -> list[str]:
        """Paddle's current webhook IPs (data.ipv4_cidrs, /32 CIDRs).

        This is the source of truth for allowlisting webhook traffic — never
        hard-code the list; it can change.
        Reference: https://api.paddle.com/ips
        """
        r = httpx.get("https://api.paddle.com/ips", timeout=30)
        r.raise_for_status()
        return r.json().get("data", {}).get("ipv4_cidrs", [])


__all__ = ["PaddleClient"]
