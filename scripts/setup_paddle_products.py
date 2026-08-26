"""Create (idempotently) the TirraMind tier products/prices in Paddle.

Reads Paddle credentials from env (see agent/payments/config.py — same
TIRRA_PADDLE_MODE/TIRRA_PADDLE_API_KEY vars used everywhere else). Creates one
Product + recurring monthly Price per tier, skipping any that already exist
(matched by name/description), then patches the placeholder price IDs in
products/brief_subscription/pricing.html so the Paddle.js checkout on that
page actually works.

Usage:
    # Preview only — makes no live/sandbox changes
    .venv/bin/python scripts/setup_paddle_products.py --dry-run

    # Create whatever's missing, patch pricing.html
    .venv/bin/python scripts/setup_paddle_products.py

Environment (see .env.example):
    TIRRA_PADDLE_MODE          sandbox | live
    TIRRA_PADDLE_API_KEY       server-side API key, needs product.write + price.write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.payments.client import PaddleClient  # noqa: E402
from agent.payments.config import PaddleConfig  # noqa: E402

_PRICING_HTML = Path(__file__).resolve().parents[1] / "products/brief_subscription/pricing.html"

# (tier key, product name, price description, amount in cents USD)
_TIERS = [
    ("data", "TirraMind — Data Platform", "Data Platform — monthly API access", "50000"),
    ("entity", "TirraMind — Entity Graph", "Entity Graph — monthly API access", "30000"),
    ("scheduler", "TirraMind — Scheduler", "Scheduler — monthly API access", "5000"),
    ("brief", "TirraMind — Opportunity Brief", "Opportunity Brief — weekly, billed monthly", "1900"),
]


def _find_product(client: PaddleClient, name: str) -> dict | None:
    for p in client.list_products(limit=200):
        if p.get("name") == name:
            return p
    return None


def _find_price(client: PaddleClient, product_id: str, description: str) -> dict | None:
    for pr in client.list_prices(product_id=product_id, limit=200):
        if pr.get("description") == description:
            return pr
    return None


def run(dry_run: bool) -> dict[str, str]:
    cfg = PaddleConfig.from_env()
    print(f"[setup-paddle] mode={cfg.mode} api_base={cfg.api_base}")
    client = PaddleClient(cfg)

    price_ids: dict[str, str] = {}
    for tier, product_name, price_desc, amount in _TIERS:
        product = _find_product(client, product_name)
        if product:
            print(f"[setup-paddle] product exists: {product_name} ({product['id']})")
        elif dry_run:
            print(f"[setup-paddle] DRY RUN — would create product: {product_name}")
        else:
            product = client.create_product(name=product_name, tax_category="saas")
            print(f"[setup-paddle] created product: {product_name} ({product['id']})")

        product_id = product["id"] if product else None

        price = _find_price(client, product_id, price_desc) if product_id else None
        if price:
            print(f"[setup-paddle] price exists: {price_desc} ({price['id']})")
            price_ids[tier] = price["id"]
        elif dry_run:
            print(f"[setup-paddle] DRY RUN — would create price: {price_desc} (${int(amount) / 100:.2f}/mo)")
        elif product_id:
            price = client.create_price(
                product_id=product_id,
                description=price_desc,
                amount=amount,
                interval="month",
                frequency=1,
            )
            print(f"[setup-paddle] created price: {price_desc} ({price['id']})")
            price_ids[tier] = price["id"]

    return price_ids


def _patch_pricing_html(price_ids: dict[str, str]) -> None:
    if not price_ids:
        return
    html = _PRICING_HTML.read_text(encoding="utf-8")
    replaced = []
    for tier, price_id in price_ids.items():
        placeholder = f"REPLACE_WITH_PRICE_ID_{tier.upper()}"
        if placeholder in html:
            html = html.replace(placeholder, price_id)
            replaced.append(tier)
    if replaced:
        _PRICING_HTML.write_text(html, encoding="utf-8")
        print(f"[setup-paddle] patched pricing.html price IDs for: {', '.join(replaced)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview only, create nothing")
    args = parser.parse_args()

    price_ids = run(dry_run=args.dry_run)

    if args.dry_run:
        print("\n[setup-paddle] dry run complete — nothing was created.")
        return

    _patch_pricing_html(price_ids)

    if price_ids:
        tier_map = ",".join(f"{pid}:{tier}" for tier, pid in price_ids.items())
        print("\n[setup-paddle] set this on your backend:")
        print(f"TIRRA_TIER_PRICE_MAP={tier_map}")
    print(
        "\n[setup-paddle] also set TIRRA_PADDLE_CLIENT_TOKEN as PADDLE_CLIENT_TOKEN in pricing.html's <script> block."
    )


if __name__ == "__main__":
    main()
