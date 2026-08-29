"""Create (idempotently) the TirraMind tier products/prices in Paddle.

Reads Paddle credentials from env (see agent/payments/config.py — same
TIRRA_PADDLE_MODE/TIRRA_PADDLE_API_KEY vars used everywhere else). Creates one
Product + recurring monthly Price per tier, skipping any that already exist
(matched by name/description), then patches
products/brief_subscription/pricing.html's Paddle.js config in place — its
PADDLE_ENV, PADDLE_CLIENT_TOKEN, and TIER_PRICE_IDS — so the checkout on that
page targets whatever mode/catalog this run targeted.

This is the intended sandbox→live cutover mechanism: run it once with
TIRRA_PADDLE_MODE=sandbox to set up sandbox testing, then again with
TIRRA_PADDLE_MODE=live (and a live TIRRA_PADDLE_API_KEY /
TIRRA_PADDLE_CLIENT_TOKEN) to flip pricing.html to live. Re-running is safe
and idempotent regardless of what's currently in the file — it does not rely
on placeholder text still being present.

Usage:
    # Preview only — makes no live/sandbox changes, does not touch pricing.html
    .venv/bin/python scripts/setup_paddle_products.py --dry-run

    # Create whatever's missing in Paddle, patch pricing.html to match
    .venv/bin/python scripts/setup_paddle_products.py

Environment (see .env.example):
    TIRRA_PADDLE_MODE          sandbox | live — no --mode flag, env var only
    TIRRA_PADDLE_API_KEY       server-side API key, needs product.write + price.write
    TIRRA_PADDLE_CLIENT_TOKEN  public client-side token — patched into pricing.html if set
"""

from __future__ import annotations

import argparse
import re
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
    ("brief", "TirraMind — Opportunity Brief", "Opportunity Brief — weekly, billed monthly", "2900"),
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


def run(dry_run: bool) -> tuple[PaddleConfig, dict[str, str]]:
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

    return cfg, price_ids


def _patch_pricing_html(cfg: PaddleConfig, price_ids: dict[str, str]) -> None:
    """Wire pricing.html's Paddle.js config to whatever mode/catalog `cfg` and
    `price_ids` describe.

    Idempotent and mode-agnostic: on a *first* run, `TIER_PRICE_IDS` still
    holds the `REPLACE_WITH_PRICE_ID_<TIER>` placeholders shipped in the repo.
    On every run *after* that (including a later sandbox→live cutover run),
    those placeholders are gone — replaced by whatever real price ID the
    previous run wrote. Matching only the placeholder string (the old
    behavior) meant every run after the first silently patched nothing: no
    error, no output, and pricing.html quietly kept pointing at sandbox even
    after a clean `--mode`-less live run. Instead, match each tier's *key* in
    the TIER_PRICE_IDS object and replace its value unconditionally, whatever
    that value currently is.
    """
    html = _PRICING_HTML.read_text(encoding="utf-8")
    original = html

    # PADDLE_ENV must track the mode we just ran setup against — leaving a
    # stale "sandbox" here after a live run means Paddle.Environment.set()
    # still targets sandbox even once live price IDs are wired in, which
    # Paddle.js will reject (or silently checkout against the wrong catalog).
    html = re.sub(r'const PADDLE_ENV = "[^"]*";', f'const PADDLE_ENV = "{cfg.mode}";', html)

    # PADDLE_CLIENT_TOKEN — only overwrite if a token is actually configured
    # for this run; never blank out a working token because the env var
    # happened to be unset when this script ran.
    if cfg.client_token:
        html = re.sub(
            r'const PADDLE_CLIENT_TOKEN = "[^"]*";',
            f'const PADDLE_CLIENT_TOKEN = "{cfg.client_token}";',
            html,
        )

    # Restrict price-ID replacement to inside the TIER_PRICE_IDS object literal
    # only. PAUSED_TIERS (further down) also keys its messages by tier name
    # (e.g. `entity: "This tier's description..."`), so a replace across the
    # whole file would match that too and overwrite a pause message with a
    # price ID — confirmed by testing against a copy of this file before this
    # scoped version shipped.
    replaced: list[str] = []
    block_pattern = re.compile(r"(const TIER_PRICE_IDS = \{)(.*?)(\n\s*\};)", re.S)
    block_match = block_pattern.search(html)
    if block_match:
        block = block_match.group(2)
        for tier, price_id in price_ids.items():
            tier_pattern = re.compile(rf'(\b{re.escape(tier)}:\s*)"[^"]*"')
            block, n = tier_pattern.subn(rf'\1"{price_id}"', block, count=1)
            if n:
                replaced.append(tier)
        html = (
            html[: block_match.start()]
            + block_match.group(1)
            + block
            + block_match.group(3)
            + html[block_match.end() :]
        )
    elif price_ids:
        print("[setup-paddle] WARNING: could not find TIER_PRICE_IDS block in pricing.html — price IDs not patched")

    if html != original:
        _PRICING_HTML.write_text(html, encoding="utf-8")
        print(
            f"[setup-paddle] patched pricing.html — PADDLE_ENV={cfg.mode}, "
            f"client_token={'updated' if cfg.client_token else 'left unchanged (TIRRA_PADDLE_CLIENT_TOKEN not set)'}, "
            f"price IDs for: {', '.join(replaced) or 'none matched'}"
        )
    else:
        print("[setup-paddle] pricing.html already matches this config — no changes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview only, create nothing")
    args = parser.parse_args()

    cfg, price_ids = run(dry_run=args.dry_run)

    if args.dry_run:
        print("\n[setup-paddle] dry run complete — nothing was created.")
        return

    _patch_pricing_html(cfg, price_ids)

    if price_ids:
        tier_map = ",".join(f"{pid}:{tier}" for tier, pid in price_ids.items())
        print("\n[setup-paddle] set this on your backend:")
        print(f"TIRRA_TIER_PRICE_MAP={tier_map}")
    if not cfg.client_token:
        print(
            "\n[setup-paddle] TIRRA_PADDLE_CLIENT_TOKEN was not set in the environment this ran "
            "against, so pricing.html's PADDLE_CLIENT_TOKEN was left unchanged — set "
            "TIRRA_PADDLE_CLIENT_TOKEN and re-run, or edit pricing.html's <script> block by hand."
        )


if __name__ == "__main__":
    main()
