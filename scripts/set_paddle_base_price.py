#!/usr/bin/env python3
"""Set the BASE (USD) unit price on live Paddle prices.

Sibling of set_paddle_inr_pricing.py (which only writes
`unit_price_overrides`) and set_paddle_trial_periods.py (only
`trial_period`). Neither of those touches the base `unit_price`, so the
$19 -> $29 Brief reprice had no execution path at all until this existed:
the storefront said $29 while Paddle still charged 1900.

Default (no flags, or --dry-run) prints the exact PATCH payload per tier and
makes NO calls. --execute additionally requires
--i-understand-this-hits-paddle-now.

Repricing in place keeps the price id, so TIRRA_TIER_PRICE_MAP and
pricing.html's TIER_PRICE_IDS need no change. Safe here because production
has zero subscribers; with live subscribers, confirm Paddle's own semantics
for an in-place reprice before running this.

    .venv/bin/python scripts/set_paddle_base_price.py            # dry run
    .venv/bin/python scripts/set_paddle_base_price.py --execute \
        --i-understand-this-hits-paddle-now
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.payments.client import PaddleClient  # noqa: E402
from agent.payments.config import PaddleConfig  # noqa: E402

# Target base prices in minor units (cents). Only tiers whose USD price is
# CHANGING belong here — listing an unchanged tier is a needless live write.
_BASE_PRICES = {
    "brief": {"usd": 29, "amount": "2900"},
}


# Reuse the sibling's resolver rather than re-deriving it: it reads
# TIER_PRICE_IDS straight out of pricing.html, so the page the customer is
# charged from and the price we patch can never drift apart.
from scripts.set_paddle_inr_pricing import _load_tier_price_ids  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--execute", action="store_true", default=False)
    ap.add_argument("--i-understand-this-hits-paddle-now", action="store_true", default=False)
    args = ap.parse_args()

    cfg = PaddleConfig.from_env()
    client = PaddleClient(cfg)
    ids = _load_tier_price_ids()

    live = args.execute and args.i_understand_this_hits_paddle_now
    if args.execute and not live:
        print("[base-price] --execute requires --i-understand-this-hits-paddle-now", file=sys.stderr)
        return 2

    print(f"[base-price] mode={cfg.mode}  api_base={cfg.api_base}")
    if cfg.mode != "live":
        print("[base-price] WARNING: not in live mode — this would patch SANDBOX, not production.")

    missing = set(_BASE_PRICES) - set(ids)
    if missing:
        print(f"[base-price] no price id for tier(s): {sorted(missing)}", file=sys.stderr)
        return 2

    for tier, spec in _BASE_PRICES.items():
        pid = ids[tier]
        payload = {"unit_price": {"amount": spec["amount"], "currency_code": "USD"}}
        print(f"\n-- {tier} ({pid}) -> ${spec['usd']} --")
        print(f"PATCH {cfg.api_base}/prices/{pid}")
        print(json.dumps(payload, indent=2))
        if live:
            res = client.update_price(pid, amount=spec["amount"], currency_code="USD")
            got = (res.get("data", {}).get("unit_price") or {}) if isinstance(res, dict) else {}
            print(f"   APPLIED -> {got.get('amount')} {got.get('currency_code')}")

    print("\n[base-price] " + ("applied." if live else "dry run complete — no calls were made."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
