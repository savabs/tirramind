"""Regression tests for scripts/setup_paddle_products.py's pricing.html patcher.

Covers a real bug found during a live-cutover audit: `_patch_pricing_html`
used to only replace the `REPLACE_WITH_PRICE_ID_<TIER>` placeholders shipped
in the repo. Those placeholders are consumed by the first-ever run (sandbox
setup already happened), so every subsequent run — including the eventual
sandbox→live cutover run — silently patched nothing, with no error and no
warning. A naive fix (match each tier key anywhere in the file) introduced a
second bug: PAUSED_TIERS also keys its pause messages by tier name (e.g.
`entity: "This tier's description..."`), so an unscoped replace would
overwrite a pause message with a price ID. Both are covered here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import setup_paddle_products as sp  # noqa: E402


class _Cfg:
    def __init__(self, mode: str, client_token: str = "") -> None:
        self.mode = mode
        self.client_token = client_token


# Synthetic fixture, not the real products/brief_subscription/pricing.html.
#
# These tests originally read the live file directly. That coupled them to
# whatever the payments/product owner currently ships there -- which legitimately
# changes independently of this patcher's own logic (PAUSED_TIERS was emptied
# out 2026-08-26 when the Entity Graph tier was re-enabled; the sandbox price
# IDs were replaced with real live ones during the sandbox->live cutover).
# Both changes broke this file's hardcoded "before" assertions even though
# _patch_pricing_html itself was untouched -- a test coupled to a file it
# does not own, not a regression in the code under test. Fixed 2026-08-27:
# a small embedded HTML snippet reproduces the exact same shapes
# (TIER_PRICE_IDS with a real, non-placeholder ID already set; PAUSED_TIERS
# keying a tier name to a pause message) without depending on the live file.
_SYNTHETIC_PRICING_HTML = """<!doctype html>
<html><head></head><body>
<script>
    const PADDLE_ENV = "sandbox";
    const PADDLE_CLIENT_TOKEN = "test_faketoken_existing";
    const TIER_PRICE_IDS = {
      data: "pri_SANDBOX_OLD_DATA_ID",
      entity: "pri_SANDBOX_OLD_ENTITY_ID",
      scheduler: "pri_SANDBOX_OLD_SCHEDULER_ID",
      brief: "pri_SANDBOX_OLD_BRIEF_ID",
    };
    const PAUSED_TIERS = {
      entity: "This tier's description doesn't yet match what you get -- paused pending copy review.",
    };
</script>
</body></html>
"""


def _pricing_html_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "pricing.html"
    dst.write_text(_SYNTHETIC_PRICING_HTML, encoding="utf-8")
    return dst


def test_patch_updates_price_ids_env_and_token(tmp_path, monkeypatch):
    html_path = _pricing_html_copy(tmp_path)
    monkeypatch.setattr(sp, "_PRICING_HTML", html_path)

    cfg = _Cfg(mode="live", client_token="live_faketoken")
    price_ids = {"data": "pri_LIVE_1", "entity": "pri_LIVE_2", "scheduler": "pri_LIVE_3", "brief": "pri_LIVE_4"}
    sp._patch_pricing_html(cfg, price_ids)

    html = html_path.read_text(encoding="utf-8")
    assert 'const PADDLE_ENV = "live";' in html
    assert 'const PADDLE_CLIENT_TOKEN = "live_faketoken";' in html
    block = re.search(r"const TIER_PRICE_IDS = \{.*?\};", html, re.S).group()
    assert 'data: "pri_LIVE_1"' in block
    assert 'entity: "pri_LIVE_2"' in block
    assert 'scheduler: "pri_LIVE_3"' in block
    assert 'brief: "pri_LIVE_4"' in block


def test_patch_does_not_corrupt_paused_tiers(tmp_path, monkeypatch):
    """The bug: an unscoped tier-name replace would leak into PAUSED_TIERS,
    which also has an `entity: "..."` entry (the pause message), overwriting
    a human-readable pause reason with a Paddle price ID."""
    html_path = _pricing_html_copy(tmp_path)
    monkeypatch.setattr(sp, "_PRICING_HTML", html_path)

    cfg = _Cfg(mode="live", client_token="live_faketoken")
    price_ids = {"entity": "pri_LIVE_ENTITY"}
    sp._patch_pricing_html(cfg, price_ids)

    html = html_path.read_text(encoding="utf-8")
    paused_block = re.search(r"const PAUSED_TIERS = \{.*?\};", html, re.S).group()
    assert "pri_LIVE_ENTITY" not in paused_block
    assert "This tier's description doesn't yet match" in paused_block


def test_patch_is_idempotent_on_rerun(tmp_path, monkeypatch):
    """Re-running with the same config a second time (the sandbox→live
    scenario, or just re-running setup) must not error and must report no
    further changes — this was the core bug: the old placeholder-only match
    silently did nothing on any run after the first."""
    html_path = _pricing_html_copy(tmp_path)
    monkeypatch.setattr(sp, "_PRICING_HTML", html_path)

    cfg = _Cfg(mode="live", client_token="live_faketoken")
    price_ids = {"data": "pri_LIVE_1"}
    sp._patch_pricing_html(cfg, price_ids)
    first = html_path.read_text(encoding="utf-8")

    sp._patch_pricing_html(cfg, price_ids)
    second = html_path.read_text(encoding="utf-8")

    assert first == second
    assert 'data: "pri_LIVE_1"' in second


def test_patch_rewrites_prior_real_price_id_not_just_placeholder(tmp_path, monkeypatch):
    """This is the exact bug: pricing.html in the repo already has real
    sandbox price IDs (not REPLACE_WITH_PRICE_ID_* placeholders), because the
    sandbox setup run already happened. A live cutover run must still be able
    to overwrite those sandbox IDs with live ones."""
    html_path = _pricing_html_copy(tmp_path)
    monkeypatch.setattr(sp, "_PRICING_HTML", html_path)

    before = html_path.read_text(encoding="utf-8")
    assert "REPLACE_WITH_PRICE_ID_" not in before  # sanity: placeholders already gone
    assert "pri_SANDBOX_OLD_DATA_ID" in before  # the (synthetic) sandbox data price ID

    cfg = _Cfg(mode="live", client_token="live_faketoken")
    sp._patch_pricing_html(cfg, {"data": "pri_LIVE_DATA_NEW"})

    after = html_path.read_text(encoding="utf-8")
    assert "pri_SANDBOX_OLD_DATA_ID" not in after
    assert "pri_LIVE_DATA_NEW" in after


def test_patch_never_blanks_client_token_when_unset(tmp_path, monkeypatch):
    html_path = _pricing_html_copy(tmp_path)
    monkeypatch.setattr(sp, "_PRICING_HTML", html_path)

    before = html_path.read_text(encoding="utf-8")
    token_before = re.search(r'const PADDLE_CLIENT_TOKEN = "([^"]*)";', before).group(1)

    cfg = _Cfg(mode="live", client_token="")  # not configured for this run
    sp._patch_pricing_html(cfg, {})

    after = html_path.read_text(encoding="utf-8")
    token_after = re.search(r'const PADDLE_CLIENT_TOKEN = "([^"]*)";', after).group(1)
    assert token_after == token_before
