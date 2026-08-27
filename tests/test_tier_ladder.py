"""The tier ladder must stay monotonic in price.

TirraMind sells the integrated surface itself, so this ladder IS the product
definition. It was inverted once: _ENTITY_GRAPH_TIERS and _DATA_PLATFORM_TIERS
both contained "scheduler", so a $50 subscriber reached all three surfaces
while a $500 subscriber reached two — the cheapest infrastructure tier bought
strictly more than the most expensive.

Nothing caught it because every existing test checks ONE tier against ONE
route. Those all passed while the ladder as a whole was incoherent. These tests
assert the relationships BETWEEN tiers instead.

See docs/research/tier_ladder_inversion.md.
"""

from __future__ import annotations

import pytest

from agent.brief_server import (
    _BRIEF_TIERS,
    _DATA_PLATFORM_TIERS,
    _ENTITY_GRAPH_TIERS,
    _SCHEDULER_TIERS,
)

# Cheapest first. Prices are the ones printed on products/brief_subscription/pricing.html.
TIERS_BY_PRICE: list[tuple[str, int]] = [
    ("brief", 19),
    ("scheduler", 50),
    ("entity", 300),
    ("data", 500),
]

SURFACES: dict[str, set[str]] = {
    "brief": _BRIEF_TIERS,
    "scheduler": _SCHEDULER_TIERS,
    "entity-graph": _ENTITY_GRAPH_TIERS,
    "data-platform": _DATA_PLATFORM_TIERS,
}


def _reachable(tier: str) -> set[str]:
    """The set of surface names this tier can reach."""
    return {name for name, allowed in SURFACES.items() if tier in allowed}


@pytest.mark.parametrize(("tier", "price"), TIERS_BY_PRICE)
def test_every_tier_reaches_at_least_one_surface(tier: str, price: int) -> None:
    """A paid tier that unlocks nothing is not a product."""
    assert _reachable(tier), f"${price} {tier} reaches no surface at all"


def test_access_is_monotonic_in_price() -> None:
    """Each tier must reach a superset of every cheaper tier.

    This is the assertion that would have caught the original inversion:
    scheduler ($50) reached {brief, scheduler, entity-graph, data-platform}
    while data ($500) reached {brief, entity-graph, data-platform} — not a
    superset, and missing dag-runs entirely.
    """
    for i, (lower, lower_price) in enumerate(TIERS_BY_PRICE):
        for higher, higher_price in TIERS_BY_PRICE[i + 1 :]:
            lo, hi = _reachable(lower), _reachable(higher)
            missing = lo - hi
            assert not missing, (
                f"${higher_price} {higher} cannot reach {sorted(missing)}, "
                f"but ${lower_price} {lower} can — the ladder is inverted"
            )


def test_brief_tier_is_a_real_tier_not_a_giveaway() -> None:
    """/brief.* must be gated on _BRIEF_TIERS, which every paid tier includes.

    The bug: the route was gated on _valid_key, i.e.
    _authorized_for(key, allowed_tiers=None), which returns True for ANY active
    subscriber. "brief" appeared in no tier set at all.
    """
    assert "brief" in _BRIEF_TIERS, "the brief tier must be able to read its own product"
    for tier, _ in TIERS_BY_PRICE:
        assert tier in _BRIEF_TIERS, f"{tier} paid more than $19 and must still get the brief"


def test_cheapest_tier_does_not_reach_paid_infrastructure() -> None:
    """$19 buys the brief, not the entity graph or the data platform."""
    assert "brief" not in _ENTITY_GRAPH_TIERS
    assert "brief" not in _DATA_PLATFORM_TIERS
    assert "brief" not in _SCHEDULER_TIERS


def test_flagship_tier_reaches_everything() -> None:
    """The $500 tier is the flagship; it must not be missing a surface."""
    assert _reachable("data") == set(SURFACES), "data ($500) must reach every surface — it is the top of the ladder"
