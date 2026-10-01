"""Portfolio construction and, from here on, portfolio *ingestion*.

``agent.portfolio.holdings`` is the foundation of the public portfolio-structure
demo: pasted text in, an aligned currency-consistent returns panel out, with an
explicit account of everything that was dropped and why.

Exports are lazy. ``agent.models.gnn.trainer`` imports
``agent.portfolio.constructor`` on a hot path, and nothing there should pay for
pandas/yfinance import time just because this package grew a new module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent.portfolio.holdings import (
        MIN_ALIGNED_DAYS as MIN_ALIGNED_DAYS,
    )
    from agent.portfolio.holdings import (
        Exclusion as Exclusion,
    )
    from agent.portfolio.holdings import (
        HoldingEntry as HoldingEntry,
    )
    from agent.portfolio.holdings import (
        Holdings as Holdings,
    )
    from agent.portfolio.holdings import (
        ParseProblem as ParseProblem,
    )
    from agent.portfolio.holdings import (
        PricePanel as PricePanel,
    )
    from agent.portfolio.holdings import (
        Resolution as Resolution,
    )
    from agent.portfolio.holdings import (
        Unit as Unit,
    )
    from agent.portfolio.holdings import (
        fetch_prices as fetch_prices,
    )
    from agent.portfolio.holdings import (
        parse_holdings as parse_holdings,
    )

_HOLDINGS_EXPORTS = {
    "Exclusion",
    "HoldingEntry",
    "Holdings",
    "MIN_ALIGNED_DAYS",
    "ParseProblem",
    "PricePanel",
    "Resolution",
    "Unit",
    "fetch_prices",
    "parse_holdings",
}

__all__ = sorted(_HOLDINGS_EXPORTS)


def __getattr__(name: str):
    if name in _HOLDINGS_EXPORTS:
        from agent.portfolio import holdings as _holdings  # noqa: PLC0415

        return getattr(_holdings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _HOLDINGS_EXPORTS)
