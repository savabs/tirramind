"""Look-through: what a user actually owns once index funds are unpacked.

A typical Indian retail book is an index fund plus the big stocks held
directly. HDFC Bank is about 10% of the Nifty 50 and Reliance about 8%, so
someone holding 5% HDFCBANK alongside 20% NIFTYBEES does not own 5% of HDFC
Bank — they own roughly 7%. The method is addition. There is no statistic, no
window and no p-value in it.

This package's foundation layer is :mod:`agent.lookthrough.indices`, which
answers "what is in this index and in what proportion". NSE publishes the
membership; it does not publish the weights, so the weights here are computed
from free-float market cap and every one of them carries a measured error. See
:attr:`agent.lookthrough.indices.Index.weight_accuracy_note`.

Nothing in this package recommends, suggests or forecasts anything.
"""

from __future__ import annotations

from agent.lookthrough.indices import (
    CapQuote,
    Constituent,
    Index,
    IndexUnavailable,
    WeightComparison,
    compare_to_published,
    fetch_index,
    known_indices,
)

__all__ = [
    "CapQuote",
    "Constituent",
    "Index",
    "IndexUnavailable",
    "WeightComparison",
    "compare_to_published",
    "fetch_index",
    "known_indices",
]
