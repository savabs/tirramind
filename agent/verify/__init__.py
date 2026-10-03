"""TirraMind — verification layer (the statistical half of a verdict).

The product sells a REFEREE, not a bet. A customer states a hypothesis — "when
X happens, Y moves within N days" — and receives a verdict plus a computed
explanation. The verdict is whatever it is: a null result delivered with its own
power calculation is the deliverable, not a failure, and it is showable to a risk
committee precisely because we had no stake in the answer.

This package holds the statistical half. ``agent.mechanism`` holds the structural
half (which routes through the evidence graph connect cause to effect, and how
many genuinely independent sources witness them).

``agent.verify.stats`` is the foundation: pure functions over arrays, no
database, no I/O, no clock, no state. It is deliberately importable in
milliseconds — nothing here reaches for torch.

The reference implementation these primitives were extracted from is
``scripts/cftc_event_study.py``, and the study they produced is
``docs/publications/cot_null_result.md`` (0 of 51 hypotheses surviving
Benjamini-Hochberg at alpha=0.05, at roughly 10% power). Both are treated as
fixed: the tests reproduce the published figures rather than the code being
adjusted to match new ones.
"""

from __future__ import annotations

from agent.verify.stats import (
    MIN_HISTORY,
    BHResult,
    BootstrapResult,
    benjamini_hochberg,
    block_bootstrap_ci,
    causal_zscore,
    effective_sample_size,
    power_estimate,
    sample_size_for_power,
)

__all__ = [
    "BHResult",
    "BootstrapResult",
    "MIN_HISTORY",
    "benjamini_hochberg",
    "block_bootstrap_ci",
    "causal_zscore",
    "effective_sample_size",
    "power_estimate",
    "sample_size_for_power",
]
