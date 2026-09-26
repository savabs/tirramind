"""A realistic epoch origin for test fixtures.

``PipelineStore`` rejects any ``observed_at`` before 1990-01-01 (audit P1.1 —
the live DB held rows dated 1920 and 2030, and ``trainer.py`` derives its
train/val/test boundaries from ``MIN``/``MAX(observed_at)``, so one bad row
moves the whole evaluation window).

Fixtures written before that guard existed seeded observations at small
relative offsets: ``observed_at=1000.0`` means 1970-01-01, which the guard
correctly refuses.  Anchor those offsets here instead — ``T(1000.0)`` keeps
every lag, ordering and window width identical and moves only the origin.
This is the same shift ``SYNTHETIC_BASE_TIME`` applies in
``agent/models/gnn/trainer.py``; the two deliberately share a value.

When a fixture writes a timestamp through one path (``store_entity_observation``)
and compares it against another (a raw-SQL ``created_at``, a ``until=`` window
bound, an assertion on ``observed_at``), **every one of them must shift
together** — shifting only the writes inverts the comparison and a
time-gating test will pass for the wrong reason.

Do not lower the store's floor to make a fixture pass.
"""

from __future__ import annotations

# 2024-01-01T00:00:00Z — comfortably inside [1990-01-01, now].
TEST_BASE_TIME = 1_704_067_200.0


def T(offset: float) -> float:
    """A fixture timestamp *offset* seconds after :data:`TEST_BASE_TIME`."""
    return TEST_BASE_TIME + offset
