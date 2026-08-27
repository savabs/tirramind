"""Processed-webhook-event ledger — the actual replay defense.

`verify_webhook_signature`'s timestamp check (webhook.py) only bounds *how
long* a captured, validly-signed webhook stays replayable — it does not stop
a replay *within* that window. A captured `subscription.activated` webhook,
sniffed off a proxy access log or a captured browser devtools session, is a
byte-for-byte valid request until its `ts` ages out. Within the window it can
be POSTed again (e.g. to flip a subsequently-canceled subscription back to
active) and the signature alone will not catch it.

Paddle's payload includes a stable, globally-unique `event_id` on every
delivery (e.g. "evt_01h..."), including retries of the same event — a retry
re-delivers the same event_id, not a new one. This ledger records event_ids
we've already applied and rejects a repeat, whether that repeat is a benign
Paddle retry (should be a no-op, not a double-write — see idempotency) or a
malicious replay (should be a no-op, not a re-activation).

Bounded: entries are pruned by age on every write, AND hard-capped by count,
so the ledger cannot grow without bound even under a clock anomaly or a
burst of traffic. Persisted to disk (same JSON-file pattern as
SubscriberStore) so it survives a process restart — an in-memory-only ledger
would forget every event on deploy/crash and reopen the exact replay window
the timestamp check was tightened to close.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_DEFAULT_LEDGER_PATH = ".tirra_opportunities/processed_events.json"

# How long we remember an event_id. Generous relative to the 300s signature
# window (webhook.py) on purpose: this is defense-in-depth, not the primary
# clock — even if the signature window is ever loosened (e.g. for clock-skew
# tolerance) or a delivery is delayed by a Paddle-side retry backoff, we still
# want to catch a replay of the *same* event_id. 1 hour bounds worst-case
# memory/disk use even at a sustained high webhook rate.
_DEFAULT_RETENTION_S = 3600.0

# Hard cap on ledger size regardless of age-based pruning, in case the clock
# is wrong (e.g. NTP drift, VM pause) and age-based pruning under-prunes.
_DEFAULT_MAX_ENTRIES = 10_000

_LOCK = threading.Lock()


class ProcessedEventLedger:
    """Tracks which Paddle `event_id`s have already been applied.

    Not thread-safe across processes (file-based, last-writer-wins on save —
    fine for this deployment's single-process ThreadingHTTPServer; flagged
    as a risk if the deployment ever becomes a multi-process/horizontal fleet,
    same caveat as SubscriberStore's own JSON-file store).
    """

    def __init__(
        self,
        path: str = _DEFAULT_LEDGER_PATH,
        *,
        retention_s: float = _DEFAULT_RETENTION_S,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._retention_s = retention_s
        self._max_entries = max_entries
        self._data: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._data = {str(k): float(v) for k, v in raw.items()}
        except Exception:
            self._data = {}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")

    def _prune(self, now: float) -> None:
        cutoff = now - self._retention_s
        if self._data:
            self._data = {eid: ts for eid, ts in self._data.items() if ts >= cutoff}
        if len(self._data) > self._max_entries:
            # Hard cap: drop the oldest entries first.
            ordered = sorted(self._data.items(), key=lambda kv: kv[1])
            keep = ordered[-self._max_entries :]
            self._data = dict(keep)

    def seen(self, event_id: str, *, now: float | None = None) -> bool:
        """True if `event_id` has already been recorded (and hasn't aged out)."""
        if not event_id:
            return False
        now_f = time.time() if now is None else now
        with _LOCK:
            self._load()
            self._prune(now_f)
            return event_id in self._data

    def mark_seen(self, event_id: str, *, now: float | None = None) -> None:
        """Record `event_id` as processed. Idempotent (re-marking is a no-op write)."""
        if not event_id:
            return
        now_f = time.time() if now is None else now
        with _LOCK:
            self._load()
            self._data[event_id] = now_f
            self._prune(now_f)
            self._save()

    def check_and_mark(self, event_id: str, *, now: float | None = None) -> bool:
        """Atomic "was this seen before, then mark it seen" — the check the
        webhook handler actually needs (avoids a check/mark race between two
        near-simultaneous deliveries of the same retried event).

        Returns True if this is the FIRST time `event_id` is seen (caller
        should process it); False if it's a replay/retry (caller should skip
        processing but still ack 200 — Paddle retries expect a 2xx, not an
        error, for an event it already successfully delivered).
        """
        if not event_id:
            # No event_id to key on (e.g. a legacy/malformed payload) — can't
            # dedupe, so don't block processing on it.
            return True
        now_f = time.time() if now is None else now
        with _LOCK:
            self._load()
            self._prune(now_f)
            if event_id in self._data:
                return False
            self._data[event_id] = now_f
            self._prune(now_f)
            self._save()
            return True

    def all(self) -> dict[str, float]:
        return dict(self._data)


__all__ = ["ProcessedEventLedger"]
