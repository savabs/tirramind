"""Tests for the SQLite event bus."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.awos.events.bus import EventBus
from agent.awos.events.schema import Event, EventStatus, TriggerCategory


def _make_event(
    source: str = "t",
    cat: TriggerCategory = TriggerCategory.WORKFLOW_PATTERN,
    conf: float = 0.8,
    payload: dict | None = None,
) -> Event:
    return Event(
        source=source,
        category=cat,
        confidence=conf,
        payload=payload or {"k": "v"},
    )


def test_publish_and_fetch(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    ev = _make_event()
    stored = bus.publish(ev)
    assert stored.id
    found = bus.fetch(limit=10)
    assert len(found) == 1
    assert found[0].id == stored.id


def test_dedup_window(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db", dedup_window_s=600)
    a = bus.publish(_make_event(payload={"x": 1}))
    b = bus.publish(_make_event(payload={"x": 1}))
    assert a.id == b.id  # deduped
    assert bus.count() == 1


def test_dedup_does_not_cross_categories(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    bus.publish(_make_event(cat=TriggerCategory.WORKFLOW_PATTERN))
    bus.publish(_make_event(cat=TriggerCategory.ARCHITECTURAL))
    assert bus.count() == 2


def test_dedup_does_not_cross_sources(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    bus.publish(_make_event(source="a"))
    bus.publish(_make_event(source="b"))
    assert bus.count() == 2


def test_mark_and_filter(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    e = bus.publish(_make_event())
    bus.mark(e.id, EventStatus.PROCESSED)
    assert bus.count(EventStatus.NEW) == 0
    assert bus.count(EventStatus.PROCESSED) == 1


def test_fetch_filter_by_category(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    bus.publish(_make_event(cat=TriggerCategory.WORKFLOW_PATTERN))
    bus.publish(_make_event(cat=TriggerCategory.DRIFT, payload={"z": 1}))
    res = bus.fetch(category=TriggerCategory.DRIFT, limit=10)
    assert len(res) == 1
    assert res[0].category == TriggerCategory.DRIFT


def test_large_payload_truncated(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db", max_payload_bytes=100)
    big = {"data": "x" * 500}
    stored = bus.publish(_make_event(payload=big))
    assert stored.payload_truncated is True
    fetched = bus.fetch(limit=1)[0]
    assert fetched.payload.get("truncated") is True


def test_bulk_mark(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    ids = [bus.publish(_make_event(source=f"s{i}")).id for i in range(5)]
    bus.bulk_mark(ids, EventStatus.IGNORED)
    assert bus.count(EventStatus.IGNORED) == 5


def test_get_by_id_returns_none_if_missing(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    assert bus.get("nonexistent-id") is None


def test_empty_fetch(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    assert bus.fetch() == []


def test_invalid_confidence_rejected() -> None:
    with pytest.raises(Exception):
        Event(source="x", category=TriggerCategory.ROUTINE, confidence=1.5)


def test_dedup_expires_after_window(tmp_path: Path) -> None:
    # Use negative window so that cutoff is in the future — any event
    # published will be outside the window and never deduped.
    bus = EventBus(tmp_path / "e.db", dedup_window_s=-10)
    a = bus.publish(_make_event(payload={"q": 1}))
    b = bus.publish(_make_event(payload={"q": 1}))
    assert a.id != b.id


def test_persistence_across_bus_instances(tmp_path: Path) -> None:
    path = tmp_path / "e.db"
    b1 = EventBus(path)
    e = b1.publish(_make_event())
    b1.close()
    b2 = EventBus(path)
    got = b2.get(e.id)
    assert got is not None
    assert got.id == e.id
