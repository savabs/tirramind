"""Tests for watchers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

from agent.awos.classifiers.heuristic import HeuristicClassifier
from agent.awos.events.bus import EventBus
from agent.awos.events.schema import EventStatus, TriggerCategory
from agent.awos.watchers.chat_log import ChatLogWatcher
from agent.awos.watchers.drift import DriftWatcher
from agent.awos.watchers.obsidian import ObsidianWatcher
from agent.awos.watchers.staleness import StalenessWatcher


# ----- drift -------------------------------------------------------------
def test_drift_watcher_no_script(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    w = DriftWatcher(bus, tmp_path)
    assert w.scan() == []


def test_drift_watcher_parses_findings(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    fake = scripts / "fact_lint.py"
    fake.write_text("import sys, json\nprint(json.dumps([{'code':'FL01','path':'x.md','line':1,'msg':'drift'}]))\n")
    w = DriftWatcher(bus, tmp_path)
    events = w.scan()
    assert len(events) == 1
    assert events[0].category == TriggerCategory.DRIFT


def test_drift_watcher_no_findings(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "fact_lint.py").write_text("print('[]')\n")
    w = DriftWatcher(bus, tmp_path)
    assert w.scan() == []


# ----- staleness --------------------------------------------------------
def test_staleness_detects_old_tasks(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    tasks = tmp_path / "tasks" / "active"
    tasks.mkdir(parents=True)
    old = tasks / "old.md"
    old.write_text("old")
    past = time.time() - 30 * 86400
    import os as _os

    _os.utime(old, (past, past))
    w = StalenessWatcher(bus, tmp_path, stale_task_days=7)
    events = w.scan()
    assert len(events) == 1
    assert events[0].category == TriggerCategory.STALENESS


def test_staleness_no_old_tasks(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    (tmp_path / "tasks" / "active").mkdir(parents=True)
    (tmp_path / "tasks" / "active" / "fresh.md").write_text("new")
    w = StalenessWatcher(bus, tmp_path, stale_task_days=7)
    assert w.scan() == []


def test_staleness_checkpoint_cap(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    mem = tmp_path / "docs" / "memory"
    mem.mkdir(parents=True)
    for i in range(35):
        (mem / f"checkpoint_{i}.md").write_text("x")
    w = StalenessWatcher(bus, tmp_path, stale_task_days=99999, checkpoint_soft_cap=30)
    events = w.scan()
    assert any("checkpoint" in (e.rationale or "") for e in events)


# ----- obsidian ---------------------------------------------------------
def test_obsidian_watcher_findings(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "obsidian_lint.py").write_text("import json; print(json.dumps([{'code':'FM01','path':'a.md'}]))\n")
    w = ObsidianWatcher(bus, tmp_path)
    events = w.scan()
    assert len(events) == 1
    assert events[0].category == TriggerCategory.DRIFT


def test_obsidian_filters_soft_findings(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "obsidian_lint.py").write_text("import json; print(json.dumps([{'code':'LK02','path':'a.md'}]))\n")
    w = ObsidianWatcher(bus, tmp_path)
    assert w.scan() == []


# ----- chat log ---------------------------------------------------------
def test_chat_log_no_dir(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    w = ChatLogWatcher(
        bus,
        tmp_path,
        classifier=HeuristicClassifier(),
        state_file=tmp_path / "s.json",
        log_dir=tmp_path / "does_not_exist",
    )
    assert w.scan() == []


def test_chat_log_processes_new_text(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    logs = tmp_path / "logs"
    logs.mkdir()
    f = logs / "chat.log"
    f.write_text(
        "Lots of routine chatter. We should always write checkpoints before ending a session. More routine text here."
    )
    w = ChatLogWatcher(
        bus,
        tmp_path,
        classifier=HeuristicClassifier(),
        state_file=tmp_path / "state.json",
        log_dir=logs,
    )
    events = w.scan()
    # at least one actionable event expected
    assert any(e.category == TriggerCategory.WORKFLOW_PATTERN for e in events)
    # state persisted
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["chat_log_offsets"][str(f)] == f.stat().st_size


def test_chat_log_skips_already_read(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    logs = tmp_path / "logs"
    logs.mkdir()
    f = logs / "chat.log"
    f.write_text("we should always test")
    state_file = tmp_path / "state.json"
    w = ChatLogWatcher(
        bus,
        tmp_path,
        classifier=HeuristicClassifier(),
        state_file=state_file,
        log_dir=logs,
    )
    w.scan()
    second = w.scan()
    assert second == []


def test_chat_log_classifier_error_isolated(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.log").write_text("a fairly long chunk of text that should trigger classification " * 3)
    bad = MagicMock()
    bad.classify.side_effect = RuntimeError("boom")
    w = ChatLogWatcher(
        bus,
        tmp_path,
        classifier=bad,
        state_file=tmp_path / "state.json",
        log_dir=logs,
    )
    # should not raise, just skip
    assert w.scan() == []


# ----- base watcher errors ----------------------------------------------
def test_base_watcher_catches_scan_errors(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "e.db")
    from agent.awos.events.schema import Event
    from agent.awos.watchers.base import Watcher

    class Boom(Watcher):
        name = "boom"

        def scan(self):
            raise RuntimeError("kaboom")

    w = Boom(bus, tmp_path)
    assert w.run_once() == 0  # doesn't raise
    assert bus.count(EventStatus.NEW) == 0
    _ = Event  # keep import live
