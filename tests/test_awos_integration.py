"""Integration + orchestrator + CLI + hooks tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent.awos.config import AWOSConfig
from agent.awos.events.bus import EventBus
from agent.awos.events.schema import Event, EventStatus, TriggerCategory
from agent.awos.hooks.install import hook_names, install, uninstall
from agent.awos.orchestrator.dispatcher import Dispatcher
from agent.awos.policies.engine import PolicyEngine


# --------- dispatcher ----------------------------------------------------
def test_dispatch_processes_event(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    bus = EventBus(awos_cfg.db_path, dedup_window_s=awos_cfg.dedup_window_s)
    policies = PolicyEngine.load()
    disp = Dispatcher(awos_cfg, bus, policies)
    ev = Event(
        source="t",
        category=TriggerCategory.WORKFLOW_PATTERN,
        confidence=0.9,
        rationale="rule",
        payload={"extracted_principle": "Do X."},
    )
    stored = bus.publish(ev)
    rep = disp.dispatch(stored)
    assert rep.planned >= 1
    assert rep.executed >= 1
    # event marked processed
    assert bus.get(stored.id).status == EventStatus.PROCESSED
    assert "Do X" in awos_cfg.awos_file.read_text()


def test_dispatch_unmatched_marks_ignored(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    bus = EventBus(awos_cfg.db_path)
    policies = PolicyEngine.load()
    disp = Dispatcher(awos_cfg, bus, policies)
    ev = Event(
        source="t", category=TriggerCategory.ROUTINE, confidence=0.9
    )
    stored = bus.publish(ev)
    disp.dispatch(stored)
    assert bus.get(stored.id).status == EventStatus.IGNORED


def test_dispatch_action_failure_marks_errored(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    # corrupt AWOS file path so awos_update fails
    awos_cfg.awos_file.unlink()
    bus = EventBus(awos_cfg.db_path)
    policies = PolicyEngine.load()
    disp = Dispatcher(awos_cfg, bus, policies)
    ev = Event(
        source="t",
        category=TriggerCategory.WORKFLOW_PATTERN,
        confidence=0.9,
        payload={"extracted_principle": "X."},
    )
    stored = bus.publish(ev)
    rep = disp.dispatch(stored)
    assert rep.failed == 1
    assert bus.get(stored.id).status == EventStatus.ERRORED


def test_drain_processes_all_new(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    bus = EventBus(awos_cfg.db_path)
    policies = PolicyEngine.load()
    disp = Dispatcher(awos_cfg, bus, policies)
    for i in range(3):
        bus.publish(
            Event(
                source=f"s{i}",
                category=TriggerCategory.WORKFLOW_PATTERN,
                confidence=0.9,
                payload={"extracted_principle": f"Rule {i}."},
            )
        )
    reports = disp.drain()
    assert len(reports) == 3
    assert bus.count(EventStatus.NEW) == 0


# --------- hooks ---------------------------------------------------------
def test_install_hooks_creates_files(tmp_path: Path) -> None:
    git = tmp_path / ".git" / "hooks"
    git.mkdir(parents=True)
    installed = install(tmp_path)
    assert len(installed) == len(hook_names())
    for p in installed:
        assert p.exists()
        assert p.stat().st_mode & 0o100  # user-executable


def test_install_hooks_no_git_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        install(tmp_path)


def test_install_appends_to_existing(tmp_path: Path) -> None:
    hooks_dir = tmp_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)
    existing = hooks_dir / "post-commit"
    existing.write_text("#!/usr/bin/env bash\necho pre-existing\n")
    install(tmp_path)
    body = existing.read_text()
    assert "echo pre-existing" in body
    assert "AWOS" in body


def test_uninstall_removes_awos_portion(tmp_path: Path) -> None:
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    install(tmp_path)
    removed = uninstall(tmp_path)
    assert len(removed) >= 1


# --------- full integration (scan → dispatch → AWOS file) ---------------
def test_e2e_chat_to_awos(tmp_path: Path, awos_cfg) -> None:
    """Scan chat log → classify → publish → dispatch → AWOS updated."""
    from agent.awos.classifiers.heuristic import HeuristicClassifier
    from agent.awos.watchers.chat_log import ChatLogWatcher

    awos_cfg.ensure_dirs()
    logs = tmp_path / "chat_logs"
    logs.mkdir()
    (logs / "sess.log").write_text(
        "We should always architecturally separate layers. "
        "quant code never fetches data, that is the rule."
    )
    bus = EventBus(awos_cfg.db_path)
    w = ChatLogWatcher(
        bus, tmp_path,
        classifier=HeuristicClassifier(),
        state_file=awos_cfg.state_file,
        log_dir=logs,
    )
    n = w.run_once()
    assert n >= 1
    policies = PolicyEngine.load()
    disp = Dispatcher(awos_cfg, bus, policies)
    disp.drain()
    body = awos_cfg.awos_file.read_text()
    # either direct write or proposal — at minimum the changelog or a proposal
    has_write = "awos-sig:" in body
    has_proposal = any(awos_cfg.proposals_dir.glob("*.md"))
    assert has_write or has_proposal


# --------- CLI ----------------------------------------------------------
def test_cli_help() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "agent.awos.cli", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0
    assert "daemon" in r.stdout
    assert "status" in r.stdout


def test_cli_status(tmp_path: Path, awos_cfg, monkeypatch) -> None:
    awos_cfg.ensure_dirs()
    env = {
        "TIRRA_AWOS_REPO_ROOT": str(tmp_path),
        "TIRRA_AWOS_STATE_DIR": str(awos_cfg.state_dir),
        "TIRRA_AWOS_AWOS_FILE": str(awos_cfg.awos_file),
        "TIRRA_AWOS_CLASSIFIER_MODE": "heuristic",
        "PATH": "/usr/bin:/bin",
    }
    import os
    env["PYTHONPATH"] = os.path.abspath(".")
    r = subprocess.run(
        [sys.executable, "-m", "agent.awos.cli", "status"],
        capture_output=True, text=True, timeout=10, env=env,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["classifier_mode"] == "heuristic"
    assert data["awos_exists"] is True


def test_cli_publish_and_dispatch(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    import os
    env = {
        "TIRRA_AWOS_REPO_ROOT": str(awos_cfg.repo_root),
        "TIRRA_AWOS_STATE_DIR": str(awos_cfg.state_dir),
        "TIRRA_AWOS_AWOS_FILE": str(awos_cfg.awos_file),
        "TIRRA_AWOS_CLASSIFIER_MODE": "heuristic",
        "PYTHONPATH": os.path.abspath("."),
        "PATH": "/usr/bin:/bin",
    }
    r = subprocess.run(
        [
            sys.executable, "-m", "agent.awos.cli", "publish",
            "--category", "workflow_pattern",
            "--confidence", "0.9",
            "--principle", "CLI integration rule.",
            "--dispatch",
        ],
        capture_output=True, text=True, timeout=10, env=env,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["planned"] >= 1
    body = awos_cfg.awos_file.read_text()
    assert "CLI integration rule" in body
