"""Tests for AWOSConfig."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.awos.config import AWOSConfig, _coerce_scalar


def test_defaults(tmp_path: Path) -> None:
    cfg = AWOSConfig(repo_root=tmp_path)
    assert cfg.db_path == cfg.state_dir / "events.db"
    assert cfg.proposals_dir == cfg.state_dir / "proposals"
    assert cfg.state_file == cfg.state_dir / "state.json"
    assert cfg.classifier_mode == "hybrid"


def test_env_override(tmp_path: Path) -> None:
    env = {
        "TIRRA_AWOS_REPO_ROOT": str(tmp_path),
        "TIRRA_AWOS_CLASSIFIER_MODE": "heuristic",
        "TIRRA_AWOS_WATCHER_INTERVAL_S": "42",
        "TIRRA_AWOS_DRIFT_WATCHER_ENABLED": "false",
        "TIRRA_AWOS_ANTHROPIC_TIMEOUT_S": "1.5",
    }
    cfg = AWOSConfig.from_env(env=env)
    assert cfg.classifier_mode == "heuristic"
    assert cfg.watcher_interval_s == 42
    assert cfg.drift_watcher_enabled is False
    assert cfg.anthropic_timeout_s == pytest.approx(1.5)


def test_yaml_override(tmp_path: Path) -> None:
    y = tmp_path / "cfg.yaml"
    y.write_text(
        "classifier_mode: llm\n"
        "watcher_interval_s: 77\n"
    )
    cfg = AWOSConfig.from_env(env={}, yaml_path=y)
    assert cfg.classifier_mode == "llm"
    assert cfg.watcher_interval_s == 77


def test_env_beats_yaml(tmp_path: Path) -> None:
    y = tmp_path / "cfg.yaml"
    y.write_text("watcher_interval_s: 1\n")
    cfg = AWOSConfig.from_env(
        env={"TIRRA_AWOS_WATCHER_INTERVAL_S": "999"}, yaml_path=y
    )
    assert cfg.watcher_interval_s == 999


def test_ensure_dirs(tmp_path: Path) -> None:
    cfg = AWOSConfig(repo_root=tmp_path, state_dir=tmp_path / "s")
    cfg.ensure_dirs()
    assert cfg.state_dir.exists()
    assert cfg.proposals_dir.exists()


def test_path_expansion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = AWOSConfig(repo_root="~/project")
    assert str(cfg.repo_root).startswith(str(tmp_path))


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("False", False),
        ("no", False),
        ("1", True),
        ("0", False),
        ("3", 3),
        ("3.14", 3.14),
        ("hello", "hello"),
    ],
)
def test_coerce_scalar(raw, expected) -> None:
    assert _coerce_scalar(raw) == expected


def test_invalid_classifier_mode(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        AWOSConfig(repo_root=tmp_path, classifier_mode="bogus")  # type: ignore[arg-type]


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    y = tmp_path / "cfg.yaml"
    y.write_text("- just a list\n")
    with pytest.raises(ValueError):
        AWOSConfig.from_env(env={}, yaml_path=y)


def test_unknown_env_field_does_not_crash(tmp_path: Path) -> None:
    # Extra fields are silently ignored by pydantic's default config —
    # verify that behavior here so future strict-mode changes surface.
    cfg = AWOSConfig.from_env(env={"TIRRA_AWOS_TOTALLY_FAKE_FIELD": "x"})
    assert cfg.classifier_mode == "hybrid"
