"""Tests for actions + text helpers + idempotency."""

from __future__ import annotations

from pathlib import Path

from agent.awos.actions._text import (
    SELF_MARKER,
    append_changelog,
    append_to_section,
    atomic_write,
    find_section,
)
from agent.awos.actions.awos_update import AwosUpdateAction
from agent.awos.actions.base import build_action, registered_types
from agent.awos.actions.proposals import (
    AdrProposeAction,
    DriftTriageAction,
    RoadmapProposeAction,
)
from agent.awos.events.schema import Event, TriggerCategory
from agent.awos.policies.engine import PlannedAction


# -------- text helpers ---------------------------------------------------
def test_atomic_write(tmp_path: Path) -> None:
    p = tmp_path / "a" / "b" / "f.txt"
    atomic_write(p, "hello")
    assert p.read_text() == "hello"


def test_find_section_basic() -> None:
    body = "# T\n\n## A\n\nline\n\n## B\n\nother\n"
    loc = find_section(body, "A")
    assert loc is not None
    start, end = loc
    assert body.splitlines()[start] == "## A"
    assert body.splitlines()[end] == "## B"


def test_find_section_missing() -> None:
    assert find_section("# Title\n", "Nope") is None


def test_append_to_section_creates_if_missing() -> None:
    body = "# Title\n"
    out = append_to_section(body, "New", "- hello")
    assert "## New" in out
    assert "- hello" in out


def test_append_to_section_dedup() -> None:
    body = "## A\n\n- already here\n"
    out = append_to_section(body, "A", "- already here", dedup=True)
    assert out.count("already here") == 1


def test_append_changelog_adds_marker() -> None:
    body = "# X\n\n## 11. Changelog\n\n"
    out = append_changelog(body, "something happened")
    assert SELF_MARKER in out
    assert "something happened" in out


# -------- registry -------------------------------------------------------
def test_registry_has_known_types() -> None:
    types = set(registered_types())
    for t in ("awos_update", "adr_propose", "drift_triage", "roadmap_propose"):
        assert t in types


def test_build_action_returns_none_for_unknown(awos_cfg) -> None:
    assert build_action("nope", awos_cfg) is None


# -------- awos_update ----------------------------------------------------
def _planned(ev: Event, **params) -> PlannedAction:
    return PlannedAction(type="awos_update", params=params, rule_id="r", event=ev)


def test_awos_update_direct_write(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    action = AwosUpdateAction(awos_cfg)
    ev = Event(
        source="t",
        category=TriggerCategory.WORKFLOW_PATTERN,
        confidence=0.9,
        payload={"extracted_principle": "Always test."},
    )
    r = action.run(_planned(ev, section_hint="3. Agent Operating Principles"))
    assert r.ok
    body = awos_cfg.awos_file.read_text()
    assert "Always test" in body
    assert "awos-sig:" in body


def test_awos_update_idempotent(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    action = AwosUpdateAction(awos_cfg)
    ev = Event(
        source="t",
        category=TriggerCategory.WORKFLOW_PATTERN,
        confidence=0.9,
        payload={"extracted_principle": "Always test."},
    )
    action.run(_planned(ev, section_hint="3. Agent Operating Principles"))
    action.run(_planned(ev, section_hint="3. Agent Operating Principles"))
    body = awos_cfg.awos_file.read_text()
    assert body.count("Always test") == 1


def test_awos_update_low_confidence_writes_proposal(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    action = AwosUpdateAction(awos_cfg)
    ev = Event(
        source="t",
        category=TriggerCategory.WORKFLOW_PATTERN,
        confidence=0.5,  # below 0.7 threshold
        payload={"extracted_principle": "Maybe rule."},
    )
    r = action.run(_planned(ev, section_hint="3. Agent Operating Principles"))
    assert r.ok
    props = list(awos_cfg.proposals_dir.glob("*.md"))
    assert len(props) == 1
    body = awos_cfg.awos_file.read_text()
    assert "Maybe rule" not in body  # not written directly


def test_awos_update_missing_awos_file_fails(tmp_path: Path) -> None:
    from agent.awos.config import AWOSConfig

    cfg = AWOSConfig(
        repo_root=tmp_path,
        state_dir=tmp_path / ".awos",
        awos_file=tmp_path / "does_not_exist.md",
    )
    cfg.ensure_dirs()
    action = AwosUpdateAction(cfg)
    ev = Event(
        source="t",
        category=TriggerCategory.WORKFLOW_PATTERN,
        confidence=0.9,
        payload={"extracted_principle": "x."},
    )
    r = action.run(_planned(ev))
    assert not r.ok


def test_awos_update_no_principle_fails(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    action = AwosUpdateAction(awos_cfg)
    ev = Event(
        source="t",
        category=TriggerCategory.WORKFLOW_PATTERN,
        confidence=0.9,
        payload={},
        rationale="",
    )
    r = action.run(_planned(ev))
    assert not r.ok


def test_awos_update_changelog_only(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    action = AwosUpdateAction(awos_cfg)
    ev = Event(
        source="t",
        category=TriggerCategory.STALENESS,
        confidence=0.9,
        rationale="stale tasks",
    )
    r = action.run(_planned(ev, changelog_only=True))
    assert r.ok
    body = awos_cfg.awos_file.read_text()
    assert "stale tasks" in body


# -------- proposal actions ----------------------------------------------
def test_adr_proposal(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    action = AdrProposeAction(awos_cfg)
    ev = Event(
        source="t",
        category=TriggerCategory.DECISION,
        confidence=0.8,
        rationale="go with library X",
    )
    r = action.run(PlannedAction(type="adr_propose", params={}, rule_id="r", event=ev))
    assert r.ok
    assert any("adr" in p.name for p in awos_cfg.proposals_dir.glob("*.md"))


def test_drift_triage(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    action = DriftTriageAction(awos_cfg)
    ev = Event(
        source="drift",
        category=TriggerCategory.DRIFT,
        confidence=0.8,
        rationale="FL01",
        payload={"findings": [{"code": "FL01"}]},
    )
    r = action.run(PlannedAction(type="drift_triage", params={}, rule_id="r", event=ev))
    assert r.ok


def test_roadmap_proposal(awos_cfg) -> None:
    awos_cfg.ensure_dirs()
    action = RoadmapProposeAction(awos_cfg)
    ev = Event(
        source="t",
        category=TriggerCategory.ROADMAP_SHIFT,
        confidence=0.7,
        rationale="shift phase 9 earlier",
    )
    r = action.run(PlannedAction(type="roadmap_propose", params={}, rule_id="r", event=ev))
    assert r.ok
