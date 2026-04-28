"""Tests for policy engine."""

from __future__ import annotations

from agent.awos.events.schema import Event, TriggerCategory
from agent.awos.policies.engine import PolicyEngine, PolicyRule


def _ev(cat: TriggerCategory, conf: float = 0.9, **payload) -> Event:
    return Event(source="test", category=cat, confidence=conf, payload=payload)


def test_defaults_load() -> None:
    engine = PolicyEngine.load()
    assert len(engine.rules) >= 5
    ids = {r.id for r in engine.rules}
    assert "workflow_pattern_to_awos" in ids


def test_plan_workflow_pattern() -> None:
    engine = PolicyEngine.load()
    acts = engine.plan(_ev(TriggerCategory.WORKFLOW_PATTERN, 0.9))
    types = [a.type for a in acts]
    assert "awos_update" in types


def test_plan_low_confidence_blocked() -> None:
    engine = PolicyEngine.load()
    acts = engine.plan(_ev(TriggerCategory.WORKFLOW_PATTERN, 0.1))
    assert acts == []


def test_plan_routine_has_no_actions() -> None:
    engine = PolicyEngine.load()
    acts = engine.plan(_ev(TriggerCategory.ROUTINE, 0.9))
    assert acts == []


def test_plan_drift() -> None:
    engine = PolicyEngine.load()
    acts = engine.plan(_ev(TriggerCategory.DRIFT, 0.9))
    assert any(a.type == "drift_triage" for a in acts)


def test_plan_decision_goes_to_adr() -> None:
    engine = PolicyEngine.load()
    acts = engine.plan(_ev(TriggerCategory.DECISION, 0.8))
    assert any(a.type == "adr_propose" for a in acts)


def test_rule_matches_source_in() -> None:
    r = PolicyRule(
        id="x",
        description="",
        match={"category": ["drift"], "source_in": ["obsidian"]},
        actions=[],
    )
    assert r.matches(Event(source="obsidian", category=TriggerCategory.DRIFT, confidence=0.9))
    assert not r.matches(Event(source="chat_log", category=TriggerCategory.DRIFT, confidence=0.9))


def test_rule_matches_payload_contains() -> None:
    r = PolicyRule(
        id="x",
        description="",
        match={"category": ["drift"], "payload_contains": ["FL01"]},
        actions=[],
    )
    assert r.matches(
        Event(
            source="s",
            category=TriggerCategory.DRIFT,
            confidence=0.9,
            payload={"findings": [{"code": "FL01"}]},
        )
    )
    assert not r.matches(
        Event(
            source="s",
            category=TriggerCategory.DRIFT,
            confidence=0.9,
            payload={"findings": []},
        )
    )


def test_user_rules_take_precedence(tmp_path) -> None:
    p = tmp_path / "user.yaml"
    p.write_text(
        "rules:\n"
        "  - id: my_rule\n"
        "    description: test\n"
        "    match:\n"
        "      category: [workflow_pattern]\n"
        "      min_confidence: 0.0\n"
        "    actions:\n"
        "      - type: awos_update\n"
        "        params: {section_hint: 'Custom'}\n"
    )
    engine = PolicyEngine.load(p)
    acts = engine.plan(_ev(TriggerCategory.WORKFLOW_PATTERN, 0.9))
    # user rule is first
    assert acts[0].rule_id == "my_rule"
    assert acts[0].params["section_hint"] == "Custom"


def test_malformed_yaml_returns_empty(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(": : not yaml\n")
    engine = PolicyEngine.load(p)
    # default rules still load
    assert len(engine.rules) >= 5
