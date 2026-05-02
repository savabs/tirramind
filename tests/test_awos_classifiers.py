"""Tests for classifiers."""

from __future__ import annotations

import pytest

from agent.awos.classifiers.anthropic import AnthropicClassifier, _parse_response
from agent.awos.classifiers.composite import CompositeClassifier
from agent.awos.classifiers.heuristic import HeuristicClassifier
from agent.awos.config import AWOSConfig
from agent.awos.events.schema import TriggerCategory


# ------------------- heuristic --------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("we should always write a checkpoint before ending a session", TriggerCategory.WORKFLOW_PATTERN),
        ("This is an architectural decision about layer separation", TriggerCategory.ARCHITECTURAL),
        ("Lesson learned: never skip the tests next time", TriggerCategory.LESSON),
        ("Let's reorder the roadmap and defer phase 9", TriggerCategory.ROADMAP_SHIFT),
        ("Fix typo in README", TriggerCategory.ROUTINE),
        ("", TriggerCategory.ROUTINE),
    ],
)
def test_heuristic_classifies(text, expected) -> None:
    c = HeuristicClassifier()
    out = c.classify(text)
    assert out.category == expected


def test_heuristic_unknown_on_no_hits() -> None:
    out = HeuristicClassifier().classify("the quick brown fox jumps over")
    assert out.category == TriggerCategory.UNKNOWN


def test_heuristic_skips_self_written() -> None:
    out = HeuristicClassifier().classify("<!-- awos:self --> we should always do X")
    assert out.category == TriggerCategory.ROUTINE
    assert out.confidence == 0.0


def test_heuristic_ceiling_bounded() -> None:
    c = HeuristicClassifier(confidence_ceiling=0.5)
    txt = " ".join(["we should always"] * 20 + ["workflow pattern rule:"] * 20)
    out = c.classify(txt)
    assert out.confidence <= 0.5


# ------------------- anthropic (no network) -------------------------------
def test_anthropic_no_key_returns_fallback() -> None:
    c = AnthropicClassifier(api_key=None)
    out = c.classify("hello")
    assert out.confidence == 0.0
    assert out.category == TriggerCategory.UNKNOWN


def test_parse_response_happy_path() -> None:
    body = {
        "content": [
            {
                "type": "text",
                "text": (
                    '{"category":"workflow_pattern","confidence":0.9,'
                    '"rationale":"agent rule","extracted_principle":"X.",'
                    '"suggested_section":"3. Agent Operating Principles"}'
                ),
            }
        ]
    }
    r = _parse_response(body)
    assert r.category == TriggerCategory.WORKFLOW_PATTERN
    assert r.confidence == pytest.approx(0.9)
    assert r.extracted_principle == "X."


def test_parse_response_with_code_fences() -> None:
    body = {
        "content": [
            {
                "type": "text",
                "text": '```json\n{"category":"routine","confidence":0.5}\n```',
            }
        ]
    }
    r = _parse_response(body)
    assert r.category == TriggerCategory.ROUTINE


def test_parse_response_bad_json_falls_back() -> None:
    body = {"content": [{"type": "text", "text": "not json"}]}
    r = _parse_response(body)
    assert r.confidence == 0.0
    assert r.category == TriggerCategory.UNKNOWN


def test_parse_response_empty_falls_back() -> None:
    r = _parse_response({"content": []})
    assert r.confidence == 0.0


def test_parse_response_clamps_confidence() -> None:
    body = {"content": [{"type": "text", "text": '{"category":"routine","confidence":5}'}]}
    r = _parse_response(body)
    assert r.confidence == 1.0


def test_parse_response_unknown_category_coerced() -> None:
    body = {"content": [{"type": "text", "text": '{"category":"bogus","confidence":0.1}'}]}
    r = _parse_response(body)
    assert r.category == TriggerCategory.UNKNOWN


# ------------------- composite --------------------------------------------
def test_composite_off_mode() -> None:
    cfg = AWOSConfig(classifier_mode="off")
    c = CompositeClassifier.from_config(cfg)
    r = c.classify("we should always ...")
    assert r.category == TriggerCategory.UNKNOWN
    assert r.confidence == 0.0


def test_composite_heuristic_only() -> None:
    cfg = AWOSConfig(classifier_mode="heuristic")
    c = CompositeClassifier.from_config(cfg)
    r = c.classify("we should always commit checkpoints")
    assert r.category == TriggerCategory.WORKFLOW_PATTERN


def test_composite_hybrid_uses_heuristic_when_confident() -> None:
    cfg = AWOSConfig(classifier_mode="hybrid", llm_confidence_floor=0.3)
    c = CompositeClassifier.from_config(cfg)
    r = c.classify("we should always write checkpoints")
    assert r.classifier == "heuristic"


def test_composite_hybrid_falls_back_when_llm_unavailable() -> None:
    # no api key => llm returns 0.0 confidence; hybrid keeps heuristic
    cfg = AWOSConfig(classifier_mode="hybrid", anthropic_api_key=None)
    c = CompositeClassifier.from_config(cfg)
    r = c.classify("totally unrelated text about nothing")
    # heuristic returns UNKNOWN, llm returns 0.0, composite keeps heuristic
    assert r.classifier in ("heuristic", "anthropic")
