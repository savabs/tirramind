"""Tests for agent.awos.learning — the embedded self-improving runtime modules."""

from __future__ import annotations

import json

import pytest

from agent.awos.learning import (
    ErrorPattern,
    ErrorPatternStore,
    LearningCore,
    LinUCBRouter,
    LiveToolSynthesizer,
    OperationFeatureExtractor,
    PromptEvolver,
    ReplayGate,
    RewardStore,
    SkillLibrary,
    compute_reward,
)


@pytest.fixture
def state_dir(tmp_path):
    return str(tmp_path / "state")


class TestErrorPatternStore:
    def test_save_retrieve_order(self, tmp_path):
        store = ErrorPatternStore(str(tmp_path / "ep.jsonl"))
        store.save(ErrorPattern(task_id="t1", signal_name="gov_contracts", source_tool="gov_contracts", error_type="PARSE", error_msg="bad", critique="validate first"))
        store.save(ErrorPattern(task_id="t1", signal_name="gov_contracts", source_tool="gov_contracts", error_type="TIMEOUT", error_msg="timeout", critique="retry"))
        got = store.retrieve("t1", "gov_contracts", top_n=2)
        assert len(got) == 2
        assert got[0].error_type == "TIMEOUT"  # newest first (insertion tiebreak)
        assert store.retrieve("other", "gov_contracts") == []

    def test_filter_signal(self, tmp_path):
        store = ErrorPatternStore(str(tmp_path / "ep.jsonl"))
        store.save(ErrorPattern(task_id="t1", signal_name="cftc", source_tool="cftc", error_type="X", error_msg="m", critique="c"))
        assert store.retrieve("t1", "gov_contracts") == []
        assert len(store.retrieve("t1", "cftc")) == 1

    def test_summary_counts(self, tmp_path):
        store = ErrorPatternStore(str(tmp_path / "ep.jsonl"))
        store.save(ErrorPattern(task_id="a", signal_name="g", source_tool="s", error_type="E1", error_msg="m", critique="c"))
        store.save(ErrorPattern(task_id="b", signal_name="g", source_tool="s", error_type="E1", error_msg="m", critique="c"))
        assert store.summary()["by_error_type"] == {"E1": 2}


class TestSkillLibrary:
    def test_record_and_context(self, tmp_path):
        lib = SkillLibrary(tmp_path / "skills")
        lib.record(signal_name="gov_contracts", source_tool="gov_contracts", strategy="zscore", operation="score tenders")
        ctx = lib.get_context("gov_contracts", "gov_contracts")
        assert "zscore" in ctx
        assert lib.total_entries() == 1

    def test_classification(self, tmp_path):
        lib = SkillLibrary(tmp_path / "skills")
        lib.record(signal_name="cftc", source_tool="cftc", strategy="hmm", operation="fuse regime evidence")
        assert "fusion" in lib.summary()

    def test_persist_reload(self, tmp_path):
        d = tmp_path / "skills"
        SkillLibrary(d).record(signal_name="a", source_tool="a", strategy="s", operation="score thing")
        assert SkillLibrary(d).total_entries() == 1


class TestRewardStore:
    def test_reward_math_bounds(self):
        assert abs(compute_reward(True, 0, 0.001) - (1 - 0.05 * 0.02)) < 1e-3
        cheap_fail = compute_reward(False, 0, 0.001)
        exp_fail = compute_reward(False, 4, 0.05)
        assert cheap_fail > exp_fail  # cheap failure is penalized less
        assert -0.36 <= exp_fail <= 0

    def test_store_and_persist(self, tmp_path):
        rs = RewardStore(tmp_path / "reward.jsonl")
        rs.store({"task_id": "t1", "action": "score"}, action_id=0, features=[1.0] * 10, success=True, cost_usd=0.001)
        assert rs.total_episodes() == 1
        assert RewardStore(tmp_path / "reward.jsonl").total_episodes() == 1

    def test_replay_gate_threshold(self, tmp_path):
        gate = ReplayGate()
        rs = RewardStore(tmp_path / "reward.jsonl")
        # a strong, novel episode (high reward, unseen action) admits
        ep = rs.store({"task_id": "t1", "action": "score"}, action_id=0, features=[1.0] * 10, success=True, cost_usd=0.001)
        assert gate.admit(ep) is True
        assert gate.total_episodes if hasattr(gate, "total_episodes") else gate.admit_rate() == 1.0


class TestPromptEvolver:
    def test_evolve_from_evidence(self, tmp_path):
        d = tmp_path / "s"
        (d / "skills").mkdir(parents=True)
        (d / "error_patterns.jsonl").write_text(
            json.dumps({"error_type": "PARSE", "critique": "strip BOM"}) + "\n"
        )
        (d / "skills" / "index.json").write_text(json.dumps([{"approach": "scoring", "win_rate": 0.9, "keywords": ["z"], "strategy": "z"}]))
        fake = lambda p: json.dumps({"guidelines": [{"section": "f", "guideline": "Strip BOM first", "reason": "PARSE", "confidence": 0.9}]})
        pe = PromptEvolver(store_path=str(d), cheap_call=fake)
        g = pe.evolve()
        assert "STRIP BOM FIRST" in g.upper() or "Strip BOM" in g
        pe.persist(g, session_count=10)
        assert "Strip BOM" in PromptEvolver(store_path=str(d)).load_evolved_guidelines()

    def test_no_llm_returns_empty(self, tmp_path):
        pe = PromptEvolver(store_path=str(tmp_path))
        assert pe.evolve() == ""


class TestLiveToolSynthesizer:
    def _fake_llm(self, prompt):
        if "should_create" in prompt:
            return json.dumps({"should_create": True, "tool_purpose": "strip BOM from text"})
        return 'import sys, json\n"""Strip BOM."""\ndata = json.loads(sys.stdin.read())\nprint(data.get("text", "").lstrip("\\ufeff"))\n'

    def test_reflect_requires_attempt_2(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TIRRA_AWOS_LIVE_TOOLS", "true")
        synth = LiveToolSynthesizer(tools_dir=str(tmp_path / "t"), cheap_call=self._fake_llm)
        assert synth.reflect("fetch", "PARSE", "bom", attempt=1) is None
        tool = synth.reflect("fetch", "PARSE", "bom", attempt=2)
        assert tool is not None
        assert len(synth.list_tools()) == 1

    def test_gate_env_disables(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TIRRA_AWOS_LIVE_TOOLS", raising=False)
        synth = LiveToolSynthesizer(tools_dir=str(tmp_path / "t"), cheap_call=self._fake_llm)
        assert synth.reflect("fetch", "PARSE", "bom", attempt=3) is None


class TestLinUCBRouter:
    def test_extractor_and_select(self, tmp_path):
        ext = OperationFeatureExtractor()
        f = ext.extract("score gov_contracts zscore")
        assert f.shape == (10,)
        r = LinUCBRouter(weights_path=tmp_path / "l.pkl")
        assert 0 <= r.select(f) < 6

    def test_learning_and_persist(self, tmp_path):
        wpath = tmp_path / "l.pkl"
        ext = OperationFeatureExtractor()
        r = LinUCBRouter(weights_path=wpath)
        for i in range(25):
            f = ext.extract(("fetch" if i % 2 else "score") + " op")
            r.update(f, action_id=1, reward=0.9)
        assert r.is_ready()
        assert LinUCBRouter(weights_path=wpath).total_updates() == 20  # saves at 10 and 20 (save cadence = 10)


class TestLearningCore:
    def test_integration_records_and_persists(self, state_dir):
        core = LearningCore(state_dir=state_dir)
        core.record_outcome(task_id="t1", operation="score gov_contracts", action_id=2, success=True, cost_usd=0.01, signal_name="gov_contracts", source_tool="gov_contracts")
        core.record_outcome(task_id="t2", operation="fetch gov_contracts", action_id=0, success=False, cost_usd=0.001, signal_name="gov_contracts", source_tool="gov_contracts", error_type="PARSE", error_msg="bad", critique="c", attempts=2)
        s = core.summary()
        assert s["total_episodes"] == 2
        assert s["errors"]["by_error_type"]["PARSE"] == 1
        assert core.context_for("score gov_contracts", "gov_contracts", "gov_contracts")
        assert 0 <= core.route_method("fetch gov_contracts") < 6

        # reload
        core2 = LearningCore(state_dir=state_dir)
        assert core2.rewards.total_episodes() == 2
        assert len(core2.errors.retrieve_all()) == 1


class TestLearningAction:
    def test_registered_and_runs(self, tmp_path):
        from agent.awos.actions.base import build_action, registered_types
        from agent.awos.config import AWOSConfig
        from agent.awos.orchestrator.dispatcher import Dispatcher  # noqa: F401  (registers)
        from agent.awos.policies.engine import PlannedAction

        assert "record_learning" in registered_types()
        cfg = AWOSConfig(state_dir=str(tmp_path))
        action = build_action("record_learning", cfg)
        assert action is not None
        res = action.run(
            PlannedAction(
                type="record_learning",
                params={
                    "task_id": "t9",
                    "operation": "score gov_contracts",
                    "action_id": 2,
                    "success": True,
                    "signal_name": "gov_contracts",
                    "source_tool": "gov_contracts",
                    "cost_usd": 0.01,
                },
                rule_id="r",
                event=None,
            )
        )
        assert res.ok, res.message
        # outcome persisted to state dir
        from agent.awos.learning.learning_core import LearningCore
        core = LearningCore(state_dir=str(tmp_path))
        assert core.rewards.total_episodes() == 1
