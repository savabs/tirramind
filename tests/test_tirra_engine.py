"""Tests for the unified TirraEngine entrypoint (collect/brief/deliver/bid/email)."""

from __future__ import annotations

import time

from scripts.tirra_engine import email_brief, record_bid


def test_record_bid_personalizes_pwin(tmp_path):
    learner = str(tmp_path / "win.jsonl")
    # prior for VA small bucket is 1/3
    from agent.quant.contract_opportunity import WinProbabilityLearner

    assert abs(WinProbabilityLearner(learner).probability_of("VA", 50000.0) - 1.0 / 3.0) < 1e-9
    # a win raises it
    res = record_bid("VA", 50000.0, True, learner)
    assert res["new_p_win"] > 1.0 / 3.0
    assert res["basis"] == "learned"


def test_record_bid_losses_lower_pwin(tmp_path):
    learner = str(tmp_path / "win.jsonl")
    record_bid("USDA", 50000.0, False, learner)
    record_bid("USDA", 50000.0, False, learner)
    res = record_bid("USDA", 50000.0, False, learner)
    assert res["new_p_win"] < 1.0 / 3.0  # three losses → below prior


def test_email_brief_noop_without_smtp(tmp_path, monkeypatch):
    monkeypatch.delenv("TIRRA_SMTP_HOST", raising=False)
    res = email_brief("# brief", ["a@b.com"])
    assert res["emailed"] is False
    assert "TIRRA_SMTP_HOST" in res["reason"]


def test_email_brief_noop_without_recipients(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_SMTP_HOST", "smtp.example.com")
    monkeypatch.delenv("TIRRA_BRIEF_TO", raising=False)
    res = email_brief("# brief", [])
    assert res["emailed"] is False


class TestCollectionBlockingContract:
    """Regression coverage for the fire-and-forget bug: `--full-collect --once`
    used to spawn the DAG on a daemon thread and exit immediately, silently
    killing the thread before it persisted anything. run_collection_sync()
    must actually block until the DAG finishes; run_collection() must not."""

    def test_run_collection_sync_blocks_until_dag_finishes(self, monkeypatch):
        import scripts.tirra_engine as engine

        finished = {"done": False}

        def _slow_dag(config):
            time.sleep(0.15)
            finished["done"] = True
            return {"dag": "daily_collection", "status": "completed", "nodes_ok": 1, "nodes_total": 1}

        monkeypatch.setattr(engine, "_run_daily_collection_dag", _slow_dag)
        result = engine.run_collection_sync(config=object())

        assert finished["done"] is True  # only true if the call actually waited
        assert result["status"] == "completed"

    def test_run_collection_returns_before_dag_finishes(self, monkeypatch):
        import scripts.tirra_engine as engine

        finished = {"done": False}

        def _slow_dag(config):
            time.sleep(0.15)
            finished["done"] = True
            return {"dag": "daily_collection", "status": "completed"}

        monkeypatch.setattr(engine, "_run_daily_collection_dag", _slow_dag)
        result = engine.run_collection(config=object())

        assert result["status"] == "started_background"
        assert finished["done"] is False  # returned before the slow DAG completed

        time.sleep(0.3)
        assert finished["done"] is True  # but it does eventually run, in the background
