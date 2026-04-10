"""Tests for insider_filings L2 — Phase 10b.1.7: MI measurement integration.

Standalone integration test that:
1. Creates PipelineStore in :memory:
2. Populates L2 entity observations via InsiderFilingsTool._persist_entities
3. Creates simulated L1 observations (aggregate cluster count per day)
4. Computes conditional MI of L2 vs L1 against a synthetic target
5. Stores a depth_evaluation result
6. Asserts MI(L2|L1) > 0 — entity-level data adds signal beyond aggregates

Trusted sources:
- MI estimation: Kraskov et al. (2004), KSG estimator via sklearn
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agent.pipeline.depth_eval import compute_conditional_mi, run_depth_evaluation
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore
from agent.tools.insider_filings import InsiderFilingsTool


@pytest.fixture()
def store() -> PipelineStore:
    return PipelineStore(db_path=":memory:")


class TestInsiderFilingsMIIntegration:
    """End-to-end: tool persistence → L1/L2 observation → MI evaluation."""

    def test_l2_adds_signal_beyond_l1(self, store: PipelineStore):
        """Conditional MI(L2; target | L1) > 0 when L2 is a cleaner signal."""
        rng = np.random.default_rng(42)
        n = 200

        # Hidden signal: insider conviction (drives future return)
        signal = rng.normal(0, 1, n)

        # L1 aggregate: noisy daily cluster count (high noise, low resolution)
        l1_obs = signal + rng.normal(0, 3.0, n)

        # L2 entity-level: per-insider purchase size, much closer to true signal
        l2_obs = signal + rng.normal(0, 0.5, n)

        # Target: next-30-day abnormal return (tight to signal)
        target = signal + rng.normal(0, 0.3, n)

        # Populate store with L2 observations via the tool
        tool = InsiderFilingsTool(pipeline_store=store)
        txns = [
            {
                "ticker": "AAPL", "company": "Apple Inc.",
                "name": f"INSIDER_{i}", "role": "Director",
                "type": "P", "shares": float(1000 + l2_obs[i] * 100),
                "price": 150.0, "date": f"2026-01-{(i % 28) + 1:02d}",
                "reporter_cik": f"000{i:07d}", "issuer_cik": "0000320193",
            }
            for i in range(n)
        ]
        tool._persist_entities(txns)

        # Verify observations were actually stored
        sample_eid = entity_id_from_key("person", f"000{0:07d}")
        obs = store.query_entity_observations(sample_eid, source_tool="insider_filings")
        assert len(obs) >= 1, "L2 observations should be stored"

        # Run depth evaluation: MI(L2; target | L1)
        result = run_depth_evaluation(
            store=store,
            tool_name="insider_filings",
            depth_level=2,
            target_variable="equity_return_30d",
            observations_new=l2_obs,
            targets=target,
            observations_existing=l1_obs,
        )

        # MI gain should be positive: L2 adds information beyond L1
        assert result["mi_gain"] is not None
        assert not math.isnan(result["mi_gain"]), "MI should not be NaN with n=200"
        assert result["mi_gain"] > 0.0, (
            f"L2 should add signal beyond L1, got MI gain = {result['mi_gain']}"
        )
        assert result["sample_size"] == n
        assert result["row_id"] is not None

        # Verify evaluation stored in DB
        evals = store.query_depth_evaluations("insider_filings")
        assert len(evals) >= 1
        eval_rec = evals[0]
        assert eval_rec["depth_level"] == 2
        assert eval_rec["target_variable"] == "equity_return_30d"
        assert eval_rec["mi_gain"] is not None
        assert eval_rec["mi_gain"] > 0.0

    def test_l2_marginal_mi_without_l1_baseline(self, store: PipelineStore):
        """Without L1 baseline, marginal MI(L2; target) should also be positive."""
        rng = np.random.default_rng(99)
        n = 150
        signal = rng.normal(0, 1, n)
        l2_obs = signal + rng.normal(0, 0.5, n)
        target = signal + rng.normal(0, 0.3, n)

        result = run_depth_evaluation(
            store=store,
            tool_name="insider_filings",
            depth_level=2,
            target_variable="equity_return_30d",
            observations_new=l2_obs,
            targets=target,
            observations_existing=None,
        )

        assert result["mi_gain"] > 0.0

    def test_no_signal_yields_near_zero_mi(self, store: PipelineStore):
        """When L2 is pure noise independent of target, MI ≈ 0."""
        rng = np.random.default_rng(7)
        n = 200
        l2_obs = rng.normal(0, 1, n)      # pure noise
        target = rng.normal(0, 1, n)       # independent noise
        l1_obs = rng.normal(0, 1, n)       # also independent

        result = run_depth_evaluation(
            store=store,
            tool_name="insider_filings",
            depth_level=2,
            target_variable="equity_return_30d",
            observations_new=l2_obs,
            targets=target,
            observations_existing=l1_obs,
        )

        # MI should be near zero (sklearn KSG clamps negative to 0)
        assert result["mi_gain"] is not None
        assert result["mi_gain"] < 0.1, (
            f"Independent noise should yield MI near 0, got {result['mi_gain']}"
        )
