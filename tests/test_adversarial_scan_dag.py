"""Tests for the adversarial_scan DAG operator (Phase 22).

Before the fix, ``run_adversarial_scan`` called ``store.put(...)`` — a
method that does not exist on ``PipelineStore`` — so every flag write
raised ``AttributeError``. Because that call sits inside the DAG's only
node with no enclosing try/except, the operator raised, the executor
recorded the node as genuinely failed, and the DAG never produced a
single row in ``adversarial_flags`` regardless of how many flags the
scanner found: a permanent no-op.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.adversarial.flags import AdversarialFlag
from agent.pipeline.dag import DAG
from agent.pipeline.dags.adversarial_scan import (
    DAG_NAME,
    DEPENDS_ON,
    build_adversarial_scan_dag,
    run_adversarial_scan,
)
from agent.pipeline.store import PipelineStore


@pytest.fixture()
def store(tmp_path: Path) -> PipelineStore:
    return PipelineStore(str(tmp_path / "pipeline.db"))


class TestConstants:
    def test_dag_name(self) -> None:
        assert DAG_NAME == "adversarial_scan"

    def test_depends_on(self) -> None:
        assert ["convergence_detection"] == DEPENDS_ON


class TestDAGStructure:
    def test_builds_without_error(self) -> None:
        dag = build_adversarial_scan_dag()
        assert isinstance(dag, DAG)

    def test_has_scan_node(self) -> None:
        dag = build_adversarial_scan_dag()
        assert "scan_adversarial" in dag.nodes


class TestStoreAdversarialFlag:
    """The schema and store method the operator must be using."""

    def test_adversarial_flags_table_exists(self, store: PipelineStore) -> None:
        conn = store._get_conn()
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='adversarial_flags'").fetchone()
        assert row is not None

    def test_store_has_no_put_method(self, store: PipelineStore) -> None:
        """Regression guard for the exact bug: store.put() never existed."""
        assert not hasattr(store, "put")

    def test_store_adversarial_flag_roundtrip(self, store: PipelineStore) -> None:
        row_id = store.store_adversarial_flag(
            flag_type="edge_decay",
            severity=0.8,
            confidence=0.9,
            flagged_at=1700000000.0,
            entity_id="ent_0",
            signal_name="sig_a",
            evidence={"sharpe": -0.2},
        )
        assert row_id is not None

        rows = store.query_adversarial_flags()
        assert len(rows) == 1
        assert rows[0]["flag_type"] == "edge_decay"
        assert rows[0]["entity_id"] == "ent_0"


class TestRunAdversarialScan:
    def test_does_not_raise_and_returns_summary(self, store: PipelineStore) -> None:
        """With no flags produced (the real default: empty scan inputs),
        the operator must complete cleanly — not raise AttributeError on
        a nonexistent store.put()."""
        result = run_adversarial_scan(
            params={"db_path": str(store._db_path)},
            upstream={},
        )
        assert result["n_flags"] == 0
        assert result["flags_stored"] == 0
        assert result["flag_types"] == []

    def test_flags_are_actually_persisted(self, store: PipelineStore) -> None:
        """Force the scanner to produce flags and verify they land in
        adversarial_flags via store_adversarial_flag, not a no-op."""
        fake_flags = [
            AdversarialFlag(
                flag_type="vpin_spike",
                severity=0.6,
                confidence=0.7,
                entity_id=None,
                signal_name=None,
                evidence={"vpin": 0.9},
                timestamp=1700000000.0,
            ),
            AdversarialFlag(
                flag_type="crowding_risk",
                severity=0.4,
                confidence=0.5,
                entity_id="ent_1",
                signal_name="sig_b",
                evidence={},
                timestamp=1700000001.0,
            ),
        ]

        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = fake_flags

        with patch(
            "agent.pipeline.dags.adversarial_scan.AdversarialScanner",
            return_value=mock_scanner,
        ):
            result = run_adversarial_scan(
                params={"db_path": str(store._db_path)},
                upstream={},
            )

        assert result["n_flags"] == 2
        assert result["flags_stored"] == 2
        assert set(result["flag_types"]) == {"vpin_spike", "crowding_risk"}

        stored = store.query_adversarial_flags(limit=10)
        assert len(stored) == 2
        assert {r["flag_type"] for r in stored} == {"vpin_spike", "crowding_risk"}
        crowding_row = next(r for r in stored if r["flag_type"] == "crowding_risk")
        assert crowding_row["entity_id"] == "ent_1"
        assert crowding_row["signal_name"] == "sig_b"
