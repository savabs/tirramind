"""Silent-success regressions in the pipeline executor and layer 5-6 DAGs.

Every test here pins one instance of the same defect class: *success
reported over an empty result*
(``docs/publications/thirteen_ways_a_pipeline_lies.md``).

Three mechanisms, all confirmed in the 2026-09-23 audit:

A1. A node returning ``{"status": "skipped", ...}`` was recorded
    ``completed``, because the executor marked any non-raising operator a
    success and every layer 3-6 node is a ``FunctionOperator`` that passes
    its dict straight through.
A2. The zero-rows guard counted ``nr.stored`` — one bool per node, i.e. a
    count of *nodes* whose summary envelope reached the generic
    ``pipeline_data`` table. A DAG that wrote nothing whatsoever to its own
    domain table still scored ``rows_written > 0`` and sailed through
    (Way 2).
B.  ``adversarial_scan`` called the scanner with hardcoded empty literals,
    so layer 6 could never emit a flag, and reported ``completed`` six
    times over a zero-row ``adversarial_flags``.

Every test in this file fails against the pre-fix executor/DAGs.

No test here touches the live database: each builds its own
``PipelineStore`` under pytest's ``tmp_path``.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from agent.pipeline.dag import DAG
from agent.pipeline.dags.adversarial_scan import (
    DOMAIN_TABLE as ADVERSARIAL_DOMAIN_TABLE,
)
from agent.pipeline.dags.adversarial_scan import (
    build_adversarial_scan_dag,
    run_adversarial_scan,
)
from agent.pipeline.dags.world_model_update import build_world_model_dag
from agent.pipeline.executor import DAGExecutor, DagRun, NodeResult
from agent.pipeline.operators import (
    DOMAIN_TABLES_PARAM_KEY,
    classify_payload_status,
    payload_reason,
)
from agent.pipeline.store import PipelineStore


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[PipelineStore]:
    """A real, file-backed store in a temp dir — never the live DB."""
    s = PipelineStore(str(tmp_path / "pipeline.db"))
    yield s
    s.close()


def _count(store: PipelineStore, table: str) -> int:
    """Row count through a strictly read-only connection."""
    conn = sqlite3.connect(f"file:{store._db_path}?mode=ro", uri=True)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # noqa: S608
    finally:
        conn.close()


# ── A1. a skipped payload is not a completed node ────────────────────


class TestSkippedPayloadIsNotCompleted:
    def test_status_skipped_payload_marks_the_node_skipped(self, store: PipelineStore) -> None:
        """The literal envelope found in the live DB's latest ``inference``
        run: recorded ``completed`` while saying it did nothing."""

        def node_fn(params: dict, upstream: dict) -> dict:
            return {"status": "skipped", "reason": "no_sac_model", "weights": {}}

        dag = DAG(name="skips")
        dag.add("n1", operator=node_fn)
        run = DAGExecutor(store=store).execute(dag)

        nr = run.node_results["n1"]
        assert nr.status == "skipped"
        assert nr.error is not None
        assert "no_sac_model" in nr.error

    def test_skipped_node_does_not_write_a_success_envelope(self, store: PipelineStore) -> None:
        """A skipped node must not leave a ``pipeline_data`` row that later
        reads as evidence the DAG produced something."""

        def node_fn(params: dict, upstream: dict) -> dict:
            return {"status": "skipped", "reason": "no_sac_model"}

        dag = DAG(name="skips_no_envelope")
        dag.add("n1", operator=node_fn)
        run = DAGExecutor(store=store).execute(dag)

        assert run.node_results["n1"].stored is False
        assert _count(store, "pipeline_data") == 0

    def test_boolean_skipped_flag_is_honoured(self, store: PipelineStore) -> None:
        """world_model_update's helpers use ``{"skipped": True}``, not a
        status string."""

        def node_fn(params: dict, upstream: dict) -> dict:
            return {"skipped": True, "reason": "fit_enabled=False"}

        dag = DAG(name="bool_skip")
        dag.add("n1", operator=node_fn)
        run = DAGExecutor(store=store).execute(dag)

        assert run.node_results["n1"].status == "skipped"

    def test_error_payload_marks_the_node_and_run_failed(self, store: PipelineStore) -> None:
        def node_fn(params: dict, upstream: dict) -> dict:
            return {"status": "error", "error": "model checkpoint missing"}

        dag = DAG(name="errors")
        dag.add("n1", operator=node_fn)
        run = DAGExecutor(store=store).execute(dag)

        assert run.node_results["n1"].status == "failed"
        assert run.status == "failed"

    def test_success_false_payload_marks_the_node_failed(self, store: PipelineStore) -> None:
        def node_fn(params: dict, upstream: dict) -> dict:
            return {"success": False, "error": "upstream refused"}

        dag = DAG(name="success_false")
        dag.add("n1", operator=node_fn)
        run = DAGExecutor(store=store).execute(dag)

        assert run.node_results["n1"].status == "failed"

    def test_downstream_of_a_skipped_node_does_not_run(self, store: PipelineStore) -> None:
        """The real consequence: ``emit_portfolio`` used to run on a skipped
        ``_sac_inference`` payload and record ``completed`` with
        ``started_at == finished_at``."""
        ran: list[str] = []

        def upstream_fn(params: dict, upstream: dict) -> dict:
            ran.append("upstream")
            return {"status": "skipped", "reason": "no_sac_model"}

        def downstream_fn(params: dict, upstream: dict) -> dict:
            ran.append("downstream")
            return {"weights": {}}

        dag = DAG(name="chain")
        dag.add("up", operator=upstream_fn)
        dag.add("down", operator=downstream_fn, depends_on=["up"])
        run = DAGExecutor(store=store).execute(dag)

        assert ran == ["upstream"]
        assert run.node_results["down"].status == "skipped"

    @pytest.mark.parametrize(
        "payload",
        [
            {"status": "completed"},
            {"status": "completed_fully"},
            {"status": "ready"},
            {"n_flags": 3},
            {"success": True},
            ["not", "a", "mapping"],
            None,
        ],
    )
    def test_success_payloads_are_left_alone(self, store: PipelineStore, payload: Any) -> None:
        """The reclassification is a closed vocabulary. Anything else keeps
        the meaning its DAG gave it — guessing would be the same
        shot-in-the-dark this whole bug class is made of."""

        def node_fn(params: dict, upstream: dict) -> Any:
            return payload

        dag = DAG(name="successes")
        dag.add("n1", operator=node_fn)
        run = DAGExecutor(store=store).execute(dag)

        assert run.node_results["n1"].status == "completed"

    def test_classify_payload_status_unit(self) -> None:
        assert classify_payload_status({"status": "SKIPPED"}) == "skipped"
        assert classify_payload_status({"status": " failed "}) == "failed"
        assert classify_payload_status({"status": "completed_fully"}) == "completed"
        assert classify_payload_status({"skipped": False}) == "completed"
        assert classify_payload_status(42) == "completed"

    def test_payload_reason_never_invents_one(self) -> None:
        assert payload_reason({"reason": "no_sac_model"}) == "no_sac_model"
        assert payload_reason({"status": "skipped"}) == "<no reason given>"


# ── A2. the guard measures the domain table, not the envelope ────────


def _flag_writer(n: int):
    """A node function that writes *n* real rows to adversarial_flags."""

    def node_fn(params: dict, upstream: dict) -> dict:
        s = PipelineStore(params["db_path"])
        try:
            for i in range(n):
                s.store_adversarial_flag(
                    flag_type="vpin_spike",
                    severity=0.5,
                    confidence=0.5,
                    flagged_at=time.time(),
                    entity_id=f"ent_{i}",
                    signal_name="sig",
                    evidence={},
                )
        finally:
            s.close()
        return {"n_flags": n}

    return node_fn


class TestDomainTableZeroRowsGuard:
    def test_zero_domain_rows_fails_the_run(self, store: PipelineStore) -> None:
        """The headline regression. Pre-fix, this node stored its envelope
        into ``pipeline_data``, ``rows_written`` became 1, and the run was
        reported ``completed`` with the domain table still empty."""

        def node_fn(params: dict, upstream: dict) -> dict:
            return {"n_flags": 0, "flags_stored": 0}

        dag = DAG(name="writes_nothing")
        dag.add(
            "n1",
            operator=node_fn,
            params={"db_path": str(store._db_path), DOMAIN_TABLES_PARAM_KEY: [ADVERSARIAL_DOMAIN_TABLE]},
        )
        run = DAGExecutor(store=store).execute(dag)

        assert run.node_results["n1"].status == "completed"
        # The envelope still landed — which is exactly why the old proxy
        # could not fire.
        assert run.rows_written == 1
        assert run.status == "failed"
        assert ADVERSARIAL_DOMAIN_TABLE in run.error
        assert run.domain_rows_written == {ADVERSARIAL_DOMAIN_TABLE: 0}
        assert _count(store, ADVERSARIAL_DOMAIN_TABLE) == 0

    def test_real_domain_rows_pass_the_guard(self, store: PipelineStore) -> None:
        dag = DAG(name="writes_rows")
        dag.add(
            "n1",
            operator=_flag_writer(2),
            params={"db_path": str(store._db_path), DOMAIN_TABLES_PARAM_KEY: [ADVERSARIAL_DOMAIN_TABLE]},
        )
        run = DAGExecutor(store=store).execute(dag)

        assert run.status == "completed"
        assert run.error is None
        assert run.domain_rows_written == {ADVERSARIAL_DOMAIN_TABLE: 2}

    def test_guard_measures_the_delta_not_the_total(self, store: PipelineStore) -> None:
        """A table that already had rows must not excuse a run that adds
        none — otherwise the guard stops firing the moment the table is
        non-empty once."""
        store.store_adversarial_flag(
            flag_type="pre_existing",
            severity=0.1,
            confidence=0.1,
            flagged_at=time.time(),
        )

        def node_fn(params: dict, upstream: dict) -> dict:
            return {"n_flags": 0}

        dag = DAG(name="adds_nothing")
        dag.add(
            "n1",
            operator=node_fn,
            params={DOMAIN_TABLES_PARAM_KEY: [ADVERSARIAL_DOMAIN_TABLE]},
        )
        run = DAGExecutor(store=store).execute(dag)

        assert run.status == "failed"
        assert run.domain_rows_written == {ADVERSARIAL_DOMAIN_TABLE: 0}

    def test_pipeline_data_may_not_be_declared_as_a_domain_table(self, store: PipelineStore) -> None:
        """Declaring the envelope table would rebuild the original defect."""
        dag = DAG(name="bad_decl")
        dag.add("n1", operator=lambda p, u: {}, params={DOMAIN_TABLES_PARAM_KEY: ["pipeline_data"]})

        with pytest.raises(ValueError, match="pipeline_data"):
            DAGExecutor(store=store).execute(dag)

    @pytest.mark.parametrize("bad", [[], "", 17, ["ok", ""], ["drop table x"]])
    def test_malformed_declaration_raises(self, store: PipelineStore, bad: Any) -> None:
        dag = DAG(name="malformed")
        dag.add("n1", operator=lambda p, u: {}, params={DOMAIN_TABLES_PARAM_KEY: bad})

        with pytest.raises(ValueError):
            DAGExecutor(store=store).execute(dag)

    def test_uncountable_table_fails_loudly_rather_than_reading_as_zero(self, store: PipelineStore) -> None:
        """A declared table that does not exist is a bug in the DAG. It must
        not silently behave like 'the table was empty', and it must not be
        swallowed into a passing run either."""
        dag = DAG(name="missing_table")
        dag.add(
            "n1",
            operator=lambda p, u: {"ok": True},
            params={DOMAIN_TABLES_PARAM_KEY: ["no_such_table"]},
        )
        run = DAGExecutor(store=store).execute(dag)

        assert run.status == "failed"
        assert run.domain_rows_written == {"no_such_table": None}
        assert "unverifiable" in run.error

    def test_skipped_node_domain_table_is_not_held_against_it(self, store: PipelineStore) -> None:
        """A node that skipped never attempted its write; its own 'skipped'
        status is the honest signal, and relabelling it as a zero-rows
        failure would just swap one true statement for another."""

        def node_fn(params: dict, upstream: dict) -> dict:
            return {"status": "skipped", "reason": "nothing to do"}

        dag = DAG(name="skip_decl")
        dag.add("n1", operator=node_fn, params={DOMAIN_TABLES_PARAM_KEY: [ADVERSARIAL_DOMAIN_TABLE]})
        run = DAGExecutor(store=store).execute(dag)

        assert run.node_results["n1"].status == "skipped"
        assert run.domain_rows_written == {}
        assert run.status == "completed"

    def test_reserved_key_is_not_passed_to_the_node_function(self, store: PipelineStore) -> None:
        """``__domain_tables__`` configures the executor, not the callee."""
        seen: dict = {}

        def node_fn(params: dict, upstream: dict) -> dict:
            seen.update(params)
            return {"ok": True}

        dag = DAG(name="reserved")
        dag.add(
            "n1",
            operator=node_fn,
            params={"db_path": str(store._db_path), DOMAIN_TABLES_PARAM_KEY: [ADVERSARIAL_DOMAIN_TABLE]},
        )
        DAGExecutor(store=store).execute(dag)

        assert DOMAIN_TABLES_PARAM_KEY not in seen
        assert seen["db_path"] == str(store._db_path)

    def test_undeclared_dags_keep_the_old_envelope_backstop(self, store: PipelineStore) -> None:
        """DAGs that declare nothing must not silently lose the weaker
        check they already had."""
        run = DagRun(run_id="r1", dag_name="d1", started_at=time.time())
        run.node_results["n1"] = NodeResult(node_id="n1", status="completed", stored=False)
        run.status = "completed"
        DAGExecutor._apply_zero_rows_guard(run, eligible_store_nodes=1, any_failure=False)
        assert run.status == "failed"
        assert "zero rows" in run.error


# ── B. adversarial_scan must not report success over nothing ─────────


class TestAdversarialScanCannotReportEmptySuccess:
    def test_empty_inputs_raise_instead_of_returning_a_summary(self, store: PipelineStore) -> None:
        """Pre-fix this returned ``{"n_flags": 0, "flags_stored": 0, ...}``
        and the node was recorded ``completed`` — six times, on top of a
        zero-row ``adversarial_flags``."""
        with pytest.raises(RuntimeError) as exc:
            run_adversarial_scan(params={"db_path": str(store._db_path)}, upstream={})

        message = str(exc.value)
        assert "every scanner input was empty" in message
        # The error names what is missing, so the next reader does not have
        # to rediscover it.
        assert "signal_returns" in message
        assert "clusters" in message

    def test_dag_run_is_failed_and_the_table_stays_empty(self, store: PipelineStore) -> None:
        dag = build_adversarial_scan_dag(db_path=str(store._db_path))
        run = DAGExecutor(store=store).execute(dag)

        assert run.node_results["scan_adversarial"].status == "failed"
        assert run.status == "failed"
        assert _count(store, ADVERSARIAL_DOMAIN_TABLE) == 0
        # And no "successful scan" envelope was written either.
        assert _count(store, "pipeline_data") == 0

    def test_dag_declares_its_domain_table(self) -> None:
        dag = build_adversarial_scan_dag()
        params = dag.nodes["scan_adversarial"].params
        assert params[DOMAIN_TABLES_PARAM_KEY] == [ADVERSARIAL_DOMAIN_TABLE]

    def test_world_model_dag_declares_beliefs(self) -> None:
        dag = build_world_model_dag()
        params = dag.nodes["update_beliefs"].params
        assert params[DOMAIN_TABLES_PARAM_KEY] == ["beliefs"]
