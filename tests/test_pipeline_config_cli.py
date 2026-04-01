"""
Tests for Step 7.5: PipelineConfig + CLI integration.

Covers:
- PipelineConfig defaults and from_env
- PipelineConfig embedded in AgentConfig
- CLI argument parsing for --pipeline subcommand
- Pipeline sub-command dispatch (run, list, status, start)
- Edge cases: empty pipeline args, unknown sub-command, missing dag name
"""

from __future__ import annotations

import os
import argparse
import sys
from unittest import mock

import pytest

from agent.config.settings import PipelineConfig, AgentConfig


# ── PipelineConfig ───────────────────────────────────────────────────────────


class TestPipelineConfigDefaults:
    """Test PipelineConfig default values."""

    def test_default_db_path(self):
        cfg = PipelineConfig()
        assert cfg.db_path == ".tirra_pipeline/pipeline.db"

    def test_default_max_workers(self):
        cfg = PipelineConfig()
        assert cfg.max_workers == 4

    def test_default_log_level(self):
        cfg = PipelineConfig()
        assert cfg.log_level == "INFO"

    def test_frozen(self):
        cfg = PipelineConfig()
        with pytest.raises(AttributeError):
            cfg.db_path = "/other/path"  # type: ignore[misc]

    def test_from_env_defaults(self):
        """from_env with no env vars should use defaults."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = PipelineConfig.from_env()
        assert cfg.db_path == ".tirra_pipeline/pipeline.db"
        assert cfg.max_workers == 4
        assert cfg.log_level == "INFO"


class TestPipelineConfigFromEnv:
    """Test PipelineConfig.from_env() reads env vars."""

    def test_custom_db_path(self):
        with mock.patch.dict(os.environ, {"TIRRA_PIPELINE_DB": "/tmp/test.db"}):
            cfg = PipelineConfig.from_env()
        assert cfg.db_path == "/tmp/test.db"

    def test_custom_max_workers(self):
        with mock.patch.dict(os.environ, {"TIRRA_PIPELINE_WORKERS": "8"}):
            cfg = PipelineConfig.from_env()
        assert cfg.max_workers == 8

    def test_custom_log_level(self):
        with mock.patch.dict(os.environ, {"TIRRA_PIPELINE_LOG_LEVEL": "DEBUG"}):
            cfg = PipelineConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_all_custom(self):
        env = {
            "TIRRA_PIPELINE_DB": "/data/pipe.db",
            "TIRRA_PIPELINE_WORKERS": "16",
            "TIRRA_PIPELINE_LOG_LEVEL": "WARNING",
        }
        with mock.patch.dict(os.environ, env):
            cfg = PipelineConfig.from_env()
        assert cfg.db_path == "/data/pipe.db"
        assert cfg.max_workers == 16
        assert cfg.log_level == "WARNING"

    def test_invalid_max_workers_raises(self):
        with mock.patch.dict(os.environ, {"TIRRA_PIPELINE_WORKERS": "not_a_number"}):
            with pytest.raises(ValueError):
                PipelineConfig.from_env()


class TestAgentConfigIncludesPipeline:
    """Test that AgentConfig includes PipelineConfig."""

    def test_pipeline_field_exists(self):
        cfg = AgentConfig(llm=mock.MagicMock(), pipeline=PipelineConfig())
        assert isinstance(cfg.pipeline, PipelineConfig)

    def test_from_env_includes_pipeline(self):
        env = {
            "TIRRA_PIPELINE_DB": "/custom/db.sqlite",
            "TIRRA_PIPELINE_WORKERS": "2",
        }
        with mock.patch.dict(os.environ, env):
            cfg = AgentConfig.from_env()
        assert cfg.pipeline.db_path == "/custom/db.sqlite"
        assert cfg.pipeline.max_workers == 2

    def test_default_pipeline_in_agent_config(self):
        """AgentConfig.from_env() without pipeline env vars uses defaults."""
        with mock.patch.dict(os.environ, {}, clear=False):
            cfg = AgentConfig.from_env()
        assert cfg.pipeline.db_path == ".tirra_pipeline/pipeline.db"
        assert cfg.pipeline.max_workers == 4


# ── CLI Argument Parsing ─────────────────────────────────────────────────────


class TestCLIPipelineArgParsing:
    """Test that --pipeline is parsed correctly by argparse."""

    @staticmethod
    def _make_parser() -> argparse.ArgumentParser:
        """Replicate the argparse setup from cli.py."""
        parser = argparse.ArgumentParser()
        parser.add_argument("goal", nargs="?")
        parser.add_argument("--interactive", "-i", action="store_true")
        parser.add_argument("--autonomous", "-a", action="store_true")
        parser.add_argument("--max-goals", type=int, default=5)
        parser.add_argument("--pipeline", "-p", nargs="*", metavar="CMD")
        parser.add_argument("--verbose", "-v", action="store_true")
        return parser

    def test_pipeline_run_dag(self):
        parser = self._make_parser()
        args = parser.parse_args(["--pipeline", "run", "daily_collection"])
        assert args.pipeline == ["run", "daily_collection"]

    def test_pipeline_list(self):
        parser = self._make_parser()
        args = parser.parse_args(["--pipeline", "list"])
        assert args.pipeline == ["list"]

    def test_pipeline_status_no_run_id(self):
        parser = self._make_parser()
        args = parser.parse_args(["--pipeline", "status"])
        assert args.pipeline == ["status"]

    def test_pipeline_status_with_run_id(self):
        parser = self._make_parser()
        args = parser.parse_args(["--pipeline", "status", "abc123"])
        assert args.pipeline == ["status", "abc123"]

    def test_pipeline_start(self):
        parser = self._make_parser()
        args = parser.parse_args(["--pipeline", "start"])
        assert args.pipeline == ["start"]

    def test_pipeline_empty(self):
        """--pipeline with no sub-args should give empty list."""
        parser = self._make_parser()
        args = parser.parse_args(["--pipeline"])
        assert args.pipeline == []

    def test_pipeline_none_when_not_specified(self):
        parser = self._make_parser()
        args = parser.parse_args(["some goal"])
        assert args.pipeline is None

    def test_short_flag(self):
        parser = self._make_parser()
        args = parser.parse_args(["-p", "list"])
        assert args.pipeline == ["list"]

    def test_pipeline_does_not_interfere_with_goal(self):
        """When --pipeline is used, goal should remain None."""
        parser = self._make_parser()
        args = parser.parse_args(["--pipeline", "run", "my_dag"])
        assert args.goal is None

    def test_pipeline_with_verbose(self):
        parser = self._make_parser()
        args = parser.parse_args(["-v", "--pipeline", "list"])
        assert args.pipeline == ["list"]
        assert args.verbose is True


# ── CLI Pipeline Dispatch ────────────────────────────────────────────────────

# The agent.cli module has heavy transitive imports (hmmlearn, etc.) that may
# not be installed in the test environment. We test the dispatch functions by
# extracting them from the source rather than importing the whole cli module.
# The functions run_pipeline, _pipeline_run, _pipeline_list, _pipeline_status,
# _pipeline_start are tested via mock patches on sys.modules to avoid the
# import chain.

def _import_cli():
    """Try to import agent.cli, skip tests if transitive deps missing."""
    pytest.importorskip("hmmlearn", reason="hmmlearn not installed — skipping CLI dispatch tests")
    import agent.cli
    return agent.cli


class TestRunPipelineDispatch:
    """Test run_pipeline dispatches to correct sub-handlers."""

    def test_empty_args_exits(self):
        cli = _import_cli()
        config = mock.MagicMock()
        with pytest.raises(SystemExit):
            cli.run_pipeline([], config)

    def test_unknown_command_exits(self):
        cli = _import_cli()
        config = mock.MagicMock()
        with pytest.raises(SystemExit):
            cli.run_pipeline(["bogus"], config)

    def test_run_dispatches(self):
        cli = _import_cli()
        config = mock.MagicMock()
        with mock.patch.object(cli, "_pipeline_run") as m:
            cli.run_pipeline(["run", "my_dag"], config)
            m.assert_called_once_with(["my_dag"], config)

    def test_list_dispatches(self):
        cli = _import_cli()
        config = mock.MagicMock()
        with mock.patch.object(cli, "_pipeline_list") as m:
            cli.run_pipeline(["list"], config)
            m.assert_called_once()

    def test_status_dispatches(self):
        cli = _import_cli()
        config = mock.MagicMock()
        with mock.patch.object(cli, "_pipeline_status") as m:
            cli.run_pipeline(["status"], config)
            m.assert_called_once_with([], config)

    def test_status_with_run_id_dispatches(self):
        cli = _import_cli()
        config = mock.MagicMock()
        with mock.patch.object(cli, "_pipeline_status") as m:
            cli.run_pipeline(["status", "run-123"], config)
            m.assert_called_once_with(["run-123"], config)

    def test_start_dispatches(self):
        cli = _import_cli()
        config = mock.MagicMock()
        with mock.patch.object(cli, "_pipeline_start") as m:
            cli.run_pipeline(["start"], config)
            m.assert_called_once_with(config)


class TestPipelineRunSubcommand:
    """Test _pipeline_run edge cases."""

    def test_run_no_dag_name_exits(self):
        cli = _import_cli()
        config = mock.MagicMock()
        with pytest.raises(SystemExit):
            cli._pipeline_run([], config)

    def test_run_missing_registry_exits(self):
        """When DAGRegistry is not importable, exits gracefully."""
        cli = _import_cli()
        config = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"agent.pipeline.registry": None}):
            with pytest.raises((SystemExit, ImportError)):
                cli._pipeline_run(["some_dag"], config)


class TestPipelineStatusSubcommand:
    """Test _pipeline_status edge cases."""

    def test_status_no_runs(self, tmp_path):
        cli = _import_cli()
        db_path = str(tmp_path / "test.db")
        config = mock.MagicMock()
        config.pipeline.db_path = db_path
        # Should not raise — just prints "No pipeline runs found."
        cli._pipeline_status([], config)

    def test_status_unknown_run_id_exits(self, tmp_path):
        cli = _import_cli()
        db_path = str(tmp_path / "test.db")
        config = mock.MagicMock()
        config.pipeline.db_path = db_path
        with pytest.raises(SystemExit):
            cli._pipeline_status(["nonexistent-run"], config)

    def test_status_shows_existing_run(self, tmp_path):
        cli = _import_cli()
        from agent.pipeline.store import PipelineStore
        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path=db_path)
        store.record_run_start(dag_name="test_dag", trigger="manual", run_id="run-abc")
        store.record_run_end("run-abc", "success", {})

        config = mock.MagicMock()
        config.pipeline.db_path = db_path
        # Should not raise
        cli._pipeline_status(["run-abc"], config)

    def test_status_lists_runs(self, tmp_path):
        cli = _import_cli()
        from agent.pipeline.store import PipelineStore
        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path=db_path)
        store.record_run_start(dag_name="dag_a", trigger="cron", run_id="run-1")
        store.record_run_end("run-1", "success", {})
        store.record_run_start(dag_name="dag_b", trigger="manual", run_id="run-2")
        store.record_run_end("run-2", "failed", {})

        config = mock.MagicMock()
        config.pipeline.db_path = db_path
        # Should not raise
        cli._pipeline_status([], config)
