"""Unit tests for pipeline_orchestrator.py helpers.

Tests cover:
  - find_latest_checkpoint: empty dir, single file, multiple files
  - is_collapsed: no metrics, normal metrics, loss explosion, negative IC
  - build_retrain_cmd: without config-file, with config-file
  - write_session_summary: correct frontmatter fields
  - sync_state_to_github: git commands called with correct args
  - load_config_if_current: missing, fresh, stale
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

# Add repo root to path so we can import without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pipeline_orchestrator import (  # noqa: E402
    PipelineState,
    build_retrain_cmd,
    find_latest_checkpoint,
    is_collapsed,
    load_config_if_current,
    sync_state_to_github,
    write_session_summary,
)

# ---------------------------------------------------------------------------
# find_latest_checkpoint
# ---------------------------------------------------------------------------


class TestFindLatestCheckpoint:
    def test_empty_directory_returns_zero(self, tmp_path):
        assert find_latest_checkpoint(tmp_path) == 0

    def test_single_checkpoint(self, tmp_path):
        (tmp_path / "epoch_007.pt").touch()
        assert find_latest_checkpoint(tmp_path) == 7

    def test_multiple_checkpoints_returns_max(self, tmp_path):
        for n in [3, 15, 22, 8]:
            (tmp_path / f"epoch_{n:03d}.pt").touch()
        assert find_latest_checkpoint(tmp_path) == 22

    def test_ignores_non_epoch_files(self, tmp_path):
        (tmp_path / "model_best.pt").touch()
        (tmp_path / "epoch_005.pt").touch()
        assert find_latest_checkpoint(tmp_path) == 5

    def test_malformed_filename_ignored(self, tmp_path):
        (tmp_path / "epoch_abc.pt").touch()
        (tmp_path / "epoch_010.pt").touch()
        assert find_latest_checkpoint(tmp_path) == 10


# ---------------------------------------------------------------------------
# is_collapsed
# ---------------------------------------------------------------------------


def _write_metrics(checkpoint_dir: Path, records: list[dict]) -> None:
    p = checkpoint_dir / "metrics.jsonl"
    with p.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


class TestIsCollapsed:
    def test_no_metrics_file_returns_false(self, tmp_path):
        assert is_collapsed(tmp_path) is False

    def test_fewer_than_3_records_returns_false(self, tmp_path):
        _write_metrics(
            tmp_path,
            [
                {"epoch": 1, "loss": {"total": 1.0, "return": 0.5}},
                {"epoch": 2, "loss": {"total": 0.9, "return": 0.4}},
            ],
        )
        assert is_collapsed(tmp_path) is False

    def test_normal_healthy_training_returns_false(self, tmp_path):
        _write_metrics(
            tmp_path,
            [
                {
                    "epoch": i,
                    "loss": {"total": 1.0 - i * 0.05, "return": 0.5 - i * 0.02},
                }
                for i in range(10)
            ],
        )
        assert is_collapsed(tmp_path) is False

    def test_loss_explosion_returns_true(self, tmp_path):
        # First epoch total=1.0, last 3 epochs total >> 10.0
        records = [{"epoch": 1, "loss": {"total": 1.0, "return": 0.5}}]
        records += [{"epoch": i, "loss": {"total": 50.0 + i, "return": 0.3}} for i in range(2, 12)]
        _write_metrics(tmp_path, records)
        assert is_collapsed(tmp_path) is True

    def test_negative_return_loss_proxy_returns_true(self, tmp_path):
        # 3 consecutive records with return < -0.05
        _write_metrics(
            tmp_path,
            [{"epoch": i, "loss": {"total": 2.0, "return": -0.1}} for i in range(3)],
        )
        assert is_collapsed(tmp_path) is True

    def test_mixed_negative_return_not_all_negative_returns_false(self, tmp_path):
        # Only 2 of 3 tail records are negative
        _write_metrics(
            tmp_path,
            [
                {"epoch": 1, "loss": {"total": 1.0, "return": 0.2}},
                {"epoch": 2, "loss": {"total": 1.0, "return": -0.1}},
                {"epoch": 3, "loss": {"total": 1.0, "return": -0.2}},
            ],
        )
        assert is_collapsed(tmp_path) is False


# ---------------------------------------------------------------------------
# build_retrain_cmd
# ---------------------------------------------------------------------------


class TestBuildRetrainCmd:
    def _base_args(self, tmp_path):
        return dict(
            work_dir=tmp_path,
            checkpoint_dir=tmp_path / "checkpoints" / "h_g",
            db_path=tmp_path / "pipeline.db",
            target_epoch=35,
            resume_epoch=30,
            device="cuda",
            config_file=None,
        )

    def test_happy_path_contains_required_flags(self, tmp_path):
        cmd = build_retrain_cmd(**self._base_args(tmp_path))
        cmd_str = " ".join(cmd)
        assert "retrain_gnn.py" in cmd_str
        assert "--epochs" in cmd_str
        assert "35" in cmd_str
        assert "--resume" in cmd_str
        assert "30" in cmd_str
        assert "--device" in cmd_str
        assert "cuda" in cmd_str
        assert "--config-file" not in cmd_str

    def test_config_file_appended_when_exists(self, tmp_path):
        cfg = tmp_path / "next_config.json"
        cfg.write_text('{"pattern": "dt_dominance"}')
        args = self._base_args(tmp_path)
        args["config_file"] = cfg
        cmd = build_retrain_cmd(**args)
        assert "--config-file" in cmd
        assert str(cfg) in cmd

    def test_config_file_not_appended_when_missing(self, tmp_path):
        args = self._base_args(tmp_path)
        args["config_file"] = tmp_path / "nonexistent_config.json"
        cmd = build_retrain_cmd(**args)
        assert "--config-file" not in cmd

    def test_all_standard_flags_present(self, tmp_path):
        cmd = build_retrain_cmd(**self._base_args(tmp_path))
        for flag in [
            "--hidden-dim",
            "--num-layers",
            "--num-heads",
            "--lr",
            "--backup",
            "--window-size",
            "--gdelt-frac",
            "--max-windows",
            "--auto-tune",
            "--listnet",
            "--return-log-var-max",
            "--skip-eval",
            "--checkpoint-dir",
            "--model-out",
        ]:
            assert flag in cmd, f"Missing flag: {flag}"


# ---------------------------------------------------------------------------
# write_session_summary
# ---------------------------------------------------------------------------


class TestWriteSessionSummary:
    def _call(self, tmp_path, state=PipelineState.SESSION_END, **kwargs):
        defaults = dict(
            knowledge_dir=tmp_path / "knowledge",
            state=state,
            start_epoch=22,
            end_epoch=27,
            blocks_completed=1,
            session_duration_hours=10.5,
            last_pattern="improving",
            flag_overrides=None,
        )
        defaults.update(kwargs)
        return write_session_summary(**defaults)

    def test_creates_file_in_knowledge_dir(self, tmp_path):
        path = self._call(tmp_path)
        assert path.exists()
        assert path.parent == tmp_path / "knowledge"

    def test_frontmatter_contains_state(self, tmp_path):
        path = self._call(tmp_path, state=PipelineState.STRUCTURAL_HALT)
        content = path.read_text()
        assert "state: structural_halt" in content

    def test_frontmatter_contains_epoch(self, tmp_path):
        path = self._call(tmp_path, end_epoch=42)
        content = path.read_text()
        assert "epoch: 42" in content

    def test_flag_overrides_serialised_as_json(self, tmp_path):
        path = self._call(
            tmp_path,
            flag_overrides={"return_weight": 2.0},
        )
        content = path.read_text()
        assert '"return_weight": 2.0' in content

    def test_filename_starts_with_session_summary(self, tmp_path):
        path = self._call(tmp_path)
        assert path.name.startswith("session_summary_")

    def test_creates_knowledge_dir_if_missing(self, tmp_path):
        knowledge_dir = tmp_path / "nonexistent" / "knowledge"
        assert not knowledge_dir.exists()
        path = self._call(tmp_path, knowledge_dir=knowledge_dir)
        assert path.exists()


# ---------------------------------------------------------------------------
# sync_state_to_github
# ---------------------------------------------------------------------------


class TestSyncStateToGitHub:
    def _base_kwargs(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        summary = knowledge_dir / "session_summary_test.md"
        summary.write_text("---\nstate: session_end\n---\n")
        metrics = ckpt_dir / "metrics.jsonl"
        metrics.write_text('{"epoch": 27}\n')
        return dict(
            work_dir=tmp_path,
            checkpoint_dir=ckpt_dir,
            knowledge_dir=knowledge_dir,
            summary_path=summary,
            state=PipelineState.SESSION_END,
            end_epoch=27,
            github_token=None,
        )

    def test_happy_path_calls_git_add_commit_push(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("scripts.pipeline_orchestrator.subprocess.run", return_value=completed) as mock_run:
            result = sync_state_to_github(**kwargs)
        assert result is True
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("add" in c for c in calls)
        assert any("commit" in c for c in calls)
        assert any("push" in c for c in calls)

    def test_git_add_failure_returns_false(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")
        with patch("scripts.pipeline_orchestrator.subprocess.run", return_value=failed):
            result = sync_state_to_github(**kwargs)
        assert result is False

    def test_push_failure_returns_false(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)

        def side_effect(cmd, **kw):
            if "push" in cmd:
                return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="rejected")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

        with patch("scripts.pipeline_orchestrator.subprocess.run", side_effect=side_effect):
            result = sync_state_to_github(**kwargs)
        assert result is False

    def test_token_sets_remote_url(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        kwargs["github_token"] = "ghp_testtoken"  # noqa: S105
        ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("scripts.pipeline_orchestrator.subprocess.run", return_value=ok) as mock_run:
            sync_state_to_github(**kwargs)
        all_cmds = [str(c) for c in mock_run.call_args_list]
        assert any("set-url" in c and "ghp_testtoken" in c for c in all_cmds)

    def test_no_state_files_returns_true_without_git_calls(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        # Remove all state files
        for f in [
            kwargs["checkpoint_dir"] / "metrics.jsonl",
            kwargs["summary_path"],
        ]:
            f.unlink(missing_ok=True)
        kwargs["summary_path"] = tmp_path / "nonexistent_summary.md"
        with patch("scripts.pipeline_orchestrator.subprocess.run") as mock_run:
            result = sync_state_to_github(**kwargs)
        assert result is True
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# load_config_if_current
# ---------------------------------------------------------------------------


class TestLoadConfigIfCurrent:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_config_if_current(tmp_path / "missing.json", 27, 5) is None

    def test_fresh_config_returns_path(self, tmp_path):
        cfg = tmp_path / "next_config.json"
        cfg.write_text(json.dumps({"based_on_epoch": 25, "pattern": "improving"}))
        result = load_config_if_current(cfg, 27, 5)
        assert result == cfg

    def test_stale_config_returns_none(self, tmp_path):
        cfg = tmp_path / "next_config.json"
        # based_on_epoch=5, current=27, block_size=5 → 22 > 10 → stale
        cfg.write_text(json.dumps({"based_on_epoch": 5, "pattern": "improving"}))
        result = load_config_if_current(cfg, 27, 5)
        assert result is None

    def test_missing_based_on_epoch_uses_current_epoch(self, tmp_path):
        cfg = tmp_path / "next_config.json"
        # No based_on_epoch — defaults to current_epoch → freshness = 0 → not stale
        cfg.write_text(json.dumps({"pattern": "improving"}))
        result = load_config_if_current(cfg, 27, 5)
        assert result == cfg

    def test_corrupt_json_falls_through_and_returns_path(self, tmp_path):
        cfg = tmp_path / "next_config.json"
        cfg.write_text("{bad json}")
        # Corrupt but file exists — return it anyway (retrain_gnn will validate)
        result = load_config_if_current(cfg, 27, 5)
        assert result == cfg
