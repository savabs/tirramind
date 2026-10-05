"""Edge-case tests for workflow preflight enforcement."""

from __future__ import annotations

from pathlib import Path

from agent.workflow_guard import infer_task_from_changed_files, validate_preflight


def _write_task(repo_root: Path, task_name: str, body: str) -> str:
    task_path = repo_root / "tasks" / "active" / task_name
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(body, encoding="utf-8")
    return task_path.relative_to(repo_root).as_posix()


class TestValidatePreflight:
    """Test workflow guard validation edge cases."""

    def test_workflow_only_paths_pass_without_task(self, tmp_path: Path):
        """Workflow artifacts alone should not require a selected task."""
        changed = [
            "docs/research/example.md",
            "docs/specs/example_spec.md",
            "tasks/active/example.md",
            "docs/memory/chat_checkpoint_2026-04-01.md",
        ]
        assert validate_preflight(tmp_path, changed) == []

    def test_non_workflow_change_requires_task(self, tmp_path: Path):
        """Implementation changes should fail without a governing task."""
        errors = validate_preflight(tmp_path, ["agent/core/orchestrator.py"])
        assert len(errors) == 1
        assert "require a governing task file" in errors[0]

    def test_empty_change_set_fails(self, tmp_path: Path):
        """An empty change set should be rejected explicitly."""
        errors = validate_preflight(tmp_path, [])
        assert errors == ["No files were provided for workflow validation."]

    def test_missing_task_file_fails(self, tmp_path: Path):
        """A selected task path must exist under tasks/active/."""
        errors = validate_preflight(
            tmp_path,
            ["agent/core/orchestrator.py"],
            task_file="tasks/active/missing_task.md",
        )
        assert len(errors) == 1
        assert "Task file does not exist" in errors[0]

    def test_task_missing_research_line_fails(self, tmp_path: Path):
        """Malformed task files missing Research metadata should fail."""
        task_file = _write_task(
            tmp_path,
            "bad_task.md",
            "# Task: bad\n\nStatus: active\nSpec: docs/specs/bad_spec.md\n",
        )
        errors = validate_preflight(
            tmp_path,
            ["agent/core/orchestrator.py"],
            task_file=task_file,
        )
        assert len(errors) == 1
        assert "missing a Research:" in errors[0]

    def test_task_with_missing_linked_files_fails(self, tmp_path: Path):
        """Tasks that reference missing research or spec files should fail."""
        task_file = _write_task(
            tmp_path,
            "missing_links.md",
            "# Task: missing_links\n\nStatus: active\n"
            "Research: docs/research/missing.md\n"
            "Spec: docs/specs/missing_spec.md\n",
        )
        errors = validate_preflight(
            tmp_path,
            ["agent/core/orchestrator.py"],
            task_file=task_file,
        )
        assert len(errors) == 1
        assert "Research file does not exist" in errors[0]

    def test_valid_task_allows_non_workflow_change(self, tmp_path: Path):
        """A valid task with existing research and spec should satisfy preflight."""
        (tmp_path / "docs" / "research").mkdir(parents=True)
        (tmp_path / "docs" / "specs").mkdir(parents=True)
        (tmp_path / "docs" / "research" / "feature.md").write_text("# Feature\n", encoding="utf-8")
        (tmp_path / "docs" / "specs" / "feature_spec.md").write_text("# Spec\n", encoding="utf-8")
        task_file = _write_task(
            tmp_path,
            "feature.md",
            "# Task: feature\n\nStatus: active\nResearch: docs/research/feature.md\nSpec: docs/specs/feature_spec.md\n",
        )

        errors = validate_preflight(
            tmp_path,
            ["agent/workflow_guard.py"],
            task_file=task_file,
        )
        assert errors == []

    def test_changed_task_file_is_inferred_when_unique(self, tmp_path: Path):
        """A single changed task file should be inferable as the governing task."""
        changed = ["tasks/active/feature.md", "agent/workflow_guard.py"]
        assert infer_task_from_changed_files(changed) == "tasks/active/feature.md"

    def test_multiple_changed_task_files_do_not_infer(self):
        """Multiple changed task files should force explicit task selection."""
        changed = ["tasks/active/a.md", "tasks/active/b.md", "agent/workflow_guard.py"]
        assert infer_task_from_changed_files(changed) is None

    def test_path_outside_repo_fails(self, tmp_path: Path):
        """Absolute paths outside the repo should be rejected."""
        outside_path = Path("/tmp/outside_file.py")
        errors = validate_preflight(tmp_path, [outside_path])
        assert len(errors) == 1
        assert "outside repo root" in errors[0]

    def test_relative_path_traversal_fails(self, tmp_path: Path):
        """Relative traversal outside the repo should be rejected."""
        errors = validate_preflight(tmp_path, ["../outside_file.py"])
        assert len(errors) == 1
        assert "outside repo root" in errors[0]


# ---------------------------------------------------------------------------
# Pre-completion gate (scripts/quality_gate.py)
#
# The gate is the back half of the same workflow this file's preflight tests
# cover, so its selection rules live here rather than in a file of their own.
#
# Defect it pins: check_tests() used to shell out to a bare `pytest tests/`,
# while CI runs `-m "not live and not slow"`. The gate therefore executed the
# live-network tests CI deselects, and `make quality-gate` went red whenever an
# upstream API had no data for the current date (NYISO: "No demand data
# available for <today>") — a failure that says nothing about the change being
# gated. These tests fail against that old behaviour.
# ---------------------------------------------------------------------------

import importlib.util
import re as _re

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_quality_gate():
    spec = importlib.util.spec_from_file_location("quality_gate_under_test", _REPO_ROOT / "scripts" / "quality_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ci_marker_expression() -> str:
    """The -m expression the CI workflow actually runs pytest with."""
    ci = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    matches = _re.findall(r'pytest\s+tests/[^\n]*?-m\s+"([^"]+)"', ci)
    assert matches, 'no `pytest tests/ ... -m "..."` step found in ci.yml'
    assert len(set(matches)) == 1, f"ci.yml runs pytest with conflicting markers: {set(matches)}"
    return matches[0]


class TestQualityGateTestSelection:
    @pytest.fixture
    def gate(self):
        return _load_quality_gate()

    @pytest.fixture
    def captured_cmd(self, gate, monkeypatch):
        """Capture the argv check_tests() would have executed."""
        seen: list[list[str]] = []

        def fake_run(cmd, timeout=300):
            seen.append(list(cmd))
            return 0, "1 passed"

        monkeypatch.setattr(gate, "_run", fake_run)
        return seen

    @staticmethod
    def _marker_arg(cmd: list[str]) -> str | None:
        """The pytest -m value, ignoring the `python -m pytest` prefix's own -m."""
        after = cmd[cmd.index("pytest") + 1 :]
        if "-m" not in after:
            return None
        return after[after.index("-m") + 1]

    def test_default_run_deselects_live_and_slow(self, gate, captured_cmd):
        gate.check_tests()
        assert len(captured_cmd) == 1
        cmd = captured_cmd[0]
        assert self._marker_arg(cmd) == "not live and not slow", f"gate ran: {cmd}"

    def test_gate_marker_matches_ci_exactly(self, gate):
        """Drift guard: the gate and CI must select the same tests."""
        assert _ci_marker_expression() == gate.CI_MARKER_EXPR

    def test_include_live_opts_back_in(self, gate, captured_cmd):
        gate.check_tests(include_live=True)
        assert self._marker_arg(captured_cmd[0]) is None

    def test_summary_records_the_selection(self, gate, captured_cmd):
        passed, summary = gate.check_tests()
        assert passed
        # The gate must not report a filtered run as if it were the full suite.
        assert "not live and not slow" in summary

    def test_failure_is_still_reported(self, gate, monkeypatch):
        """Filtering must not turn a real pytest failure into a pass."""
        monkeypatch.setattr(gate, "_run", lambda cmd, timeout=300: (1, "1 failed, 2 passed"))
        passed, summary = gate.check_tests()
        assert passed is False
        assert "1 failed" in summary
