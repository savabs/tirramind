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
