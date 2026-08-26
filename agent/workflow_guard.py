"""Workflow preflight guard for repository-level process enforcement."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

WORKFLOW_PREFIXES = (
    "docs/research/",
    "docs/specs/",
    "tasks/active/",
)
CHECKPOINT_PREFIX = "docs/memory/chat_checkpoint_"


@dataclass(frozen=True)
class TaskMetadata:
    """Minimal metadata required to validate workflow preflight."""

    task_path: Path
    research_path: Path
    spec_path: Path


def _normalize_repo_path(path: str | Path, repo_root: Path) -> str:
    raw_path = Path(path)
    candidate_path = raw_path.resolve() if raw_path.is_absolute() else (repo_root / raw_path).resolve()
    try:
        relative_path = candidate_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path is outside repo root: {raw_path}") from exc
    return relative_path.as_posix().lstrip("./")


def is_workflow_file(path: str) -> bool:
    """Return True when the path is allowed before implementation preflight."""
    normalized = path.lstrip("./")
    if normalized.startswith(CHECKPOINT_PREFIX) and normalized.endswith(".md"):
        return True
    return any(normalized.startswith(prefix) for prefix in WORKFLOW_PREFIXES)


def collect_staged_files(repo_root: Path) -> list[str]:
    """Return staged file paths relative to the repository root."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git is not installed or not available on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or "unknown git error"
        raise RuntimeError(f"Unable to inspect staged files: {stderr}") from exc

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def parse_task_metadata(repo_root: Path, task_file: str | Path) -> TaskMetadata:
    """Parse task metadata from a task tracking file."""
    task_relative = _normalize_repo_path(task_file, repo_root)
    if not task_relative.startswith("tasks/active/"):
        raise ValueError("Task file must live under tasks/active/ for workflow enforcement")

    task_path = repo_root / task_relative
    if not task_path.exists():
        raise ValueError(f"Task file does not exist: {task_relative}")

    research_ref: str | None = None
    spec_ref: str | None = None

    for line in task_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Research:"):
            research_ref = line.split(":", 1)[1].strip().strip("`")
        elif line.startswith("Spec:"):
            spec_ref = line.split(":", 1)[1].strip().strip("`")

    if not research_ref:
        raise ValueError(f"Task file is missing a Research: line: {task_relative}")
    if not spec_ref:
        raise ValueError(f"Task file is missing a Spec: line: {task_relative}")

    research_path = repo_root / _normalize_repo_path(research_ref, repo_root)
    spec_path = repo_root / _normalize_repo_path(spec_ref, repo_root)

    if not research_path.exists():
        raise ValueError(f"Research file does not exist: {research_ref}")
    if not spec_path.exists():
        raise ValueError(f"Spec file does not exist: {spec_ref}")

    return TaskMetadata(
        task_path=task_path,
        research_path=research_path,
        spec_path=spec_path,
    )


def infer_task_from_changed_files(changed_files: list[str]) -> str | None:
    """Infer the task when exactly one task file is part of the change set."""
    task_candidates = sorted(
        {path for path in changed_files if path.startswith("tasks/active/") and path.endswith(".md")}
    )
    if len(task_candidates) == 1:
        return task_candidates[0]
    return None


def validate_preflight(
    repo_root: Path,
    changed_files: list[str | Path],
    task_file: str | Path | None = None,
) -> list[str]:
    """Validate workflow preflight for the given change set."""
    errors: list[str] = []

    normalized_files: list[str] = []
    for changed_file in changed_files:
        try:
            normalized_files.append(_normalize_repo_path(changed_file, repo_root))
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        return errors

    if not normalized_files:
        return ["No files were provided for workflow validation."]

    if all(is_workflow_file(path) for path in normalized_files):
        return []

    selected_task = task_file or os.environ.get("TIRRA_WORKFLOW_TASK")
    if not selected_task:
        selected_task = infer_task_from_changed_files(normalized_files)

    if not selected_task:
        return [
            "Non-workflow changes require a governing task file. "
            "Pass --task tasks/active/<task>.md or set TIRRA_WORKFLOW_TASK."
        ]

    try:
        parse_task_metadata(repo_root, selected_task)
    except ValueError as exc:
        return [str(exc)]

    return []


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the workflow guard."""
    parser = argparse.ArgumentParser(description="Validate TirraMind workflow preflight before implementation commits")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Explicit paths to validate relative to the repo root",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Validate currently staged git paths",
    )
    parser.add_argument(
        "--task",
        help="Task file governing the current implementation change",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing docs/, tasks/, and agent/",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    candidate_files = list(args.paths)

    if args.staged:
        try:
            candidate_files.extend(collect_staged_files(repo_root))
        except RuntimeError as exc:
            print(f"workflow guard: {exc}", file=sys.stderr)
            return 2

    errors = validate_preflight(repo_root, candidate_files, task_file=args.task)
    if errors:
        for error in errors:
            print(f"workflow guard: {error}", file=sys.stderr)
        return 1

    print("workflow guard: preflight satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
