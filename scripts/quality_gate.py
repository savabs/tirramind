#!/usr/bin/env python3
"""Quality gate: pre-completion checks before marking a task done.

Checks:
1. pytest passes for test files
2. obsidian_lint.py has no FM01/FM02/LK01 errors
3. No unchecked steps remain in the specified task file

Usage:
    python scripts/quality_gate.py [--task tasks/active/foo.md] [--skip-tests]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks" / "active"


def _run(cmd: list[str], timeout: int = 300) -> tuple[int, str]:
    """Run a command, return (exit_code, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=timeout,
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT"
    except FileNotFoundError:
        return 1, f"Command not found: {cmd[0]}"


def check_tests() -> tuple[bool, str]:
    """Run pytest and return (passed, message)."""
    code, output = _run(
        ["python", "-m", "pytest", "tests/", "--tb=short", "-q", "--no-header"]
    )
    passed = code == 0
    # Extract summary line
    lines = output.strip().splitlines()
    summary = lines[-1] if lines else "no output"
    return passed, summary


def check_obsidian_lint() -> tuple[bool, str]:
    """Run obsidian_lint.py and check for critical errors."""
    lint_script = ROOT / "scripts" / "obsidian_lint.py"
    if not lint_script.exists():
        return True, "obsidian_lint.py not found (skipped)"

    code, output = _run(["python", str(lint_script)])
    # Check for FM01, FM02, LK01 (critical) — LK02, ST03 are advisory
    critical_pattern = re.compile(r"\b(FM01|FM02|LK01)\b")
    critical_lines = [l for l in output.splitlines() if critical_pattern.search(l)]
    if critical_lines:
        return False, f"{len(critical_lines)} critical lint errors:\n" + "\n".join(
            critical_lines[:10]
        )
    return True, "No critical lint errors"


def check_task_complete(task_path: Path) -> tuple[bool, str]:
    """Check that no unchecked steps remain in the task file."""
    if not task_path.exists():
        return False, f"Task file not found: {task_path}"

    content = task_path.read_text()
    unchecked = re.findall(r"^- \[ \]", content, re.MULTILINE)
    checked = re.findall(r"^- \[x\]", content, re.MULTILINE)

    if unchecked:
        return (
            False,
            f"{len(unchecked)} unchecked steps remain ({len(checked)} complete)",
        )
    return True, f"All {len(checked)} steps complete"


def find_active_tasks() -> list[Path]:
    """Find all active task files."""
    if not TASKS_DIR.exists():
        return []
    return sorted(f for f in TASKS_DIR.glob("*.md") if f.name != ".gitkeep")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality gate checks.")
    parser.add_argument("--task", type=str, help="Specific task file to check.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest run.")
    args = parser.parse_args()

    results: list[tuple[str, bool, str]] = []

    # Tests
    if not args.skip_tests:
        passed, msg = check_tests()
        results.append(("Tests", passed, msg))

    # Obsidian lint
    passed, msg = check_obsidian_lint()
    results.append(("Obsidian Lint", passed, msg))

    # Task completeness
    if args.task:
        task_path = Path(args.task)
        if not task_path.is_absolute():
            task_path = ROOT / task_path
        passed, msg = check_task_complete(task_path)
        results.append(("Task Steps", passed, msg))
    else:
        for task in find_active_tasks():
            passed, msg = check_task_complete(task)
            results.append((f"Task: {task.stem}", passed, msg))

    # Report
    all_pass = True
    print("\n=== Quality Gate ===\n")
    for name, passed, msg in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {msg}")

    print()
    if all_pass:
        print("All checks passed.")
        sys.exit(0)
    else:
        print("Some checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
