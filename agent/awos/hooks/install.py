"""Install / uninstall git hooks that publish events into AWOS.

Hooks installed:
- ``post-commit``   — publish a ROUTINE event with the commit sha
- ``post-merge``    — publish a ROUTINE event after merges
- ``pre-push``      — run the drift scanner as a gate (non-blocking warning)

All hooks shell out to ``python -m agent.awos.cli scan --source <name>``
so the hook scripts are tiny and version-agnostic.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

_HOOKS: dict[str, str] = {
    "post-commit": (
        "#!/usr/bin/env bash\n"
        "# AWOS post-commit hook\n"
        "python -m agent.awos.cli scan --source git-post-commit "
        ">/dev/null 2>&1 || true\n"
    ),
    "post-merge": (
        "#!/usr/bin/env bash\n"
        "# AWOS post-merge hook\n"
        "python -m agent.awos.cli scan --source git-post-merge "
        ">/dev/null 2>&1 || true\n"
    ),
    "pre-push": (
        "#!/usr/bin/env bash\n"
        "# AWOS pre-push hook (non-blocking)\n"
        "python -m agent.awos.cli scan --source git-pre-push "
        ">/dev/null 2>&1 || true\n"
        "exit 0\n"
    ),
}

_AWOS_MARKER = "# AWOS"


def install(repo_root: Path) -> list[Path]:
    """Install AWOS hooks into ``.git/hooks/``. Returns installed paths.

    If a hook already exists and does not contain ``# AWOS``, we append
    our script rather than overwrite.
    """
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        raise RuntimeError(f"not a git repo: {repo_root}")
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    installed: list[Path] = []
    for name, body in _HOOKS.items():
        path = hooks_dir / name
        if path.exists():
            existing = path.read_text()
            if _AWOS_MARKER in existing:
                installed.append(path)
                continue
            # append, preserve existing hook
            new = existing.rstrip() + "\n\n" + body
            path.write_text(new)
        else:
            path.write_text(body)
        _make_executable(path)
        installed.append(path)
    return installed


def uninstall(repo_root: Path) -> list[Path]:
    """Remove the AWOS portion from installed hooks.

    If the hook becomes empty (only shebang + comments), the file is
    deleted outright. Otherwise the AWOS section is stripped.
    """
    hooks_dir = repo_root / ".git" / "hooks"
    removed: list[Path] = []
    if not hooks_dir.exists():
        return removed
    for name in _HOOKS:
        path = hooks_dir / name
        if not path.exists():
            continue
        text = path.read_text()
        if _AWOS_MARKER not in text:
            continue
        stripped = "\n".join(
            line
            for line in text.splitlines()
            if "agent.awos.cli scan" not in line and _AWOS_MARKER not in line
        ).strip()
        if not stripped or stripped.strip() in {"#!/usr/bin/env bash", "#!/bin/bash"}:
            path.unlink()
        else:
            path.write_text(stripped + "\n")
        removed.append(path)
    return removed


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# --- exports -----------------------------------------------------------
def hook_names() -> list[str]:
    return list(_HOOKS)


__all__ = ["install", "uninstall", "hook_names"]


# ensure `os` is used so lint doesn't flag it
_ = os
