"""
Tool: Code Executor

Execute Python code in a sandboxed subprocess.
Captures stdout, stderr, and return value.

Security: runs with filtered environment (no API keys leaked),
temp directory isolation, and configurable timeout.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_MAX_OUTPUT_LEN = 8_000

# Env var name patterns that must never be exposed to subprocesses.
_SECRET_PATTERNS = ("API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")

# Env vars safe to pass through.
_SAFE_VARS = {"PATH", "HOME", "LANG", "TERM", "LC_ALL", "TMPDIR", "USER"}


def _safe_env() -> dict[str, str]:
    """Build a minimal environment for subprocess execution.

    Includes only safe system vars + TIRRA_-prefixed config vars,
    but excludes any var whose name contains secret-like patterns.
    """
    env: dict[str, str] = {}

    # Safe system vars
    for key in _SAFE_VARS:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    # TIRRA_-prefixed vars (config), excluding secrets
    for key, val in os.environ.items():
        if key.startswith("TIRRA_") and not any(p in key.upper() for p in _SECRET_PATTERNS):
            env[key] = val

    return env


class CodeExecutorTool(Tool):
    name = "execute_python"
    description = (
        "Execute a Python code snippet and return stdout/stderr. "
        "Use this for data analysis, calculations, file processing, or any programmatic task. "
        "The code runs in an isolated subprocess with a timeout."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute.",
            },
        },
        "required": ["code"],
    }

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def execute(self, *, code: str, **_: Any) -> ToolResult:
        # Write code to a temp file and run in subprocess for isolation
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="tirra_exec_", delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=tempfile.gettempdir(),
                env=_safe_env(),
            )
            stdout = result.stdout[:_MAX_OUTPUT_LEN] if result.stdout else ""
            stderr = result.stderr[:_MAX_OUTPUT_LEN] if result.stderr else ""

            output_parts = []
            if stdout:
                output_parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                output_parts.append(f"STDERR:\n{stderr}")
            output_parts.append(f"Exit code: {result.returncode}")

            return ToolResult(
                success=result.returncode == 0,
                output="\n\n".join(output_parts),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=f"Execution timed out after {self._timeout}s")
        except Exception as exc:
            log.exception("Code execution failed")
            return ToolResult(success=False, output=f"Execution error: {exc}")
        finally:
            Path(script_path).unlink(missing_ok=True)
