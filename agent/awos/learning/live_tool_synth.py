"""LiveToolSynthesizer — on-the-fly helper synthesis during recurring failures.

Adapted from AWOS live_tool_synth.py into the learning subpackage.

When a signal operation (a fetch, a parse, a scorer) fails on the same pattern
twice, a reflection prompt fires: "Would writing a small helper script prevent
this class of error?" If yes, the runtime synthesises a script, validates it
(AST parse + sandbox run), saves it, and registers it in the index.

Future operations with similar descriptions get the helper's output reused.

Reference: Live-SWE-agent (arXiv:2511.13646)
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_TOOLS_INDEX_FILE = "index.json"
_TOOL_TIMEOUT_SECS = 8
_MAX_OUTPUT_CHARS = 1000
_MIN_KEYWORD_OVERLAP = 0.4


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _keyword_overlap(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


@dataclass
class SynthesizedTool:
    """A synthesized helper script persisted to state_dir/tools/."""

    name: str
    description: str
    script_path: str
    trigger_pattern: str
    created_at: str = field(default_factory=_now_iso)
    use_count: int = 0


class LiveToolSynthesizer:
    """Synthesizes and registers helper scripts on the fly during failures."""

    def __init__(
        self,
        tools_dir: str = ".awos/tools",
        cheap_call: Callable[[str], str] | None = None,
    ) -> None:
        self._tools_dir = Path(tools_dir)
        self._index_path = self._tools_dir / _TOOLS_INDEX_FILE
        self._cheap_call = cheap_call

    # ── Public API ──────────────────────────────────────────────────────────
    def reflect(
        self,
        operation: str,
        error_type: str,
        error: str,
        attempt: int,
    ) -> SynthesizedTool | None:
        """On attempt >= 2 failure: ask LLM if a helper would prevent this class."""
        if os.getenv("TIRRA_AWOS_LIVE_TOOLS", "").lower() != "true" or self._cheap_call is None:
            return None
        if attempt < 2:
            return None

        reflect_prompt = (
            f"A signal-intelligence runtime failed twice on this operation: {operation!r}\n"
            f"Error class: {error_type!r}; Error: {error[:300]!r}\n"
            f"Would writing a small Python helper script (called as a subprocess) "
            f"help prevent or diagnose this class of error?\n"
            f"Answer ONLY valid JSON: "
            f'{{ "should_create": true/false, "tool_purpose": "..." }}'
        )
        try:
            raw = self._cheap_call(reflect_prompt)
            decision = self._parse_reflect_decision(raw)
        except Exception as exc:
            logger.debug("[LiveToolSynth] reflect parse failed: %s", exc)
            return None

        if not decision.get("should_create"):
            return None
        tool_purpose = decision.get("tool_purpose", "")
        if not tool_purpose:
            return None
        return self._synthesize(operation, error, tool_purpose)

    def find_relevant_tool(self, operation: str) -> SynthesizedTool | None:
        best: SynthesizedTool | None = None
        best_score = _MIN_KEYWORD_OVERLAP
        for tool in self.list_tools():
            score = _keyword_overlap(operation, tool.trigger_pattern)
            if score > best_score:
                best_score = score
                best = tool
        return best

    def run_tool(self, tool: SynthesizedTool, operation: str) -> str:
        script = Path(tool.script_path)
        if not script.exists():
            logger.warning("[LiveToolSynth] tool script missing: %s", tool.script_path)
            return ""
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                input=json.dumps({"operation": operation}),
                capture_output=True,
                text=True,
                timeout=_TOOL_TIMEOUT_SECS,
            )
            output = (result.stdout or "").strip()
            if result.returncode != 0:
                logger.debug(
                    "[LiveToolSynth] tool exited %d: %s",
                    result.returncode,
                    result.stderr[:200],
                )
                return ""
            self._increment_use_count(tool)
            return output[:_MAX_OUTPUT_CHARS]
        except subprocess.TimeoutExpired:
            logger.warning("[LiveToolSynth] tool timed out: %s", tool.name)
            return ""
        except Exception as exc:
            logger.debug("[LiveToolSynth] tool run error: %s", exc)
            return ""

    def list_tools(self) -> list[SynthesizedTool]:
        if not self._index_path.exists():
            return []
        try:
            data = json.loads(self._index_path.read_text())
            return [SynthesizedTool(**entry) for entry in data]
        except Exception as exc:
            logger.debug("[LiveToolSynth] index read error: %s", exc)
            return []

    # ── Internal helpers ────────────────────────────────────────────────────
    def _synthesize(
        self, operation: str, error: str, tool_purpose: str
    ) -> SynthesizedTool | None:
        tool_name = self._make_tool_name(tool_purpose)
        synth_prompt = (
            f"Write a Python script named {tool_name}.py that {tool_purpose}.\n"
            f"Context: This helper serves a signal-intelligence runtime that failed on: {operation!r}\n"
            f"Error encountered: {error[:200]!r}\n\n"
            f"Requirements:\n"
            f"- The script reads a JSON object from stdin (use sys.stdin.read())\n"
            f"- It prints its analysis or result to stdout\n"
            f"- Include a clear one-line docstring at the top\n"
            f"- Be minimal: under 40 lines\n"
            f"- Handle errors gracefully (try/except, exit code 0 on partial failure)\n\n"
            f"Output ONLY the Python code, nothing else."
        )
        try:
            code = self._cheap_call(synth_prompt)
        except Exception as exc:
            logger.warning("[LiveToolSynth] synthesis call failed: %s", exc)
            return None

        code = self._strip_fences(code)
        if not self._validate_syntax(code):
            logger.info("[LiveToolSynth] synthesized tool failed syntax check")
            return None
        if not self._validate_runtime(code):
            logger.info("[LiveToolSynth] synthesized tool failed runtime check")
            return None
        return self._persist_tool(tool_name, code, tool_purpose, operation)

    def _make_tool_name(self, purpose: str) -> str:
        slug = "".join(c if c.isalnum() else "_" for c in purpose.lower()[:30]).strip("_")
        h = hashlib.md5(purpose.encode()).hexdigest()[:6]  # noqa: S324 — non-crypto name slug
        return f"helper_{slug}_{h}"

    @staticmethod
    def _strip_fences(code: str) -> str:
        code = code.strip()
        if code.startswith("```"):
            code = "\n".join(code.split("\n")[1:])
        if code.endswith("```"):
            code = "\n".join(code.split("\n")[:-1])
        return code.strip()

    @staticmethod
    def _validate_syntax(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _validate_runtime(code: str) -> bool:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(code)
                tmp_path = f.name
            result = subprocess.run(
                [sys.executable, tmp_path],
                input="{}",
                capture_output=True,
                text=True,
                timeout=_TOOL_TIMEOUT_SECS,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _persist_tool(
        self, name: str, code: str, purpose: str, trigger_pattern: str
    ) -> SynthesizedTool | None:
        try:
            self._tools_dir.mkdir(parents=True, exist_ok=True)
            script_path = self._tools_dir / f"{name}.py"
            script_path.write_text(code)
            tool = SynthesizedTool(
                name=name,
                description=purpose[:200],
                script_path=str(script_path),
                trigger_pattern=trigger_pattern,
                created_at=_now_iso(),
                use_count=0,
            )
            self._register_tool(tool)
            logger.info("[LiveToolSynth] persisted new helper: %s", name)
            return tool
        except Exception as exc:
            logger.warning("[LiveToolSynth] persist failed: %s", exc)
            return None

    def _register_tool(self, tool: SynthesizedTool) -> None:
        existing = self.list_tools()
        if tool.name not in {t.name for t in existing}:
            existing.append(tool)
        self._tools_dir.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps([asdict(t) for t in existing], indent=2))

    def _increment_use_count(self, tool: SynthesizedTool) -> None:
        tools = self.list_tools()
        for t in tools:
            if t.name == tool.name:
                t.use_count += 1
                break
        self._index_path.write_text(json.dumps([asdict(t) for t in tools], indent=2))

    @staticmethod
    def _parse_reflect_decision(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        return json.loads(raw.strip())


__all__ = ["SynthesizedTool", "LiveToolSynthesizer"]
