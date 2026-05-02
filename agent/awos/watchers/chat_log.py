"""Chat log watcher.

Scans VS Code / Copilot Chat workspace debug logs for new text chunks
and classifies each chunk via the composite classifier. On actionable
classifications, publishes events to the bus.

Log location defaults to the Copilot Chat debug log folder under
``~/.config/Code/User/workspaceStorage/*/GitHub.copilot-chat/debug-logs/``.

This watcher is best-effort: if no log dir is configured and no logs
are found, it returns an empty list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agent.awos.classifiers.base import Classifier
from agent.awos.events.schema import Event, EventStatus
from agent.awos.watchers.base import Watcher

log = logging.getLogger(__name__)


class ChatLogWatcher(Watcher):
    name = "chat_log"

    def __init__(
        self,
        bus,
        repo_root: Path,
        *,
        classifier: Classifier,
        state_file: Path,
        log_dir: Path | None = None,
        max_chunk_bytes: int = 16_000,
        min_chunk_bytes: int = 80,
    ) -> None:
        super().__init__(bus, repo_root)
        self.classifier = classifier
        self.state_file = Path(state_file)
        self.log_dir = log_dir
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.min_chunk_bytes = int(min_chunk_bytes)

    # ------------------------------------------------------------------
    def scan(self) -> list[Event]:
        log_dir = self.log_dir or _default_log_dir()
        if log_dir is None or not log_dir.exists():
            return []

        state = _load_state(self.state_file)
        events: list[Event] = []

        for log_file in sorted(log_dir.glob("**/*.log")):
            try:
                size = log_file.stat().st_size
            except OSError:
                continue
            key = str(log_file)
            offset = int(state.get(key, 0))
            if offset >= size:
                continue
            chunk = self._read_chunk(log_file, offset, size)
            if not chunk or len(chunk) < self.min_chunk_bytes:
                state[key] = size
                continue

            try:
                result = self.classifier.classify(chunk)
            except Exception as e:
                log.exception("classifier error: %s", e)
                state[key] = size
                continue

            state[key] = size
            if not result.is_actionable:
                continue

            events.append(
                Event(
                    source=self.name,
                    category=result.category,
                    confidence=result.confidence,
                    status=EventStatus.NEW,
                    rationale=result.rationale,
                    payload={
                        "chunk_prefix": chunk[:1500],
                        "log_file": str(log_file),
                        "classifier": result.classifier,
                        "suggested_section": result.suggested_section,
                        "extracted_principle": result.extracted_principle,
                    },
                )
            )

        _save_state(self.state_file, state)
        return events

    # ------------------------------------------------------------------
    def _read_chunk(self, path: Path, start: int, end: int) -> str:
        start = max(0, start)
        end = max(start, end)
        size = end - start
        if size > self.max_chunk_bytes:
            start = end - self.max_chunk_bytes
            size = self.max_chunk_bytes
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                raw = fh.read(size)
            return raw.decode("utf-8", errors="replace")
        except OSError:
            return ""


# ======================================================================
def _default_log_dir() -> Path | None:
    base = Path.home() / ".config" / "Code" / "User" / "workspaceStorage"
    if not base.exists():
        return None
    # return the parent folder; caller will glob recursively
    return base


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return raw.get("chat_log_offsets", {}) if isinstance(raw, dict) else {}


def _save_state(path: Path, offsets: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text()) or {}
        except (json.JSONDecodeError, OSError):
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing["chat_log_offsets"] = offsets
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2))
    tmp.replace(path)


__all__ = ["ChatLogWatcher"]
