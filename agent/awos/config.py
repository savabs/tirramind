"""AWOS configuration.

Layered config: defaults → environment variables (``TIRRA_AWOS_*``)
→ optional ``.awos/config.yaml`` override.

The AWOS file path defaults to the Copilot memory-tool location so that
updates are immediately visible in the editor's memory UI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

ClassifierMode = Literal["heuristic", "llm", "hybrid", "off"]

DEFAULT_AWOS_FILE = Path.home() / (
    ".config/Code/User/globalStorage/github.copilot-chat/memory-tool/memories/agent_workflow_os.md"
)


class AWOSConfig(BaseModel):
    """Runtime configuration for the AWOS system."""

    # --- filesystem layout -------------------------------------------------
    repo_root: Path = Field(default_factory=Path.cwd)
    state_dir: Path = Field(default_factory=lambda: Path.cwd() / ".awos")
    db_path: Path | None = None  # defaults to state_dir / "events.db"
    proposals_dir: Path | None = None  # defaults to state_dir / "proposals"
    state_file: Path | None = None  # defaults to state_dir / "state.json"
    awos_file: Path = DEFAULT_AWOS_FILE
    policies_file: Path | None = None  # user override; merged with default

    # --- classifiers -------------------------------------------------------
    classifier_mode: ClassifierMode = "hybrid"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5-20251001"
    anthropic_max_tokens: int = 512
    anthropic_timeout_s: float = 15.0
    llm_confidence_floor: float = 0.5
    heuristic_confidence_ceiling: float = 0.7

    # --- watchers ----------------------------------------------------------
    watcher_interval_s: int = 300
    drift_watcher_enabled: bool = True
    staleness_watcher_enabled: bool = True
    obsidian_watcher_enabled: bool = True
    chat_log_watcher_enabled: bool = True
    chat_log_dir: Path | None = None  # if None, auto-detect VS Code path
    stale_task_days: int = 7
    watcher_timeout_s: float = 30.0

    # --- bus / actions -----------------------------------------------------
    dedup_window_s: int = 600
    max_payload_bytes: int = 1_000_000
    awos_auto_update_confidence: float = 0.7  # below → proposal, above → direct

    # ----------------------------------------------------------------------
    @field_validator(
        "repo_root",
        "state_dir",
        "db_path",
        "proposals_dir",
        "state_file",
        "awos_file",
        "policies_file",
        "chat_log_dir",
        mode="before",
    )
    @classmethod
    def _expand_path(cls, v: Any) -> Any:
        if v is None or isinstance(v, Path):
            return v
        return Path(str(v)).expanduser()

    def model_post_init(self, __context: Any) -> None:
        # fill derived paths so consumers don't need to null-check
        if self.db_path is None:
            object.__setattr__(self, "db_path", self.state_dir / "events.db")
        if self.proposals_dir is None:
            object.__setattr__(self, "proposals_dir", self.state_dir / "proposals")
        if self.state_file is None:
            object.__setattr__(self, "state_file", self.state_dir / "state.json")

    # ----------------------------------------------------------------------
    def ensure_dirs(self) -> None:
        """Create state/proposals directories if they don't exist."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        assert self.proposals_dir is not None  # narrowed by post_init
        self.proposals_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        env: dict[str, str] | None = None,
        yaml_path: Path | None = None,
    ) -> AWOSConfig:
        """Build a config from environment + optional YAML overrides.

        Precedence (low → high): defaults → YAML → env.
        Env var prefix is ``TIRRA_AWOS_`` (upper-snake → snake_case).
        """
        data: dict[str, Any] = {}
        if yaml_path is not None and yaml_path.exists():
            loaded = yaml.safe_load(yaml_path.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"AWOS config YAML at {yaml_path} must be a mapping")
            data.update(loaded)

        e = env if env is not None else dict(os.environ)
        prefix = "TIRRA_AWOS_"
        for key, value in e.items():
            if not key.startswith(prefix):
                continue
            field = key[len(prefix) :].lower()
            data[field] = _coerce_scalar(value)

        # Fall back to shared LLM key if no AWOS-specific key was set
        if not data.get("anthropic_api_key") and e.get("TIRRA_LLM_API_KEY"):
            data["anthropic_api_key"] = e["TIRRA_LLM_API_KEY"]

        return cls(**data)


def _coerce_scalar(value: str) -> Any:
    """Coerce env var strings into the natural Python scalar type."""
    low = value.lower().strip()
    if low in {"true", "yes", "1"}:
        return True
    if low in {"false", "no", "0"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
