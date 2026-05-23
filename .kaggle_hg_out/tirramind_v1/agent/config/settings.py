"""
TirraMind Agent — Configuration

All config via environment variables. No secrets in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMConfig:
    """LLM backend configuration. Supports OpenAI-compatible APIs (including Ollama)."""

    provider: str = "openai"  # "openai" | "ollama"
    model: str = "gpt-4o"  # model name
    base_url: str | None = None  # override endpoint (e.g. http://localhost:11434/v1 for Ollama)
    api_key: str = ""  # set via TIRRA_LLM_API_KEY
    temperature: float = 0.2
    max_tokens: int = 4096

    @classmethod
    def from_env(cls) -> LLMConfig:
        provider = os.getenv("TIRRA_LLM_PROVIDER", "openai")
        if provider == "ollama":
            return cls(
                provider="ollama",
                model=os.getenv("TIRRA_LLM_MODEL", "llama3"),
                base_url=os.getenv("TIRRA_LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama",
                temperature=float(os.getenv("TIRRA_LLM_TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv("TIRRA_LLM_MAX_TOKENS", "4096")),
            )
        return cls(
            provider=provider,
            model=os.getenv("TIRRA_LLM_MODEL", "gpt-4o"),
            base_url=os.getenv("TIRRA_LLM_BASE_URL") or None,
            api_key=os.getenv("TIRRA_LLM_API_KEY", ""),
            temperature=float(os.getenv("TIRRA_LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("TIRRA_LLM_MAX_TOKENS", "4096")),
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline Layer configuration."""

    db_path: str = ".tirra_pipeline/pipeline.db"
    max_workers: int = 4
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> PipelineConfig:
        return cls(
            db_path=os.getenv("TIRRA_PIPELINE_DB", ".tirra_pipeline/pipeline.db"),
            max_workers=int(os.getenv("TIRRA_PIPELINE_WORKERS", "4")),
            log_level=os.getenv("TIRRA_PIPELINE_LOG_LEVEL", "INFO"),
        )


@dataclass(frozen=True)
class AgentConfig:
    """Top-level agent configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig.from_env)
    fred_api_key: str = ""  # FRED (Federal Reserve Economic Data) API key
    max_steps: int = 30  # hard limit on agent loop iterations
    max_plan_depth: int = 3  # hierarchical planning depth
    memory_dir: str = ".tirra_memory"  # local memory persistence path
    tool_timeout: int = 30  # subprocess timeout for code/shell tools (seconds)
    lesson_min_support: int = 3  # min episodes before lesson promotion
    lesson_min_runs: int = 2  # min distinct runs before lesson promotion
    episode_ttl_days: int = 30  # episodic memory decay window
    verbose: bool = False

    @classmethod
    def from_env(cls) -> AgentConfig:
        return cls(
            llm=LLMConfig.from_env(),
            pipeline=PipelineConfig.from_env(),
            fred_api_key=os.getenv("TIRRA_FRED_API_KEY", ""),
            max_steps=int(os.getenv("TIRRA_MAX_STEPS", "30")),
            max_plan_depth=int(os.getenv("TIRRA_MAX_PLAN_DEPTH", "3")),
            memory_dir=os.getenv("TIRRA_MEMORY_DIR", ".tirra_memory"),
            tool_timeout=int(os.getenv("TIRRA_TOOL_TIMEOUT", "30")),
            lesson_min_support=int(os.getenv("TIRRA_LESSON_MIN_SUPPORT", "3")),
            lesson_min_runs=int(os.getenv("TIRRA_LESSON_MIN_RUNS", "2")),
            episode_ttl_days=int(os.getenv("TIRRA_EPISODE_TTL_DAYS", "30")),
            verbose=os.getenv("TIRRA_VERBOSE", "").lower() in ("1", "true", "yes"),
        )

    def validate(self) -> list[str]:
        """Check config for common problems. Returns list of error messages."""
        errors: list[str] = []
        if self.llm.provider == "openai" and not self.llm.api_key:
            errors.append("TIRRA_LLM_API_KEY is not set. Set it in .env or as an environment variable.")
        return errors
