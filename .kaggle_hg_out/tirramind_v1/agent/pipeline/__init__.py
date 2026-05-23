"""TirraMind — Pipeline Layer (Deterministic DAG Scheduler)."""

from agent.pipeline.storage_backend import (
    PostgresBackend,
    SQLiteBackend,
    StorageBackend,
)
from agent.pipeline.store import PipelineStore

__all__ = ["PipelineStore", "PipelineScheduler", "StorageBackend", "SQLiteBackend", "PostgresBackend"]


def __getattr__(name: str):
    if name == "PipelineScheduler":
        from agent.pipeline.scheduler import PipelineScheduler

        return PipelineScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
