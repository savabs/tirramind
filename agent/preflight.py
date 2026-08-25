"""
TirraMind — Feature Preflight System

Answers the question: "Is this feature actually ready to run?"
before any real work starts.

Three failure categories are distinguished:

  MISSING_CONFIG   An API key, env var, or required config field is absent.
                   Fix: set the env var / config field.

  NO_DATA          The required data hasn't been ingested yet (or is too
                   stale).  Feature would return empty/useless results.
                   Fix: run the relevant data tool first.

  MODEL_NOT_READY  A trained GNN model is needed but hasn't been built or
                   trained yet.
                   Fix: call trainer.build_model() + run training epochs.

  OK               All prerequisites satisfied.

Usage
-----
    from agent.preflight import FeaturePreflight, FailureReason

    ok, result = FeaturePreflight.for_nightlight(firms_key, store)
    if not ok:
        log.warning("Nightlight not ready: %s — %s", result.reason, result.detail)
        return ToolResult(success=False, output=result.user_message)

    ok, result = FeaturePreflight.for_attribution(model, store, id_map)
    if not ok:
        return {}

Each check is a pure function: no side-effects, no network calls.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)

_HOURS = 3600.0


# ═══════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════


class FailureReason(str, Enum):
    OK             = "OK"
    MISSING_CONFIG = "MISSING_CONFIG"
    NO_DATA        = "NO_DATA"
    MODEL_NOT_READY = "MODEL_NOT_READY"


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of a preflight check.

    Attributes
    ----------
    ok     : True if the feature may proceed.
    reason : Failure category (OK when ok=True).
    detail : One-line technical explanation.
    fix    : Actionable fix hint shown to operators.
    """

    ok: bool
    reason: FailureReason
    detail: str
    fix: str = ""

    @property
    def user_message(self) -> str:
        if self.ok:
            return "Preflight passed."
        return f"[{self.reason.value}] {self.detail}. Fix: {self.fix}"

    # Convenience factory
    @staticmethod
    def passed() -> "PreflightResult":
        return PreflightResult(ok=True, reason=FailureReason.OK, detail="")

    @staticmethod
    def missing_config(detail: str, fix: str = "") -> "PreflightResult":
        return PreflightResult(
            ok=False, reason=FailureReason.MISSING_CONFIG,
            detail=detail, fix=fix,
        )

    @staticmethod
    def no_data(detail: str, fix: str = "") -> "PreflightResult":
        return PreflightResult(
            ok=False, reason=FailureReason.NO_DATA,
            detail=detail, fix=fix,
        )

    @staticmethod
    def model_not_ready(detail: str, fix: str = "") -> "PreflightResult":
        return PreflightResult(
            ok=False, reason=FailureReason.MODEL_NOT_READY,
            detail=detail, fix=fix,
        )


# ═══════════════════════════════════════════════════════════════
# FeaturePreflight — one static method per feature / idea
# ═══════════════════════════════════════════════════════════════


class FeaturePreflight:
    """Collection of preflight checks, one per major feature.

    Every method returns ``(bool, PreflightResult)``.  The bool is True
    when the feature is ready.  No method raises — failures are returned
    as PreflightResult.
    """

    # ── Idea 14: Nightlight / NDVI ─────────────────────────────────────────

    @staticmethod
    def for_nightlight(
        firms_api_key: str | None,
        store: Any | None = None,
        mode: str = "both",
        max_stale_hours: float = 336.0,
    ) -> tuple[bool, PreflightResult]:
        """Check NASA FIRMS key is set (required for nightlight mode).

        Args:
            firms_api_key   : The key string (may be empty/None).
            store           : Optional PipelineStore — checks for stale data.
            mode            : "nightlight" | "ndvi" | "both".
            max_stale_hours : Warn if last observation older than this.
        """
        if mode in ("nightlight", "both"):
            key = firms_api_key or os.getenv("FIRMS_API_KEY", "")
            if not key:
                return False, PreflightResult.missing_config(
                    detail="FIRMS_API_KEY is not set — nightlight mode requires a NASA FIRMS key",
                    fix="Export FIRMS_API_KEY=<your-key> "
                        "(free at https://firms.modaps.eosdis.nasa.gov/api/map_key/)",
                )

        if store is not None:
            result = FeaturePreflight._check_data_staleness(
                store, "nightlight_activity", max_stale_hours,
                fix="Run NightlightActivityTool.execute(mode='both') to ingest data.",
            )
            if not result.ok:
                return False, result

        return True, PreflightResult.passed()

    # ── Idea 12: Barra Attribution ────────────────────────────────────────

    @staticmethod
    def for_attribution(
        model: Any | None,
        store: Any | None = None,
        id_map: Any | None = None,
        min_instrument_nodes: int = 1,
    ) -> tuple[bool, PreflightResult]:
        """Check model is trained and has instrument nodes in graph.

        Args:
            model               : HetTGN instance (or None if not built).
            store               : PipelineStore — checks entity count.
            id_map              : IDMap — checks instrument nodes present.
            min_instrument_nodes: Minimum instruments required.
        """
        if model is None:
            return False, PreflightResult.model_not_ready(
                detail="GNN model has not been built",
                fix="Call trainer.build_model() before compute_attribution().",
            )

        if id_map is not None:
            n_instr = len(id_map.type_local.get("instrument", {}))
            if n_instr < min_instrument_nodes:
                return False, PreflightResult.no_data(
                    detail=f"Graph has {n_instr} instrument nodes "
                           f"(need ≥ {min_instrument_nodes})",
                    fix="Ingest instrument data via instrument_universe tool "
                        "or run SyntheticGraphGenerator.",
                )

        return True, PreflightResult.passed()

    # ── Idea 11: Portfolio Construction ───────────────────────────────────

    @staticmethod
    def for_portfolio(
        store: Any | None,
        return_preds: dict | None = None,
        min_assets: int = 2,
        min_price_rows: int = 10,
    ) -> tuple[bool, PreflightResult]:
        """Check return predictions exist and price history is sufficient.

        Args:
            store        : PipelineStore.
            return_preds : Dict of entity_id → predicted return (may be None).
            min_assets   : Minimum assets for covariance to be meaningful.
            min_price_rows: Minimum price observations per asset.
        """
        if return_preds is not None and len(return_preds) < min_assets:
            return False, PreflightResult.no_data(
                detail=f"Only {len(return_preds)} return predictions "
                       f"(need ≥ {min_assets} for covariance estimation)",
                fix="Ensure at least 2 instruments are present in the graph.",
            )

        if store is not None:
            result = FeaturePreflight._check_min_observations(
                store, "price", min_price_rows,
                fix="Ingest price data via macro_data or instrument_universe tool.",
            )
            if not result.ok:
                return False, result

        return True, PreflightResult.passed()

    # ── Idea 13: Data Catalog ─────────────────────────────────────────────

    @staticmethod
    def for_data_catalog(
        store: Any | None,
        use_data_catalog: bool = False,
    ) -> tuple[bool, PreflightResult]:
        """Check use_data_catalog flag and store is reachable.

        Args:
            store            : PipelineStore.
            use_data_catalog : TrainerConfig flag value.
        """
        if not use_data_catalog:
            return False, PreflightResult.missing_config(
                detail="use_data_catalog=False in TrainerConfig",
                fix="Set TrainerConfig(use_data_catalog=True) to enable.",
            )
        if store is None:
            return False, PreflightResult.missing_config(
                detail="PipelineStore is None — cannot run freshness checks",
                fix="Pass a valid PipelineStore to Trainer.",
            )
        return True, PreflightResult.passed()

    # ── Generic GNN inference gate ────────────────────────────────────────

    @staticmethod
    def for_gnn_inference(
        model: Any | None,
        store: Any | None,
        min_entity_types: int = 1,
    ) -> tuple[bool, PreflightResult]:
        """Common gate for any feature that runs GNN inference.

        Args:
            model            : HetTGN instance.
            store            : PipelineStore.
            min_entity_types : Minimum node types in graph.
        """
        if model is None:
            return False, PreflightResult.model_not_ready(
                detail="GNN model is None — not built yet",
                fix="Call trainer.build_model().",
            )

        if store is not None:
            result = FeaturePreflight._check_store_has_entities(
                store, min_entity_types,
            )
            if not result.ok:
                return False, result

        return True, PreflightResult.passed()

    # ── Tool-level API key check ──────────────────────────────────────────

    @staticmethod
    def for_api_key(
        key_value: str | None,
        env_var: str,
        tool_name: str,
        signup_url: str = "",
    ) -> tuple[bool, PreflightResult]:
        """Generic check for any tool that requires an API key.

        Args:
            key_value : The key string (may be empty/None).
            env_var   : Name of the environment variable.
            tool_name : Human-readable tool name for the error message.
            signup_url: URL where the user can get a key.
        """
        val = key_value or os.getenv(env_var, "")
        if not val:
            fix = f"Export {env_var}=<your-key>"
            if signup_url:
                fix += f" (get one at {signup_url})"
            return False, PreflightResult.missing_config(
                detail=f"{tool_name} requires {env_var} but it is not set",
                fix=fix,
            )
        return True, PreflightResult.passed()

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _check_data_staleness(
        store: Any,
        source_tool: str,
        max_stale_hours: float,
        fix: str = "",
    ) -> PreflightResult:
        """Return NO_DATA if source_tool's last observation is too old."""
        try:
            conn = store._get_conn()
            rows = conn.execute(
                "SELECT MAX(observed_at) FROM entity_observations WHERE source_tool=?",
                (source_tool,),
            ).fetchall()
            val = rows[0][0] if rows else None
            if val is None:
                return PreflightResult.no_data(
                    detail=f"No observations from '{source_tool}' in the pipeline store",
                    fix=fix,
                )
            age_hours = (time.time() - float(val)) / _HOURS
            if age_hours > max_stale_hours:
                return PreflightResult.no_data(
                    detail=f"Last '{source_tool}' data is {age_hours:.1f}h old "
                           f"(limit: {max_stale_hours}h)",
                    fix=fix,
                )
            return PreflightResult.passed()
        except Exception as exc:
            log.warning("Preflight staleness check failed: %s", exc)
            return PreflightResult.passed()  # don't block on DB errors

    @staticmethod
    def _check_min_observations(
        store: Any,
        observation_type: str,
        min_rows: int,
        fix: str = "",
    ) -> PreflightResult:
        """Return NO_DATA if fewer than min_rows rows of given type exist."""
        try:
            conn = store._get_conn()
            rows = conn.execute(
                "SELECT COUNT(*) FROM entity_observations WHERE observation_type=?",
                (observation_type,),
            ).fetchall()
            count = rows[0][0] if rows else 0
            if count < min_rows:
                return PreflightResult.no_data(
                    detail=f"Only {count} '{observation_type}' observations "
                           f"(need ≥ {min_rows})",
                    fix=fix,
                )
            return PreflightResult.passed()
        except Exception as exc:
            log.warning("Preflight observation count check failed: %s", exc)
            return PreflightResult.passed()

    @staticmethod
    def _check_store_has_entities(
        store: Any,
        min_entity_types: int,
    ) -> PreflightResult:
        """Return NO_DATA if the store has fewer than min_entity_types distinct types."""
        try:
            conn = store._get_conn()
            rows = conn.execute(
                "SELECT COUNT(DISTINCT source_tool) FROM entity_observations"
            ).fetchall()
            count = rows[0][0] if rows else 0
            if count < min_entity_types:
                return PreflightResult.no_data(
                    detail=f"Store has {count} distinct entity sources "
                           f"(need ≥ {min_entity_types})",
                    fix="Run data ingestion tools or SyntheticGraphGenerator.",
                )
            return PreflightResult.passed()
        except Exception as exc:
            log.warning("Preflight entity check failed: %s", exc)
            return PreflightResult.passed()
