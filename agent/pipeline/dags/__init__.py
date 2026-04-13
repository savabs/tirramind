"""
TirraMind — Built-in DAG Definitions

Each module in this package defines one or more DAGs. The get_default_dags()
function collects them all for the DAGRegistry.
"""

from __future__ import annotations

from typing import Any

from agent.pipeline.dag import DAG


def get_default_dags(tool_registry: Any) -> list[DAG]:
    """Return all built-in DAGs.

    tool_registry is available for DAG builders that need to verify
    tool existence, but DAG nodes reference tools by string name —
    resolution happens at execution time in the operator layer.
    """
    from agent.pipeline.dags.daily_collection import build_daily_collection_dag
    from agent.pipeline.dags.convergence_detection import (
        build_convergence_detection_dag,
    )
    from agent.pipeline.dags.feature_generation import (
        build_feature_generation_dag,
    )
    from agent.pipeline.dags.entity_scoring import (
        build_entity_scoring_dag,
    )
    from agent.pipeline.dags.gnn_inference import (
        build_gnn_inference_dag,
    )
    from agent.pipeline.dags.whale_tracking import (
        build_whale_tracking_dag,
        build_whale_scoring_dag,
    )
    from agent.pipeline.dags.world_model_update import (
        build_world_model_dag,
    )
    from agent.pipeline.dags.rl_training import (
        build_rl_training_dag,
    )
    from agent.pipeline.dags.adversarial_scan import (
        build_adversarial_scan_dag,
    )

    return [
        build_daily_collection_dag(),
        build_whale_tracking_dag(),
        build_whale_scoring_dag(),
        build_convergence_detection_dag(),
        build_gnn_inference_dag(),
        build_entity_scoring_dag(),
        build_feature_generation_dag(),
        build_world_model_dag(),
        build_adversarial_scan_dag(),
        build_rl_training_dag(),
    ]
