"""
Integration test for M9 microstructure features in graph builder.

Verifies that the graph builder correctly adds 11-dimensional microstructure 
feature vectors to instrument nodes.
"""

import pytest
import torch
from agent.models.gnn.graph_builder import (
    GraphBuilder,
    BASE_FEAT_DIM,
    M15_QUANT_DIM,
    MICROSTRUCTURE_DIM,
    PRICE_FEAT_DIM,
)
from agent.pipeline.store import PipelineStore


def test_graph_builder_microstructure_integration(tmp_path):
    """T9: graph_builder adds microstructure features to instrument nodes."""
    
    # Create a test database
    db_path = tmp_path / "test.db"
    store = PipelineStore(str(db_path))
    
    # Register test entities
    store.register_entity("instrument", "AAPL", "AAPL", {"source": "test"})
    store.register_entity("instrument", "MSFT", "MSFT", {"source": "test"})
    store.register_entity("country", "USA", "USA", {"source": "test"})
    
    # Store enough daily bars for M9 micro (min 30 days)
    for day in range(40):
        ts = 86400.0 * day
        close = 150.0 + 0.1 * day
        store.store_entity_observation(
            entity_id="AAPL",
            source_tool="test",
            observed_at=ts,
            observation_type="instrument_daily",
            value={
                "close": close,
                "log_return": 0.001,
                "volume": 10000.0 + day * 100,
            },
        )
    store.store_entity_observation(
        entity_id="MSFT",
        source_tool="test",
        observed_at=1000.0,
        observation_type="instrument_daily",
        value={"close": 300.0, "log_return": 0.0, "volume": 5000.0},
    )
    
    # Build graph
    builder = GraphBuilder(store)
    data, id_map, events = builder.build()
    
    # Verify instrument nodes have correct feature dimensionality
    instrument_features = data["instrument"].x
    
    # BASE (14) + PRICE (9) + MICRO (11) + M15 quant (15) = 49
    expected_dim = BASE_FEAT_DIM + PRICE_FEAT_DIM + MICROSTRUCTURE_DIM + M15_QUANT_DIM
    assert instrument_features.shape[1] == expected_dim, \
        f"Expected instrument feature dim {expected_dim}, got {instrument_features.shape[1]}"
    
    # AAPL has 40 days → non-zero micro block; MSFT has 1 day → zeros
    micro_offset = 14 + 9  # BASE + PRICE
    micro_aapl = instrument_features[0, micro_offset:micro_offset + MICROSTRUCTURE_DIM]
    assert micro_aapl.abs().sum() > 0, "AAPL micro features should be populated"
    assert micro_aapl.shape[0] == MICROSTRUCTURE_DIM
    
    # Verify other node types don't have microstructure features
    country_features = data["country"].x
    # Country should have BASE (14) only
    assert country_features.shape[1] == 14, \
        f"Country nodes should not have microstructure features, got {country_features.shape[1]}"


def test_graph_builder_with_enrichment():
    """Verify microstructure features offset correctly with enrichment."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        store = PipelineStore(tmp.name)
        store.register_entity("instrument", "TEST", "TEST", {"source": "test"})
        store.store_entity_observation(
            entity_id="TEST",
            source_tool="test",
            observed_at=1000.0,
            observation_type="instrument_daily",
            value={"close": 100.0}
        )
        
        builder = GraphBuilder(store)
        
        # Build with enrichment
        enrichment = {"TEST": {"cusum": 1.0, "hawkes": 2.0}}
        data, _, _ = builder.build(enrichment=enrichment)
        
        # BASE(14) + ENRICHMENT(55) + PRICE(9) + MICRO(11) + M15(15) = 104
        expected_dim = (
            BASE_FEAT_DIM + 55 + PRICE_FEAT_DIM + MICROSTRUCTURE_DIM + M15_QUANT_DIM
        )
        assert data["instrument"].x.shape[1] == expected_dim, \
            f"Expected dim {expected_dim} with enrichment, got {data['instrument'].x.shape[1]}"
        
        store.close()
        import os
        os.unlink(tmp.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
