"""TirraMind — RL Policy Module

Phase 21: Surprise-driven portfolio allocation.
Closes the loop: entity surprise → learned actions → P&L → reward → updated policy.

Sub-modules:
    config              Configuration dataclasses
    symlog              DreamerV3 symlog/symexp transforms
    asset_mapper        Entity-to-asset resolution
    reward_fn           Extrinsic + intrinsic reward
    weight_learner      Phase 21a differentiable backtest
    state_assembler     Heterogeneous signal → fixed-size state tensor
    replay_buffer       Circular numpy-backed buffer for SAC
    sac                 Soft Actor-Critic (Haarnoja 2018)
    portfolio_strategy  Strategy ABC adapters for walk-forward evaluation
"""
