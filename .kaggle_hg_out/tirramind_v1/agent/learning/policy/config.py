"""TirraMind — RL Policy Configuration

Frozen dataclasses that parameterize Phase 21a (weight learning)
and Phase 21b (SAC actor-critic).  All default values are documented
with their mathematical or empirical justification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.learning.policy.feature_gate import FeatureGateConfig


@dataclass(frozen=True)
class RewardConfig:
    """Reward-function parameters.

    Extrinsic reward is Sharpe-normalized return with CVaR penalty:
        r_ext = portfolio_return / max(σ_W, ε) − κ · max(0, −CVaR₀.₀₅)

    Intrinsic reward is mean composite surprise of alerted entities,
    decaying linearly so the policy shifts from exploration to exploitation:
        r_int = mean(composite_surprise) · λ(t)
        λ(t)  = λ₀ · (1 − t/T)
    """

    cvar_confidence: float = 0.95
    cvar_penalty: float = 1.0  # κ
    intrinsic_weight_initial: float = 0.1  # λ₀
    intrinsic_decay: bool = True
    rolling_vol_window: int = 20
    vol_floor: float = 1e-8  # ε — prevents division-by-zero
    adversarial_penalty: float = 1.0  # β — scale of adversarial flag penalty


@dataclass(frozen=True)
class WeightLearnerConfig:
    """Phase 21a: differentiable backtest config.

    Learns 5 surprise weights by maximising the differentiable Sharpe
    ratio (Moody & Saffell 2001) under walk-forward cross-validation.

    Differentiable Sharpe:
        S = r̄ / √(r̄² − r̄² + ε)
    where r_t = wᵀ s_t · ret_{t+1} and w is softmax-normalised.
    """

    learning_rate: float = 0.01
    max_epochs: int = 200
    patience: int = 20  # early stopping epochs without improvement
    grad_clip_norm: float = 1.0
    min_train_periods: int = 104  # 2 years weekly
    test_periods: int = 52  # 1 year weekly
    walk_forward_step: int = 26  # 6-month step
    l2_reg: float = 1e-4


@dataclass(frozen=True)
class SACConfig:
    """Phase 21b: Soft Actor-Critic hyperparameters.

    SAC (Haarnoja 2018, arXiv:1801.01290) maximises:
        J(π) = E[Σ γᵗ (r_t + α H[π(·|s_t)])]

    Temperature α is auto-tuned to a target entropy of
        H_target = scale · dim(A)
    (negative half the action dim, per Haarnoja's heuristic).
    """

    hidden_dim: int = 128
    num_hidden: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # Polyak soft-target update coefficient
    batch_size: int = 256
    buffer_size: int = 100_000
    target_entropy_scale: float = -0.5  # H_target = scale * dim(A)
    max_position: float = 0.5  # half-Kelly cap as fraction of Kelly-optimal
    leverage_limit: float = 1.0  # Σ|w_i| ≤ L
    warmup_steps: int = 1000  # random actions before policy is used
    # Model-based Kalman augmentation (Phase B — differentiable bypass)
    aux_kalman_weight: float = 0.01  # scale for auxiliary actor loss through Kalman
    kalman_lr: float = 1e-4  # separate optimizer LR for Kalman params
    kalman_grad_clip: float = 0.1  # max gradient norm for Kalman params


@dataclass(frozen=True)
class StateEncoderConfig:
    """Learned state encoder hyperparameters (Change 6).

    A multihead-attention encoder that replaces the hand-designed
    top-K truncation in StateAssembler with learned entity attention.

    entity_embed_dim must be divisible by n_heads.
    """

    entity_embed_dim: int = 32
    n_heads: int = 4
    n_attention_layers: int = 1
    dropout: float = 0.1
    max_entities: int = 50
    surprise_dim: int = 5
    belief_dim: int = 4
    market_dim: int = 8
    adversarial_dim: int = 4


@dataclass(frozen=True)
class PolicyConfig:
    """Top-level aggregation of all RL policy configs.

    After Phase 21a training, ``surprise_weights`` is populated
    with the learned 5-tuple which replaces the hard-coded defaults
    in SurpriseExtractor.
    """

    reward: RewardConfig = field(default_factory=RewardConfig)
    weight_learner: WeightLearnerConfig = field(default_factory=WeightLearnerConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    surprise_weights: tuple[float, ...] | None = None
    state_encoder: StateEncoderConfig | None = None
    feature_gate: FeatureGateConfig | None = None
