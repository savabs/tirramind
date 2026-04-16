"""TirraMind — Learned State Encoder (Change 6)

Replaces hand-designed top-K truncation in StateAssembler with a learnable
multihead-attention encoder that produces a compact, fixed-dim state for SAC.

Architecture:
    1. Parse flat state vector from StateAssembler into entity block + global block
    2. Per-entity embedding: Linear(entity_feat_dim → entity_embed_dim) + ReLU
    3. Prepend learnable [CLS] token, apply MultiheadAttention with padding mask
    4. Extract [CLS] output as entity summary
    5. Concatenate: [z_CLS ; global_features] → compact state for SAC

The encoder sits between the replay buffer (which stores the 463-dim assembler
output) and the SAC actor/critic.  Gradients flow:
    SAC loss → actor MLP → encoder weights

This mirrors standard deep RL encoders (e.g., CNN in Atari DQN).

Reference: Vaswani et al. 2017 (MHA), Lee et al. ICML 2019 (Set Transformer
for permutation-invariant set encoding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from agent.learning.policy.feature_gate import FeatureGate


@dataclass(frozen=True)
class StateEncoderConfig:
    """Hyperparameters for the learned state encoder.

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


class LearnedStateEncoder(nn.Module):
    """Attention-based encoder over variable-length entity sets.

    Takes the flat state vector produced by ``StateAssembler.assemble()``
    (E*surprise_dim + E*belief_dim + market_dim + 1 + adversarial_dim),
    reshapes into per-entity tokens and global features, runs MHA with a
    learnable [CLS] token, and outputs a compact state vector.

    Parameters
    ----------
    config : StateEncoderConfig
        Encoder hyperparameters.
    """

    def __init__(self, config: StateEncoderConfig | None = None) -> None:
        super().__init__()
        cfg = config or StateEncoderConfig()
        self._cfg = cfg

        E = cfg.max_entities
        entity_feat_dim = cfg.surprise_dim + cfg.belief_dim  # 9

        self._E = E
        self._surprise_dim = cfg.surprise_dim
        self._belief_dim = cfg.belief_dim
        self._entity_feat_dim = entity_feat_dim
        self._market_dim = cfg.market_dim
        self._adv_dim = cfg.adversarial_dim

        # Global block size: market + entity_count + adversarial
        self._global_dim = cfg.market_dim + 1 + cfg.adversarial_dim

        # Offset computations for parsing the flat state vector
        self._surprise_end = E * cfg.surprise_dim
        self._belief_end = self._surprise_end + E * cfg.belief_dim
        self._global_start = self._belief_end
        self._expected_state_dim = (
            E * cfg.surprise_dim
            + E * cfg.belief_dim
            + cfg.market_dim
            + 1
            + cfg.adversarial_dim
        )

        # Entity embedding: 9 → entity_embed_dim
        self._entity_embed = nn.Sequential(
            nn.Linear(entity_feat_dim, cfg.entity_embed_dim),
            nn.ReLU(),
        )

        # Learnable [CLS] token
        self._cls_token = nn.Parameter(torch.randn(1, 1, cfg.entity_embed_dim) * 0.02)

        # Self-attention layers
        self._attention_layers = nn.ModuleList()
        for _ in range(cfg.n_attention_layers):
            self._attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=cfg.entity_embed_dim,
                    num_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
            )
            # Layer norm after attention (Pre-LN pattern)
            self._attention_layers.append(nn.LayerNorm(cfg.entity_embed_dim))

        # Output dim = entity_embed_dim + global_dim
        self._output_dim = cfg.entity_embed_dim + self._global_dim

        # Optional feature gate (Change 11) — set via set_feature_gate()
        self._feat_gate: FeatureGate | None = None

    @property
    def output_dim(self) -> int:
        """Dimensionality of the encoded state vector."""
        return self._output_dim

    @property
    def input_dim(self) -> int:
        """Expected dimensionality of the flat assembler state."""
        return self._expected_state_dim

    @property
    def feature_gate(self) -> FeatureGate | None:
        """Return the attached feature gate, if any."""
        return self._feat_gate

    def set_feature_gate(self, gate: FeatureGate) -> None:
        """Attach a feature gate module (Change 11).

        The gate's parameters become part of this encoder's parameter
        graph, so they are included in any optimizer that covers the encoder.
        """
        from agent.learning.policy.feature_gate import FeatureGate as _FG

        if not isinstance(gate, _FG):
            raise TypeError(f"Expected FeatureGate, got {type(gate)}")
        self._feat_gate = gate
        # Register as submodule so params are discoverable
        self.add_module("feat_gate", gate)

    def forward(
        self,
        state_flat: Tensor,
        regime_context: Tensor | None = None,
    ) -> Tensor:
        """Encode a flat assembler state into a compact representation.

        Parameters
        ----------
        state_flat : Tensor of shape (batch, state_dim) or (state_dim,)
            The raw state from StateAssembler.assemble().
        regime_context : Tensor of shape (batch, regime_dim) or (regime_dim,), optional
            HMM regime posterior for feature gating (Change 11).
            Ignored if no feature gate is attached.

        Returns
        -------
        Tensor of shape (batch, output_dim) — compact encoded state.
        """
        squeezed = False
        if state_flat.dim() == 1:
            state_flat = state_flat.unsqueeze(0)
            if regime_context is not None and regime_context.dim() == 1:
                regime_context = regime_context.unsqueeze(0)
            squeezed = True

        # ── Apply feature gate (Change 11) if attached ────────
        if self._feat_gate is not None and regime_context is not None:
            state_flat = self._feat_gate(state_flat, regime_context)

        B = state_flat.shape[0]
        E = self._E
        sd = self._surprise_dim
        bd = self._belief_dim

        # ── Parse flat state into entity + global blocks ──────
        surprise_flat = state_flat[:, : self._surprise_end]  # (B, E*5)
        belief_flat = state_flat[:, self._surprise_end : self._belief_end]  # (B, E*4)
        global_features = state_flat[:, self._global_start :]  # (B, global_dim)

        # Reshape to per-entity features
        surprise = surprise_flat.reshape(B, E, sd)  # (B, E, 5)
        belief = belief_flat.reshape(B, E, bd)  # (B, E, 4)
        entity_features = torch.cat([surprise, belief], dim=-1)  # (B, E, 9)

        # ── Build padding mask ────────────────────────────────
        # Zero-padded entities have all-zero features. Mask them out.
        entity_active = entity_features.abs().sum(dim=-1) > 0  # (B, E) bool
        # Mask for attention: True = IGNORE this position
        # Prepend False for [CLS] (never masked)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=state_flat.device)
        # key_padding_mask: (B, 1+E), True where position should be ignored
        key_padding_mask = torch.cat([cls_mask, ~entity_active], dim=1)  # (B, 1+E)

        # ── Entity embedding ──────────────────────────────────
        h = self._entity_embed(entity_features)  # (B, E, embed_dim)

        # Prepend [CLS] token
        cls_expanded = self._cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        tokens = torch.cat([cls_expanded, h], dim=1)  # (B, 1+E, embed_dim)

        # ── Self-attention ────────────────────────────────────
        for i in range(0, len(self._attention_layers), 2):
            attn_layer = self._attention_layers[i]
            norm_layer = self._attention_layers[i + 1]
            residual = tokens
            attn_out, _ = attn_layer(
                tokens,
                tokens,
                tokens,
                key_padding_mask=key_padding_mask,
            )
            tokens = norm_layer(residual + attn_out)

        # Extract [CLS] output
        z_cls = tokens[:, 0, :]  # (B, embed_dim)

        # ── Concatenate with global features ──────────────────
        compact_state = torch.cat([z_cls, global_features], dim=-1)  # (B, output_dim)

        if squeezed:
            compact_state = compact_state.squeeze(0)

        return compact_state
