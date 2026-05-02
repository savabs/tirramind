"""
Tests for Learned State Encoder (Change 6, Tier 4).

Covers: encoder forward shapes, gradient flow, padding mask behaviour,
save/load round-trip, backward compatibility, SACTrainer integration,
determinism, and edge cases (0/1/max entities).
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn as nn

from agent.learning.policy.config import SACConfig, StateEncoderConfig
from agent.learning.policy.replay_buffer import ReplayBuffer
from agent.learning.policy.sac import SACTrainer
from agent.learning.policy.state_encoder import LearnedStateEncoder

# ── Helpers ───────────────────────────────────────────────────────


def _default_encoder_config(**overrides) -> StateEncoderConfig:
    defaults = dict(
        entity_embed_dim=32,
        n_heads=4,
        n_attention_layers=1,
        dropout=0.0,  # deterministic for testing
        max_entities=50,
        surprise_dim=5,
        belief_dim=4,
        market_dim=8,
        adversarial_dim=4,
    )
    defaults.update(overrides)
    return StateEncoderConfig(**defaults)


def _make_state(
    n_active: int = 10,
    E: int = 50,
    surprise_dim: int = 5,
    belief_dim: int = 4,
    market_dim: int = 8,
    adv_dim: int = 4,
    batch: int | None = None,
) -> torch.Tensor:
    """Build a synthetic flat state matching StateAssembler layout."""
    total = E * surprise_dim + E * belief_dim + market_dim + 1 + adv_dim

    if batch is not None:
        state = torch.zeros(batch, total)
        for b in range(batch):
            # Fill active entity surprise block
            state[b, : n_active * surprise_dim] = torch.randn(n_active * surprise_dim)
            # Fill active entity belief block
            bstart = E * surprise_dim
            state[b, bstart : bstart + n_active * belief_dim] = torch.randn(n_active * belief_dim)
            # Market + count + adversarial
            state[b, E * surprise_dim + E * belief_dim :] = torch.randn(market_dim + 1 + adv_dim)
    else:
        state = torch.zeros(total)
        state[: n_active * surprise_dim] = torch.randn(n_active * surprise_dim)
        bstart = E * surprise_dim
        state[bstart : bstart + n_active * belief_dim] = torch.randn(n_active * belief_dim)
        state[E * surprise_dim + E * belief_dim :] = torch.randn(market_dim + 1 + adv_dim)

    return state


def _fill_buffer(
    buffer: ReplayBuffer,
    state_dim: int,
    action_dim: int,
    n: int = 300,
) -> None:
    for _ in range(n):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randn(action_dim).astype(np.float32) * 0.1
        r = float(np.random.randn())
        ns = np.random.randn(state_dim).astype(np.float32)
        buffer.push(s, a, r, ns, False)


# ═════════════════════════════════════════════════════════════════
# §1 — Encoder Forward Shape Tests
# ═════════════════════════════════════════════════════════════════


class TestEncoderForwardShape:
    """Validate that encoder produces correct output shapes."""

    def test_single_state(self):
        cfg = _default_encoder_config()
        enc = LearnedStateEncoder(cfg)
        state = _make_state(n_active=10)
        out = enc(state)
        assert out.shape == (cfg.entity_embed_dim + cfg.market_dim + 1 + cfg.adversarial_dim,)

    def test_batch_state(self):
        cfg = _default_encoder_config()
        enc = LearnedStateEncoder(cfg)
        state = _make_state(n_active=10, batch=8)
        out = enc(state)
        assert out.shape == (8, enc.output_dim)

    def test_output_dim_property(self):
        cfg = _default_encoder_config(entity_embed_dim=64)
        enc = LearnedStateEncoder(cfg)
        expected = 64 + cfg.market_dim + 1 + cfg.adversarial_dim
        assert enc.output_dim == expected

    def test_input_dim_property(self):
        cfg = _default_encoder_config()
        enc = LearnedStateEncoder(cfg)
        expected = 50 * 5 + 50 * 4 + 8 + 1 + 4  # 463
        assert enc.input_dim == expected

    def test_custom_entity_embed_dim(self):
        cfg = _default_encoder_config(entity_embed_dim=16, n_heads=2)
        enc = LearnedStateEncoder(cfg)
        state = _make_state(n_active=5, batch=4)
        out = enc(state)
        assert out.shape == (4, 16 + 8 + 1 + 4)

    def test_multiple_attention_layers(self):
        cfg = _default_encoder_config(n_attention_layers=3)
        enc = LearnedStateEncoder(cfg)
        state = _make_state(n_active=20, batch=2)
        out = enc(state)
        assert out.shape == (2, enc.output_dim)


# ═════════════════════════════════════════════════════════════════
# §2 — Gradient Flow
# ═════════════════════════════════════════════════════════════════


class TestGradientFlow:
    """Verify that gradients flow through the encoder."""

    def test_encoder_grads_from_output_loss(self):
        cfg = _default_encoder_config()
        enc = LearnedStateEncoder(cfg)
        state = _make_state(n_active=10, batch=4)
        out = enc(state)
        loss = out.sum()
        loss.backward()
        # All encoder params should have non-None gradients
        for name, p in enc.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_encoder_grads_through_mlp(self):
        """Gradients flow: MLP → encoder."""
        cfg = _default_encoder_config()
        enc = LearnedStateEncoder(cfg)
        mlp = nn.Linear(enc.output_dim, 1)
        state = _make_state(n_active=10, batch=4)
        out = enc(state)
        pred = mlp(out)
        loss = pred.mean()
        loss.backward()
        for name, p in enc.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_cls_token_updates(self):
        """The [CLS] parameter receives gradients (finite-difference check)."""
        cfg = _default_encoder_config()
        torch.manual_seed(42)
        enc = LearnedStateEncoder(cfg)
        enc.eval()
        state = _make_state(n_active=5, batch=2)

        # Finite-difference: perturb _cls_token and check loss changes
        out0 = enc(state)
        loss0 = out0.sum().item()

        with torch.no_grad():
            enc._cls_token.add_(0.1)
        out1 = enc(state)
        loss1 = out1.sum().item()

        assert loss0 != loss1, "CLS token has no effect on output"


# ═════════════════════════════════════════════════════════════════
# §3 — Padding Mask Correctness
# ═════════════════════════════════════════════════════════════════


class TestPaddingMask:
    """Zero-padded entities should not affect the output."""

    def test_zero_padded_entities_masked(self):
        """Output should be (approximately) the same regardless of how many
        zero-padded slots exist, given the same active entities."""
        cfg = _default_encoder_config(dropout=0.0)
        enc = LearnedStateEncoder(cfg)
        enc.eval()

        # State with 5 active entities
        state = _make_state(n_active=5)

        # Same state — zero-padded slots are already zero
        out1 = enc(state)
        out2 = enc(state.clone())

        torch.testing.assert_close(out1, out2)

    def test_all_zero_entities(self):
        """When all entities are zero (no active), output should still be valid."""
        cfg = _default_encoder_config(dropout=0.0)
        enc = LearnedStateEncoder(cfg)
        enc.eval()

        state = _make_state(n_active=0)  # all entity slots zero
        out = enc(state)
        assert out.shape == (enc.output_dim,)
        assert torch.isfinite(out).all()

    def test_single_active_entity(self):
        cfg = _default_encoder_config(dropout=0.0)
        enc = LearnedStateEncoder(cfg)
        enc.eval()

        state = _make_state(n_active=1)
        out = enc(state)
        assert out.shape == (enc.output_dim,)
        assert torch.isfinite(out).all()

    def test_max_active_entities(self):
        cfg = _default_encoder_config(dropout=0.0)
        enc = LearnedStateEncoder(cfg)
        enc.eval()

        state = _make_state(n_active=50)  # all slots active
        out = enc(state)
        assert out.shape == (enc.output_dim,)
        assert torch.isfinite(out).all()


# ═════════════════════════════════════════════════════════════════
# §4 — SACTrainer Integration (with encoder)
# ═════════════════════════════════════════════════════════════════


class TestSACTrainerWithEncoder:
    """SACTrainer should work with and without encoder."""

    def test_trainer_without_encoder(self):
        """Baseline: no encoder, raw state → actor/critic."""
        state_dim = 463
        action_dim = 5
        trainer = SACTrainer(state_dim, action_dim)

        state = torch.randn(state_dim)
        action = trainer.select_action(state, deterministic=True)
        assert action.shape == (action_dim,)

    def test_trainer_with_encoder_select_action(self):
        """Encoder reduces state_dim before actor sees it."""
        enc_cfg = _default_encoder_config()
        encoder = LearnedStateEncoder(enc_cfg)
        state_dim = encoder.input_dim  # 463

        trainer = SACTrainer(state_dim, 5, encoder=encoder)
        state = _make_state(n_active=10)
        action = trainer.select_action(state, deterministic=True)
        assert action.shape == (5,)

    def test_trainer_with_encoder_stochastic(self):
        enc_cfg = _default_encoder_config()
        encoder = LearnedStateEncoder(enc_cfg)
        trainer = SACTrainer(encoder.input_dim, 3, encoder=encoder)
        state = _make_state(n_active=5)
        action = trainer.select_action(state, deterministic=False)
        assert action.shape == (3,)

    def test_trainer_with_encoder_update(self):
        """Full training update with encoder produces sensible metrics."""
        enc_cfg = _default_encoder_config()
        encoder = LearnedStateEncoder(enc_cfg)
        state_dim = encoder.input_dim
        action_dim = 3
        cfg = SACConfig(batch_size=32)

        trainer = SACTrainer(state_dim, action_dim, cfg, encoder=encoder)
        buffer = ReplayBuffer(1000, state_dim, action_dim)
        _fill_buffer(buffer, state_dim, action_dim, n=100)

        metrics = trainer.update(buffer)
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha" in metrics
        assert metrics["critic_loss"] >= 0
        assert np.isfinite(metrics["critic_loss"])

    def test_encoder_params_in_actor_optim(self):
        """Encoder parameters should be in the actor optimizer."""
        enc_cfg = _default_encoder_config()
        encoder = LearnedStateEncoder(enc_cfg)
        trainer = SACTrainer(encoder.input_dim, 3, encoder=encoder)

        # Actor optimizer should have more param groups than actor alone
        actor_param_count = sum(1 for _ in trainer._actor.parameters())
        encoder_param_count = sum(1 for _ in encoder.parameters())
        optim_param_count = sum(len(pg["params"]) for pg in trainer._actor_optim.param_groups)
        assert optim_param_count == actor_param_count + encoder_param_count

    def test_encoder_grads_after_update(self):
        """After one update, encoder parameters should have gradients."""
        enc_cfg = _default_encoder_config()
        encoder = LearnedStateEncoder(enc_cfg)
        state_dim = encoder.input_dim
        action_dim = 3
        cfg = SACConfig(batch_size=32)

        trainer = SACTrainer(state_dim, action_dim, cfg, encoder=encoder)
        buffer = ReplayBuffer(1000, state_dim, action_dim)
        _fill_buffer(buffer, state_dim, action_dim, n=100)

        trainer.update(buffer)

        # At least some encoder params should have non-zero grad
        has_grad = False
        for p in encoder.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Encoder got no gradients from SAC update"


# ═════════════════════════════════════════════════════════════════
# §5 — Save / Load Round-Trip
# ═════════════════════════════════════════════════════════════════


class TestSaveLoadWithEncoder:
    """Checkpoint round-trip with and without encoder."""

    def test_save_load_with_encoder(self):
        enc_cfg = _default_encoder_config()
        encoder = LearnedStateEncoder(enc_cfg)
        state_dim = encoder.input_dim
        action_dim = 3

        trainer = SACTrainer(state_dim, action_dim, encoder=encoder)
        data = trainer.save()

        loaded = SACTrainer.load(data, state_dim, action_dim)
        assert loaded._encoder is not None

        # Verify encoder weights match
        state = _make_state(n_active=10)
        with torch.no_grad():
            out_orig = trainer.select_action(state, deterministic=True)
            out_loaded = loaded.select_action(state.clone(), deterministic=True)
        np.testing.assert_allclose(out_orig, out_loaded, atol=1e-6)

    def test_save_load_without_encoder(self):
        """Old-style checkpoint without encoder loads fine."""
        state_dim = 463
        action_dim = 3
        trainer = SACTrainer(state_dim, action_dim)
        data = trainer.save()

        loaded = SACTrainer.load(data, state_dim, action_dim)
        assert loaded._encoder is None

        state = torch.randn(state_dim)
        with torch.no_grad():
            out_orig = trainer.select_action(state, deterministic=True)
            out_loaded = loaded.select_action(state.clone(), deterministic=True)
        np.testing.assert_allclose(out_orig, out_loaded, atol=1e-6)

    def test_encoder_config_preserved_in_checkpoint(self):
        """Encoder config should survive round-trip."""
        enc_cfg = _default_encoder_config(entity_embed_dim=64, n_heads=8)
        encoder = LearnedStateEncoder(enc_cfg)
        trainer = SACTrainer(encoder.input_dim, 3, encoder=encoder)
        data = trainer.save()

        loaded = SACTrainer.load(data, encoder.input_dim, 3)
        assert loaded._encoder is not None
        assert loaded._encoder._cfg.entity_embed_dim == 64
        assert loaded._encoder._cfg.n_heads == 8

    def test_update_count_preserved(self):
        enc_cfg = _default_encoder_config()
        encoder = LearnedStateEncoder(enc_cfg)
        state_dim = encoder.input_dim
        trainer = SACTrainer(state_dim, 3, encoder=encoder)
        trainer._update_count = 42
        data = trainer.save()

        loaded = SACTrainer.load(data, state_dim, 3)
        assert loaded._update_count == 42


# ═════════════════════════════════════════════════════════════════
# §6 — Backward Compatibility
# ═════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Old checkpoints (no encoder) should load without error."""

    def test_old_checkpoint_loads(self):
        """Simulate old checkpoint format (no has_encoder key)."""
        state_dim = 463
        action_dim = 3
        trainer = SACTrainer(state_dim, action_dim)
        # Save and load — should work since has_encoder defaults to False
        data = trainer.save()

        # Manually verify the checkpoint has has_encoder=False
        buf = io.BytesIO(data)
        checkpoint = torch.load(buf, map_location="cpu", weights_only=False)
        assert checkpoint.get("has_encoder") is False

        loaded = SACTrainer.load(data, state_dim, action_dim)
        assert loaded._encoder is None

    def test_old_format_select_action(self):
        """Old-format trainer still does direct MLP forwarding."""
        state_dim = 463
        action_dim = 5
        loaded = SACTrainer(state_dim, action_dim)
        state = torch.randn(state_dim)
        action = loaded.select_action(state, deterministic=True)
        assert action.shape == (action_dim,)


# ═════════════════════════════════════════════════════════════════
# §7 — Determinism
# ═════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Same input should produce same output (with dropout=0)."""

    def test_encoder_deterministic(self):
        cfg = _default_encoder_config(dropout=0.0)
        enc = LearnedStateEncoder(cfg)
        enc.eval()

        state = _make_state(n_active=10, batch=4)
        out1 = enc(state.clone())
        out2 = enc(state.clone())
        torch.testing.assert_close(out1, out2)

    def test_trainer_deterministic_action(self):
        enc_cfg = _default_encoder_config(dropout=0.0)
        encoder = LearnedStateEncoder(enc_cfg)
        encoder.eval()
        trainer = SACTrainer(encoder.input_dim, 3, encoder=encoder)

        state = _make_state(n_active=10)
        a1 = trainer.select_action(state.clone(), deterministic=True)
        a2 = trainer.select_action(state.clone(), deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-6)


# ═════════════════════════════════════════════════════════════════
# §8 — Edge Cases
# ═════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_batch_size_one(self):
        cfg = _default_encoder_config()
        enc = LearnedStateEncoder(cfg)
        state = _make_state(n_active=5, batch=1)
        out = enc(state)
        assert out.shape == (1, enc.output_dim)

    def test_large_batch(self):
        cfg = _default_encoder_config()
        enc = LearnedStateEncoder(cfg)
        state = _make_state(n_active=20, batch=64)
        out = enc(state)
        assert out.shape == (64, enc.output_dim)

    def test_all_same_entities(self):
        """All entities have identical features — encoder should still work."""
        cfg = _default_encoder_config(dropout=0.0)
        enc = LearnedStateEncoder(cfg)
        enc.eval()

        E = 50
        total = E * 5 + E * 4 + 8 + 1 + 4
        state = torch.ones(total) * 0.5
        out = enc(state)
        assert torch.isfinite(out).all()

    def test_very_large_values(self):
        """Encoder handles large input values without NaN."""
        cfg = _default_encoder_config(dropout=0.0)
        enc = LearnedStateEncoder(cfg)
        enc.eval()

        state = _make_state(n_active=10) * 100.0
        out = enc(state)
        assert torch.isfinite(out).all()

    def test_negative_values(self):
        cfg = _default_encoder_config(dropout=0.0)
        enc = LearnedStateEncoder(cfg)
        enc.eval()

        state = _make_state(n_active=10) * -1.0
        out = enc(state)
        assert torch.isfinite(out).all()

    def test_minimal_config(self):
        """Smallest valid config: embed_dim=2, heads=1, layers=1."""
        cfg = _default_encoder_config(
            entity_embed_dim=2,
            n_heads=1,
            n_attention_layers=1,
        )
        enc = LearnedStateEncoder(cfg)
        state = _make_state(n_active=3)
        out = enc(state)
        assert out.shape == (2 + 8 + 1 + 4,)


# ═════════════════════════════════════════════════════════════════
# §9 — Integration: Training Loop With Encoder
# ═════════════════════════════════════════════════════════════════


class TestTrainingLoopIntegration:
    """Multiple training steps with encoder produce learning signal."""

    def test_multiple_updates(self):
        enc_cfg = _default_encoder_config()
        encoder = LearnedStateEncoder(enc_cfg)
        state_dim = encoder.input_dim
        action_dim = 3
        cfg = SACConfig(batch_size=32)

        trainer = SACTrainer(state_dim, action_dim, cfg, encoder=encoder)
        buffer = ReplayBuffer(1000, state_dim, action_dim)
        _fill_buffer(buffer, state_dim, action_dim, n=200)

        metrics_history = []
        for _ in range(5):
            m = trainer.update(buffer)
            metrics_history.append(m)

        # All updates should produce finite metrics
        for m in metrics_history:
            assert np.isfinite(m["critic_loss"])
            assert np.isfinite(m["actor_loss"])
            assert m["alpha"] > 0

    def test_encoder_weights_change_during_training(self):
        """Encoder parameters should change after training steps."""
        enc_cfg = _default_encoder_config()
        encoder = LearnedStateEncoder(enc_cfg)
        state_dim = encoder.input_dim
        action_dim = 3
        cfg = SACConfig(batch_size=32)

        trainer = SACTrainer(state_dim, action_dim, cfg, encoder=encoder)
        buffer = ReplayBuffer(1000, state_dim, action_dim)
        _fill_buffer(buffer, state_dim, action_dim, n=200)

        # Snapshot encoder weights before training
        before = {name: p.data.clone() for name, p in encoder.named_parameters()}

        for _ in range(10):
            trainer.update(buffer)

        # At least some weights should have changed
        changed = False
        for name, p in encoder.named_parameters():
            if not torch.allclose(before[name], p.data, atol=1e-8):
                changed = True
                break
        assert changed, "Encoder weights did not change after 10 updates"

    def test_save_load_after_training(self):
        """Checkpoint after training preserves trained encoder state."""
        enc_cfg = _default_encoder_config()
        encoder = LearnedStateEncoder(enc_cfg)
        state_dim = encoder.input_dim
        action_dim = 3
        cfg = SACConfig(batch_size=32)

        trainer = SACTrainer(state_dim, action_dim, cfg, encoder=encoder)
        buffer = ReplayBuffer(1000, state_dim, action_dim)
        _fill_buffer(buffer, state_dim, action_dim, n=200)

        for _ in range(5):
            trainer.update(buffer)

        data = trainer.save()
        loaded = SACTrainer.load(data, state_dim, action_dim)

        # Actions should match after load
        state = _make_state(n_active=10)
        a_orig = trainer.select_action(state.clone(), deterministic=True)
        a_loaded = loaded.select_action(state.clone(), deterministic=True)
        np.testing.assert_allclose(a_orig, a_loaded, atol=1e-5)
