---
title: "Research: Learned State Encoder (Change 6)"
tags:
  - doc/research
  - phase/25
  - topic/self-improving
  - topic/state-representation
  - layer/learning
---

# Research: Learned State Encoder (Change 6)

## Problem Statement

`InstrumentStateAssembler` constructs the SAC state vector via hand-designed layout:
- Instrument surprises (N×5)
- Entity surprises (top-E by composite_surprise) (E×5)
- Entity beliefs (E×4)
- Market features (M)
- Entity count (1)
- Adversarial summary (4)

Total: N×5 + E×9 + M + 5 with E=50, M=8.

**Three hand-coded decisions that limit learning:**
1. **Top-K truncation** by composite_surprise — hard attention that discards potentially useful entities
2. **Fixed feature grouping** — surprise and belief are separate blocks; no cross-entity interaction
3. **Zero-padding** — wastes capacity; SAC MLP sees many zeros when fewer than E entities are active

## Current Architecture Analysis

### StateAssembler (training)
- `state_dim = E*5 + E*4 + M + 1 + 4 = 463` (E=50, M=8)
- Used in `rl_training.py` — no instrument block
- Output: contiguous numpy → torch tensor

### InstrumentStateAssembler (inference)
- `state_dim = N*5 + E*5 + E*4 + M + 1 + 4` where N = len(instrument_tickers)
- Used in `inference.py` — adds per-instrument surprise prefix
- **Mismatch:** Training assembler has different state_dim than inference assembler

### SAC Policy
- `GaussianActor`: `_trunk = _build_mlp(state_dim, hidden_dim, hidden_dim, num_hidden-1)` → `_mu_head`, `_log_std_head`
- `TwinCritic`: `_q1 = _build_mlp(state_dim+action_dim, 1, hidden_dim, num_hidden)`, same for `_q2`
- Both consume flat state vector — no structure awareness

### Gradient Flow Path
- Training: `states (from replay buffer) → actor.sample() → critic() → loss.backward()`
- Actor optimizer: `Adam(actor.parameters(), lr=3e-4)`
- If encoder is added: encoder params must be in actor optimizer so gradients flow through

### Checkpoint Format
- `SACTrainer.save()`: serializes actor/critic/target_critic/optimizers/alpha state dicts
- `SACTrainer.load()`: reconstructs from `(data_bytes, state_dim, action_dim)`
- Encoder state dict must be included in checkpoint

## Design Decision: Multihead Attention (MHA) Encoder

### Why not Set Transformer (ISAB)?
- ISAB uses inducing points for O(n) complexity, but our entity set is bounded at E=50 — O(n²) standard attention is fine (50² = 2500 operations, trivial)
- Standard MHA is simpler, well-understood, fewer hyperparameters
- PyTorch native `nn.MultiheadAttention` — no extra dependencies

### Architecture: Permutation-Invariant Attention Encoder

```
Input: variable-length set of entity feature vectors
  entity_i = [surprise(5) ; belief(4)] = 9-dim

Step 1: Entity Embedding
  h_i = ReLU(Linear(9 → entity_embed_dim))    per entity

Step 2: Self-Attention
  [h₁, ..., h_K] → MultiheadAttention → [z₁, ..., z_K]
  Use learnable [CLS] token prepended: [CLS, h₁, ..., h_K]
  z_CLS = entity summary (fixed dim regardless of K)

Step 3: Concatenate global features
  state = [z_CLS ; market_features ; entity_count ; adversarial_summary]

Output dim: entity_embed_dim + M + 1 + 4
```

### Key Design Choices

1. **Learnable [CLS] token** — standard approach from BERT/ViT for set→vector pooling. Alternative: mean-pool over entity tokens (simpler but less expressive).

2. **No instrument block in encoder** — the instrument block in InstrumentStateAssembler duplicates entity data (instruments are entities). The encoder naturally handles instruments as entities with attention. Instrument identity can be encoded via ticker embedding if needed later.

3. **Padding mask** — entities beyond n_active are padding; attention mask prevents them from contributing. PyTorch MHA supports `key_padding_mask`.

4. **Cold-start fallback** — before encoder is trained (no checkpoint), fall back to `StateAssembler.assemble()` to produce a flat state. This requires maintaining backward compatibility with old state_dim for existing SAC checkpoints.

### Hyperparameters
- `entity_embed_dim`: 32 (small — 50 entities × 9 features is not high-dimensional)
- `n_heads`: 4 (entity_embed_dim must be divisible by n_heads)
- `n_attention_layers`: 1 (single layer sufficient for 50 entities; more would overfit)
- `dropout`: 0.1

### Output State Dim
- `entity_embed_dim + market_dim + 1 + 4 = 32 + 8 + 1 + 4 = 45`
- Much smaller than current 463 — SAC now gets a compact, learned representation
- Trade-off: richer per-entity representation → compressed bottleneck → SAC MLP

## Integration Plan

### In SACTrainer
1. Add optional `encoder: LearnedStateEncoder | None` parameter
2. If encoder is present, include its parameters in `_actor_optim`
3. In `update()`: run states through encoder before passing to actor/critic
4. In `save()`/`load()`: include encoder state dict

### In Training DAG
1. When creating SACTrainer, also create encoder
2. Pass raw entity features (not assembled state) to replay buffer — OR: store assembled state and re-encode at training time
3. **Decision: store pre-encoded state in buffer** — simpler, avoids storing variable-length entity lists. The encoder is applied during `assemble()` to produce a fixed-dim state, which goes into the buffer. Gradients still flow because encoder is called freshly at each update step on batch states.
4. **Wait — that breaks gradient flow.** Buffer stores detached tensors. To get gradients through the encoder, we need to re-encode at training time.

### Gradient Flow Solution
**Approach: Two-phase encoding**
1. At collect time (inference/rollout): `encoder.encode(raw_inputs) → state_tensor` (detached, stored in buffer)
2. At train time: encode raw inputs again with gradients enabled
3. **Problem:** buffer only stores (state, action, reward, next_state, done) — no raw inputs

**Simpler approach: Treat encoder as part of the assembler, NOT as an nn.Module in the SAC graph**
- Encoder runs at assemble time, produces a fixed-dim state
- State goes into buffer (no gradients — standard off-policy pattern)
- Encoder is still learnable, but trained with an auxiliary loss (reconstruction, contrastive) rather than policy gradient
- **This matches how CURL, DrQ, SPR work** — representation learning + RL, but decoder doesn't get RL gradients directly

**Even simpler: End-to-end with replay recomputation**
- Store raw entity features in a separate buffer alongside (state, action, reward, ...)
- At training time, recompute encoder(raw_features) and use that as state input to actor/critic
- Gradients flow: critic_loss → actor → encoder

This requires expanding the replay buffer to store raw features. But it's the cleanest way to get end-to-end gradients.

### Final Architecture Decision

**Chosen approach: Encoder wraps the assembler, produces a new state_dim output. Train end-to-end by storing raw features alongside transitions in the replay buffer.**

However, this is complex integration. The spec calls for this being the highest-effort Tier 4 change.

**Pragmatic first step:** Implement the encoder as a module that transforms the assembler output (463-dim or N*5+463-dim) into a compact learned representation. The encoder input is the existing flat state vector. This avoids changing the replay buffer format while still providing learned representation.

```
existing_state (463-dim) → LearnedStateEncoder → compact_state (45-dim) → SAC actor/critic
```

Gradients flow through: `SAC loss → actor MLP → encoder → (stop here, state is from buffer)`

**Wait — same problem.** Buffer stores the 463-dim state. Encoder runs on it at train time. But the 463-dim state is detached. So encoder gradients from SAC loss DO flow through encoder weights, even though they don't flow back to the data. This is fine — the encoder learns to extract useful features from the fixed-format state.

**This is exactly how feature extraction networks work in standard deep RL** (e.g., CNN encoder in Atari). The encoder is part of the policy network, takes the state from the buffer, and is trained end-to-end with policy gradients. The state in the buffer is just pixels/raw features — the encoding IS the learned part.

### Final Design

```
            ┌─────────────────────────────────────────────┐
            │            LearnedStateEncoder               │
            │                                              │
raw_state   │  entity_block → emb → self-attn → z_CLS     │
(463-dim)   │                                              │  → compact
from buffer │  [z_CLS ; market ; count ; adv]              │    state
            │                                              │    (45-dim)
            └─────────────────────────────────────────────┘
                              │
                    actor/critic MLP
                              │
                         SAC loss
```

- Buffer stores 463-dim state (no change to ReplayBuffer)
- Encoder is part of SACTrainer, runs at train AND inference time
- Gradients flow: loss → actor MLP → encoder weights
- Encoder learns which entity features matter via attention

## Risks

1. **Encoder overfitting** — 50 entities × 9 features = small input. Single attention layer + low embed_dim mitigates
2. **State dim mismatch** — old checkpoints have actor weights for 463-dim input. Encoder changes this to 45-dim. Need version flag in checkpoint
3. **Cold start** — no encoder in old checkpoints → fall back to no-encoder (MLP takes 463-dim directly)
4. **Training/inference assembler mismatch** — currently training uses StateAssembler (no instrument block), inference uses InstrumentStateAssembler. Encoder must handle both or we need to standardize

## Trusted References

- Vaswani et al. 2017. "Attention Is All You Need." — MultiheadAttention architecture
- Lee et al. ICML 2019. "Set Transformer: A Framework for Attention-based Permutation-Invariant Input." — set encoding with attention
- Devlin et al. 2019. "[CLS] token pattern" from BERT for sequence→vector
- Standard deep RL with CNN encoder (Mnih et al. 2015, Nature DQN) — encoder as learnable part of policy network, buffer stores raw observations

## Related

- [[learned_vs_handcoded_architecture_spec]]
- [[learned_architecture_impl]]
- [[tier4_learn_dag_structure]]
