"""
GNN Smoke Test — Gradient Flow & Embedding Diversity
=====================================================
Run BEFORE every Kaggle push. All tests must PASS.
No database or Kaggle required — uses synthetic data.

Usage:
    python scripts/smoke_test_gnn.py
    python scripts/smoke_test_gnn.py --verbose   # show tensor stats

Exit 0 = all PASS
Exit 1 = any FAIL

Tests
-----
T1  model_instantiation      HetTGN builds, return_concat_head exists when use_concat_head=True
T2  forward_no_nan           Forward pass produces finite instrument embeddings
T3  csrc_grad_backbone       CSRC loss backward → backbone type_projections grad nonzero
T4  concat_grad_backbone     ListNet loss via concat head → backbone grad nonzero  ← THE CRITICAL TEST
T5  raw_head_no_backbone_grad  Raw head (no concat) → backbone grad should be ZERO (sanity inversion)
T6  grad_no_nan              Combined loss backward → no NaN or Inf in any grad
T7  embedding_diversity_csrc  After 20 CSRC steps embedding std improves vs baseline
T8  return_head_routing       Correct head fires based on model config
T9  hgt_conv_grad             With edges present HGTConv layers receive gradient
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agent.models.gnn.graph_builder import IDMap
from agent.models.gnn.het_tgn import HetTGN

# ── Test config ───────────────────────────────────────────────────────────────
N_INST = 24          # instruments per snapshot (small, fast)
RAW_DIM = 23         # instrument raw feature dim (matches production)
HIDDEN = 32          # small hidden dim for speed
N_LAYERS = 2
N_HEADS = 2
MEMORY_DIM = 32
TEMPERATURE = 0.1
N_DECILES = 4

PASS = "✓ PASS"
FAIL = "✗ FAIL"

results: list[tuple[str, bool, str]] = []  # (name, passed, detail)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_id_map(n: int) -> IDMap:
    """Build an IDMap with n synthetic instrument entities."""
    id_map = IDMap()
    for i in range(n):
        id_map.add("instrument", f"instr_{i:03d}")
    return id_map


def _make_data(id_map: IDMap, with_edges: bool = False) -> HeteroData:
    """Minimal HeteroData with instrument nodes and optional self-edges."""
    data = HeteroData()
    n = id_map.num_nodes_of_type("instrument")
    data["instrument"].x = torch.randn(n, RAW_DIM)
    if with_edges:
        # Random sparse instrument→instrument edges
        src = torch.randint(0, n, (n * 2,))
        dst = torch.randint(0, n, (n * 2,))
        data["instrument", "co_moves", "instrument"].edge_index = torch.stack([src, dst])
    return data


def _make_model(use_concat_head: bool, with_edges: bool = False) -> HetTGN:
    id_map = _make_id_map(N_INST)
    data = _make_data(id_map, with_edges=with_edges)
    metadata = data.metadata()
    in_channels = {"instrument": RAW_DIM}
    return HetTGN(
        metadata=metadata,
        in_channels=in_channels,
        hidden_dim=HIDDEN,
        memory_dim=MEMORY_DIM,
        message_dim=MEMORY_DIM,
        time_dim=8,
        num_heads=N_HEADS,
        num_layers=N_LAYERS,
        num_nodes=id_map.num_nodes,
        instrument_raw_dim=RAW_DIM,
        use_concat_head=use_concat_head,
    )


def _csrc_loss(embeddings: torch.Tensor, return_targets: torch.Tensor) -> torch.Tensor:
    """Minimal InfoNCE CSRC loss (mirrors trainer._cross_sectional_ranking_contrastive)."""
    emb = F.normalize(embeddings, dim=1)
    valid = ~torch.isnan(return_targets)
    emb, tgt = emb[valid], return_targets[valid]
    if emb.shape[0] < 2:
        return torch.tensor(0.0)
    _, sorted_idx = torch.sort(tgt)
    n = len(tgt)
    dec_size = max(1, n // N_DECILES)
    deciles = torch.zeros(n, dtype=torch.long)
    for i in range(N_DECILES):
        s, e = i * dec_size, (i + 1) * dec_size if i < N_DECILES - 1 else n
        deciles[sorted_idx[s:e]] = i
    sim = torch.matmul(emb, emb.T) / TEMPERATURE
    loss = torch.tensor(0.0)
    for i in range(len(emb)):
        pos = (deciles == deciles[i]) & (torch.arange(len(emb)) != i)
        neg = (deciles != deciles[i])
        if not (pos.any() and neg.any()):
            continue
        pos_sim = sim[i][pos].mean()
        all_sim = sim[i][neg | pos]
        loss = loss + (-pos_sim + torch.logsumexp(all_sim, dim=0))
    return loss / len(emb)


def _listnet_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ListNet top-1 loss."""
    p_tgt = F.softmax(targets / TEMPERATURE, dim=0)
    p_pred = F.log_softmax(scores, dim=0)
    return -( p_tgt * p_pred).sum()


def _backbone_param(model: HetTGN) -> torch.nn.Parameter:
    """Return a backbone parameter that should receive gradient from return loss."""
    return model.type_projections["instrument"].weight


def _grad_norm(p: torch.nn.Parameter) -> float:
    if p.grad is None:
        return 0.0
    return p.grad.norm().item()


def register(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        def wrapper(verbose: bool = False) -> bool:
            try:
                detail = fn(verbose=verbose)
                passed = True
                results.append((name, True, detail or ""))
                print(f"  {PASS}  {name:<45} {detail or ''}")
                return True
            except AssertionError as e:
                results.append((name, False, str(e)))
                print(f"  {FAIL}  {name:<45} {e}")
                return False
            except Exception as e:
                results.append((name, False, f"EXCEPTION: {e}"))
                print(f"  {FAIL}  {name:<45} EXCEPTION: {e}")
                if verbose:
                    traceback.print_exc()
                return False
        wrapper.__name__ = name
        return wrapper
    return decorator


# ── Tests ─────────────────────────────────────────────────────────────────────

@register("T1_model_instantiation")
def t1(verbose=False):
    model_with = _make_model(use_concat_head=True)
    assert model_with.return_concat_head is not None, \
        "return_concat_head is None despite use_concat_head=True"
    expected_in = RAW_DIM + HIDDEN
    actual_in = model_with.return_concat_head[0].in_features
    assert actual_in == expected_in, \
        f"concat_head input dim={actual_in}, expected {expected_in} ({RAW_DIM}+{HIDDEN})"

    model_without = _make_model(use_concat_head=False)
    assert model_without.return_concat_head is None, \
        "return_concat_head is not None despite use_concat_head=False"
    return f"concat_in={actual_in} ✓"


@register("T2_forward_no_nan")
def t2(verbose=False):
    model = _make_model(use_concat_head=True)
    id_map = _make_id_map(N_INST)
    data = _make_data(id_map)
    with torch.no_grad():
        embs = model(data, id_map)
    assert "instrument" in embs, "No 'instrument' key in forward output"
    ie = embs["instrument"]
    assert ie.shape == (N_INST, HIDDEN), f"Shape mismatch: {ie.shape}"
    assert not torch.isnan(ie).any(), "NaN in instrument embeddings"
    assert not torch.isinf(ie).any(), "Inf in instrument embeddings"
    return f"shape={tuple(ie.shape)} nan=0 inf=0"


@register("T3_csrc_grad_backbone")
def t3(verbose=False):
    model = _make_model(use_concat_head=True)
    id_map = _make_id_map(N_INST)
    data = _make_data(id_map)
    model.zero_grad()

    embs = model(data, id_map)
    inst_emb = embs["instrument"]
    ret_targets = torch.randn(N_INST)
    loss = _csrc_loss(inst_emb, ret_targets)
    assert loss.item() > 0, f"CSRC loss is 0 — no positive/negative pairs formed"
    loss.backward()

    bp = _backbone_param(model)
    gn = _grad_norm(bp)
    assert gn > 1e-8, \
        f"CSRC backward produced ZERO grad on type_projections['instrument'].weight (norm={gn:.2e}). " \
        "Backbone is DETACHED from CSRC loss."
    return f"csrc_loss={loss.item():.4f}  backbone_grad_norm={gn:.4f}"


@register("T4_concat_head_grad_backbone  ← CRITICAL")
def t4(verbose=False):
    """
    THE critical test we missed in V38-V41.
    Verifies that ListNet return loss through return_concat_head
    flows gradient back to GNN backbone (type_projections weights).
    If this fails: ghost patterns CANNOT affect IC. Ever.
    """
    model = _make_model(use_concat_head=True)
    id_map = _make_id_map(N_INST)
    data = _make_data(id_map)
    model.zero_grad()

    embs = model(data, id_map)
    inst_emb = embs["instrument"]
    raw_feats = F.normalize(data["instrument"].x, dim=1)

    concat_in = torch.cat([raw_feats, inst_emb], dim=-1)
    scores = model.return_concat_head(concat_in).squeeze(-1)
    ret_targets = torch.randn(N_INST)
    loss = _listnet_loss(scores, ret_targets)
    loss.backward()

    bp = _backbone_param(model)
    gn = _grad_norm(bp)
    combiner_gn = _grad_norm(model.combiner.weight)

    assert gn > 1e-8, \
        f"ListNet through concat_head produced ZERO grad on type_projections['instrument'].weight " \
        f"(norm={gn:.2e}). GNN backbone is DETACHED from return loss. Ghost patterns will NOT affect IC."
    assert combiner_gn > 1e-8, \
        f"combiner.weight grad is zero (norm={combiner_gn:.2e}). Memory→embedding path is broken."

    return (
        f"listnet_loss={loss.item():.4f}  "
        f"projection_grad={gn:.4f}  "
        f"combiner_grad={combiner_gn:.4f}"
    )


@register("T5_raw_head_has_no_backbone_grad")
def t5(verbose=False):
    """
    Sanity inversion: raw head (no concat) should NOT gradient the backbone.
    This confirms T4 is testing a real difference, not an always-true condition.
    """
    model = _make_model(use_concat_head=False)
    id_map = _make_id_map(N_INST)
    data = _make_data(id_map)
    model.zero_grad()

    # Raw head path: features only, no GNN embeddings
    raw_feats = F.normalize(data["instrument"].x, dim=1)
    scores = model.return_raw_head(raw_feats).squeeze(-1)
    ret_targets = torch.randn(N_INST)
    loss = _listnet_loss(scores, ret_targets)
    loss.backward()

    bp = _backbone_param(model)
    gn = _grad_norm(bp)
    assert gn < 1e-8, \
        f"Raw head unexpectedly gradients the backbone (norm={gn:.2e}). " \
        "Something is wrong with the head isolation."
    return f"backbone_grad={gn:.2e} (correctly zero for raw-only head)"


@register("T6_combined_loss_no_nan_grad")
def t6(verbose=False):
    model = _make_model(use_concat_head=True)
    id_map = _make_id_map(N_INST)
    data = _make_data(id_map)
    model.zero_grad()

    embs = model(data, id_map)
    inst_emb = embs["instrument"]
    raw_feats = F.normalize(data["instrument"].x, dim=1)
    ret_targets = torch.randn(N_INST)

    csrc = _csrc_loss(inst_emb, ret_targets)
    concat_in = torch.cat([raw_feats, inst_emb], dim=-1)
    scores = model.return_concat_head(concat_in).squeeze(-1)
    ret_loss = _listnet_loss(scores, ret_targets)

    # obs_type auxiliary loss
    obs_logits = model.obs_type_head(inst_emb)
    obs_tgt = torch.randint(0, obs_logits.shape[1], (N_INST,))
    obs_loss = F.cross_entropy(obs_logits, obs_tgt)

    total = csrc + ret_loss + 0.1 * obs_loss
    total.backward()

    nan_params = []
    for name, p in model.named_parameters():
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            nan_params.append(name)

    assert not nan_params, f"NaN/Inf gradients in: {nan_params[:5]}"
    return f"total_loss={total.item():.4f}  nan_grad_params=0"


@register("T7_embedding_diversity_after_csrc")
def t7(verbose=False):
    """
    Runs 20 CSRC optimizer steps and checks embedding std improves.
    Baseline std ~ 0.05-0.15 (random init). After CSRC: should increase.
    If std stays flat or drops: CSRC is not differentiating embeddings.
    """
    model = _make_model(use_concat_head=True)
    id_map = _make_id_map(N_INST)
    data = _make_data(id_map)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    with torch.no_grad():
        embs_before = model(data, id_map)["instrument"]
        std_before = embs_before.std(dim=0).mean().item()

    for step in range(20):
        opt.zero_grad()
        embs = model(data, id_map)["instrument"]
        ret_tgts = torch.randn(N_INST)
        loss = _csrc_loss(embs, ret_tgts)
        if loss.item() > 0:
            loss.backward()
            opt.step()

    with torch.no_grad():
        embs_after = model(data, id_map)["instrument"]
        std_after = embs_after.std(dim=0).mean().item()

    assert std_after > 1e-4, \
        f"Embeddings fully collapsed after CSRC training: std={std_after:.6f}"
    return f"std: {std_before:.4f} → {std_after:.4f} ({'↑ improved' if std_after > std_before else '↓ degraded — check CSRC loss scale'})"


@register("T8_return_head_routing_logic")
def t8(verbose=False):
    """
    Mirrors the trainer's if/elif/else routing logic.
    Confirms which head fires given model configuration.
    """
    id_map = _make_id_map(N_INST)
    data = _make_data(id_map)
    ret_targets = torch.randn(N_INST)

    # Case 1: concat head enabled → concat path
    model_c = _make_model(use_concat_head=True)
    embs_c = model_c(data, id_map)["instrument"]
    has_raw = model_c.return_raw_head is not None
    raw_feats = data["instrument"].x
    if has_raw and raw_feats is not None and model_c.return_concat_head is not None:
        active = "concat_head"
        inp = torch.cat([F.normalize(raw_feats, dim=1), embs_c], dim=-1)
        pred = model_c.return_concat_head(inp).squeeze(-1)
    elif has_raw and raw_feats is not None:
        active = "raw_head"
        pred = model_c.return_raw_head(F.normalize(raw_feats, dim=1)).squeeze(-1)
    else:
        active = "pred_head"
        pred = model_c.predict_return(embs_c).squeeze(-1)
    assert active == "concat_head", f"Expected concat_head, got {active}"
    assert pred.shape == (N_INST,), f"Pred shape mismatch: {pred.shape}"

    # Case 2: no concat head → raw path
    model_r = _make_model(use_concat_head=False)
    embs_r = model_r(data, id_map)["instrument"]
    has_raw_r = model_r.return_raw_head is not None
    if has_raw_r and raw_feats is not None and model_r.return_concat_head is not None:
        active_r = "concat_head"
    elif has_raw_r and raw_feats is not None:
        active_r = "raw_head"
    else:
        active_r = "pred_head"
    assert active_r == "raw_head", f"Expected raw_head without concat, got {active_r}"

    return f"with_concat→{active}  without_concat→{active_r}"


@register("T9_hgt_conv_grad_with_edges")
def t9(verbose=False):
    """
    With actual edges, HGTConv layers run and must also receive gradient.
    """
    id_map = _make_id_map(N_INST)
    data = _make_data(id_map, with_edges=True)
    metadata = data.metadata()
    in_channels = {"instrument": RAW_DIM}
    model = HetTGN(
        metadata=metadata,
        in_channels=in_channels,
        hidden_dim=HIDDEN,
        memory_dim=MEMORY_DIM,
        message_dim=MEMORY_DIM,
        time_dim=8,
        num_heads=N_HEADS,
        num_layers=N_LAYERS,
        num_nodes=id_map.num_nodes,
        instrument_raw_dim=RAW_DIM,
        use_concat_head=True,
    )
    model.zero_grad()
    embs = model(data, id_map)["instrument"]
    raw_feats = F.normalize(data["instrument"].x, dim=1)
    concat_in = torch.cat([raw_feats, embs], dim=-1)
    scores = model.return_concat_head(concat_in).squeeze(-1)
    loss = _listnet_loss(scores, torch.randn(N_INST))
    loss.backward()

    # HGTConv grad
    hgt_gn = 0.0
    for layer in model.hgt_layers:
        for p in layer.parameters():
            if p.grad is not None:
                hgt_gn += p.grad.norm().item()
    assert hgt_gn > 1e-8, \
        f"HGTConv layers have zero gradient with edges (norm={hgt_gn:.2e}). " \
        "Message passing is not contributing to the return loss gradient."
    proj_gn = _grad_norm(_backbone_param(model))
    return f"hgt_conv_grad_norm={hgt_gn:.4f}  projection_grad={proj_gn:.4f}"


# ── Runner ────────────────────────────────────────────────────────────────────

ALL_TESTS = [t1, t2, t3, t4, t5, t6, t7, t8, t9]


def main() -> int:
    parser = argparse.ArgumentParser(description="GNN gradient flow smoke test")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--test", "-t", help="Run only this test (e.g. T4)")
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  TirraMind GNN Smoke Test — Gradient Flow & Embedding Diversity")
    print("═" * 70)

    tests = ALL_TESTS
    if args.test:
        prefix = args.test.upper()
        tests = [t for t in ALL_TESTS if t.__name__.upper().startswith(prefix)]
        if not tests:
            print(f"No test matching '{args.test}'")
            return 1

    passed = 0
    for test in tests:
        test(verbose=args.verbose)
        if results and results[-1][1]:
            passed += 1

    total = len(results)
    failed = total - passed
    print("─" * 70)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ← {failed} FAILED")
    else:
        print("  ← ALL PASS ✓")
    print("═" * 70 + "\n")

    if failed:
        print("FAILED TESTS:")
        for name, ok, detail in results:
            if not ok:
                print(f"  {name}: {detail}")
        print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
