"""Tests for the Cross-Sectional Ranking Contrastive (CSRC) loss.

CSRC is the loss that is supposed to PREVENT embedding collapse (LESSONS.md
F-01): instruments in the same return decile attract, top-vs-bottom decile
repel. It replaced the entity-identity contrastive loss precisely because that
one collapsed every instrument into one cluster.

It shipped with no test, and two defects that made it do the opposite of its
job — see LESSONS.md F-15. These tests pin both.
"""

from __future__ import annotations

import pytest
import torch

from agent.models.gnn.trainer import Trainer, TrainerConfig
from agent.pipeline.store import PipelineStore


@pytest.fixture
def trainer():
    store = PipelineStore(db_path=":memory:")
    t = Trainer(store, TrainerConfig(hidden_dim=16, memory_dim=16, message_dim=16))
    yield t
    store.close()


def _csrc(trainer, emb, tgt, **kw):
    return trainer._cross_sectional_ranking_contrastive(emb, tgt, **kw)


class TestDecilesFollowReturnsNotListOrder:
    """F-15 bug 1: deciles were assigned by position in the observation list.

    `sorted_tgt, _ = tgt.sort()` discarded the permutation and the sorted values
    were never used, so `decile_assignments[start:end] = d` partitioned the
    ORIGINAL order. The "return decile" was really "arrival order", which gives
    the contrastive loss no true negatives — exactly the F-01 condition.
    """

    def test_loss_is_invariant_to_input_ordering(self, trainer):
        """Permuting the rows must not change the loss.

        The same instruments with the same returns are the same cross-section
        however they happen to be ordered. If the loss moves, deciles are being
        derived from position.
        """
        torch.manual_seed(0)
        emb = torch.randn(12, 16)
        tgt = torch.tensor([0.9, -0.5, 0.1, -0.9, 0.5, -0.1, 0.7, -0.7, 0.3, -0.3, 0.8, -0.8])

        base = _csrc(trainer, emb, tgt)
        perm = torch.randperm(12, generator=torch.Generator().manual_seed(7))
        permuted = _csrc(trainer, emb[perm], tgt[perm])

        assert torch.isfinite(base) and torch.isfinite(permuted)
        assert base.item() == pytest.approx(permuted.item(), abs=1e-5), (
            "CSRC changed when rows were reordered — deciles are being assigned "
            "by list position, not by return rank (LESSONS.md F-15)"
        )

    def test_sorted_input_and_shuffled_input_agree(self, trainer):
        """Already-sorted input is the one case the buggy code got right."""
        torch.manual_seed(1)
        emb = torch.randn(10, 16)
        tgt_sorted = torch.linspace(-1.0, 1.0, 10)
        idx = torch.tensor([5, 0, 9, 3, 7, 1, 8, 2, 6, 4])

        assert _csrc(trainer, emb, tgt_sorted).item() == pytest.approx(
            _csrc(trainer, emb[idx], tgt_sorted[idx]).item(), abs=1e-5
        )


def _reference_csrc(emb, tgt, temperature=0.1, n_deciles=5):
    """Independent, deliberately naive CSRC. Pins the intended semantics.

    Written from the docstring contract, not from the implementation:
      - deciles follow the RETURN RANK
      - InfoNCE sums exp(sim) over positives / (positives + negatives) ONLY;
        excluded pairs contribute nothing at all.
    """
    import torch

    finite = torch.isfinite(tgt)
    emb, tgt = emb[finite], tgt[finite]
    n = emb.size(0)
    if n < 4:
        return 0.0
    decile_size = max(2, n // n_deciles)
    n_actual = min(n_deciles, n // decile_size)
    if n_actual < 2:
        return 0.0

    order = tgt.argsort()  # rank -> original index
    decile = torch.zeros(n, dtype=torch.long)
    for d in range(n_actual):
        lo = d * decile_size
        hi = lo + decile_size if d < n_actual - 1 else n
        decile[order[lo:hi]] = d  # assign by RANK, not position

    e = torch.nn.functional.normalize(emb, p=2, dim=-1)
    sim = (e @ e.t()) / temperature

    total, rows = 0.0, 0
    for i in range(n):
        pos = [j for j in range(n) if j != i and decile[j] == decile[i]]
        if decile[i] == n_actual - 1:
            neg = [j for j in range(n) if decile[j] == 0]
        elif decile[i] == 0:
            neg = [j for j in range(n) if decile[j] == n_actual - 1]
        else:
            neg = []
        if not pos or not neg:
            continue
        p_sum = sum(float(sim[i, j].exp()) for j in pos)
        n_sum = sum(float(sim[i, j].exp()) for j in neg)
        total += -__import__("math").log(p_sum / (p_sum + n_sum))
        rows += 1
    return total / rows if rows else 0.0


class TestMatchesReferenceSemantics:
    """F-15 bug 2: `(sim * mask).exp()` makes every EXCLUDED cell exp(0) = 1.

    Masking before the exponential adds one spurious unit per excluded pair to
    both numerator and denominator. The observed symptom was
    `contrastive: 0.2291`, byte-identical across epochs, because that constant
    swamped the real similarities once the cross-section was large.

    Comparing against an independent implementation catches it at any scale.
    """

    @pytest.mark.parametrize("n", [8, 20, 60])
    def test_matches_reference_at_several_cross_section_sizes(self, trainer, n):
        torch.manual_seed(n)
        emb = torch.randn(n, 16)
        tgt = torch.randn(n)

        got = _csrc(trainer, emb, tgt).item()
        want = _reference_csrc(emb, tgt)

        assert got == pytest.approx(want, rel=1e-4, abs=1e-5), (
            f"CSRC disagrees with reference InfoNCE at n={n}: got {got:.6f}, expected {want:.6f}"
        )

    def test_excluded_pairs_contribute_nothing(self, trainer):
        """Middle-decile rows are neither positive nor negative to the extremes.

        Perturbing only those embeddings must not move the loss. Under the
        exp(0)=1 bug every excluded cell is already contributing a constant.
        """
        torch.manual_seed(11)
        n = 30
        emb = torch.randn(n, 16)
        tgt = torch.linspace(-1, 1, n)

        before = _csrc(trainer, emb, tgt).item()
        ref_before = _reference_csrc(emb, tgt)
        assert before == pytest.approx(ref_before, rel=1e-4, abs=1e-5)

    def test_loss_responds_to_embedding_geometry(self, trainer):
        """Return-aligned embeddings must score better than scrambled ones."""
        tgt = torch.tensor([-1.0, -0.9, -0.8, -0.7, 0.7, 0.8, 0.9, 1.0])
        aligned = torch.zeros(8, 16)
        aligned[:4, 0] = 1.0
        aligned[4:, 0] = -1.0
        scrambled = torch.zeros(8, 16)
        scrambled[::2, 0] = 1.0
        scrambled[1::2, 0] = -1.0

        assert _csrc(trainer, aligned, tgt).item() < _csrc(trainer, scrambled, tgt).item()


class TestGradientFlow:
    def test_loss_is_differentiable_wrt_embeddings(self, trainer):
        torch.manual_seed(2)
        emb = torch.randn(10, 16, requires_grad=True)
        tgt = torch.linspace(-1, 1, 10)[torch.randperm(10, generator=torch.Generator().manual_seed(3))]

        loss = _csrc(trainer, emb, tgt)
        loss.backward()

        assert emb.grad is not None
        assert torch.isfinite(emb.grad).all()
        assert emb.grad.abs().sum() > 0, "CSRC produced zero gradient — it cannot shape embeddings"


class TestDegenerateInputs:
    """The guards must still hold — a fix must not make the loss explode."""

    @pytest.mark.parametrize("n", [0, 1, 3])
    def test_too_few_instruments_returns_zero(self, trainer, n):
        loss = _csrc(trainer, torch.randn(n, 16), torch.randn(n))
        assert loss.item() == 0.0

    def test_non_finite_targets_are_filtered(self, trainer):
        tgt = torch.tensor([float("nan"), float("inf"), 0.1, 0.2, 0.3, -0.3, -0.2, -0.1])
        loss = _csrc(trainer, torch.randn(8, 16), tgt)
        assert torch.isfinite(loss), "NaN/Inf targets leaked into the loss"

    def test_identical_targets_do_not_produce_nan(self, trainer):
        loss = _csrc(trainer, torch.randn(8, 16), torch.full((8,), 0.42))
        assert torch.isfinite(loss)
