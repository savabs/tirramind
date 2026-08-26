"""Regression test for scripts/retrain_gnn.py CLI defaults.

LESSONS.md F-02: use_concat_head defaults to False on the TrainerConfig
dataclass, so the return-prediction path silently bypasses the GNN embedding
entirely (raw-price-only linear head) while loss still converges and looks
fine. The prevention rule ("print the active branch") only tells you AFTER
a run that this happened — it does not stop a bare invocation of the actual
training entrypoint from regressing into the bypass again.

This test locks in that scripts/retrain_gnn.py's CLI — the real training
entrypoint used on Kaggle — defaults --use-concat-head to True, so a bare
`python scripts/retrain_gnn.py` (no flags) does not silently reintroduce the
GNN bypass. --no-use-concat-head remains available for deliberate ablations.
"""

from __future__ import annotations

from scripts.retrain_gnn import build_arg_parser


class TestUseConcatHeadDefault:
    def test_bare_invocation_defaults_use_concat_head_true(self):
        """No flags at all — the common case for a fresh training run —
        must NOT silently fall back to the GNN-bypassing raw head."""
        args = build_arg_parser().parse_args([])
        assert args.use_concat_head is True

    def test_explicit_flag_still_works(self):
        args = build_arg_parser().parse_args(["--use-concat-head"])
        assert args.use_concat_head is True

    def test_no_use_concat_head_opts_out_explicitly(self):
        """Deliberate ablations (raw-head-only / embedding-only) must still
        be reachable by an explicit flag."""
        args = build_arg_parser().parse_args(["--no-use-concat-head"])
        assert args.use_concat_head is False
