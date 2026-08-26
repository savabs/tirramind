"""Tests for immutable, versioned GNN checkpoint management.

CLAUDE.md §5: checkpoints are immutable once created. The `gnn_inference` DAG
previously saved straight over `.tirra_pipeline/gnn_model.pt` on every run and
called `model_path.unlink()` on a high-changepoint. That in-place rewriting is
how a checkpoint ended up recording `in_channels['instrument'] = 49` against
trained weights of width 23 — metadata from a later run, tensors from an
earlier one, in the same file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent.models.gnn.checkpoint_store import (
    archive_checkpoint,
    checkpoint_dir,
    list_checkpoints,
    save_versioned,
)


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class _FakeTrainer:
    """Minimal stand-in — save_model is the only surface used."""

    def __init__(self, payload: str = "weights-v1") -> None:
        self.payload = payload
        self.saved_to: list[Path] = []

    def save_model(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.payload)
        self.saved_to.append(p)


@pytest.fixture()
def stable(tmp_path: Path) -> Path:
    return tmp_path / "gnn_model.pt"


class TestSaveVersioned:
    def test_creates_versioned_file_in_checkpoint_dir(self, stable):
        v = save_versioned(_FakeTrainer(), stable)
        assert v.exists()
        assert v.parent == checkpoint_dir(stable)
        assert v != stable

    def test_repoints_stable_path(self, stable):
        v = save_versioned(_FakeTrainer("payload-A"), stable)
        assert stable.exists()
        assert stable.read_text() == "payload-A"
        assert _digest(stable) == _digest(v)

    def test_second_save_does_not_touch_first_artifact(self, stable):
        first = save_versioned(_FakeTrainer("payload-A"), stable)
        first_digest = _digest(first)

        save_versioned(_FakeTrainer("payload-B"), stable)

        assert first.exists(), "first immutable checkpoint was deleted"
        assert _digest(first) == first_digest, "first checkpoint was written through"

    def test_two_saves_produce_two_distinct_checkpoints(self, stable):
        first = save_versioned(_FakeTrainer("payload-A"), stable, label="run1")
        second = save_versioned(_FakeTrainer("payload-B"), stable, label="run2")
        assert first != second
        assert len(list_checkpoints(stable)) >= 2

    def test_stable_path_reflects_latest(self, stable):
        save_versioned(_FakeTrainer("payload-A"), stable)
        save_versioned(_FakeTrainer("payload-B"), stable)
        assert stable.read_text() == "payload-B"

    def test_preexisting_stable_file_is_archived_not_deleted(self, stable):
        """Regression: a file already sitting at the stable path when
        save_versioned() is first called — e.g. a direct trainer.save_model(
        current) from other calling code, before this module was ever used —
        must be archived before the pointer is repointed, not unlink()'d.

        This is the exact anti-pattern the module's docstring says it exists
        to prevent (`gnn_inference` used to save straight over the stable
        path and delete it outright). Nothing else backs this file up: it
        is not one of save_versioned's own immutable files, so silently
        unlinking it destroys the only copy.
        """
        stable.write_text("legacy-direct-save")

        save_versioned(_FakeTrainer("payload-A"), stable)

        archived = list(checkpoint_dir(stable).glob("archived_*"))
        assert archived, "pre-existing stable-path file was deleted, not archived"
        assert any(
            p.read_text() == "legacy-direct-save" for p in archived
        ), "archived file does not contain the pre-existing checkpoint's content"
        # And the new save still won.
        assert stable.read_text() == "payload-A"

    def test_archive_happens_before_repoint_not_after(self, stable):
        """The archived file must be the OLD checkpoint, never a duplicate of
        the NEW one (which would happen if archiving ran after the copy)."""
        stable.write_text("old-model")

        save_versioned(_FakeTrainer("new-model"), stable)

        archived = list(checkpoint_dir(stable).glob("archived_*"))
        assert archived
        contents = {p.read_text() for p in archived}
        assert "old-model" in contents, "archived the wrong file — old checkpoint was lost"
        assert "new-model" not in contents, (
            "archived a copy of the NEW checkpoint instead of the OLD one — "
            "archiving ran after the stable pointer was already overwritten"
        )


class TestArchiveCheckpoint:
    def test_archives_instead_of_deleting(self, stable):
        stable.write_text("the-old-model")
        archived = archive_checkpoint(stable)

        assert archived is not None
        assert archived.exists(), "checkpoint was destroyed instead of archived"
        assert archived.read_text() == "the-old-model"
        assert not stable.exists(), "stable path should be cleared for the retrain"

    def test_returns_none_when_nothing_to_archive(self, stable):
        assert archive_checkpoint(stable) is None

    def test_repeated_archive_does_not_clobber(self, stable):
        stable.write_text("model-1")
        a1 = archive_checkpoint(stable)
        stable.write_text("model-2")
        a2 = archive_checkpoint(stable)

        assert a1 != a2
        assert a1.read_text() == "model-1", "first archive was overwritten"
        assert a2.read_text() == "model-2"


class TestNoInPlaceMutation:
    """The specific regression: nothing may write through an existing checkpoint."""

    def test_archived_file_survives_subsequent_saves(self, stable):
        stable.write_text("original-may-25-model")
        archived = archive_checkpoint(stable)
        original_digest = _digest(archived)

        for payload in ("run-1", "run-2", "run-3"):
            save_versioned(_FakeTrainer(payload), stable)

        assert _digest(archived) == original_digest, "an archived checkpoint was mutated by a later run"
