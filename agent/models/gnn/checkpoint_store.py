"""Immutable, versioned checkpoint management for the HetTGN.

CLAUDE.md §5: *"Checkpoints are immutable once created. Create a new one for each
major run."* The `gnn_inference` DAG violated this twice over — it saved straight
over `.tirra_pipeline/gnn_model.pt` on every run, and on a high-changepoint it
called `model_path.unlink()`, destroying the artifact outright.

That is how the May 25 checkpoint ended up with `in_channels['instrument'] = 49`
recorded against weights of width 23: a later run rewrote the metadata in place
while the trained tensors stayed put, leaving a file that contradicted itself.

Model layer (L3) — artifact lifecycle for the world model, no pipeline concerns.

Layout::

    .tirra_pipeline/
        gnn_model.pt                     <- stable pointer, what consumers read
        checkpoints/
            gnn_model_20260826T041500.pt <- immutable, never rewritten
            gnn_model_20260826T190000.pt
            archived_20260826T041500_gnn_model.pt   <- superseded, not deleted

Never write through an existing file in ``checkpoints/``.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

log = logging.getLogger(__name__)

_ARCHIVE_DIRNAME = "checkpoints"


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def checkpoint_dir(current_path: str | Path) -> Path:
    """Directory holding immutable checkpoints, beside the stable pointer."""
    return Path(current_path).parent / _ARCHIVE_DIRNAME


def archive_checkpoint(current_path: str | Path) -> Path | None:
    """Move an existing checkpoint into the archive instead of destroying it.

    Replaces ``model_path.unlink()``. A checkpoint that a regime shift made
    stale is still the only record of what the model looked like before that
    shift — deleting it makes the change unauditable.

    Returns the archived path, or None if there was nothing to archive.
    """
    current = Path(current_path)
    if not current.exists():
        return None

    dest_dir = checkpoint_dir(current)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"archived_{_timestamp()}_{current.name}"

    # Never clobber an existing archive entry.
    n = 1
    while dest.exists():
        dest = dest_dir / f"archived_{_timestamp()}_{n}_{current.name}"
        n += 1

    shutil.move(str(current), str(dest))
    log.info("Archived checkpoint %s -> %s (not deleted).", current.name, dest.name)
    return dest


def save_versioned(trainer, current_path: str | Path, *, label: str | None = None) -> Path:
    """Write a NEW immutable checkpoint, then repoint the stable path at it.

    The stable path (``gnn_model.pt``) is a convenience for consumers; the
    versioned file under ``checkpoints/`` is the artifact of record and is never
    written through.

    Args:
        trainer: a Trainer with a built/trained model (must expose save_model).
        current_path: the stable pointer path consumers read.
        label: optional suffix for the versioned filename (e.g. a run id).

    Returns:
        Path to the immutable versioned checkpoint.
    """
    current = Path(current_path)
    dest_dir = checkpoint_dir(current)
    dest_dir.mkdir(parents=True, exist_ok=True)

    stem = current.stem
    suffix = f"_{label}" if label else ""
    versioned = dest_dir / f"{stem}_{_timestamp()}{suffix}{current.suffix}"

    n = 1
    while versioned.exists():  # never overwrite an immutable artifact
        versioned = dest_dir / f"{stem}_{_timestamp()}{suffix}_{n}{current.suffix}"
        n += 1

    trainer.save_model(versioned)
    log.info("Wrote immutable checkpoint %s.", versioned.name)

    # Archive whatever was previously at the stable path BEFORE overwriting
    # it — never unlink() it directly. This must happen before the copy
    # below: archiving after the copy would just archive a duplicate of the
    # NEW checkpoint we already wrote above, losing the OLD one instead of
    # preserving it. Do not assume `current` is already backed up under
    # checkpoints/ (e.g. a caller may have written straight to the stable
    # path before ever going through save_versioned) — archive_checkpoint()
    # is a no-op (returns None) when there's nothing there, so this is safe
    # either way.
    if current.exists():
        archive_checkpoint(current)
    shutil.copy2(versioned, current)
    log.info("Repointed %s -> %s.", current.name, versioned.name)

    return versioned


def list_checkpoints(current_path: str | Path) -> list[Path]:
    """All immutable checkpoints, newest first."""
    d = checkpoint_dir(current_path)
    if not d.exists():
        return []
    return sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)


__all__ = [
    "archive_checkpoint",
    "checkpoint_dir",
    "list_checkpoints",
    "save_versioned",
]
