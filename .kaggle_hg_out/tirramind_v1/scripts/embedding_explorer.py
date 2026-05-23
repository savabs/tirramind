"""
scripts/embedding_explorer.py
─────────────────────────────
TirraMind — GNN Embedding Space Visualizer

Extracts node embeddings from the trained HetTGN model, reduces them to 2D
using UMAP (t-SNE fallback), and produces an interactive Plotly HTML report.

The embedding space is where the ghost patterns live.  This tool lets you see
whether the GNN has learned structured geometry — do instruments cluster by
commodity class?  Do countries group by geopolitical region?  Do certain
combinations of instrument + country sit in regions that precede large moves?

Usage:
    python3 scripts/embedding_explorer.py
    python3 scripts/embedding_explorer.py --model .tirra_pipeline/gnn_model_h_g.pt
    python3 scripts/embedding_explorer.py --output reports/my_embedding.html
    python3 scripts/embedding_explorer.py --types instrument cftc_contract country
    python3 scripts/embedding_explorer.py --no-browser   # save HTML only

Dependencies (auto-installed if missing):
    umap-learn, plotly, scikit-learn
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any


# ── auto-install lightweight deps ─────────────────────────────────────────────
def _ensure(package: str, import_name: str | None = None) -> None:
    imp = import_name or package
    try:
        __import__(imp)
    except ImportError:
        print(f"[bootstrap] Installing {package} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", package],
            check=True,
        )


_ensure("umap-learn", "umap")
_ensure("plotly")
_ensure("scikit-learn", "sklearn")

# ── stdlib / project imports ──────────────────────────────────────────────────
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("embedding_explorer")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / ".tirra_pipeline" / "gnn_model.pt"
DEFAULT_DB = PROJECT_ROOT / ".tirra_pipeline" / "pipeline.db"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "embedding_space.html"

# Plotly color palette — one per entity type
_TYPE_COLORS: dict[str, str] = {
    "instrument": "#4C9BE8",  # blue
    "cftc_contract": "#F4A261",  # orange
    "country": "#2A9D8F",  # teal
    "company": "#E9C46A",  # gold
    "organization": "#E76F51",  # red-orange
    "person": "#A8DADC",  # light blue
    "domain": "#9B72CF",  # purple
    "topic": "#6BCB77",  # green
    "vessel": "#FF6B6B",  # red
    "wallet": "#C77DFF",  # violet
    "protocol": "#FFD166",  # yellow
}
_DEFAULT_COLOR = "#AAAAAA"


# ─────────────────────────────────────────────────────────────────────────────
# Label / metadata helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_entity_names(db_path: Path) -> dict[str, str]:
    """Return {entity_id: canonical_name} for all entities in the DB."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    rows = conn.execute("SELECT entity_id, canonical_name FROM entities").fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def _load_value_scores(db_path: Path, entity_ids: list[str]) -> dict[str, float]:
    """
    Return the most recent 'futures_positioning_derived' mm pct rank for
    each entity_id where available.  Used to colour-code instruments by
    positioning extremity in the hover text.
    """
    if not entity_ids:
        return {}
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    scores: dict[str, float] = {}
    for eid in entity_ids:
        row = conn.execute(
            "SELECT value_json FROM entity_observations "
            "WHERE entity_id=? AND observation_type='futures_positioning_derived' "
            "ORDER BY observed_at DESC LIMIT 1",
            (eid,),
        ).fetchone()
        if row:
            try:
                val = json.loads(row[0])
                pct = val.get("cftc_mm_pct_52w_rank")
                if pct is not None:
                    scores[eid] = float(pct)
            except Exception:
                pass
    conn.close()
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ─────────────────────────────────────────────────────────────────────────────


def _reduce_to_2d(
    matrix: np.ndarray,
    method: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce [N, D] embedding matrix to [N, 2] for plotting."""
    n = matrix.shape[0]

    # Need at least 5 points for UMAP; fall back to t-SNE/PCA for tiny sets
    if n < 5:
        from sklearn.decomposition import PCA

        print(f"  Only {n} points — using PCA(2) instead of {method}.")
        pca = PCA(n_components=min(2, n), random_state=random_state)
        reduced = pca.fit_transform(matrix)
        if reduced.shape[1] < 2:
            reduced = np.hstack([reduced, np.zeros((n, 2 - reduced.shape[1]))])
        return reduced

    if method == "umap":
        try:
            import umap as umap_lib

            print(
                f"  Running UMAP(n_neighbors={n_neighbors}, min_dist={min_dist}) on {n} nodes ..."
            )
            reducer = umap_lib.UMAP(
                n_components=2,
                n_neighbors=min(n_neighbors, n - 1),
                min_dist=min_dist,
                random_state=random_state,
                verbose=False,
            )
            return reducer.fit_transform(matrix)
        except Exception as exc:
            print(f"  UMAP failed ({exc}); falling back to t-SNE.")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, n - 1)
        print(f"  Running t-SNE(perplexity={perplexity}) on {n} nodes ...")
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
        )
        return reducer.fit_transform(matrix)

    # PCA fallback
    from sklearn.decomposition import PCA

    print(f"  Running PCA(2) on {n} nodes ...")
    return PCA(n_components=2, random_state=random_state).fit_transform(matrix)


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_chart(
    coords: np.ndarray,
    labels: list[str],  # entity_id per row
    types: list[str],  # entity_type per row
    names: list[str],  # canonical_name per row
    pct_scores: dict[str, float],
    reduction_method: str,
    model_path: Path,
) -> Any:
    """Build and return a Plotly Figure."""
    import plotly.graph_objects as go

    traces: list[go.Scatter] = []

    unique_types = sorted(set(types))
    for etype in unique_types:
        mask = [i for i, t in enumerate(types) if t == etype]
        x = coords[mask, 0]
        y = coords[mask, 1]

        hover_parts = []
        for i in mask:
            eid = labels[i]
            name = names[i]
            pct = pct_scores.get(eid)
            parts = [f"<b>{name}</b>", f"id: {eid}", f"type: {etype}"]
            if pct is not None:
                parts.append(f"COT pct rank: {pct:.1%}")
            hover_parts.append("<br>".join(parts))

        color = _TYPE_COLORS.get(etype, _DEFAULT_COLOR)
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=etype,
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.85,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover_parts,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(
                f"TirraMind — GNN Embedding Space ({reduction_method.upper()})<br>"
                f"<sub>Model: {model_path.name} | {len(labels)} nodes | "
                f"{len(unique_types)} entity types</sub>"
            ),
            font=dict(size=16),
        ),
        xaxis=dict(
            title=f"{reduction_method.upper()} dim 1", showgrid=False, zeroline=False
        ),
        yaxis=dict(
            title=f"{reduction_method.upper()} dim 2", showgrid=False, zeroline=False
        ),
        legend=dict(
            title="Entity Type",
            bgcolor="rgba(30,30,30,0.85)",
            bordercolor="#555",
            borderwidth=1,
            font=dict(color="white"),
        ),
        plot_bgcolor="#111111",
        paper_bgcolor="#1a1a2e",
        font=dict(color="white"),
        hoverlabel=dict(bgcolor="#222", font_color="white"),
        height=750,
        margin=dict(l=60, r=60, t=100, b=60),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    import torch

    sys.path.insert(0, str(PROJECT_ROOT))
    from agent.models.gnn.trainer import Trainer
    from agent.pipeline.store import PipelineStore

    model_path = Path(args.model).resolve()
    db_path = Path(args.db).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    assert model_path.exists(), f"Model not found: {model_path}"
    assert db_path.exists(), f"DB not found: {db_path}"

    # ── 1. Load model and extract embeddings ─────────────────────────────────
    print(f"\n[1/4] Loading model: {model_path.name}")
    store = PipelineStore(str(db_path))
    trainer = Trainer.load_model(str(model_path), store)

    print("[1/4] Running infer() to extract all node embeddings ...")
    embeddings, id_map = trainer.infer()

    if not embeddings:
        print("ERROR: infer() returned empty embeddings — is the model trained?")
        sys.exit(1)

    # ── 2. Filter to requested entity types and collect metadata ─────────────
    print("\n[2/4] Collecting entity labels and metadata ...")
    target_types = set(args.types) if args.types else set(embeddings.keys())
    entity_names = _load_entity_names(db_path)

    all_vecs: list[np.ndarray] = []
    all_labels: list[str] = []
    all_types: list[str] = []
    all_names: list[str] = []

    for etype, tensor in embeddings.items():
        if etype not in target_types:
            continue
        local_map = id_map.type_local.get(etype, {})
        # invert: local_idx → entity_id
        inv_local: dict[int, str] = {v: k for k, v in local_map.items()}
        n_type = tensor.shape[0]
        for local_idx in range(n_type):
            eid = inv_local.get(local_idx, f"{etype}:{local_idx}")
            name = entity_names.get(eid, eid)
            vec = tensor[local_idx].cpu().float().numpy()
            all_vecs.append(vec)
            all_labels.append(eid)
            all_types.append(etype)
            all_names.append(name)

        print(f"  {etype:20s}: {n_type} nodes, dim={tensor.shape[1]}")

    if not all_vecs:
        print(
            "No embeddings found for the requested types. Available:",
            list(embeddings.keys()),
        )
        sys.exit(1)

    matrix = np.stack(all_vecs, axis=0)  # [N_total, D]
    print(f"\n  Total: {matrix.shape[0]} nodes × {matrix.shape[1]} dims")

    # Load CFTC pct scores for instrument hover text
    instrument_ids = [
        all_labels[i] for i, t in enumerate(all_types) if t == "instrument"
    ]
    pct_scores = _load_value_scores(db_path, instrument_ids)

    # ── 3. Dimensionality reduction ───────────────────────────────────────────
    print(f"\n[3/4] Reducing to 2D via {args.method.upper()} ...")
    coords = _reduce_to_2d(
        matrix,
        method=args.method,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )

    # ── 4. Build Plotly chart and save ────────────────────────────────────────
    print(f"\n[4/4] Building interactive chart → {out_path}")
    fig = _build_chart(
        coords=coords,
        labels=all_labels,
        types=all_types,
        names=all_names,
        pct_scores=pct_scores,
        reduction_method=args.method,
        model_path=model_path,
    )

    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"\n✓ Saved: {out_path}")

    if not args.no_browser:
        import webbrowser

        webbrowser.open(out_path.as_uri())
        print("  Opened in browser.")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n── Embedding space summary ──────────────────────────────────────────")
    type_counts: dict[str, int] = {}
    for t in all_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    for etype, cnt in sorted(type_counts.items()):
        print(f"  {etype:20s}: {cnt} nodes")

    # Quick geometry check: mean pairwise cosine similarity per type
    print("\n── Mean within-type cosine similarity (higher = tighter cluster) ────")
    from sklearn.metrics.pairwise import cosine_similarity

    type_indices: dict[str, list[int]] = {}
    for i, t in enumerate(all_types):
        type_indices.setdefault(t, []).append(i)

    for etype, idxs in sorted(type_indices.items()):
        if len(idxs) < 2:
            print(f"  {etype:20s}: n/a (only {len(idxs)} node)")
            continue
        sub = matrix[idxs]
        sim = cosine_similarity(sub)
        # upper triangle excluding diagonal
        n = len(idxs)
        upper = sim[np.triu_indices(n, k=1)]
        print(f"  {etype:20s}: {upper.mean():.3f} ± {upper.std():.3f}  (n={n})")

    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TirraMind GNN embedding space visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"Path to GNN model .pt file (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help=f"Path to pipeline.db (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUT),
        help=f"Output HTML path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        metavar="TYPE",
        help=(
            "Entity types to include, e.g.: instrument cftc_contract country "
            "(default: all types with embeddings)"
        ),
    )
    parser.add_argument(
        "--method",
        choices=["umap", "tsne", "pca"],
        default="umap",
        help="Dimensionality reduction method (default: umap)",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        metavar="N",
        help="UMAP n_neighbors — lower = more local structure (default: 15)",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        metavar="F",
        help="UMAP min_dist — lower = tighter clusters (default: 0.1)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Save HTML but do not open in browser",
    )
    main(parser.parse_args())
