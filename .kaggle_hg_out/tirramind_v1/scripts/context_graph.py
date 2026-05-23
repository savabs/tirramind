#!/usr/bin/env python3
"""TirraMind Context Graph — fast project context retrieval via graph + embeddings.

Builds a typed digraph from the Obsidian vault (markdown files) and Python
source tree (modules/imports).  Provides sub-second queries for:

  Graph commands:
    related <file>     — neighbors + 2-hop context, ranked by centrality
    context <topic>    — BFS from a topic/tag/keyword, top-N relevant files
    path <A> <B>       — shortest path between two artifacts
    orphans            — nodes with no incoming edges (dead docs/code)
    hubs               — top-N most-connected nodes (key architectural files)
    rebuild            — force rebuild the graph cache

  Embedding commands:
    embed [--force]    — build/rebuild the BGE transformer embedding index
    search <query>     — semantic search by natural language
    hybrid <query>     — combined semantic + graph ranking (best quality)

Caches: .context_graph.json (graph), .context_embeddings.pkl (embeddings).
Both auto-rebuild when source files change.

Usage:
  python scripts/context_graph.py related entity_linking_layer
  python scripts/context_graph.py search "bayesian inference belief propagation"
  python scripts/context_graph.py hybrid "entity linking GNN training"
  python scripts/context_graph.py context "topic/convergence"
  python scripts/context_graph.py path insider_filings world_model
  python scripts/context_graph.py hubs 15
  python scripts/context_graph.py embed --force
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / ".context_graph.json"
EMBED_CACHE = ROOT / ".context_embeddings.pkl"

# ── Embedding model ──────────────────────────────────────────────────────────

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"  # 438MB, 768-dim, MIT license
EMBED_BATCH_SIZE = 32  # chunks per forward pass
EMBED_MAX_TOKENS = 512  # model max sequence length

_embed_model: AutoModel | None = None
_embed_tokenizer: AutoTokenizer | None = None
_embed_device: str = "cpu"


def _get_embed_device() -> str:
    """Pick best available device: CUDA > CPU."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except Exception:
            pass
    return "cpu"


def _get_embedder() -> tuple[AutoModel, AutoTokenizer, str]:
    """Lazy-load the embedding model. Cached in module globals."""
    global _embed_model, _embed_tokenizer, _embed_device
    if _embed_model is not None:
        return _embed_model, _embed_tokenizer, _embed_device

    _embed_device = _get_embed_device()
    print(f"[context_graph] Loading {EMBED_MODEL_NAME} on {_embed_device}...",
          file=sys.stderr)
    _embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    _embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
    _embed_model = _embed_model.to(_embed_device)
    _embed_model.eval()
    return _embed_model, _embed_tokenizer, _embed_device


def _encode_texts(texts: list[str]) -> np.ndarray:
    """Encode texts to normalized embeddings using BGE model.

    Returns (n_texts, 768) float32 numpy array, L2-normalized.
    """
    model, tokenizer, device = _get_embedder()
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=EMBED_MAX_TOKENS,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            # BGE uses CLS token embedding
            emb = outputs.last_hidden_state[:, 0, :]
            # L2 normalize
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)

# ── Parsing constants ────────────────────────────────────────────────────────

WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
TAG_LINE_RE = re.compile(r"^\s*-\s+(.+)$", re.MULTILINE)
HEADING_RE = re.compile(r"^#+\s+(.+)$", re.MULTILINE)

SKIP_DIRS = {
    ".obsidian",
    ".venv",
    ".git",
    "node_modules",
    "__pycache__",
    "tirramind_vault",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "htmlcov",
    "*.egg-info",
}

MD_DIRS = ["docs", "tasks", "wiki"]
PY_DIRS = ["agent", "scripts", "tests"]


# ── Graph builder ────────────────────────────────────────────────────────────


def _iter_md_files() -> list[Path]:
    """Yield all .md files in vault directories + root."""
    files: list[Path] = []
    # Root-level md files
    for f in ROOT.iterdir():
        if f.is_file() and f.suffix == ".md":
            files.append(f)
    # Vault subdirectories
    for d in MD_DIRS:
        dp = ROOT / d
        if not dp.exists():
            continue
        for f in dp.rglob("*.md"):
            if not any(part in SKIP_DIRS for part in f.parts):
                files.append(f)
    return files


def _iter_py_files() -> list[Path]:
    """Yield all .py files in source directories."""
    files: list[Path] = []
    for d in PY_DIRS:
        dp = ROOT / d
        if not dp.exists():
            continue
        for f in dp.rglob("*.py"):
            if not any(part in SKIP_DIRS for part in f.parts):
                files.append(f)
    return files


def _parse_frontmatter(content: str) -> dict[str, Any]:
    """Extract tags and title from YAML frontmatter (simple regex, no PyYAML)."""
    m = FRONTMATTER_RE.match(content)
    if not m:
        return {}
    block = m.group(1)
    result: dict[str, Any] = {}

    # Title
    for line in block.splitlines():
        if line.strip().startswith("title:"):
            result["title"] = line.split(":", 1)[1].strip().strip('"').strip("'")
            break

    # Tags
    tags: list[str] = []
    in_tags = False
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("tags:"):
            # Inline tags: tags: [a, b] or tags: a
            rest = stripped[5:].strip()
            if rest.startswith("["):
                tags.extend(
                    t.strip().strip('"').strip("'")
                    for t in rest.strip("[]").split(",")
                    if t.strip()
                )
                break
            elif rest:
                tags.append(rest.strip('"').strip("'"))
                break
            in_tags = True
            continue
        if in_tags:
            tm = TAG_LINE_RE.match(line)
            if tm:
                tags.append(tm.group(1).strip().strip('"').strip("'"))
            elif stripped and not stripped.startswith("-"):
                in_tags = False
    result["tags"] = tags
    return result


def _parse_wiki_links(content: str) -> list[str]:
    """Extract [[wiki link]] targets from markdown content."""
    return WIKI_LINK_RE.findall(content)


def _parse_headings(content: str) -> list[str]:
    """Extract markdown headings (used as concept keywords)."""
    return [m.group(1).strip() for m in HEADING_RE.finditer(content)]


def _stem(filename: str) -> str:
    """Return filename without extension, lowered for matching."""
    return Path(filename).stem.lower()


def _node_id_for_md(path: Path) -> str:
    """Stable node ID: relative path from ROOT."""
    return str(path.relative_to(ROOT))


def _node_id_for_py(path: Path) -> str:
    """Stable node ID: dotted module path."""
    rel = path.relative_to(ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].replace(".py", "")
    return ".".join(parts)


def _parse_py_imports(content: str, filepath: Path) -> list[str]:
    """Extract local (agent.*) imports from Python file."""
    imports: list[str] = []
    try:
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("agent"):
                imports.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("agent"):
                    imports.append(alias.name)
    return imports


def build_graph() -> nx.DiGraph:
    """Build the full context graph from vault + source tree."""
    G = nx.DiGraph()

    # Filename stem → node_id mapping (for resolving [[wiki links]])
    stem_to_node: dict[str, str] = {}

    # ── Pass 1: Markdown files ───────────────────────────────────────────
    md_files = _iter_md_files()
    for path in md_files:
        node_id = _node_id_for_md(path)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        fm = _parse_frontmatter(content)
        tags = fm.get("tags", [])
        title = fm.get("title", path.stem)

        # Classify by path
        rel = str(path.relative_to(ROOT))
        if rel.startswith("docs/research/"):
            ntype = "research"
        elif rel.startswith("docs/specs/"):
            ntype = "spec"
        elif rel.startswith("tasks/"):
            ntype = "task"
        elif rel.startswith("docs/memory/"):
            ntype = "checkpoint"
        elif rel.startswith("docs/adr/"):
            ntype = "adr"
        elif rel.startswith("wiki/"):
            ntype = "wiki"
        else:
            ntype = "doc"

        G.add_node(node_id, type=ntype, title=title, tags=tags, stem=path.stem.lower())
        stem_to_node[path.stem.lower()] = node_id

        # Tag edges: doc → tag node
        for tag in tags:
            tag_node = f"tag:{tag}"
            if not G.has_node(tag_node):
                G.add_node(tag_node, type="tag", title=tag, tags=[], stem=tag)
            G.add_edge(node_id, tag_node, type="tagged_with")
            G.add_edge(tag_node, node_id, type="tags")

        # Wiki link edges (targets resolved in pass 3)
        targets = _parse_wiki_links(content)
        # Store raw targets for resolution
        G.nodes[node_id]["_wiki_targets"] = targets

    # ── Pass 2: Python files ─────────────────────────────────────────────
    py_files = _iter_py_files()
    module_to_node: dict[str, str] = {}

    for path in py_files:
        node_id = _node_id_for_py(path)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        G.add_node(
            node_id,
            type="code",
            title=path.stem,
            tags=[],
            stem=path.stem.lower(),
            path=str(path.relative_to(ROOT)),
        )
        module_to_node[node_id] = node_id
        stem_to_node[path.stem.lower()] = node_id

        # Import edges
        imports = _parse_py_imports(content, path)
        for imp in imports:
            imp_node = imp
            if not G.has_node(imp_node):
                G.add_node(
                    imp_node,
                    type="code",
                    title=imp.split(".")[-1],
                    tags=[],
                    stem=imp.split(".")[-1].lower(),
                )
            G.add_edge(node_id, imp_node, type="imports")

    # ── Pass 3: Resolve wiki links ───────────────────────────────────────
    for node_id, data in list(G.nodes(data=True)):
        targets = data.pop("_wiki_targets", [])
        for target in targets:
            target_stem = _stem(target.strip())
            resolved = stem_to_node.get(target_stem)
            if resolved and resolved != node_id:
                G.add_edge(node_id, resolved, type="references")

    # ── Pass 4: Cross-link research↔spec↔task triads ────────────────────
    # Many triads share a common stem (e.g., entity_linking_layer,
    # entity_linking_layer_spec).  Add explicit triad edges where wiki
    # links didn't already create them.
    for node_id, data in G.nodes(data=True):
        ntype = data.get("type")
        stem = data.get("stem", "")
        if ntype == "spec" and stem.endswith("_spec"):
            base = stem[:-5]  # remove _spec suffix
            research_id = stem_to_node.get(base)
            if research_id and not G.has_edge(node_id, research_id):
                G.add_edge(node_id, research_id, type="specifies")
                G.add_edge(research_id, node_id, type="specified_by")

    return G


# ── Cache ────────────────────────────────────────────────────────────────────


def _latest_mtime() -> float:
    """Max mtime across all source files."""
    latest = 0.0
    for d in MD_DIRS + PY_DIRS:
        dp = ROOT / d
        if not dp.exists():
            continue
        for f in dp.rglob("*"):
            if f.is_file() and f.suffix in (".md", ".py"):
                try:
                    latest = max(latest, f.stat().st_mtime)
                except OSError:
                    pass
    # Root-level .md
    for f in ROOT.iterdir():
        if f.is_file() and f.suffix == ".md":
            try:
                latest = max(latest, f.stat().st_mtime)
            except OSError:
                pass
    return latest


def _save_cache(G: nx.DiGraph) -> None:
    """Serialize graph to JSON cache."""
    data = nx.node_link_data(G, edges="edges")
    data["_built_at"] = time.time()
    CACHE_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")


def _load_cache() -> nx.DiGraph | None:
    """Load graph from cache if fresh enough."""
    if not CACHE_FILE.exists():
        return None
    try:
        cache_mtime = CACHE_FILE.stat().st_mtime
        if _latest_mtime() > cache_mtime:
            return None
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return nx.node_link_graph(data, edges="edges")
    except Exception:
        return None


def get_graph(force_rebuild: bool = False) -> nx.DiGraph:
    """Get the context graph, rebuilding if stale."""
    if not force_rebuild:
        G = _load_cache()
        if G is not None:
            return G
    t0 = time.time()
    G = build_graph()
    _save_cache(G)
    dt = time.time() - t0
    print(
        f"[context_graph] Built: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges in {dt:.2f}s",
        file=sys.stderr,
    )
    return G


# ── Query commands ───────────────────────────────────────────────────────────

# ── Embedding index ──────────────────────────────────────────────────────────

HEADING_SPLIT_RE = re.compile(r"^(?=##\s)", re.MULTILINE)


def _py_summary(content: str, filepath: Path) -> str:
    """Extract a searchable summary from a Python file via AST."""
    parts: list[str] = []
    try:
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError:
        # Fallback: first 500 chars
        return content[:500]

    # Module docstring
    ds = ast.get_docstring(tree)
    if ds:
        parts.append(ds)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = node.name
            if isinstance(node, ast.ClassDef):
                bases = [
                    getattr(b, "id", getattr(b, "attr", "?"))
                    for b in node.bases
                ]
                sig = f"class {node.name}({', '.join(bases)})"
            else:
                args = []
                for a in node.args.args:
                    args.append(a.arg)
                sig = f"def {node.name}({', '.join(args)})"
            parts.append(sig)
            cds = ast.get_docstring(node)
            if cds:
                parts.append(f"  {cds[:200]}")

    return "\n".join(parts) if parts else content[:500]


def _chunk_md(content: str, path: Path, node_id: str,
              ntype: str, title: str, tags: list[str]) -> list[dict]:
    """Split a markdown file into embeddable chunks."""
    # Prefix every chunk with metadata for context
    meta_prefix = f"[{ntype}] {title}"
    if tags:
        meta_prefix += f" | tags: {', '.join(tags[:5])}"
    meta_prefix += "\n\n"

    # Strip frontmatter for content
    body = FRONTMATTER_RE.sub("", content).strip()

    if len(body) < 3000:
        return [{
            "id": node_id,
            "text": meta_prefix + body,
            "path": node_id,
            "node_type": ntype,
            "title": title,
            "tags": tags,
            "chunk_index": 0,
            "total_chunks": 1,
        }]

    # Split by ## headings
    sections = HEADING_SPLIT_RE.split(body)
    chunks: list[dict] = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section or len(section) < 30:
            continue
        chunks.append({
            "id": f"{node_id}#chunk{i}",
            "text": meta_prefix + section,
            "path": node_id,
            "node_type": ntype,
            "title": title,
            "tags": tags,
            "chunk_index": i,
            "total_chunks": len(sections),
        })

    return chunks or [{
        "id": node_id,
        "text": meta_prefix + body[:3000],
        "path": node_id,
        "node_type": ntype,
        "title": title,
        "tags": tags,
        "chunk_index": 0,
        "total_chunks": 1,
    }]


def _chunk_py(content: str, path: Path, node_id: str) -> list[dict]:
    """Create a single embeddable chunk from a Python module."""
    summary = _py_summary(content, path)
    if not summary.strip():
        return []
    return [{
        "id": node_id,
        "text": f"[code] {node_id}\n\n{summary}",
        "path": str(path.relative_to(ROOT)),
        "node_type": "code",
        "title": path.stem,
        "tags": [],
        "chunk_index": 0,
        "total_chunks": 1,
    }]


def build_embedding_index() -> dict:
    """Build transformer embedding index from all project files.

    Uses BAAI/bge-base-en-v1.5 (768-dim).  Prefers GPU when available.

    Returns a dict with:
      - matrix: dense numpy array (n_chunks × 768), L2-normalized
      - chunks: list of chunk metadata dicts
      - model_name: embedding model used
      - dim: embedding dimensionality
      - built_at: timestamp
    """
    all_chunks: list[dict] = []

    # Markdown files
    for path in _iter_md_files():
        node_id = _node_id_for_md(path)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        fm = _parse_frontmatter(content)
        tags = fm.get("tags", [])
        title = fm.get("title", path.stem)
        rel = str(path.relative_to(ROOT))

        if rel.startswith("docs/research/"):
            ntype = "research"
        elif rel.startswith("docs/specs/"):
            ntype = "spec"
        elif rel.startswith("tasks/"):
            ntype = "task"
        elif rel.startswith("docs/memory/"):
            ntype = "checkpoint"
        elif rel.startswith("docs/adr/"):
            ntype = "adr"
        elif rel.startswith("wiki/"):
            ntype = "wiki"
        else:
            ntype = "doc"

        all_chunks.extend(_chunk_md(content, path, node_id, ntype, title, tags))

    # Python files
    for path in _iter_py_files():
        node_id = _node_id_for_py(path)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Skip near-empty __init__.py files
        if path.name == "__init__.py" and len(content.strip()) < 50:
            continue
        all_chunks.extend(_chunk_py(content, path, node_id))

    # Encode with transformer model
    texts = [c["text"] for c in all_chunks]
    print(f"[context_graph] Encoding {len(texts)} chunks...", file=sys.stderr)
    matrix = _encode_texts(texts)

    return {
        "matrix": matrix,
        "chunks": all_chunks,
        "model_name": EMBED_MODEL_NAME,
        "dim": matrix.shape[1],
        "built_at": time.time(),
    }


def _save_embed_cache(index: dict) -> None:
    """Persist embedding index to disk."""
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_embed_cache() -> dict | None:
    """Load embedding index if fresh."""
    if not EMBED_CACHE.exists():
        return None
    try:
        cache_mtime = EMBED_CACHE.stat().st_mtime
        if _latest_mtime() > cache_mtime:
            return None
        with open(EMBED_CACHE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def get_embed_index(force_rebuild: bool = False) -> dict:
    """Get embedding index, rebuilding if stale."""
    if not force_rebuild:
        idx = _load_embed_cache()
        if idx is not None:
            return idx
    t0 = time.time()
    idx = build_embedding_index()
    _save_embed_cache(idx)
    dt = time.time() - t0
    print(
        f"[context_graph] Indexed: {len(idx['chunks'])} chunks, "
        f"{idx['dim']}d embeddings in {dt:.2f}s",
        file=sys.stderr,
    )
    return idx


# ── Embedding query commands ─────────────────────────────────────────────────

def _format_search_result(chunk: dict, score: float,
                          preview_len: int = 150) -> str:
    """Format a search result for display."""
    ntype = chunk.get("node_type", "?")
    path = chunk.get("path", "?")
    title = chunk.get("title", "?")
    tags = chunk.get("tags", [])

    # Get a content preview (skip the metadata prefix)
    text = chunk.get("text", "")
    lines = text.split("\n")
    # Skip the first 2 lines (metadata prefix)
    body_lines = [l for l in lines[2:] if l.strip()][:3]
    preview = " ".join(body_lines)[:preview_len]
    if len(preview) == preview_len:
        preview += "..."

    parts = [f"  [{ntype:10s}] {path}"]
    if title and title != Path(path).stem:
        parts.append(f"  title: {title}")
    if tags:
        parts.append(f"  tags: {', '.join(tags[:3])}")
    parts.append(f"  score: {score:.4f}")
    parts.append(f"  preview: {preview}")
    return "\n".join(parts)


def cmd_embed(force: bool = False) -> None:
    """Build or rebuild the embedding index."""
    idx = get_embed_index(force_rebuild=force)
    n_chunks = len(idx["chunks"])
    dim = idx.get("dim", idx["matrix"].shape[1])
    model = idx.get("model_name", "unknown")

    # Count by type
    type_counts: dict[str, int] = {}
    for c in idx["chunks"]:
        t = c.get("node_type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"Embedding index: {n_chunks} chunks, {dim}d ({model})")
    print(f"\nChunk types:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:12s}: {c}")


def _cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single query vector and a matrix of vectors.

    Both inputs are assumed to be L2-normalized, so dot product = cosine sim.
    """
    return matrix @ query_vec.T


def cmd_search(query: str, top_n: int = 10) -> None:
    """Pure semantic search via transformer cosine similarity."""
    idx = get_embed_index()
    matrix = idx["matrix"]
    chunks = idx["chunks"]

    q_vec = _encode_texts([query])[0]
    scores = _cosine_scores(q_vec, matrix).flatten()
    top_indices = np.argsort(scores)[::-1]

    # Deduplicate: keep best chunk per file
    seen_paths: set[str] = set()
    results: list[tuple[dict, float]] = []
    for i in top_indices:
        if scores[i] < 0.01:
            break
        path = chunks[i]["path"]
        if path in seen_paths:
            continue
        seen_paths.add(path)
        results.append((chunks[i], float(scores[i])))
        if len(results) >= top_n:
            break

    if not results:
        print(f"No results for '{query}'")
        return

    print(f"Search: '{query}' ({len(results)} results)\n")
    for chunk, score in results:
        print(_format_search_result(chunk, score))
        print()


def cmd_hybrid(query: str, top_n: int = 10,
               sem_weight: float = 0.7, graph_weight: float = 0.3) -> None:
    """Combined semantic search + graph proximity ranking."""
    idx = get_embed_index()
    G = get_graph()

    matrix = idx["matrix"]
    chunks = idx["chunks"]

    # Step 1: Semantic search — top 50 candidates
    q_vec = _encode_texts([query])[0]
    scores = _cosine_scores(q_vec, matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:50]

    # Deduplicate to unique paths, keeping best semantic score per path
    path_best: dict[str, tuple[dict, float, int]] = {}
    for i in top_indices:
        if scores[i] < 0.01:
            break
        path = chunks[i]["path"]
        if path not in path_best or scores[i] > path_best[path][1]:
            path_best[path] = (chunks[i], float(scores[i]), i)

    if not path_best:
        print(f"No results for '{query}'")
        return

    # Step 2: Find the top semantic hit's node in the graph for PPR
    top_sem_path = max(path_best.items(), key=lambda x: x[1][1])[0]
    seed_node = _match_node(G, Path(top_sem_path).stem)

    # Compute Personalized PageRank from seed
    graph_scores: dict[str, float] = {}
    if seed_node:
        try:
            ppr = nx.pagerank(G, personalization={seed_node: 1.0},
                              max_iter=100, alpha=0.5)
            # Normalize PPR to [0, 1]
            max_ppr = max(ppr.values()) if ppr else 1.0
            graph_scores = {nid: v / max_ppr for nid, v in ppr.items()}
        except Exception:
            pass

    # Step 3: Combine scores
    results: list[tuple[dict, float]] = []
    for path, (chunk, sem_score, _) in path_best.items():
        # Try to find graph node for this path
        stem = Path(path).stem.lower()
        g_node = _match_node(G, stem)
        g_score = graph_scores.get(g_node, 0.0) if g_node else 0.0
        combined = sem_weight * sem_score + graph_weight * g_score
        results.append((chunk, combined))

    results.sort(key=lambda x: -x[1])

    print(f"Hybrid search: '{query}' ({len(results[:top_n])} results)\n")
    for chunk, score in results[:top_n]:
        print(_format_search_result(chunk, score))
        print()


def _match_node(G: nx.DiGraph, query: str) -> str | None:
    """Fuzzy-match a query to a node ID.  Tries exact, stem, then substring."""
    q = query.lower().strip()

    # Exact match
    if G.has_node(query):
        return query

    # tag: prefix match
    if G.has_node(f"tag:{q}"):
        return f"tag:{q}"

    # Stem match
    for nid, data in G.nodes(data=True):
        if data.get("stem") == q:
            return nid

    # Substring match (prefer shorter node IDs = more specific)
    candidates = [
        (nid, data)
        for nid, data in G.nodes(data=True)
        if q in nid.lower()
        or q in data.get("title", "").lower()
        or q in data.get("stem", "")
    ]
    if candidates:
        candidates.sort(key=lambda x: len(x[0]))
        return candidates[0][0]

    return None


def _format_node(G: nx.DiGraph, nid: str, score: float | None = None) -> str:
    """Format a node for display."""
    data = G.nodes.get(nid, {})
    ntype = data.get("type", "?")
    title = data.get("title", nid)
    tags_str = ", ".join(data.get("tags", [])[:3])
    parts = [f"  [{ntype:10s}] {nid}"]
    if title != nid and title != Path(nid).stem:
        parts.append(f"  title: {title}")
    if tags_str:
        parts.append(f"  tags: {tags_str}")
    if score is not None:
        parts.append(f"  score: {score:.4f}")
    return "\n".join(parts)


def cmd_related(G: nx.DiGraph, query: str, depth: int = 2, top_n: int = 20) -> None:
    """Show files related to a node via graph neighbors up to `depth` hops."""
    start = _match_node(G, query)
    if not start:
        print(f"No node matching '{query}'")
        return

    print(f"Related to: {start}\n")

    # BFS with depth limit, collect nodes and distances
    visited: dict[str, int] = {start: 0}
    queue = [start]
    while queue:
        current = queue.pop(0)
        d = visited[current]
        if d >= depth:
            continue
        for nbr in set(G.successors(current)) | set(G.predecessors(current)):
            if nbr not in visited:
                visited[nbr] = d + 1
                queue.append(nbr)

    # Score: closer = higher, PageRank as tiebreaker
    try:
        pr = nx.pagerank(G, max_iter=50)
    except Exception:
        pr = {n: 0.0 for n in G.nodes}

    results: list[tuple[str, float]] = []
    for nid, dist in visited.items():
        if nid == start:
            continue
        # Skip tag nodes from output (they're intermediate)
        if G.nodes[nid].get("type") == "tag":
            continue
        score = (1.0 / (dist + 0.5)) + pr.get(nid, 0) * 10
        results.append((nid, score))

    results.sort(key=lambda x: -x[1])
    for nid, score in results[:top_n]:
        print(_format_node(G, nid, score))
        print()


def cmd_context(G: nx.DiGraph, topic: str, top_n: int = 15) -> None:
    """Find all files relevant to a topic (tag, keyword, or filename stem)."""
    start = _match_node(G, topic)
    if not start:
        print(f"No node matching '{topic}'")
        return

    print(f"Context for: {start}\n")

    # Personalized PageRank from the start node
    try:
        ppr = nx.pagerank(G, personalization={start: 1.0}, max_iter=100, alpha=0.5)
    except Exception:
        # Fallback: BFS
        cmd_related(G, topic, depth=3, top_n=top_n)
        return

    results = [
        (nid, score)
        for nid, score in ppr.items()
        if nid != start and G.nodes[nid].get("type") != "tag"
    ]
    results.sort(key=lambda x: -x[1])

    for nid, score in results[:top_n]:
        print(_format_node(G, nid, score))
        print()


def cmd_path(G: nx.DiGraph, src: str, dst: str) -> None:
    """Show shortest path between two nodes."""
    s = _match_node(G, src)
    d = _match_node(G, dst)
    if not s:
        print(f"No node matching '{src}'")
        return
    if not d:
        print(f"No node matching '{dst}'")
        return

    # Undirected shortest path
    UG = G.to_undirected()
    try:
        path = nx.shortest_path(UG, s, d)
    except nx.NetworkXNoPath:
        print(f"No path between {s} and {d}")
        return

    print(f"Path ({len(path) - 1} hops):\n")
    for i, nid in enumerate(path):
        data = G.nodes.get(nid, {})
        ntype = data.get("type", "?")
        indent = "  " * i
        # Show edge type to next node
        if i < len(path) - 1:
            edge_data = G.edges.get(
                (nid, path[i + 1]), G.edges.get((path[i + 1], nid), {})
            )
            etype = edge_data.get("type", "~")
            print(f"{indent}[{ntype}] {nid}")
            print(f"{indent}  --({etype})-->")
        else:
            print(f"{indent}[{ntype}] {nid}")


def cmd_orphans(G: nx.DiGraph) -> None:
    """Show nodes with no incoming edges (potential dead artifacts)."""
    orphans = [
        nid
        for nid in G.nodes
        if G.in_degree(nid) == 0 and G.nodes[nid].get("type") not in ("tag",)
    ]
    orphans.sort()
    print(f"Orphan nodes ({len(orphans)} with 0 incoming edges):\n")
    for nid in orphans:
        data = G.nodes.get(nid, {})
        ntype = data.get("type", "?")
        out = G.out_degree(nid)
        print(f"  [{ntype:10s}] {nid}  (out_degree={out})")


def cmd_hubs(G: nx.DiGraph, top_n: int = 20) -> None:
    """Show the most connected nodes (key architectural files)."""
    try:
        pr = nx.pagerank(G, max_iter=50)
    except Exception:
        pr = {n: float(G.degree(n)) for n in G.nodes}

    results = [(nid, pr[nid]) for nid in G.nodes if G.nodes[nid].get("type") != "tag"]
    results.sort(key=lambda x: -x[1])

    print(f"Top {top_n} hub nodes:\n")
    for nid, score in results[:top_n]:
        data = G.nodes.get(nid, {})
        ntype = data.get("type", "?")
        deg = G.degree(nid)
        print(f"  [{ntype:10s}] {nid}  (degree={deg}, pr={score:.5f})")


def cmd_stats(G: nx.DiGraph) -> None:
    """Print graph statistics."""
    type_counts: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    edge_type_counts: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        t = data.get("type", "unknown")
        edge_type_counts[t] = edge_type_counts.get(t, 0) + 1

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"\nNode types:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:12s}: {c}")
    print(f"\nEdge types:")
    for t, c in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:15s}: {c}")

    # Connected components (undirected)
    UG = G.to_undirected()
    components = list(nx.connected_components(UG))
    print(f"\nConnected components: {len(components)}")
    if len(components) > 1:
        sizes = sorted([len(c) for c in components], reverse=True)
        print(f"  Sizes: {sizes[:10]}{'...' if len(sizes) > 10 else ''}")

    # Embedding index stats
    idx = _load_embed_cache()
    if idx:
        n_chunks = len(idx["chunks"])
        dim = idx.get("dim", idx["matrix"].shape[1])
        model = idx.get("model_name", "unknown")
        embed_type_counts: dict[str, int] = {}
        for c in idx["chunks"]:
            t = c.get("node_type", "?")
            embed_type_counts[t] = embed_type_counts.get(t, 0) + 1
        print(f"\nEmbedding index: {n_chunks} chunks, {dim}d ({model})")
        for t, ct in sorted(embed_type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t:12s}: {ct}")
    else:
        print("\nEmbedding index: not built (run 'embed' command)")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TirraMind Context Graph — fast project context retrieval"
    )
    sub = parser.add_subparsers(dest="command")

    # Graph commands
    p_related = sub.add_parser("related", help="Files related to a node")
    p_related.add_argument("query")
    p_related.add_argument("-n", "--top", type=int, default=20)
    p_related.add_argument("-d", "--depth", type=int, default=2)

    p_context = sub.add_parser("context", help="Context for a topic")
    p_context.add_argument("topic")
    p_context.add_argument("-n", "--top", type=int, default=15)

    p_path = sub.add_parser("path", help="Shortest path between nodes")
    p_path.add_argument("src")
    p_path.add_argument("dst")

    sub.add_parser("orphans", help="Nodes with no incoming edges")

    p_hubs = sub.add_parser("hubs", help="Most connected nodes")
    p_hubs.add_argument("top_n", nargs="?", type=int, default=20)

    sub.add_parser("stats", help="Graph + embedding statistics")
    sub.add_parser("rebuild", help="Force rebuild graph + embedding caches")

    # Embedding commands
    p_embed = sub.add_parser("embed", help="Build/rebuild embedding index")
    p_embed.add_argument("--force", action="store_true",
                         help="Force rebuild even if cache is fresh")

    p_search = sub.add_parser("search", help="Semantic search by query")
    p_search.add_argument("query")
    p_search.add_argument("-n", "--top", type=int, default=10)

    p_hybrid = sub.add_parser("hybrid", help="Semantic + graph combined search")
    p_hybrid.add_argument("query")
    p_hybrid.add_argument("-n", "--top", type=int, default=10)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "rebuild":
        G = get_graph(force_rebuild=True)
        get_embed_index(force_rebuild=True)
        cmd_stats(G)
        return

    if args.command == "embed":
        cmd_embed(force=args.force)
        return

    if args.command == "search":
        cmd_search(args.query, top_n=args.top)
        return

    if args.command == "hybrid":
        cmd_hybrid(args.query, top_n=args.top)
        return

    G = get_graph()

    if args.command == "related":
        cmd_related(G, args.query, depth=args.depth, top_n=args.top)
    elif args.command == "context":
        cmd_context(G, args.topic, top_n=args.top)
    elif args.command == "path":
        cmd_path(G, args.src, args.dst)
    elif args.command == "orphans":
        cmd_orphans(G)
    elif args.command == "hubs":
        cmd_hubs(G, args.top_n)
    elif args.command == "stats":
        cmd_stats(G)


if __name__ == "__main__":
    main()
