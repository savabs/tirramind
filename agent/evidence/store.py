"""Evidence Graph Store — document → entities → relationships → searchable graph.

A durable, deterministic record of *where a fact came from*. Builds on the
existing entity graph (agent/pipeline/entity.py) but adds an evidence layer:

    documents       — one row per ingested doc (PDF/news/CSV/text), with source
    mentions        — an entity name string appearing in a doc at a location
    evidence_links  — a relationship between two entities with a confidence
                      score and the evidence that supports it

Everything is append-only: re-ingesting a doc adds new mentions/links, never
edits old ones. Confidence is explicit and deterministic (co-occurrence anchors,
not a hidden model).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT UNIQUE NOT NULL,
    source TEXT,              -- filename / url / label
    doc_type TEXT,            -- pdf | text | csv | news
    title TEXT,
    ingested_at REAL NOT NULL,
    text_hash TEXT,           -- sha256 of raw text (dedup)
    meta_json TEXT
);
CREATE TABLE IF NOT EXISTS evidence_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_key TEXT NOT NULL,   -- normalized canonical key
    raw_name TEXT NOT NULL,
    sentence TEXT,             -- the sentence it appeared in (evidence)
    position INTEGER,          -- byte offset
    confidence REAL NOT NULL,
    FOREIGN KEY(doc_id) REFERENCES evidence_documents(doc_id)
);
CREATE TABLE IF NOT EXISTS evidence_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    entity_a TEXT NOT NULL,    -- canonical entity key
    entity_b TEXT NOT NULL,
    relation TEXT NOT NULL,    -- e.g. "co_occur", "same_sentence", "acquired_by"
    confidence REAL NOT NULL,
    evidence TEXT,             -- supporting snippet
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mentions_doc ON evidence_mentions(doc_id);
CREATE INDEX IF NOT EXISTS idx_mentions_key ON evidence_mentions(entity_key);
CREATE INDEX IF NOT EXISTS idx_links_ab ON evidence_links(entity_a, entity_b);
"""


class EvidenceGraphStore:
    """SQLite-backed evidence graph with append-only semantics."""

    def __init__(self, path: str = ".tirra_pipeline/evidence.db") -> None:
        import sqlite3

        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self._path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ── Documents ────────────────────────────────────────────────────────────
    def add_document(
        self,
        *,
        doc_id: str,
        source: str,
        doc_type: str,
        title: str,
        text_hash: str,
        meta: dict[str, Any] | None = None,
    ) -> bool:
        """Register a document. Returns False if it already exists (dedup)."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO evidence_documents "
            "(doc_id, source, doc_type, title, ingested_at, text_hash, meta_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (doc_id, source, doc_type, title, time.time(), text_hash, json.dumps(meta or {})),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def has_document(self, doc_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM evidence_documents WHERE doc_id=?", (doc_id,)).fetchone() is not None

    # ── Mentions ─────────────────────────────────────────────────────────────
    def add_mention(
        self,
        *,
        doc_id: str,
        entity_type: str,
        entity_key: str,
        raw_name: str,
        sentence: str,
        position: int,
        confidence: float,
    ) -> None:
        self.conn.execute(
            "INSERT INTO evidence_mentions "
            "(doc_id, entity_type, entity_key, raw_name, sentence, position, confidence) "
            "VALUES (?,?,?,?,?,?,?)",
            (doc_id, entity_type, entity_key, raw_name, sentence, position, confidence),
        )
        self.conn.commit()

    # ── Links ────────────────────────────────────────────────────────────────
    def add_link(
        self,
        *,
        doc_id: str,
        entity_a: str,
        entity_b: str,
        relation: str,
        confidence: float,
        evidence: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO evidence_links "
            "(doc_id, entity_a, entity_b, relation, confidence, evidence, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (doc_id, entity_a, entity_b, relation, confidence, evidence, time.time()),
        )
        self.conn.commit()

    # ── Search / query (the "API" surface) ───────────────────────────────────
    def search_entity(self, entity_key: str, limit: int = 50) -> list[dict[str, Any]]:
        """Return mentions + outgoing links for an entity."""
        mentions = [
            dict(r)
            for r in self.conn.execute(
                "SELECT doc_id, raw_name, sentence, confidence FROM evidence_mentions "
                "WHERE entity_key=? ORDER BY confidence DESC LIMIT ?",
                (entity_key, limit),
            ).fetchall()
        ]
        links = [
            dict(r)
            for r in self.conn.execute(
                "SELECT entity_b, relation, confidence, evidence FROM evidence_links "
                "WHERE entity_a=? ORDER BY confidence DESC LIMIT ?",
                (entity_key, limit),
            ).fetchall()
        ]
        return {"entity": entity_key, "mentions": mentions, "links": links}

    def related(self, entity_key: str, min_confidence: float = 0.3, limit: int = 30) -> list[dict[str, Any]]:
        """Neighbors of an entity above a confidence threshold (graph search)."""
        return [
            dict(r)
            for r in self.conn.execute(
                "SELECT entity_b AS neighbor, relation, confidence, evidence FROM evidence_links "
                "WHERE entity_a=? AND confidence>=? ORDER BY confidence DESC LIMIT ?",
                (entity_key, min_confidence, limit),
            ).fetchall()
        ]

    def document(self, doc_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM evidence_documents WHERE doc_id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def stats(self) -> dict[str, int]:
        return {
            "documents": self.conn.execute("SELECT COUNT(*) FROM evidence_documents").fetchone()[0],
            "mentions": self.conn.execute("SELECT COUNT(*) FROM evidence_mentions").fetchone()[0],
            "links": self.conn.execute("SELECT COUNT(*) FROM evidence_links").fetchone()[0],
        }

    # ── Cross-document analytics (the real signal) ───────────────────────────
    def co_occurrences(self, entity_key: str, min_docs: int = 1, limit: int = 20) -> list[dict[str, Any]]:
        """Entities that co-occur with `entity_key` across DOCUMENTS.

        Counts distinct documents where both entities appear. More documents =
        stronger (the kind of signal funds care about). Ignores within-doc
        duplicates so a single noisy doc can't dominate.
        """
        return [
            dict(r)
            for r in self.conn.execute(
                """
            SELECT m2.entity_key AS neighbor,
                   COUNT(DISTINCT m1.doc_id) AS n_docs,
                   COUNT(DISTINCT m1.doc_id || '|' || m2.entity_key) AS pair_docs,
                   MAX(m1.confidence * m2.confidence) AS max_conf
            FROM evidence_mentions m1
            JOIN evidence_mentions m2
              ON m1.doc_id = m2.doc_id AND m1.entity_key != m2.entity_key
            WHERE m1.entity_key = ?
            GROUP BY m2.entity_key
            HAVING n_docs >= ?
            ORDER BY n_docs DESC, max_conf DESC
            LIMIT ?
            """,
                (entity_key, min_docs, limit),
            ).fetchall()
        ]

    def cross_doc_pairs(self, min_docs: int = 2, limit: int = 30) -> list[dict[str, Any]]:
        """Strongest cross-document entity pairs (the 'recurring relationship' signal)."""
        return [
            dict(r)
            for r in self.conn.execute(
                """
            SELECT m1.entity_key AS a, m2.entity_key AS b,
                   COUNT(DISTINCT m1.doc_id) AS n_docs
            FROM evidence_mentions m1
            JOIN evidence_mentions m2
              ON m1.doc_id = m2.doc_id AND m1.entity_key < m2.entity_key
            GROUP BY m1.entity_key, m2.entity_key
            HAVING n_docs >= ?
            ORDER BY n_docs DESC
            LIMIT ?
            """,
                (min_docs, limit),
            ).fetchall()
        ]

    # ── Graph export (networkx-ready) ────────────────────────────────────────
    def all_edges(self, min_confidence: float = 0.0, limit: int = 5000) -> list[tuple[str, str, dict]]:
        """All evidence links as (entity_a, entity_b, attrs) — ready for networkx.

        Includes per-document distinct evidence weight when available (recurring
        pairs get higher weight). If multiple links exist, we aggregate to the
        highest-confidence relationship between the pair.
        """
        rows = self.conn.execute(
            """
            SELECT entity_a, entity_b,
                   MAX(confidence) AS confidence,
                   COUNT(DISTINCT doc_id) AS n_docs,
                   MAX(evidence) AS evidence
            FROM evidence_links
            WHERE confidence >= ?
            GROUP BY entity_a, entity_b
            ORDER BY n_docs DESC, confidence DESC
            LIMIT ?
            """,
            (min_confidence, limit),
        ).fetchall()
        return [
            (
                r["entity_a"],
                r["entity_b"],
                {"confidence": r["confidence"], "n_docs": r["n_docs"], "evidence": r["evidence"]},
            )
            for r in rows
        ]

    def graph_nodes(self) -> list[str]:
        """All unique entity keys that appear in at least one link."""
        return [
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT entity_a AS e FROM evidence_links "
                "UNION SELECT DISTINCT entity_b FROM evidence_links"
            ).fetchall()
        ]

    def graph_export(self, min_confidence: float = 0.0, limit: int = 5000) -> dict[str, Any]:
        """Adjacency-list graph export: {nodes, edges[{source,target,attrs}]}."""
        nodes = self.graph_nodes()
        edges = []
        for a, b, attrs in self.all_edges(min_confidence=min_confidence, limit=limit):
            edges.append(
                {
                    "source": a,
                    "target": b,
                    "n_docs": attrs["n_docs"],
                    "confidence": attrs["confidence"],
                    "evidence": attrs["evidence"],
                }
            )
        return {"nodes": nodes, "edges": edges, "n_nodes": len(nodes), "n_edges": len(edges)}


__all__ = ["EvidenceGraphStore"]
