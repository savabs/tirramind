"""Evidence in-gestor — deterministically extract entities + relationships.

Converts raw documents (text, PDF via pypdf, CSV, news string) into
(mentions, relations) with explicit confidence scores. Deterministic — no LLM,
no hidden model — so the evidence graph is auditable and honest about *why* an
entity/link was extracted.

Extraction approach:
  1. Text split into sentences.
  2. Candidate entities matched against a seed dictionary (company tickers,
     orgs) OR detected by capitalization pattern fallback.
  3. Canonical key via normalize_company_name (reuses the entity graph).
  4. A relationship (co-occurrence) is recorded when two entities appear in the
     same sentence ("same_sentence") or adjacent sentences ("nearby"), with a
     deterministic confidence that drops with distance.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.pipeline.entity import normalize_company_name

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_UPPER_WORDS = re.compile(r"\b[A-Z][A-Za-z0-9&\'.\-]{1,}(?:\s+[A-Z][A-Za-z0-9&\'.\-]{1,}){0,3}\b")

# Hard bounds on relation-building, enforced regardless of document size.
#
# _build_relations used to pair EVERY mention against every other mention in
# the whole document -- O(mentions^2) with no cap. Confirmed in production:
# a single 267-mention document produced 31,685 links (89% of all 35,511
# possible pairs), 99.9% of the entire evidence_links table.
#
# Two entities mentioned in unrelated parts of a long document aren't good
# evidence of a relationship anyway, so linking is now scoped to sentence
# proximity (same sentence = strong, adjacent sentence = weak) -- a
# signal-quality fix that happens to also be the cost fix. The per-sentence
# and per-document caps below are the hard backstop: even a pathological
# "sentence" (e.g. a flattened CSV row with no punctuation) can't regress
# this back to unbounded growth.
MAX_MENTIONS_PER_SENTENCE = 12  # bounds same-sentence/adjacent-sentence pairing
MAX_LINKS_PER_DOCUMENT = 2000  # absolute backstop; should rarely trigger given the cap above

# Known entities to match before falling back to case-detection. Extensible;
# seeded with a few so the demo is meaningful.
SEED_ENTITIES: dict[str, str] = {  # normalized_key -> display label
    "apple": "Apple Inc.",
    "microsoft": "Microsoft Corporation",
    "openai": "OpenAI",
    "nvidia": "NVIDIA",
    "tesla": "Tesla",
    "amazon": "Amazon",
    "qualcomm": "Qualcomm",
}


def seed_entities_from_registry(
    db_path: str = ".tirra_pipeline/pipeline.db",
    entity_types: tuple[str, ...] = ("company", "organization", "person", "country", "instrument"),
    limit: int = 2000,
) -> dict[str, str]:
    """Build a SEED_ENTITIES dict from the existing entity registry.

    Maps canonical entity key → display name, so extraction is far richer than
    the tiny demo seed. Reuses the graph's normalization so keys align with the
    existing entity graph (linking evidence mentions to real entities).
    """
    from agent.pipeline.store import PipelineStore

    store = PipelineStore(db_path)
    cur = store._conn.cursor()
    placeholders = ",".join("?" * len(entity_types))
    rows = cur.execute(
        f"SELECT entity_type, canonical_name FROM entities "
        f"WHERE entity_type IN ({placeholders}) ORDER BY canonical_name LIMIT ?",
        (*entity_types, limit),
    ).fetchall()
    seeds: dict[str, str] = {}
    for _etype, name in rows:
        if not name or not str(name).strip():
            continue
        label = str(name).strip()
        # filter obvious non-entity noise (SEC signatures, digits, generic)
        if label.startswith("/s/") or label.isdigit() or len(label) < 2:
            continue
        try:
            key = normalize_company_name(label)
        except Exception:
            key = label.lower()
        if key and len(key) >= 2 and key not in seeds:
            seeds[key] = label
    return seeds


@dataclass
class Extraction:
    doc_id: str
    source: str
    doc_type: str
    title: str
    mentions: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class EvidenceIngestor:
    """Parses documents into mentions + relations with confidence scores."""

    def __init__(self, seed_entities: dict[str, str] | None = None) -> None:
        self._normalize = normalize_company_name
        self._seed = seed_entities or SEED_ENTITIES

    @classmethod
    def from_registry(cls, db_path: str = ".tirra_pipeline/pipeline.db", limit: int = 2000) -> EvidenceIngestor:
        """Build an ingestor seeded from the real entity registry (rich extraction)."""
        return cls(seed_entities=seed_entities_from_registry(db_path=db_path, limit=limit))

    # ── Public API ───────────────────────────────────────────────────────────
    def ingest_text(
        self, *, doc_id: str, text: str, source: str = "", title: str = "", doc_type: str = "text"
    ) -> Extraction:
        """Process raw text into an Extraction (mentions + relations)."""
        text = (text or "").strip()
        ext = Extraction(doc_id=doc_id, source=source, doc_type=doc_type, title=title, text=text)
        if not text:
            return ext
        sentences = _SENT_SPLIT.split(text)
        for si, sent in enumerate(sentences):
            self._extract_sentence(ext, sent.strip(), si)
        self._build_relations(ext)
        return ext

    def ingest_pdf(self, *, doc_id: str, path: str, title: str = "", source: str = "") -> Extraction:
        """Read a PDF and ingest its text."""
        from pypdf import PdfReader

        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return self.ingest_text(
            doc_id=doc_id, text=text, source=source or str(path), title=title or Path(path).stem, doc_type="pdf"
        )

    def ingest_csv(self, *, doc_id: str, path: str, title: str = "", source: str = "") -> Extraction:
        """Flatten a CSV into text and ingest (coarse but deterministic)."""
        import csv

        rows = []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                rows.append(" ".join(c for c in row if c))
        text = "\n".join(rows)
        return self.ingest_text(
            doc_id=doc_id, text=text, source=source or str(path), title=title or Path(path).stem, doc_type="csv"
        )

    # ── Extraction ───────────────────────────────────────────────────────────
    def _extract_sentence(self, ext: Extraction, sentence: str, si: int) -> None:
        # 1. Seed matches (canonical, high confidence)
        for key, label in self._seed.items():
            if re.search(rf"\b{re.escape(label)}\b", sentence, re.IGNORECASE):
                ext.mentions.append(self._mention("company", key, label, sentence, 0, 0.95, si))
        # 2. Capitalized-name fallback (lower confidence, audit-flagged)
        for match in _UPPER_WORDS.finditer(sentence):
            raw = match.group().strip()
            # skip sentence starts that are just ordinary words / digit-led
            if " " not in raw and _is_likely_ticker(raw):
                continue
            try:
                key = self._normalize(raw)
            except ValueError:
                continue
            if key in {m["entity_key"] for m in ext.mentions}:
                continue
            ext.mentions.append(self._mention("org", key, raw, sentence, match.start(), 0.6, si))

    @staticmethod
    def _mention(
        entity_type: str, key: str, raw: str, sentence: str, pos: int, confidence: float, sentence_index: int
    ) -> dict[str, Any]:
        return {
            "entity_type": entity_type,
            "entity_key": key,
            "raw_name": raw,
            "sentence": sentence[:200],
            "position": pos,
            "confidence": confidence,
            "sentence_index": sentence_index,
        }

    def _build_relations(self, ext: Extraction) -> None:
        """Link mentions that co-occur within the same sentence (strong signal)
        or in immediately adjacent sentences (weak signal) -- NOT anywhere else
        in the document. See module-level comment on MAX_MENTIONS_PER_SENTENCE
        for why (this used to be unbounded whole-document pairwise linking).
        """
        by_sentence: dict[int, list[dict[str, Any]]] = {}
        for m in ext.mentions:
            by_sentence.setdefault(m["sentence_index"], []).append(m)

        for si, ms in by_sentence.items():
            if len(ms) > MAX_MENTIONS_PER_SENTENCE:
                logger.warning(
                    "evidence ingest: doc=%s sentence=%d has %d entity mentions; "
                    "capping to %d for linking (dropping %d) to bound pairwise growth",
                    ext.doc_id,
                    si,
                    len(ms),
                    MAX_MENTIONS_PER_SENTENCE,
                    len(ms) - MAX_MENTIONS_PER_SENTENCE,
                )
                by_sentence[si] = ms[:MAX_MENTIONS_PER_SENTENCE]

        relations: list[dict[str, Any]] = []

        def _pair(mi: dict[str, Any], mj: dict[str, Any], relation: str, conf: float, snip: str) -> None:
            if mi["entity_key"] == mj["entity_key"]:
                return
            relations.append(
                {
                    "doc_id": ext.doc_id,
                    "entity_a": mi["entity_key"],
                    "entity_b": mj["entity_key"],
                    "relation": relation,
                    "confidence": conf,
                    "evidence": snip,
                }
            )

        indices = sorted(by_sentence)
        for si in indices:
            ms = by_sentence[si]
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    _pair(ms[i], ms[j], "same_sentence", 0.9, (ms[i]["sentence"] or "")[:200])

        # Adjacent sentences only (weaker signal) -- not the whole document.
        for si, nxt in zip(indices, indices[1:]):
            if nxt - si != 1:
                continue
            for mi in by_sentence[si]:
                for mj in by_sentence[nxt]:
                    _pair(mi, mj, "nearby", 0.4, f"{mi['raw_name']} .. {mj['raw_name']}")

        if len(relations) > MAX_LINKS_PER_DOCUMENT:
            logger.warning(
                "evidence ingest: doc=%s produced %d candidate links (> cap %d); "
                "truncating to the highest-confidence %d",
                ext.doc_id,
                len(relations),
                MAX_LINKS_PER_DOCUMENT,
                MAX_LINKS_PER_DOCUMENT,
            )
            relations.sort(key=lambda r: r["confidence"], reverse=True)
            relations = relations[:MAX_LINKS_PER_DOCUMENT]

        ext.relations = relations


def _is_likely_ticker(raw: str) -> bool:
    """Heuristic: uppercase run of 1 word with only letters could be a ticker."""
    if not raw.isupper():
        return False
    return len(raw) <= 5 and all(c.isalpha() for c in raw)


def ingest_to_store(
    store, ingestor: EvidenceIngestor, *, doc_id, text=None, path=None, source="", title="", doc_type="text"
) -> bool:
    """Convenience: run an ingestion and write mentions+relations into a store.

    Returns True if the document was newly ingested, False if duplicate.
    """
    if text is not None:
        ext = ingestor.ingest_text(doc_id=doc_id, text=text, source=source, title=title, doc_type=doc_type)
    else:
        # path-based: dispatch on doc_type
        if doc_type == "pdf":
            ext = ingestor.ingest_pdf(doc_id=doc_id, path=path, title=title, source=source)
        elif doc_type == "csv":
            ext = ingestor.ingest_csv(doc_id=doc_id, path=path, title=title, source=source)
        else:
            ext = ingestor.ingest_text(
                doc_id=doc_id,
                text=Path(path).read_text(encoding="utf-8"),
                source=source,
                title=title,
                doc_type=doc_type,
            )

    new = store.add_document(
        doc_id=doc_id, source=ext.source, doc_type=ext.doc_type, title=ext.title, text_hash=ext.text_hash
    )
    if not new:
        return False  # duplicate
    for m in ext.mentions:
        store.add_mention(
            doc_id=doc_id,
            entity_type=m["entity_type"],
            entity_key=m["entity_key"],
            raw_name=m["raw_name"],
            sentence=m["sentence"],
            position=m["position"],
            confidence=m["confidence"],
        )
    for r in ext.relations:
        store.add_link(
            doc_id=doc_id,
            entity_a=r["entity_a"],
            entity_b=r["entity_b"],
            relation=r["relation"],
            confidence=r["confidence"],
            evidence=r["evidence"],
        )
    return True


__all__ = ["EvidenceIngestor", "Extraction", "ingest_to_store"]
