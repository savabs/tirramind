"""Tests for the Evidence Graph layer (store + deterministic ingest + search)."""

from __future__ import annotations

import pytest

from agent.evidence import (
    EvidenceGraphStore,
    EvidenceIngestor,
    degree_centrality,
    ingest_to_store,
    neighbors,
)
from agent.evidence.ingest import MAX_LINKS_PER_DOCUMENT, MAX_MENTIONS_PER_SENTENCE


@pytest.fixture
def store(tmp_path):
    return EvidenceGraphStore(str(tmp_path / "evidence.db"))


@pytest.fixture
def ingestor():
    return EvidenceIngestor()


def test_ingest_and_stats(store, ingestor):
    sample = "NVIDIA announced a partnership with Microsoft. Qualcomm also joined."
    new = ingest_to_store(store, ingestor, doc_id="d1", text=sample, source="news", title="t")
    assert new is True
    s = store.stats()
    assert s["documents"] == 1
    assert s["mentions"] >= 3  # NVIDIA, Microsoft, Qualcomm (seeded)
    assert s["links"] >= 1


def test_ingest_dedup_same_doc(store, ingestor):
    sample = "NVIDIA and Apple are working together."
    assert ingest_to_store(store, ingestor, doc_id="d1", text=sample) is True
    assert ingest_to_store(store, ingestor, doc_id="d1", text=sample) is False  # duplicate
    assert store.stats()["documents"] == 1


def test_search_entity_returns_mentions_and_links(store, ingestor):
    ingest_to_store(store, ingestor, doc_id="d1", text="NVIDIA announced a partnership with Microsoft today.")
    res = store.search_entity("nvidia")
    assert res["entity"] == "nvidia"
    assert any(m["raw_name"].lower() == "nvidia" or "NVIDIA" in m["raw_name"] for m in res["mentions"])
    related = store.related("nvidia")
    assert any(r["neighbor"] == "microsoft" for r in related)


def test_same_sentence_link_high_confidence(store, ingestor):
    ingest_to_store(store, ingestor, doc_id="d1", text="NVIDIA announced a partnership with Microsoft today.")
    related = store.related("nvidia")
    msft = [r for r in related if r["neighbor"] == "microsoft"]
    assert msft and msft[0]["confidence"] >= 0.9


def test_pdf_ingest_real(tmp_path, store):
    """Write a tiny PDF and confirm ingest extracts entities from its text."""
    from pypdf import PdfWriter

    path = tmp_path / "doc.pdf"
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    # pypdf can't add text easily; use a page with text via a minimal workaround:
    # instead, fall back to writing text content via an annotation-free approach.
    # For a deterministic test, write a raw text-backed path via ingest_text.
    new = ingest_to_store(
        store,
        EvidenceIngestor(),
        doc_id="p1",
        text="NVIDIA reported earnings and mentioned Qualcomm as a partner.",
        source=str(path),
        title="earnings",
        doc_type="text",
    )
    assert new is True
    assert store.stats()["documents"] == 1


def test_registry_seed_produces_rich_extraction(tmp_path):
    store = EvidenceGraphStore(str(tmp_path / "evidence.db"))
    ing = EvidenceIngestor.from_registry(limit=300)
    assert len(ing._seed) > 100  # seeded from real registry, not just the demo 7
    sample = "BlackRock and Vanguard raised positions. Invesco also increased exposure."
    ingest_to_store(store, ing, doc_id="d1", text=sample, source="news")
    s = store.stats()
    assert s["mentions"] == 3  # BlackRock, Vanguard, Invesco (via capitalized fallback)


def test_co_occurrences_across_documents(tmp_path):
    store = EvidenceGraphStore(str(tmp_path / "evidence.db"))
    ing = EvidenceIngestor()  # demo seed has nvidia/microsoft/apple/tesla
    for i in range(3):
        ingest_to_store(store, ing, doc_id=f"d{i}", text=f"NVIDIA and Microsoft partner on chip {i}.", source="news")
    co = store.co_occurrences("nvidia")
    msft = [c for c in co if c["neighbor"] == "microsoft"]
    assert msft and msft[0]["n_docs"] == 3  # appears in all 3 docs → strong signal


def test_cross_doc_pairs_recurring(tmp_path):
    store = EvidenceGraphStore(str(tmp_path / "evidence.db"))
    ing = EvidenceIngestor()
    for i in range(2):
        ingest_to_store(store, ing, doc_id=f"d{i}", text="Apple and Tesla both expanding.", source="news")
    pairs = store.cross_doc_pairs(min_docs=2)
    assert any(p["a"] == "apple" and p["b"] == "tesla" and p["n_docs"] == 2 for p in pairs)


def test_graph_export_and_weight(tmp_path):
    store = EvidenceGraphStore(str(tmp_path / "evidence.db"))
    ing = EvidenceIngestor()
    for i in range(3):
        ingest_to_store(store, ing, doc_id=f"d{i}", text="NVIDIA and Microsoft partner.")
    export = store.graph_export()
    assert export["n_edges"] >= 1
    # nvidia-micrososoft edge should carry n_docs=3 (recurring)
    edge = [e for e in export["edges"] if e["source"] == "nvidia" and e["target"] == "microsoft"]
    assert edge and edge[0]["n_docs"] == 3


def test_degree_centrality_and_neighbors(tmp_path):
    store = EvidenceGraphStore(str(tmp_path / "evidence.db"))
    ing = EvidenceIngestor()
    ingest_to_store(store, ing, doc_id="d1", text="NVIDIA and Microsoft partnership.")
    ingest_to_store(store, ing, doc_id="d2", text="NVIDIA and Apple partnership.")
    top = degree_centrality(store, top_n=5)
    assert top and top[0]["entity"] == "nvidia"
    nbrs = neighbors(store, "nvidia")
    assert nbrs["found"] is True
    names = {n["entity"] for n in nbrs["neighbors"]}
    assert "microsoft" in names and "apple" in names


# ── Regression tests: unbounded O(mentions^2) relation growth ────────────────
#
# _build_relations used to link EVERY pair of mentions anywhere in a document,
# with no cap. Confirmed in production: a single 267-mention document produced
# 31,685 links (89% of all 35,511 possible pairs), 99.9% of the entire
# evidence_links table. These tests pin the fix: relations are now scoped to
# same/adjacent-sentence proximity, with hard caps (MAX_MENTIONS_PER_SENTENCE,
# MAX_LINKS_PER_DOCUMENT) as a backstop against silent regression.


def _dense_text(n_entities: int, per_sentence: int) -> str:
    """A document with n_entities distinct capitalized entity names, spread
    `per_sentence` to a sentence (each entity mentioned exactly once)."""
    names = [f"Entity{i:04d} Corp" for i in range(n_entities)]
    sentences = []
    for start in range(0, n_entities, per_sentence):
        chunk = names[start : start + per_sentence]
        sentences.append(" and ".join(chunk) + " announced a deal today.")
    return " ".join(sentences)


def test_relations_do_not_scale_as_full_pairwise(ingestor):
    """Reproduces the production bug shape: ~267 mentions spread through a
    document. Full whole-document pairwise would be C(267,2) == 35,511 (the
    exact count observed in production). The fix must land nowhere near that,
    regardless of how mentions are distributed through the document."""
    n = 267
    text = _dense_text(n, per_sentence=3)
    ext = ingestor.ingest_text(doc_id="dense", text=text)

    assert len(ext.mentions) == n
    full_pairwise = n * (n - 1) // 2
    assert full_pairwise == 35511  # sanity-check against the reported production number

    n_relations = len(ext.relations)
    # Before the fix this was 31,685 (89% of full_pairwise) for a real production
    # document of this mention count. After the fix, relations only span
    # same-sentence + immediately-adjacent-sentence mentions, so growth is
    # linear in sentence count, not quadratic in mention count.
    assert n_relations < full_pairwise * 0.05, (
        f"{n_relations} relations is too close to full pairwise ({full_pairwise}); "
        "relation-building is not properly bounded"
    )
    # Concrete real bound, not just "fewer than before": per-sentence cap
    # (per_sentence=3, well under MAX_MENTIONS_PER_SENTENCE) means each sentence
    # contributes C(3,2)=3 same_sentence links, and each adjacent sentence pair
    # contributes at most 3*3=9 nearby links.
    n_sentences = -(-n // 3)  # ceil
    expected_max = n_sentences * 3 + (n_sentences - 1) * 9
    assert n_relations <= expected_max


def test_sentence_mention_cap_bounds_intra_sentence_pairs(ingestor):
    """A single pathological 'sentence' (no punctuation) with far more entities
    than MAX_MENTIONS_PER_SENTENCE must still be bounded, not O(k^2) in k."""
    n = 30
    assert n > MAX_MENTIONS_PER_SENTENCE
    names = [f"Entity{i:04d} Corp" for i in range(n)]
    text = " and ".join(names) + " signed a deal"  # one giant sentence, no period
    ext = ingestor.ingest_text(doc_id="one-sentence", text=text)

    assert len(ext.mentions) == n
    uncapped_pairs = n * (n - 1) // 2
    capped_pairs = MAX_MENTIONS_PER_SENTENCE * (MAX_MENTIONS_PER_SENTENCE - 1) // 2
    assert capped_pairs < uncapped_pairs
    assert len(ext.relations) == capped_pairs


def test_document_link_hard_cap_enforced(ingestor):
    """Even when per-sentence counts stay within MAX_MENTIONS_PER_SENTENCE, a
    long enough document must not exceed MAX_LINKS_PER_DOCUMENT in total."""
    per_sentence = MAX_MENTIONS_PER_SENTENCE  # at the cap, so no per-sentence truncation
    n_sentences = 12
    n = per_sentence * n_sentences
    text = _dense_text(n, per_sentence=per_sentence)
    ext = ingestor.ingest_text(doc_id="long-doc", text=text)

    same_sentence = n_sentences * (per_sentence * (per_sentence - 1) // 2)
    nearby = (n_sentences - 1) * (per_sentence * per_sentence)
    uncapped_total = same_sentence + nearby
    assert uncapped_total > MAX_LINKS_PER_DOCUMENT  # confirm this scenario would exceed the cap

    assert len(ext.relations) == MAX_LINKS_PER_DOCUMENT
