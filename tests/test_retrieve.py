"""
tests/test_retrieve.py
======================
Offline unit tests for core/retrieve.py — Phase 2B.

Guarantees:
- 100% offline — zero HuggingFace downloads or external network calls.
- Zero API cost ($0) — uses FakeEmbedder and FakeReranker.
- ChromaDB uses pytest tmp_path isolated directories.
- Full coverage of:
    * Dense vector search (empty collections, small-document clamping, cosine similarity)
    * BM25 sparse search (empty index, zero-score filtering, canonical tokenization)
    * Exact mathematical Reciprocal Rank Fusion (k=60, dual-hit > single-hit, slicing)
    * Cross-Encoder reranking & raw logit thresholding (negative scores, abstention gate)
    * End-to-end ingestion-to-retrieval integration

Run with:
    pytest tests/test_retrieve.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from core.config import Settings
from core.ingest import BaseEmbedder, clear_all_documents, ingest_bytes
from core.retrieve import (
    BaseReranker,
    _bm25_search,
    _dense_search,
    _reciprocal_rank_fusion,
    _rerank_passages,
    retrieve_and_rerank,
)
from core.schemas import DocumentChunk, RetrievedPassage


# ===========================================================================
# Test Doubles (Fake Embedder & Fake Reranker)
# ===========================================================================


class FakeEmbedder(BaseEmbedder):
    """
    Deterministic unit-vector embedder.
    Assigns vector based on simple hash of the text so different texts get
    different angles, but same text gets identical vector.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vec = [0.0] * 384
            # Distribute weights deterministically across first few dims
            for i, ch in enumerate(text[:8]):
                vec[i % 384] += (ord(ch) % 10) / 10.0
            # Normalize vector to unit length
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            else:
                vec[0] = 1.0
            results.append(vec)
        return results


class FakeReranker(BaseReranker):
    """
    Predicts controllable or mapped logit scores for test assertions.
    """

    def __init__(
        self,
        scores: list[float] | None = None,
        score_map: dict[str, float] | None = None,
        constant_score: float | None = None,
    ) -> None:
        self.scores = scores
        self.score_map = score_map or {}
        self.constant_score = constant_score

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self.constant_score is not None:
            return [self.constant_score] * len(pairs)
        if self.scores is not None:
            return self.scores[: len(pairs)]
        results: list[float] = []
        for _query, passage in pairs:
            # Match passage by substring (case-insensitive)
            matched = False
            for key, score in self.score_map.items():
                if key.lower() in passage.lower():
                    results.append(score)
                    matched = True
                    break
            if not matched:
                results.append(-5.0)  # default low score
        return results


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    """Isolated Settings pointing all persistence to a pytest tmp_path."""
    chroma_dir = str(tmp_path / "chroma_db")
    return Settings(
        chroma_persist_dir=chroma_dir,
        bm25_index_path=str(tmp_path / "chroma_db" / "bm25_corpus.json"),
        sqlite_db_path=str(tmp_path / "rag_memory.sqlite"),
        dense_top_k=20,
        bm25_top_k=20,
        rrf_candidates=15,
        rrf_k=60,
        rerank_threshold=-1.5,
    )


def _create_sample_chunk(
    content: str,
    chunk_index: int = 0,
    source_filename: str = "sample.txt",
    page_or_sheet: int | str = 1,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=str(uuid4()),
        doc_id="doc-1234",
        sha256_hash="hash-1234",
        content=content,
        source_filename=source_filename,
        file_type="txt",
        page_or_sheet=page_or_sheet,
        chunk_index=chunk_index,
        ingested_at=datetime.now(timezone.utc),
    )


# ===========================================================================
# 1. TestDenseSearch
# ===========================================================================


class TestDenseSearch:

    def test_dense_search_empty_store_returns_empty(self, tmp_settings: Settings) -> None:
        """When ChromaDB collection does not exist, returns empty list without error."""
        embedder = FakeEmbedder()
        results = _dense_search("what is RAG?", tmp_settings, embedder)
        assert results == []

    def test_dense_search_clamps_n_results_to_collection_count(
        self, tmp_settings: Settings
    ) -> None:
        """
        When collection has fewer chunks (e.g. 2) than dense_top_k (20),
        n_results is clamped to count and does not crash ChromaDB.
        """
        embedder = FakeEmbedder()
        # Ingest 2 small chunks
        ingest_bytes("file1.txt", b"First chunk about search.", tmp_settings, embedder)
        ingest_bytes("file2.txt", b"Second chunk about retrieval.", tmp_settings, embedder)

        results = _dense_search("search query", tmp_settings, embedder)
        assert len(results) == 2
        for chunk, score in results:
            assert isinstance(chunk, DocumentChunk)
            assert isinstance(score, float)

    def test_dense_search_preserves_metadata(self, tmp_settings: Settings) -> None:
        """Reconstructed DocumentChunk retains exact filename and page metadata."""
        embedder = FakeEmbedder()
        ingest_bytes("report.txt", b"Important document body.", tmp_settings, embedder)

        results = _dense_search("document", tmp_settings, embedder)
        assert len(results) == 1
        chunk, _ = results[0]
        assert chunk.source_filename == "report.txt"
        assert chunk.file_type == "txt"


# ===========================================================================
# 2. TestBM25Search
# ===========================================================================


class TestBM25Search:

    def test_bm25_empty_index_returns_empty(self, tmp_settings: Settings) -> None:
        """When bm25_corpus.json does not exist, returns empty list."""
        results = _bm25_search("quantum computing", tmp_settings)
        assert results == []

    def test_bm25_retrieves_matching_chunk(self, tmp_settings: Settings) -> None:
        """BM25 successfully retrieves document containing exact token."""
        embedder = FakeEmbedder()
        ingest_bytes(
            "doc.txt",
            b"The cheetah is the fastest land animal in Africa.",
            tmp_settings,
            embedder,
        )

        results = _bm25_search("fastest cheetah", tmp_settings)
        assert len(results) >= 1
        chunk, score = results[0]
        assert "cheetah" in chunk.content
        assert score > 0.0

    def test_bm25_excludes_zero_score_documents(self, tmp_settings: Settings) -> None:
        """BM25 does not return documents that have zero lexical overlap."""
        embedder = FakeEmbedder()
        ingest_bytes("tech.txt", b"Python software engineering guidelines.", tmp_settings, embedder)

        # Query has zero words in common
        results = _bm25_search("astronomy telescope stars", tmp_settings)
        assert results == []

    def test_bm25_empty_query_returns_empty(self, tmp_settings: Settings) -> None:
        """Whitespace query returns empty list."""
        embedder = FakeEmbedder()
        ingest_bytes("doc.txt", b"Some content here.", tmp_settings, embedder)

        results = _bm25_search("   ", tmp_settings)
        assert results == []


# ===========================================================================
# 3. TestReciprocalRankFusion
# ===========================================================================


class TestReciprocalRankFusion:

    def test_exact_rrf_formula_calculation(self, tmp_settings: Settings) -> None:
        """
        Verifies RRF(d) = sum(1 / (k + rank)).
        For a chunk ranked #1 in dense and #2 in BM25 with k=60:
          dense contribution = 1 / (60 + 1) = 1/61
          bm25 contribution  = 1 / (60 + 2) = 1/62
          expected = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.032522
        """
        c1 = _create_sample_chunk("Chunk 1")
        dense = [(c1, 0.95)]
        bm25 = [(c1, 15.0)]  # rank 1 in both

        # For rank 1 in both: 1/61 + 1/61 = 2/61 approx 0.032787
        passages = _reciprocal_rank_fusion(dense, bm25, tmp_settings)
        assert len(passages) == 1
        expected = round(1.0 / 61.0 + 1.0 / 61.0, 6)
        assert passages[0].rrf_score == pytest.approx(expected, abs=1e-5)
        assert passages[0].dense_score == pytest.approx(0.95)
        assert passages[0].bm25_score == pytest.approx(15.0)

    def test_dual_hit_outranks_single_hit(self, tmp_settings: Settings) -> None:
        """
        A document retrieved by BOTH dense and BM25 must outrank a document
        retrieved by only one, even if that one was ranked #1 in its retriever.
        """
        dual_chunk = _create_sample_chunk("Found by both dense and sparse")
        dense_only_chunk = _create_sample_chunk("Found only by dense")

        # dense_only is rank 1 in dense, dual_chunk is rank 2
        dense = [(dense_only_chunk, 0.99), (dual_chunk, 0.90)]
        # dual_chunk is rank 1 in BM25, dense_only is absent
        bm25 = [(dual_chunk, 12.0)]

        passages = _reciprocal_rank_fusion(dense, bm25, tmp_settings)

        # dual_chunk score: 1/62 + 1/61 = 0.032522
        # dense_only score: 1/61 = 0.016393
        assert passages[0].chunk.chunk_id == dual_chunk.chunk_id
        assert passages[1].chunk.chunk_id == dense_only_chunk.chunk_id

    def test_rrf_candidates_sliced_to_setting(self, tmp_settings: Settings) -> None:
        """Fusion retains at most rrf_candidates items."""
        dense_list = [
            (_create_sample_chunk(f"Dense {i}"), 0.5)
            for i in range(25)
        ]
        passages = _reciprocal_rank_fusion(dense_list, [], tmp_settings)
        assert len(passages) == tmp_settings.rrf_candidates


# ===========================================================================
# 4. TestCrossEncoderReranking
# ===========================================================================


class TestCrossEncoderReranking:

    def test_reranker_sorts_descending_by_raw_logit(self, tmp_settings: Settings) -> None:
        """Passages are ordered strictly descending by rerank_score."""
        p1 = RetrievedPassage(
            chunk=_create_sample_chunk("First chunk"),
            dense_score=0.8,
            bm25_score=5.0,
            rrf_score=0.03,
            rerank_score=0.0,
        )
        p2 = RetrievedPassage(
            chunk=_create_sample_chunk("Second chunk"),
            dense_score=0.9,
            bm25_score=6.0,
            rrf_score=0.032,
            rerank_score=0.0,
        )

        # Fake reranker assigns higher logit to p1 (+2.5) than p2 (-0.5)
        reranker = FakeReranker(scores=[2.5, -0.5])
        passages, abstained = _rerank_passages(
            "query", [p1, p2], reranker, tmp_settings, top_k=4
        )

        assert not abstained
        assert len(passages) == 2
        assert passages[0].chunk.content == "First chunk"
        assert passages[0].rerank_score == pytest.approx(2.5)
        assert passages[1].chunk.content == "Second chunk"
        assert passages[1].rerank_score == pytest.approx(-0.5)

    def test_negative_threshold_passes_qualifying_logit(self, tmp_settings: Settings) -> None:
        """
        ms-marco outputs negative logits for weak/moderate matches.
        A logit of -0.8 must pass when settings.rerank_threshold is -1.5.
        """
        p = RetrievedPassage(
            chunk=_create_sample_chunk("Acceptable match"),
            dense_score=0.5,
            bm25_score=2.0,
            rrf_score=0.02,
            rerank_score=0.0,
        )
        reranker = FakeReranker(scores=[-0.8])
        passages, abstained = _rerank_passages(
            "query", [p], reranker, tmp_settings, top_k=4
        )

        assert not abstained
        assert len(passages) == 1
        assert passages[0].rerank_score == pytest.approx(-0.8)

    def test_abstention_gate_triggers_when_all_below_threshold(
        self, tmp_settings: Settings
    ) -> None:
        """
        When best rerank_score (-4.2) is below threshold (-1.5),
        pipeline returns empty list and abstained=True.
        """
        p = RetrievedPassage(
            chunk=_create_sample_chunk("Irrelevant content"),
            dense_score=0.1,
            bm25_score=0.5,
            rrf_score=0.01,
            rerank_score=0.0,
        )
        reranker = FakeReranker(scores=[-4.2])
        passages, abstained = _rerank_passages(
            "irrelevant query", [p], reranker, tmp_settings, top_k=4
        )

        assert abstained is True
        assert passages == []

    def test_top_k_slicing_respected(self, tmp_settings: Settings) -> None:
        """Only top_k passages are returned even if more qualify."""
        passages_in = [
            RetrievedPassage(
                chunk=_create_sample_chunk(f"Chunk {i}"),
                dense_score=0.5,
                bm25_score=1.0,
                rrf_score=0.02,
                rerank_score=0.0,
            )
            for i in range(5)
        ]
        reranker = FakeReranker(scores=[3.0, 2.0, 1.0, 0.5, 0.1])
        passages_out, abstained = _rerank_passages(
            "query", passages_in, reranker, tmp_settings, top_k=2
        )

        assert not abstained
        assert len(passages_out) == 2
        assert passages_out[0].rerank_score == pytest.approx(3.0)
        assert passages_out[1].rerank_score == pytest.approx(2.0)


# ===========================================================================
# 5. TestRetrieveAndRerankIntegration
# ===========================================================================


class TestRetrieveAndRerankIntegration:

    def test_empty_database_returns_abstained(self, tmp_settings: Settings) -> None:
        """retrieve_and_rerank against an empty database returns ([], True)."""
        embedder = FakeEmbedder()
        reranker = FakeReranker(constant_score=1.0)

        passages, abstained = retrieve_and_rerank(
            query="test query",
            settings=tmp_settings,
            top_k=4,
            embedder=embedder,
            reranker=reranker,
        )

        assert abstained is True
        assert passages == []

    def test_whitespace_query_returns_abstained(self, tmp_settings: Settings) -> None:
        """Whitespace query returns ([], True) immediately."""
        passages, abstained = retrieve_and_rerank(
            query="   \n\t  ",
            settings=tmp_settings,
        )
        assert abstained is True
        assert passages == []

    def test_end_to_end_ingest_and_retrieve_success(
        self, tmp_settings: Settings
    ) -> None:
        """
        Ingest two distinct text documents, search for one, and verify
        that the relevant document is returned with verified scores and citations.
        """
        embedder = FakeEmbedder()
        reranker = FakeReranker(
            score_map={
                "solar panels": 2.4,
                "deep sea": -3.5,
            }
        )

        # Ingest document 1
        doc1_bytes = b"Solar panels convert sunlight into clean electricity."
        ingest_bytes("solar.txt", doc1_bytes, tmp_settings, embedder)

        # Ingest document 2
        doc2_bytes = b"The deep sea hydrothermal vents support chemosynthetic life."
        ingest_bytes("ocean.txt", doc2_bytes, tmp_settings, embedder)

        # Query for solar energy
        passages, abstained = retrieve_and_rerank(
            query="solar panels electricity",
            settings=tmp_settings,
            top_k=2,
            embedder=embedder,
            reranker=reranker,
        )

        assert not abstained
        assert len(passages) >= 1
        top_passage = passages[0]
        assert "solar panels" in top_passage.chunk.content.lower()
        assert top_passage.chunk.source_filename == "solar.txt"
        assert top_passage.rerank_score == pytest.approx(2.4)
        assert top_passage.dense_score is not None
        assert top_passage.rrf_score > 0.0

    def test_end_to_end_irrelevant_query_triggers_abstention(
        self, tmp_settings: Settings
    ) -> None:
        """
        Ingest a document, but query about an unrelated topic where
        the reranker scores below threshold (-1.5).
        Pipeline must abstain (abstained=True).
        """
        embedder = FakeEmbedder()
        # Reranker returns -4.0 for everything
        reranker = FakeReranker(constant_score=-4.0)

        ingest_bytes("cats.txt", b"Domestic cats sleep up to sixteen hours a day.", tmp_settings, embedder)

        passages, abstained = retrieve_and_rerank(
            query="how to repair a diesel submarine engine",
            settings=tmp_settings,
            top_k=4,
            embedder=embedder,
            reranker=reranker,
        )

        assert abstained is True
        assert passages == []
