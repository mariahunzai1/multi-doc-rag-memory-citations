"""
tests/test_contracts.py
=======================
Offline unit tests for core/config.py and core/schemas.py.

Guarantees:
- 100% offline — no model downloads, no network calls, no external servers.
- Zero cost ($0) — no paid API keys required at any point.
- Full coverage of:
    * Default $0 config loading
    * .env environment variable overrides via monkeypatch
    * Negative rerank_threshold acceptance (ms-marco raw logits)
    * Valid construction of all 6 Pydantic v2 schema models
    * ValidationError on required-field omission, bad literals, and
      out-of-range field values

Run with:
    pytest tests/test_contracts.py -v
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import Settings
from core.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentChunk,
    IngestFileResult,
    RetrievedPassage,
)


# ---------------------------------------------------------------------------
# Helpers — minimal valid payloads for each model
# ---------------------------------------------------------------------------

VALID_CHUNK_DATA: dict = {
    "chunk_id": "11111111-1111-1111-1111-111111111111",
    "doc_id": "22222222-2222-2222-2222-222222222222",
    "sha256_hash": "a" * 64,
    "content": "The quick brown fox jumps over the lazy dog.",
    "source_filename": "sample.pdf",
    "file_type": "pdf",
    "page_or_sheet": 1,
    "chunk_index": 0,
}

VALID_CITATION_DATA: dict = {
    "source_filename": "sample.pdf",
    "location": "Page 3",
    "excerpt": "The quick brown fox jumps over the lazy dog.",
    "relevance_score": 0.87,
}


# ===========================================================================
# 1.  core/config.py — Settings
# ===========================================================================


class TestConfigDefaults:
    """Verify that Settings() loads cleanly with zero .env file present."""

    def test_default_config_loads(self) -> None:
        """Settings constructs without any .env or env vars set."""
        s = Settings()
        assert s is not None

    def test_default_embedding_model(self) -> None:
        """Default embedding model is the $0 local BGE model."""
        s = Settings()
        assert s.embedding_model == "BAAI/bge-small-en-v1.5"

    def test_default_reranker_model(self) -> None:
        """Default reranker is the $0 local ms-marco CrossEncoder."""
        s = Settings()
        assert s.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_default_llm_base_url(self) -> None:
        """Default LLM endpoint is local Ollama ($0)."""
        s = Settings()
        assert s.llm_base_url == "http://localhost:11434/v1"

    def test_default_llm_api_key(self) -> None:
        """Default API key is 'ollama' — no paid key required."""
        s = Settings()
        assert s.llm_api_key == "ollama"

    def test_default_llm_model(self) -> None:
        s = Settings()
        assert s.llm_model == "llama3.2:3b"

    def test_default_llm_temperature(self) -> None:
        s = Settings()
        assert s.llm_temperature == 0.1

    def test_default_chunk_size(self) -> None:
        s = Settings()
        assert s.chunk_size == 800

    def test_default_chunk_overlap(self) -> None:
        s = Settings()
        assert s.chunk_overlap == 150

    def test_default_dense_top_k(self) -> None:
        s = Settings()
        assert s.dense_top_k == 20

    def test_default_bm25_top_k(self) -> None:
        s = Settings()
        assert s.bm25_top_k == 20

    def test_default_rrf_candidates(self) -> None:
        s = Settings()
        assert s.rrf_candidates == 15

    def test_default_rrf_k(self) -> None:
        s = Settings()
        assert s.rrf_k == 60

    def test_default_rerank_threshold(self) -> None:
        """Default rerank_threshold is exactly -1.5 (negative float)."""
        s = Settings()
        assert s.rerank_threshold == -1.5

    def test_default_chroma_persist_dir(self) -> None:
        s = Settings()
        assert s.chroma_persist_dir == "./chroma_db"

    def test_default_sqlite_db_path(self) -> None:
        s = Settings()
        assert s.sqlite_db_path == "./rag_memory.sqlite"

    def test_default_bm25_index_path(self) -> None:
        s = Settings()
        assert s.bm25_index_path == "./chroma_db/bm25_corpus.json"

    def test_default_auto_seed_samples(self) -> None:
        """Auto-seeding is enabled by default for Streamlit Community Cloud."""
        s = Settings()
        assert s.auto_seed_samples is True


class TestConfigEnvOverride:
    """.env / environment variable overrides propagate correctly."""

    def test_env_override_llm_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_MODEL env var overrides the default llama3.2:3b."""
        monkeypatch.setenv("LLM_MODEL", "mistral")
        s = Settings()
        assert s.llm_model == "mistral"

    def test_env_override_llm_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_BASE_URL can point to any OpenAI-compatible free-tier endpoint."""
        monkeypatch.setenv("LLM_BASE_URL", "https://api.together.ai/v1")
        s = Settings()
        assert s.llm_base_url == "https://api.together.ai/v1"

    def test_env_override_embedding_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        s = Settings()
        assert s.embedding_model == "BAAI/bge-base-en-v1.5"

    def test_env_override_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHUNK_SIZE", "512")
        s = Settings()
        assert s.chunk_size == 512

    def test_env_override_rerank_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """rerank_threshold can be overridden to any float, including negative."""
        monkeypatch.setenv("RERANK_THRESHOLD", "-3.0")
        s = Settings()
        assert s.rerank_threshold == -3.0

    def test_env_unknown_key_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown env vars are silently ignored (extra='ignore')."""
        monkeypatch.setenv("TOTALLY_UNKNOWN_KEY", "whatever")
        s = Settings()  # Must not raise
        assert s is not None


class TestConfigNegativeRerankThreshold:
    """
    ms-marco-MiniLM-L-6-v2 outputs raw logits — negative values are normal
    for irrelevant passage pairs.  Settings must accept them without error.
    """

    def test_negative_threshold_constructor(self) -> None:
        """Directly passing a negative float to the constructor works."""
        s = Settings(rerank_threshold=-3.7)
        assert s.rerank_threshold == pytest.approx(-3.7)

    def test_very_negative_threshold(self) -> None:
        """Extreme negative values must also be accepted."""
        s = Settings(rerank_threshold=-100.0)
        assert s.rerank_threshold == -100.0

    def test_zero_threshold(self) -> None:
        """Zero is a valid threshold (boundary between logit ranges)."""
        s = Settings(rerank_threshold=0.0)
        assert s.rerank_threshold == 0.0

    def test_positive_threshold(self) -> None:
        """Positive threshold also valid (strict confidence requirement)."""
        s = Settings(rerank_threshold=1.5)
        assert s.rerank_threshold == 1.5


# ===========================================================================
# 2.  core/schemas.py — DocumentChunk
# ===========================================================================


class TestDocumentChunk:

    def test_valid_construction(self) -> None:
        """Valid DocumentChunk constructs without error."""
        chunk = DocumentChunk(**VALID_CHUNK_DATA)
        assert chunk.chunk_id == VALID_CHUNK_DATA["chunk_id"]
        assert chunk.file_type == "pdf"
        assert chunk.chunk_index == 0

    def test_xlsx_sheet_name(self) -> None:
        """page_or_sheet accepts a string (Excel sheet name)."""
        data = {**VALID_CHUNK_DATA, "file_type": "xlsx", "page_or_sheet": "Revenue Q1"}
        chunk = DocumentChunk(**data)
        assert chunk.page_or_sheet == "Revenue Q1"

    def test_ingested_at_defaults_to_utc(self) -> None:
        """ingested_at is auto-populated as a timezone-aware UTC datetime."""
        chunk = DocumentChunk(**VALID_CHUNK_DATA)
        from datetime import timezone
        assert chunk.ingested_at.tzinfo is not None

    def test_missing_required_field_raises(self) -> None:
        """Omitting a required field raises ValidationError."""
        data = {k: v for k, v in VALID_CHUNK_DATA.items() if k != "content"}
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(**data)
        assert "content" in str(exc_info.value)

    def test_invalid_file_type_raises(self) -> None:
        """An unlisted file_type literal raises ValidationError."""
        data = {**VALID_CHUNK_DATA, "file_type": "csv"}
        with pytest.raises(ValidationError):
            DocumentChunk(**data)

    def test_frozen_model_immutable(self) -> None:
        """
        DocumentChunk is frozen — direct attribute assignment raises
        ValidationError (Pydantic v2 frozen model behaviour).
        object.__setattr__ bypasses Python descriptors entirely and is not
        a valid mutation path; use normal attribute assignment to trigger
        Pydantic's frozen guard.
        """
        from pydantic import ValidationError as PydanticValidationError

        chunk = DocumentChunk(**VALID_CHUNK_DATA)
        with pytest.raises((PydanticValidationError, TypeError)):
            chunk.content = "mutated"  # type: ignore[misc]


# ===========================================================================
# 3.  core/schemas.py — IngestFileResult
# ===========================================================================


class TestIngestFileResult:

    def test_indexed_status_valid(self) -> None:
        result = IngestFileResult(
            filename="report.pdf",
            sha256_hash="b" * 64,
            status="indexed",
            chunks_added=12,
        )
        assert result.status == "indexed"
        assert result.chunks_added == 12
        assert result.error_detail is None

    def test_skipped_status_valid(self) -> None:
        result = IngestFileResult(
            filename="report.pdf",
            sha256_hash="b" * 64,
            status="skipped",
            chunks_added=0,
        )
        assert result.status == "skipped"
        assert result.chunks_added == 0

    def test_error_status_with_detail(self) -> None:
        result = IngestFileResult(
            filename="corrupt.pdf",
            sha256_hash="c" * 64,
            status="error",
            chunks_added=0,
            error_detail="Failed to open PDF: file is encrypted.",
        )
        assert result.status == "error"
        assert result.error_detail is not None

    def test_invalid_status_literal_raises(self) -> None:
        """'done' is not a valid status literal — must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IngestFileResult(
                filename="f.pdf",
                sha256_hash="d" * 64,
                status="done",          # type: ignore[arg-type]
                chunks_added=0,
            )
        assert "done" in str(exc_info.value) or "status" in str(exc_info.value)

    def test_negative_chunks_added_raises(self) -> None:
        """chunks_added has ge=0 constraint — negative values must raise."""
        with pytest.raises(ValidationError):
            IngestFileResult(
                filename="f.pdf",
                sha256_hash="e" * 64,
                status="indexed",
                chunks_added=-1,
            )


# ===========================================================================
# 4.  core/schemas.py — RetrievedPassage
# ===========================================================================


class TestRetrievedPassage:

    def _make_chunk(self) -> DocumentChunk:
        return DocumentChunk(**VALID_CHUNK_DATA)

    def test_valid_construction(self) -> None:
        passage = RetrievedPassage(
            chunk=self._make_chunk(),
            dense_score=0.91,
            bm25_score=14.3,
            rrf_score=0.032,
            rerank_score=2.1,
        )
        assert passage.rerank_score == pytest.approx(2.1)

    def test_negative_rerank_score_accepted(self) -> None:
        """
        rerank_score may be negative — ms-marco outputs raw logits.
        The model must NOT clamp or reject negative values.
        """
        passage = RetrievedPassage(
            chunk=self._make_chunk(),
            dense_score=0.55,
            bm25_score=3.1,
            rrf_score=0.015,
            rerank_score=-2.1,
        )
        assert passage.rerank_score == pytest.approx(-2.1)

    def test_very_negative_rerank_score(self) -> None:
        """Even very negative logits (irrelevant passages) are valid."""
        passage = RetrievedPassage(
            chunk=self._make_chunk(),
            dense_score=0.1,
            bm25_score=0.5,
            rrf_score=0.005,
            rerank_score=-9.99,
        )
        assert passage.rerank_score == pytest.approx(-9.99)


# ===========================================================================
# 5.  core/schemas.py — Citation
# ===========================================================================


class TestCitation:

    def test_valid_citation(self) -> None:
        citation = Citation(**VALID_CITATION_DATA)
        assert citation.source_filename == "sample.pdf"
        assert citation.location == "Page 3"

    def test_xlsx_sheet_location(self) -> None:
        data = {**VALID_CITATION_DATA, "location": "Sheet: Revenue Q1"}
        citation = Citation(**data)
        assert "Revenue" in citation.location

    def test_missing_field_raises(self) -> None:
        data = {k: v for k, v in VALID_CITATION_DATA.items() if k != "excerpt"}
        with pytest.raises(ValidationError):
            Citation(**data)


# ===========================================================================
# 6.  core/schemas.py — ChatRequest
# ===========================================================================


class TestChatRequest:

    def test_valid_chat_request(self) -> None:
        req = ChatRequest(session_id="sess-001", question="What is RAG?")
        assert req.session_id == "sess-001"
        assert req.top_k == 4  # default

    def test_top_k_explicit(self) -> None:
        req = ChatRequest(session_id="s", question="Hello?", top_k=10)
        assert req.top_k == 10

    def test_question_min_length(self) -> None:
        """Empty question must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(session_id="s", question="")
        assert "question" in str(exc_info.value)

    def test_question_max_length(self) -> None:
        """Question exceeding 2000 chars must raise ValidationError."""
        with pytest.raises(ValidationError):
            ChatRequest(session_id="s", question="x" * 2001)

    def test_question_exactly_2000_chars(self) -> None:
        """Exactly 2000 characters is the boundary — must be valid."""
        req = ChatRequest(session_id="s", question="a" * 2000)
        assert len(req.question) == 2000

    def test_top_k_lower_bound(self) -> None:
        """top_k=0 is below ge=1 and must raise ValidationError."""
        with pytest.raises(ValidationError):
            ChatRequest(session_id="s", question="Hello?", top_k=0)

    def test_top_k_upper_bound(self) -> None:
        """top_k=16 exceeds le=15 and must raise ValidationError."""
        with pytest.raises(ValidationError):
            ChatRequest(session_id="s", question="Hello?", top_k=16)

    def test_top_k_boundary_values(self) -> None:
        """top_k=1 and top_k=15 are the valid boundaries."""
        low = ChatRequest(session_id="s", question="Q?", top_k=1)
        high = ChatRequest(session_id="s", question="Q?", top_k=15)
        assert low.top_k == 1
        assert high.top_k == 15


# ===========================================================================
# 7.  core/schemas.py — ChatResponse
# ===========================================================================


class TestChatResponse:

    def test_valid_chat_response(self) -> None:
        resp = ChatResponse(
            session_id="sess-001",
            answer="RAG stands for Retrieval-Augmented Generation.",
            standalone_query="What is RAG?",
            citations=[Citation(**VALID_CITATION_DATA)],
            abstained=False,
            latency_ms=342.7,
        )
        assert resp.abstained is False
        assert len(resp.citations) == 1

    def test_abstained_true_empty_citations(self) -> None:
        """
        When abstained=True the citations list may be empty.
        This is the expected state when rerank_score < rerank_threshold.
        """
        resp = ChatResponse(
            session_id="sess-002",
            answer="I don't have enough information to answer that question.",
            standalone_query="What is the revenue for Q5 2099?",
            citations=[],
            abstained=True,
            latency_ms=112.0,
        )
        assert resp.abstained is True
        assert resp.citations == []

    def test_default_abstained_is_false(self) -> None:
        """abstained defaults to False when not explicitly set."""
        resp = ChatResponse(
            session_id="s",
            answer="Answer.",
            standalone_query="Q?",
            latency_ms=50.0,
        )
        assert resp.abstained is False
        assert resp.citations == []

    def test_missing_required_field_raises(self) -> None:
        """Missing latency_ms raises ValidationError."""
        with pytest.raises(ValidationError):
            ChatResponse(
                session_id="s",
                answer="A",
                standalone_query="Q",
                # latency_ms intentionally omitted
            )
