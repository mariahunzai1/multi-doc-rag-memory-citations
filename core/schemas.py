"""
core/schemas.py
===============
Pydantic v2 domain contracts for the RAG engine.

Design guarantees:
- All models use strict BaseModel with full type annotations (Python 3.11+).
- DocumentChunk is immutable (frozen=True) — chunks must not be mutated
  after ingestion to preserve integrity with the vector store and BM25 index.
- RetrievedPassage wraps DocumentChunk and carries all four retrieval scores.
  rerank_score is an unclamped float because ms-marco-MiniLM-L-6-v2 outputs
  raw logits that are commonly negative for irrelevant passage pairs.
- ChatRequest enforces question length bounds and top_k range at the
  schema layer so the API layer never sees out-of-range values.
- ChatResponse.abstained is True when the best rerank_score falls below
  Settings.rerank_threshold, signalling to the UI that no grounded answer
  could be produced.

Rule compliance:
- Never imports fastapi, uvicorn, or streamlit.
- Pydantic v2 (BaseModel + ConfigDict).
- Zero use of langchain_community.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# DocumentChunk — immutable representation of one indexed text passage
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    """
    A single chunk produced by the ingestion pipeline.

    Frozen so that chunks stored in ChromaDB metadata and the BM25 corpus
    are never accidentally mutated after indexing.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        description="UUID4 string uniquely identifying this chunk."
    )
    doc_id: str = Field(
        description="UUID4 derived from the document's sha256_hash. "
                    "All chunks from the same file share the same doc_id."
    )
    sha256_hash: str = Field(
        description="Hex digest of the raw source file bytes. "
                    "Used for idempotent ingestion (skip on re-upload)."
    )
    content: str = Field(
        description="The verbatim chunk text passed to the embedder and BM25."
    )
    source_filename: str = Field(
        description="Exact original filename preserved from the upload."
    )
    file_type: Literal["pdf", "docx", "xlsx", "txt"] = Field(
        description="Parsed file format."
    )
    page_or_sheet: int | str = Field(
        description="1-based page number for PDF/DOCX, or sheet name for XLSX."
    )
    chunk_index: int = Field(
        description="0-based position of this chunk within the source document."
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this chunk was ingested."
    )


# ---------------------------------------------------------------------------
# IngestFileResult — outcome of a single file upload
# ---------------------------------------------------------------------------

class IngestFileResult(BaseModel):
    """
    Returned by the ingestion pipeline for each processed file.

    status meanings:
    - "indexed"  → file was new; chunks_added > 0
    - "skipped"  → sha256_hash already present; no duplicate chunks created
    - "error"    → parsing or indexing failed; error_detail contains the reason
    """

    model_config = ConfigDict(frozen=True)

    filename: str
    sha256_hash: str
    status: Literal["indexed", "skipped", "error"]
    chunks_added: int = Field(
        ge=0,
        description="Number of new chunks written to the vector store. "
                    "Always 0 for 'skipped' and 'error' statuses."
    )
    error_detail: str | None = Field(
        default=None,
        description="Human-readable error message. Present only when status='error'."
    )


# ---------------------------------------------------------------------------
# RetrievedPassage — a candidate chunk with all four retrieval scores
# ---------------------------------------------------------------------------

class RetrievedPassage(BaseModel):
    """
    A chunk returned by the hybrid retrieval + reranking pipeline,
    carrying scores from every retrieval stage.

    rerank_score is NOT clamped — ms-marco-MiniLM-L-6-v2 outputs raw logits
    that are commonly negative for irrelevant passages.
    """

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    dense_score: float = Field(
        description="Cosine similarity from ChromaDB dense vector search."
    )
    bm25_score: float = Field(
        description="Raw BM25 score from rank_bm25."
    )
    rrf_score: float = Field(
        description="Reciprocal Rank Fusion score (k=60) combining dense and BM25 ranks."
    )
    rerank_score: float = Field(
        description="Raw CrossEncoder logit. May be negative. "
                    "Compared against Settings.rerank_threshold for abstention."
    )


# ---------------------------------------------------------------------------
# Citation — a verifiable source reference attached to a ChatResponse
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """
    A single verifiable source reference grounding part of the generated answer.
    """

    model_config = ConfigDict(frozen=True)

    source_filename: str = Field(
        description="Exact filename of the source document."
    )
    location: str = Field(
        description="Human-readable location, e.g. 'Page 3' or 'Sheet: Revenue Q1'."
    )
    excerpt: str = Field(
        description="Short verbatim passage excerpt supporting the answer."
    )
    relevance_score: float = Field(
        description="CrossEncoder rerank_score for this citation passage."
    )


# ---------------------------------------------------------------------------
# ChatRequest — validated inbound message from the user
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """
    Inbound chat message from the user.

    question is bounded at the schema layer so the API never processes
    empty strings or payloads exceeding 2 000 characters.
    top_k is bounded 1–15 with a default of 4.
    """

    session_id: str = Field(
        description="Opaque session identifier (UUID or user-defined string). "
                    "Keys the SQLite conversation history."
    )
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The user's question. Must be 1–2000 characters."
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=15,
        description="Number of citations to include in the response (1–15)."
    )


# ---------------------------------------------------------------------------
# ChatResponse — the full grounded answer returned to the caller
# ---------------------------------------------------------------------------

class ChatResponse(BaseModel):
    """
    Full response from the RAG pipeline.

    abstained=True signals that no retrieved passage cleared the
    rerank_threshold, so the answer field will contain a safe
    "I don't know" response rather than a hallucinated answer.
    citations is empty when abstained=True.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    answer: str = Field(
        description="Generated answer grounded in the retrieved passages, "
                    "or a safe abstention message when abstained=True."
    )
    standalone_query: str = Field(
        description="Reformulated question after condensing multi-turn history. "
                    "Logged for evaluation and debugging."
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Ordered list of source citations supporting the answer."
    )
    abstained: bool = Field(
        default=False,
        description="True when the best rerank_score < Settings.rerank_threshold."
    )
    latency_ms: float = Field(
        description="End-to-end pipeline latency in milliseconds."
    )
