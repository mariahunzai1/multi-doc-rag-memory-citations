"""
core/retrieve.py
================
Phase 2B: Hybrid Retrieval (Dense + BM25 + RRF) & Cross-Encoder Reranking.

Architectural flow:
  1. Dense vector search in ChromaDB via BaseEmbedder (top dense_top_k)
  2. Sparse lexical search with rank_bm25.BM25Okapi (top bm25_top_k)
  3. Reciprocal Rank Fusion (RRF, k=60), sliced to rrf_candidates
  4. CrossEncoder reranking (raw logits, allowing negative values)
  5. Calibrated confidence threshold abstention gate

Rule compliance:
- Never imports fastapi, uvicorn, or streamlit.
- Python 3.11+ strict type annotations throughout.
- Consumes core/config.py and core/schemas.py contracts read-only.
- Zero langchain_community usage.
- Embedder and Reranker interfaces injected via ABCs for 100% offline testing.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from core.config import Settings
from core.ingest import (
    COLLECTION_NAME,
    BaseEmbedder,
    SentenceTransformerEmbedder,
    tokenize_text,
)
from core.schemas import DocumentChunk, RetrievedPassage


# ===========================================================================
# Reranker Interface & Implementations
# ===========================================================================


class BaseReranker(ABC):
    """
    Abstract reranker interface.

    Inject a FakeReranker in tests to verify score thresholds and abstention
    logic without downloading any cross-encoder model weights.
    """

    @abstractmethod
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score a list of (query, passage_content) text pairs.

        Returns:
            One raw logit score per pair. May be negative (ms-marco outputs
            unclamped logits where negative scores denote irrelevant pairs).
        """
        ...


class CrossEncoderReranker(BaseReranker):
    """
    Production CrossEncoder reranker using sentence-transformers.
    Deferred import ensures no weight download unless explicitly instantiated.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]


# ===========================================================================
# Internal Retrieval Helpers
# ===========================================================================


def _reconstruct_chunk_from_meta_and_doc(
    chunk_id: str,
    content: str,
    metadata: dict[str, object],
) -> DocumentChunk:
    """Reconstruct a DocumentChunk from ChromaDB metadata dict and content string."""
    page_val = metadata.get("page_or_sheet", "")
    if isinstance(page_val, str) and page_val.isdigit():
        page_or_sheet: int | str = int(page_val)
    elif isinstance(page_val, (int, str)):
        page_or_sheet = page_val
    else:
        page_or_sheet = str(page_val)

    ingested_str = metadata.get("ingested_at")
    if isinstance(ingested_str, str) and ingested_str:
        ingested_at = datetime.fromisoformat(ingested_str)
    else:
        ingested_at = datetime.now(timezone.utc)

    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=str(metadata.get("doc_id", "")),
        sha256_hash=str(metadata.get("sha256_hash", "")),
        content=content,
        source_filename=str(metadata.get("source_filename", "")),
        file_type=metadata.get("file_type", "txt"),  # type: ignore[arg-type]
        page_or_sheet=page_or_sheet,
        chunk_index=int(metadata.get("chunk_index", 0)),
        ingested_at=ingested_at,
    )


def _dense_search(
    query: str,
    settings: Settings,
    embedder: BaseEmbedder,
) -> list[tuple[DocumentChunk, float]]:
    """
    Execute dense vector search against ChromaDB.

    Safeguards:
      - Gracefully handles non-existent persistent directory or collection.
      - Checks collection.count() and clamps n_results = min(settings.dense_top_k, count)
        to prevent index-out-of-bounds errors on small collections.

    Returns:
        Ranked list of (DocumentChunk, dense_score).
    """
    chroma_dir = Path(settings.chroma_persist_dir)
    if not chroma_dir.exists():
        return []

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return []

    count = collection.count()
    if count == 0:
        return []

    # Safeguard: Clamp n_results to avoid Chroma index bounds error
    n_results = min(settings.dense_top_k, count)

    query_vectors = embedder.embed([query])
    if not query_vectors:
        return []
    query_vector = query_vectors[0]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    candidates: list[tuple[DocumentChunk, float]] = []
    for chunk_id, content, metadata, dist in zip(ids, docs, metas, dists):
        # Cosine distance in Chroma is 1.0 - cosine_similarity.
        # Normalize to similarity score.
        similarity = float(1.0 - dist)
        chunk = _reconstruct_chunk_from_meta_and_doc(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata or {},
        )
        candidates.append((chunk, similarity))

    return candidates


def _bm25_search(
    query: str,
    settings: Settings,
) -> list[tuple[DocumentChunk, float]]:
    """
    Execute sparse lexical search using rank_bm25.BM25Okapi over bm25_corpus.json.

    Safeguards:
      - Gracefully handles missing or empty corpus file.
      - Discards zero-score matches (no token overlap).
      - Loads content directly from corpus record or falls back to ChromaDB lookup.

    Returns:
        Ranked list of (DocumentChunk, bm25_score).
    """
    bm25_path = Path(settings.bm25_index_path)
    if not bm25_path.exists():
        return []

    try:
        with bm25_path.open("r", encoding="utf-8") as fh:
            corpus: list[dict] = json.load(fh)
    except Exception:
        return []

    if not corpus:
        return []

    from rank_bm25 import BM25Okapi

    corpus_tokens = [rec.get("tokens", []) for rec in corpus]
    bm25 = BM25Okapi(corpus_tokens)

    # Safeguard against negative/zero IDF on small corpora (standard rank_bm25 quirk
    # when terms appear in >= 50% of documents or N <= 2)
    for word, val in bm25.idf.items():
        if val <= 0.0:
            bm25.idf[word] = 0.25

    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    # Filter out documents with zero lexical match
    scored_candidates: list[tuple[int, float]] = [
        (idx, float(score))
        for idx, score in enumerate(scores)
        if score > 0.0
    ]
    if not scored_candidates:
        return []

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored_candidates[: settings.bm25_top_k]

    # Reconstruct DocumentChunk objects
    results: list[tuple[DocumentChunk, float]] = []
    missing_content_ids: list[str] = []

    for idx, score in top_candidates:
        rec = corpus[idx]
        chunk_id = rec["chunk_id"]
        content = rec.get("content")
        if content is not None:
            chunk = _reconstruct_chunk_from_meta_and_doc(
                chunk_id=chunk_id,
                content=content,
                metadata=rec,
            )
            results.append((chunk, score))
        else:
            missing_content_ids.append(chunk_id)

    # Fallback to ChromaDB if earlier versions of bm25_corpus did not store content
    if missing_content_ids:
        chroma_dir = Path(settings.chroma_persist_dir)
        if chroma_dir.exists():
            client = chromadb.PersistentClient(path=str(chroma_dir))
            try:
                coll = client.get_collection(name=COLLECTION_NAME)
                fetched = coll.get(
                    ids=missing_content_ids,
                    include=["documents", "metadatas"],
                )
                f_ids = fetched.get("ids", [])
                f_docs = fetched.get("documents", [])
                f_metas = fetched.get("metadatas", [])
                id_to_data = {
                    cid: (doc, meta)
                    for cid, doc, meta in zip(f_ids, f_docs, f_metas)
                }
                for idx, score in top_candidates:
                    rec = corpus[idx]
                    cid = rec["chunk_id"]
                    if cid in id_to_data:
                        doc, meta = id_to_data[cid]
                        chunk = _reconstruct_chunk_from_meta_and_doc(
                            chunk_id=cid,
                            content=doc,
                            metadata=meta or rec,
                        )
                        results.append((chunk, score))
            except Exception:
                pass

    return results


def _reciprocal_rank_fusion(
    dense_candidates: list[tuple[DocumentChunk, float]],
    bm25_candidates: list[tuple[DocumentChunk, float]],
    settings: Settings,
) -> list[RetrievedPassage]:
    """
    Fuse dense and sparse rankings using Reciprocal Rank Fusion:
      RRF(d) = sum_{r in {dense, bm25}} 1 / (k + rank_r(d))
    where rank_r(d) is 1-based, and k = settings.rrf_k (default 60).

    Preserves individual dense_score and bm25_score on each RetrievedPassage.
    Sorts candidates descending by rrf_score and slices to settings.rrf_candidates.
    """
    chunks_by_id: dict[str, DocumentChunk] = {}
    dense_scores: dict[str, float] = {}
    bm25_scores: dict[str, float] = {}
    rrf_scores: dict[str, float] = {}

    k = float(settings.rrf_k)

    # 1-based ranking for dense results
    for rank, (chunk, score) in enumerate(dense_candidates, start=1):
        cid = chunk.chunk_id
        chunks_by_id[cid] = chunk
        dense_scores[cid] = score
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    # 1-based ranking for BM25 results
    for rank, (chunk, score) in enumerate(bm25_candidates, start=1):
        cid = chunk.chunk_id
        chunks_by_id[cid] = chunk
        bm25_scores[cid] = score
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    passages: list[RetrievedPassage] = []
    for cid, chunk in chunks_by_id.items():
        passages.append(
            RetrievedPassage(
                chunk=chunk,
                dense_score=dense_scores.get(cid, 0.0),
                bm25_score=bm25_scores.get(cid, 0.0),
                rrf_score=round(rrf_scores[cid], 6),
                rerank_score=0.0,
            )
        )

    # Sort descending by fused RRF score
    passages.sort(key=lambda p: p.rrf_score, reverse=True)

    # Retain top rrf_candidates for reranking
    return passages[: settings.rrf_candidates]


def _rerank_passages(
    query: str,
    passages: list[RetrievedPassage],
    reranker: BaseReranker,
    settings: Settings,
    top_k: int,
) -> tuple[list[RetrievedPassage], bool]:
    """
    Score passages with the CrossEncoder and apply the abstention gate.

    Returns:
        (qualifying_passages, abstained)
        abstained=True if no passages cleared settings.rerank_threshold.
    """
    if not passages:
        return [], True

    pairs = [(query, p.chunk.content) for p in passages]
    scores = reranker.predict(pairs)

    scored_passages: list[RetrievedPassage] = [
        RetrievedPassage(
            chunk=p.chunk,
            dense_score=p.dense_score,
            bm25_score=p.bm25_score,
            rrf_score=p.rrf_score,
            rerank_score=float(score),
        )
        for p, score in zip(passages, scores)
    ]

    scored_passages.sort(key=lambda p: p.rerank_score, reverse=True)

    # Confidence abstention gate
    best_score = scored_passages[0].rerank_score
    if best_score < settings.rerank_threshold:
        return [], True

    qualifying = [
        p for p in scored_passages
        if p.rerank_score >= settings.rerank_threshold
    ]

    return qualifying[:top_k], False


# ===========================================================================
# Public Entry Point
# ===========================================================================


def retrieve_and_rerank(
    query: str,
    settings: Settings,
    top_k: int = 4,
    embedder: BaseEmbedder | None = None,
    reranker: BaseReranker | None = None,
) -> tuple[list[RetrievedPassage], bool]:
    """
    Hybrid retrieval (Dense + BM25 + RRF) followed by CrossEncoder reranking.

    Args:
        query: User question or standalone search query.
        settings: Application settings.
        top_k: Maximum number of qualifying passages to return (ge=1, le=15).
        embedder: Optional BaseEmbedder (defaults to SentenceTransformerEmbedder).
        reranker: Optional BaseReranker (defaults to CrossEncoderReranker).

    Returns:
        (passages, abstained)
        passages: list of RetrievedPassage sorted descending by rerank_score.
        abstained: True if no passage met the confidence threshold.
    """
    clean_query = query.strip()
    if not clean_query:
        return [], True

    effective_embedder = (
        embedder
        if embedder is not None
        else SentenceTransformerEmbedder(settings.embedding_model)
    )
    effective_reranker = (
        reranker
        if reranker is not None
        else CrossEncoderReranker(settings.reranker_model)
    )

    # 1. Dual-Index Retrieval
    dense_results = _dense_search(clean_query, settings, effective_embedder)
    bm25_results = _bm25_search(clean_query, settings)

    # Empty index safety: if neither retriever returned any candidates
    if not dense_results and not bm25_results:
        return [], True

    # 2. Reciprocal Rank Fusion
    fused_candidates = _reciprocal_rank_fusion(
        dense_results, bm25_results, settings
    )

    # 3. Cross-Encoder Reranking & Abstention Gate
    return _rerank_passages(
        query=clean_query,
        passages=fused_candidates,
        reranker=effective_reranker,
        settings=settings,
        top_k=top_k,
    )
