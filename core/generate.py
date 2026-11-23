"""
core/generate.py
================
Phase 2C: Grounded LCEL Generation with Verifiable Citations.

Architectural flow:
  1. Retrieve multi-turn history from SQLiteMemoryStore.
  2. Standalone Query Rewriting:
     - Turn 1 (empty history): bypass rewrite LLM entirely.
     - Turn 2+: LCEL reformulation chain (resolving pronouns & references).
     - Fallback: if rewritten query is empty/whitespace, fall back to request.question.
  3. Hybrid Retrieval & Abstention Gate:
     - Execute retrieve_and_rerank().
     - If abstained or empty, short-circuit without calling Generator LLM.
  4. Grounded Synthesis & Inline Citations:
     - Build numbered [Source N] context.
     - Generate answer strictly grounded in retrieved passages.
     - Construct Citation models with excerpts truncated cleanly up to 300 chars.
     - Persist turn in SQLiteMemoryStore and record latency_ms.

Rule compliance:
- Never imports fastapi, uvicorn, or streamlit.
- Python 3.11+ strict type annotations throughout.
- Strictly modern LCEL (prompt | llm | parser) — zero langchain_community.
- 100% $0 local Ollama or OpenAI-compatible endpoint.
- Pluggable component injection for offline pytest verification.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from core.config import Settings
from core.ingest import BaseEmbedder
from core.memory import SQLiteMemoryStore
from core.retrieve import BaseReranker, retrieve_and_rerank
from core.schemas import ChatRequest, ChatResponse, Citation


# ===========================================================================
# Prompts
# ===========================================================================

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a search query reformulation assistant. Given the following conversation history "
                "and a follow-up question, rewrite the follow-up question into a clear, standalone search query "
                "that can be fully understood without the conversation history.\n"
                "Do NOT answer the question. Do NOT add explanation or punctuation other than necessary. "
                "Return ONLY the standalone search query string."
            ),
        ),
        (
            "human",
            (
                "Conversation History:\n{history}\n\n"
                "Follow-up Question: {question}\n\n"
                "Standalone Search Query:"
            ),
        ),
    ]
)

GROUNDED_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a truthful, professional conversational RAG assistant.\n"
                "Answer the user's question based strictly and exclusively on the provided Source Documents below.\n\n"
                "Strict Grounding Rules:\n"
                "1. Every factual statement must be directly supported by the provided Source Documents.\n"
                "2. Cite your sources inline using [Source 1], [Source 2], etc. matching the numbered sources provided.\n"
                "3. If the provided sources do not contain sufficient information to answer the question, "
                "state clearly: 'I could not find reliable information in the uploaded documents to answer that question.'\n"
                "4. Do NOT extrapolate, speculate, or draw on outside knowledge."
            ),
        ),
        (
            "human",
            (
                "Source Documents:\n{context}\n\n"
                "User Question: {question}\n\n"
                "Answer:"
            ),
        ),
    ]
)


# ===========================================================================
# LLM Factory
# ===========================================================================


def build_llm(settings: Settings) -> ChatOpenAI:
    """
    Construct the ChatOpenAI client pointing to the zero-cost local endpoint
    (Ollama by default: http://localhost:11434/v1).
    """
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )


# ===========================================================================
# Public Generation Pipeline
# ===========================================================================


def generate_answer(
    request: ChatRequest,
    settings: Settings | None = None,
    llm: Runnable[Any, Any] | None = None,
    rewrite_llm: Runnable[Any, Any] | None = None,
    embedder: BaseEmbedder | None = None,
    reranker: BaseReranker | None = None,
    memory_store: SQLiteMemoryStore | None = None,
) -> ChatResponse:
    """
    End-to-end multi-turn RAG generation with query rewriting, hybrid retrieval,
    confidence abstention gating, and grounded inline citations.

    Args:
        request: Validated inbound user message.
        settings: Application settings (defaults to Settings()).
        llm: Optional answer generation runnable (defaults to build_llm(settings)).
        rewrite_llm: Optional query rewrite runnable (defaults to llm).
        embedder: Optional BaseEmbedder (passed to retrieve_and_rerank).
        reranker: Optional BaseReranker (passed to retrieve_and_rerank).
        memory_store: Optional SQLiteMemoryStore (defaults to settings.sqlite_db_path).

    Returns:
        ChatResponse with answer, standalone_query, citations, abstained, and latency_ms.
    """
    start_time = time.perf_counter()

    if settings is None:
        settings = Settings()

    if memory_store is None:
        memory_store = SQLiteMemoryStore(settings.sqlite_db_path)

    effective_llm: Runnable[Any, Any] = (
        llm if llm is not None else build_llm(settings)
    )
    effective_rewrite_llm: Runnable[Any, Any] = (
        rewrite_llm if rewrite_llm is not None else effective_llm
    )

    # ── Step 1: Standalone Query Rewriting ─────────────────────────────
    recent_history = memory_store.get_recent_history(
        session_id=request.session_id,
        max_messages=6,
    )

    if not recent_history:
        # First turn: bypass the rewrite LLM entirely
        standalone_query = request.question
    else:
        # Multi-turn: format transcript and run LCEL query rewrite chain
        history_transcript = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in recent_history
        )
        rewrite_chain = REWRITE_PROMPT | effective_rewrite_llm | StrOutputParser()
        raw_rewritten = rewrite_chain.invoke(
            {
                "history": history_transcript,
                "question": request.question,
            }
        )
        cleaned_rewritten = str(raw_rewritten).strip()

        # Defensive safeguard 1: fallback safely if rewrite produces empty/whitespace
        if not cleaned_rewritten:
            standalone_query = request.question
        else:
            standalone_query = cleaned_rewritten

    # ── Step 2: Hybrid Retrieval & Abstention Gate ─────────────────────
    passages, abstained = retrieve_and_rerank(
        query=standalone_query,
        settings=settings,
        top_k=request.top_k,
        embedder=embedder,
        reranker=reranker,
    )

    # If abstained or no passages qualified, short-circuit immediately
    if abstained or not passages:
        safe_abstention_msg = (
            "I could not find reliable information in the uploaded documents to answer that question."
        )
        # Record turn in persistent SQLite memory
        memory_store.append_turn(
            session_id=request.session_id,
            user_msg=request.question,
            assistant_msg=safe_abstention_msg,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return ChatResponse(
            session_id=request.session_id,
            answer=safe_abstention_msg,
            standalone_query=standalone_query,
            citations=[],
            abstained=True,
            latency_ms=round(latency_ms, 2),
        )

    # ── Step 3: Grounded Answer & Verifiable Citations ─────────────────
    context_blocks: list[str] = []
    citations: list[Citation] = []

    for i, passage in enumerate(passages, start=1):
        chunk = passage.chunk
        location_str = str(chunk.page_or_sheet)
        context_blocks.append(
            f"[Source {i}] (File: {chunk.source_filename}, Location: {location_str})\n"
            f"{chunk.content}"
        )

        # Defensive safeguard 2: truncate excerpt cleanly up to 300 characters
        raw_content = chunk.content.strip()
        if len(raw_content) > 300:
            excerpt = raw_content[:297] + "..."
        else:
            excerpt = raw_content

        citations.append(
            Citation(
                source_filename=chunk.source_filename,
                location=location_str,
                excerpt=excerpt,
                relevance_score=passage.rerank_score,
            )
        )

    context_str = "\n\n".join(context_blocks)

    answer_chain = GROUNDED_ANSWER_PROMPT | effective_llm | StrOutputParser()
    raw_answer = answer_chain.invoke(
        {
            "context": context_str,
            "question": standalone_query,
        }
    )
    answer = str(raw_answer).strip()

    # Record successful grounded turn in persistent SQLite memory
    memory_store.append_turn(
        session_id=request.session_id,
        user_msg=request.question,
        assistant_msg=answer,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
        standalone_query=standalone_query,
        citations=citations,
        abstained=False,
        latency_ms=round(latency_ms, 2),
    )
