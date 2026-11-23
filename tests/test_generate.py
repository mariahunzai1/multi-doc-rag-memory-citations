"""
tests/test_generate.py
======================
Offline unit tests for core/memory.py and core/generate.py — Phase 2C.

Guarantees:
- 100% offline — zero Ollama server, HuggingFace, or external network calls.
- Zero API cost ($0) — uses FakeRunnable, FakeEmbedder, and FakeReranker.
- SQLite uses isolated pytest tmp_path databases.
- Full coverage of:
    * SQLiteMemoryStore CRUD, persistence across instances, FIFO max_messages windowing
    * First-turn rewrite bypass (rewrite_llm call_count == 0)
    * Multi-turn query reformulation (rewrite_llm call_count == 1)
    * Empty rewrite string safe fallback to request.question
    * Abstention gate short-circuit (generator_llm call_count == 0)
    * Grounded answer generation and inline Citation construction
    * Clean excerpt truncation (<= 300 characters)
    * Multi-turn conversational flow across consecutive turns

Run with:
    pytest tests/test_generate.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import Runnable

from core.config import Settings
from core.generate import generate_answer
from core.ingest import BaseEmbedder, ingest_bytes
from core.memory import SQLiteMemoryStore
from core.retrieve import BaseReranker
from core.schemas import ChatRequest, ChatResponse, Citation


# ===========================================================================
# Test Doubles (Fake Runnable, Fake Embedder, Fake Reranker)
# ===========================================================================


class FakeRunnable(Runnable):
    """
    Deterministic LangChain Runnable double for LLM chains.
    Tracks invocation count and captured input.
    """

    def __init__(self, response_text: str = "Deterministic test response") -> None:
        self.response_text = response_text
        self.call_count = 0
        self.last_input: Any = None

    def invoke(self, input: Any, config: Any = None) -> str:
        self.call_count += 1
        self.last_input = input
        return self.response_text


class FakeEmbedder(BaseEmbedder):
    """Deterministic unit-vector embedder."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vec = [0.0] * 384
            for i, ch in enumerate(text[:8]):
                vec[i % 384] += (ord(ch) % 10) / 10.0
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            else:
                vec[0] = 1.0
            results.append(vec)
        return results


class FakeReranker(BaseReranker):
    """Deterministic reranker returning pre-configured scores."""

    def __init__(
        self,
        scores: list[float] | None = None,
        constant_score: float | None = None,
    ) -> None:
        self.scores = scores
        self.constant_score = constant_score

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self.constant_score is not None:
            return [self.constant_score] * len(pairs)
        if self.scores is not None:
            return self.scores[: len(pairs)]
        return [2.0] * len(pairs)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    return Settings(
        chroma_persist_dir=str(tmp_path / "chroma_db"),
        bm25_index_path=str(tmp_path / "chroma_db" / "bm25_corpus.json"),
        sqlite_db_path=str(tmp_path / "rag_memory.sqlite"),
        rerank_threshold=-1.5,
    )


@pytest.fixture()
def memory_store(tmp_settings: Settings) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(tmp_settings.sqlite_db_path)


# ===========================================================================
# 1. TestSQLiteMemoryStore
# ===========================================================================


class TestSQLiteMemoryStore:

    def test_schema_and_persistence_across_instances(
        self, tmp_settings: Settings
    ) -> None:
        """Data appended via one store instance is readable from another."""
        store1 = SQLiteMemoryStore(tmp_settings.sqlite_db_path)
        store1.append_turn("sess-1", "Hello", "Hi there!")

        store2 = SQLiteMemoryStore(tmp_settings.sqlite_db_path)
        history = store2.get_recent_history("sess-1")
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there!"}

    def test_fifo_windowing_max_messages(self, memory_store: SQLiteMemoryStore) -> None:
        """get_recent_history respects max_messages window and returns chronological order."""
        for i in range(5):
            memory_store.append_turn("sess-window", f"User {i}", f"Assistant {i}")

        # 5 turns = 10 messages. Request last 4:
        history = memory_store.get_recent_history("sess-window", max_messages=4)
        assert len(history) == 4
        # Chronological order of the last 4:
        assert history[0]["content"] == "User 3"
        assert history[1]["content"] == "Assistant 3"
        assert history[2]["content"] == "User 4"
        assert history[3]["content"] == "Assistant 4"

    def test_clear_session(self, memory_store: SQLiteMemoryStore) -> None:
        """Clearing one session leaves other sessions untouched."""
        memory_store.append_turn("sess-a", "Msg A", "Resp A")
        memory_store.append_turn("sess-b", "Msg B", "Resp B")

        memory_store.clear_session("sess-a")
        assert memory_store.get_recent_history("sess-a") == []
        assert len(memory_store.get_recent_history("sess-b")) == 2

    def test_list_sessions(self, memory_store: SQLiteMemoryStore) -> None:
        """list_sessions returns distinct session IDs ordered by recency."""
        memory_store.append_turn("sess-first", "Q1", "A1")
        memory_store.append_turn("sess-second", "Q2", "A2")

        sessions = memory_store.list_sessions()
        assert sessions == ["sess-second", "sess-first"]


# ===========================================================================
# 2. TestQueryRewriting
# ===========================================================================


class TestQueryRewriting:

    def test_first_turn_rewrite_bypassed(
        self, tmp_settings: Settings, memory_store: SQLiteMemoryStore
    ) -> None:
        """
        On turn 1 (empty session history), the rewrite LLM is never invoked
        and standalone_query equals request.question exactly.
        """
        rewrite_llm = FakeRunnable("Rewritten query")
        generator_llm = FakeRunnable("Final answer")
        embedder = FakeEmbedder()
        reranker = FakeReranker(constant_score=2.0)

        # Ingest sample content so retrieval succeeds
        ingest_bytes("doc.txt", b"RAG stands for Retrieval-Augmented Generation.", tmp_settings, embedder)

        req = ChatRequest(session_id="new-session", question="What is RAG?")
        resp = generate_answer(
            request=req,
            settings=tmp_settings,
            llm=generator_llm,
            rewrite_llm=rewrite_llm,
            embedder=embedder,
            reranker=reranker,
            memory_store=memory_store,
        )

        assert rewrite_llm.call_count == 0
        assert resp.standalone_query == "What is RAG?"
        assert resp.answer == "Final answer"

    def test_followup_turn_invokes_rewrite_llm(
        self, tmp_settings: Settings, memory_store: SQLiteMemoryStore
    ) -> None:
        """
        On turn 2+ (history exists), rewrite LLM is called once to resolve pronouns.
        """
        memory_store.append_turn("sess-followup", "What is RAG?", "RAG is Retrieval-Augmented Generation.")

        rewrite_llm = FakeRunnable("What are the advantages of Retrieval-Augmented Generation?")
        generator_llm = FakeRunnable("Grounded advantages")
        embedder = FakeEmbedder()
        reranker = FakeReranker(constant_score=2.0)

        ingest_bytes("doc.txt", b"RAG improves factual accuracy.", tmp_settings, embedder)

        req = ChatRequest(session_id="sess-followup", question="What are its advantages?")
        resp = generate_answer(
            request=req,
            settings=tmp_settings,
            llm=generator_llm,
            rewrite_llm=rewrite_llm,
            embedder=embedder,
            reranker=reranker,
            memory_store=memory_store,
        )

        assert rewrite_llm.call_count == 1
        assert resp.standalone_query == "What are the advantages of Retrieval-Augmented Generation?"

    def test_empty_rewrite_fallback_to_question(
        self, tmp_settings: Settings, memory_store: SQLiteMemoryStore
    ) -> None:
        """
        Defensive detail 1: If rewrite LLM produces whitespace or empty string,
        pipeline safely falls back to request.question.
        """
        memory_store.append_turn("sess-fallback", "Q1", "A1")

        rewrite_llm = FakeRunnable("   \n\t  ")  # empty/whitespace
        generator_llm = FakeRunnable("Answer")
        embedder = FakeEmbedder()
        reranker = FakeReranker(constant_score=2.0)

        ingest_bytes("doc.txt", b"Some content here.", tmp_settings, embedder)

        req = ChatRequest(session_id="sess-fallback", question="Fallback question?")
        resp = generate_answer(
            request=req,
            settings=tmp_settings,
            llm=generator_llm,
            rewrite_llm=rewrite_llm,
            embedder=embedder,
            reranker=reranker,
            memory_store=memory_store,
        )

        assert resp.standalone_query == "Fallback question?"


# ===========================================================================
# 3. TestAbstentionGate
# ===========================================================================


class TestAbstentionGate:

    def test_abstention_short_circuits_generator_llm(
        self, tmp_settings: Settings, memory_store: SQLiteMemoryStore
    ) -> None:
        """
        When retrieval abstains (below threshold), the generator LLM is NOT called,
        abstained=True, citations=[], and the turn is saved in SQLite memory.
        """
        generator_llm = FakeRunnable("Should never be called")
        embedder = FakeEmbedder()
        # Reranker returns -5.0 (below -1.5 threshold)
        reranker = FakeReranker(constant_score=-5.0)

        ingest_bytes("doc.txt", b"Irrelevant content about frogs.", tmp_settings, embedder)

        req = ChatRequest(session_id="sess-abstain", question="How does a rocket fly?")
        resp = generate_answer(
            request=req,
            settings=tmp_settings,
            llm=generator_llm,
            embedder=embedder,
            reranker=reranker,
            memory_store=memory_store,
        )

        # Zero generator LLM calls
        assert generator_llm.call_count == 0
        assert resp.abstained is True
        assert resp.citations == []
        assert "could not find reliable information" in resp.answer
        assert resp.latency_ms >= 0.0

        # Memory persisted
        history = memory_store.get_recent_history("sess-abstain")
        assert len(history) == 2
        assert history[0]["content"] == "How does a rocket fly?"
        assert "could not find reliable information" in history[1]["content"]


# ===========================================================================
# 4. TestGroundedAnswerAndCitations
# ===========================================================================


class TestGroundedAnswerAndCitations:

    def test_grounded_answer_and_citation_generation(
        self, tmp_settings: Settings, memory_store: SQLiteMemoryStore
    ) -> None:
        """
        Grounded answer constructs verifiable Citation objects with source_filename,
        location, relevance_score, and cleanly formatted answer.
        """
        embedder = FakeEmbedder()
        reranker = FakeReranker(constant_score=2.8)
        generator_llm = FakeRunnable("According to [Source 1], Python was created by Guido van Rossum.")

        ingest_bytes("history.txt", b"Python was created by Guido van Rossum in 1991.", tmp_settings, embedder)

        req = ChatRequest(session_id="sess-grounded", question="Who created Python?", top_k=2)
        resp = generate_answer(
            request=req,
            settings=tmp_settings,
            llm=generator_llm,
            embedder=embedder,
            reranker=reranker,
            memory_store=memory_store,
        )

        assert not resp.abstained
        assert resp.answer == "According to [Source 1], Python was created by Guido van Rossum."
        assert len(resp.citations) == 1

        citation = resp.citations[0]
        assert isinstance(citation, Citation)
        assert citation.source_filename == "history.txt"
        assert citation.location == "Text Document"
        assert citation.relevance_score == pytest.approx(2.8)
        assert "Guido van Rossum" in citation.excerpt
        assert resp.latency_ms > 0.0

    def test_excerpt_truncation_under_300_chars(
        self, tmp_settings: Settings, memory_store: SQLiteMemoryStore
    ) -> None:
        """
        Defensive detail 2: Chunks longer than 300 characters have their
        Citation.excerpt truncated cleanly to <= 300 characters with '...'.
        """
        embedder = FakeEmbedder()
        reranker = FakeReranker(constant_score=1.5)
        generator_llm = FakeRunnable("Answer grounded in long text.")

        long_text = b"Sentence number one. " * 30  # ~600 chars
        ingest_bytes("long.txt", long_text, tmp_settings, embedder)

        req = ChatRequest(session_id="sess-trunc", question="Summarize sentence one.")
        resp = generate_answer(
            request=req,
            settings=tmp_settings,
            llm=generator_llm,
            embedder=embedder,
            reranker=reranker,
            memory_store=memory_store,
        )

        assert len(resp.citations) >= 1
        for cit in resp.citations:
            assert len(cit.excerpt) <= 300
            assert cit.excerpt.endswith("...")


# ===========================================================================
# 5. TestMultiTurnConversationalFlow
# ===========================================================================


class TestMultiTurnConversationalFlow:

    def test_two_turn_conversational_integration(
        self, tmp_settings: Settings, memory_store: SQLiteMemoryStore
    ) -> None:
        """
        Verifies complete two-turn conversation flow:
          Turn 1: Direct question -> no rewrite -> grounded answer -> stored.
          Turn 2: Follow-up question -> rewrite called -> grounded answer -> stored.
        """
        embedder = FakeEmbedder()
        reranker = FakeReranker(constant_score=2.5)

        ingest_bytes("battery.txt", b"Lithium-ion batteries have an energy density of 250 Wh/kg.", tmp_settings, embedder)

        rewrite_llm = FakeRunnable("What is the energy density of Lithium-ion batteries?")
        generator_llm = FakeRunnable("The energy density is 250 Wh/kg [Source 1].")

        session_id = "sess-two-turn"

        # Turn 1
        req1 = ChatRequest(session_id=session_id, question="What battery is described?")
        resp1 = generate_answer(
            request=req1,
            settings=tmp_settings,
            llm=generator_llm,
            rewrite_llm=rewrite_llm,
            embedder=embedder,
            reranker=reranker,
            memory_store=memory_store,
        )
        assert rewrite_llm.call_count == 0  # bypassed on turn 1
        assert not resp1.abstained

        # Turn 2
        req2 = ChatRequest(session_id=session_id, question="What is its energy density?")
        resp2 = generate_answer(
            request=req2,
            settings=tmp_settings,
            llm=generator_llm,
            rewrite_llm=rewrite_llm,
            embedder=embedder,
            reranker=reranker,
            memory_store=memory_store,
        )
        assert rewrite_llm.call_count == 1  # called on turn 2
        assert resp2.standalone_query == "What is the energy density of Lithium-ion batteries?"
        assert not resp2.abstained

        # Verify full 4-message conversation stored in SQLite
        history = memory_store.get_recent_history(session_id, max_messages=10)
        assert len(history) == 4
        assert history[0]["content"] == "What battery is described?"
        assert history[2]["content"] == "What is its energy density?"
