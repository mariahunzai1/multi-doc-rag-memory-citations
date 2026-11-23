"""
tests/test_api.py
=================
Offline unit and integration tests for api/main.py — Phase 3.

Guarantees:
- 100% offline — zero Ollama server, HuggingFace, or external network calls.
- Zero API cost ($0) — uses TestClient with app.dependency_overrides.
- Isolated pytest tmp_path directories for ChromaDB and SQLite.
- Full coverage of:
    * GET /health (liveness and model configuration)
    * POST /ingest (empty payload 400, multi-file indexing, duplicate SHA-256 skipping, 0-byte/corrupt error isolation)
    * POST /chat (grounded answer with citations, confidence abstention gate, 422 Pydantic schema validation errors)
    * GET /sessions, GET /sessions/{id}/history, DELETE /sessions/{id}
    * DELETE /documents (complete index reset)

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import Runnable

from api.main import (
    app,
    get_embedder,
    get_llm,
    get_memory_store,
    get_reranker,
    get_settings,
)
from core.config import Settings
from core.ingest import BaseEmbedder
from core.memory import SQLiteMemoryStore
from core.retrieve import BaseReranker


# ===========================================================================
# Test Doubles
# ===========================================================================


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

    def __init__(self, constant_score: float = 2.0) -> None:
        self.constant_score = constant_score

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self.constant_score] * len(pairs)


class FakeRunnable(Runnable):
    """Deterministic LangChain runnable double for generation."""

    def __init__(self, response_text: str = "Grounded response from test LLM.") -> None:
        self.response_text = response_text
        self.call_count = 0

    def invoke(self, input: Any, config: Any = None) -> str:
        self.call_count += 1
        return self.response_text


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
        auto_seed_samples=False,
    )


@pytest.fixture()
def test_client(
    tmp_settings: Settings,
) -> Generator[TestClient, None, None]:
    """TestClient configured with dependency overrides for offline execution."""
    memory_store = SQLiteMemoryStore(tmp_settings.sqlite_db_path)
    embedder = FakeEmbedder()
    reranker = FakeReranker(constant_score=2.5)
    llm = FakeRunnable("Grounded response from test LLM.")

    app.dependency_overrides[get_settings] = lambda: tmp_settings
    app.dependency_overrides[get_memory_store] = lambda: memory_store
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_reranker] = lambda: reranker
    app.dependency_overrides[get_llm] = lambda: llm

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


# ===========================================================================
# 1. Test Health Endpoint
# ===========================================================================


def test_health_endpoint(test_client: TestClient, tmp_settings: Settings) -> None:
    """GET /health returns 200 with status ok and active model strings."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["llm_model"] == tmp_settings.llm_model
    assert data["embedding_model"] == tmp_settings.embedding_model


# ===========================================================================
# 2. Test Ingestion Endpoint
# ===========================================================================


def test_ingest_empty_files_returns_400(test_client: TestClient) -> None:
    """POST /ingest with no files returns HTTP 400 Bad Request."""
    resp = test_client.post("/ingest", files=[])
    assert resp.status_code == 400


def test_ingest_valid_files_indexed(test_client: TestClient) -> None:
    """POST /ingest with valid text file returns status indexed with chunks_added > 0."""
    files = [
        ("files", ("sample.txt", b"Photosynthesis produces oxygen and glucose in plants.", "text/plain"))
    ]
    resp = test_client.post("/ingest", files=files)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    assert results[0]["filename"] == "sample.txt"
    assert results[0]["status"] == "indexed"
    assert results[0]["chunks_added"] > 0
    assert len(results[0]["sha256_hash"]) == 64


def test_ingest_duplicate_file_skipped(test_client: TestClient) -> None:
    """Ingesting the exact same file bytes twice returns status skipped on second upload."""
    content = b"Deterministic payload for duplicate detection."
    files = [("files", ("first.txt", content, "text/plain"))]

    resp1 = test_client.post("/ingest", files=files)
    assert resp1.status_code == 200
    assert resp1.json()[0]["status"] == "indexed"

    # Second upload with different filename but same hash
    files2 = [("files", ("second.txt", content, "text/plain"))]
    resp2 = test_client.post("/ingest", files=files2)
    assert resp2.status_code == 200
    assert resp2.json()[0]["status"] == "skipped"
    assert resp2.json()[0]["chunks_added"] == 0


def test_ingest_corrupted_file_error_isolation(test_client: TestClient) -> None:
    """Corrupted file returns status error without crashing the batch."""
    files = [
        ("files", ("good.txt", b"Good content that indexes cleanly.", "text/plain")),
        ("files", ("bad.pdf", b"corrupted non-pdf bytes", "application/pdf")),
    ]
    resp = test_client.post("/ingest", files=files)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 2
    assert results[0]["status"] == "indexed"
    assert results[1]["status"] == "error"
    assert results[1]["error_detail"] is not None


def test_ingest_zero_byte_file_handled_gracefully(test_client: TestClient) -> None:
    """0-byte file returns status error per file without failing the request."""
    files = [("files", ("empty.txt", b"", "text/plain"))]
    resp = test_client.post("/ingest", files=files)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert "empty" in results[0]["error_detail"].lower()


# ===========================================================================
# 3. Test Chat Endpoint
# ===========================================================================


def test_chat_grounded_answer_success(test_client: TestClient) -> None:
    """POST /chat on indexed knowledge base returns grounded answer with citations."""
    # First, ingest knowledge
    files = [
        ("files", ("mars.txt", b"Mars is the fourth planet from the Sun and the second-smallest planet.", "text/plain"))
    ]
    ingest_resp = test_client.post("/ingest", files=files)
    assert ingest_resp.status_code == 200

    # Execute chat query
    chat_payload = {
        "session_id": "sess-api-1",
        "question": "What is Mars?",
        "top_k": 2,
    }
    resp = test_client.post("/chat", json=chat_payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_id"] == "sess-api-1"
    assert data["abstained"] is False
    assert data["answer"] == "Grounded response from test LLM."
    assert data["standalone_query"] == "What is Mars?"
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["source_filename"] == "mars.txt"
    assert data["latency_ms"] >= 0.0


def test_chat_abstention_flow(test_client: TestClient) -> None:
    """When reranker confidence is below threshold, POST /chat returns abstained=True."""
    # Override reranker with low confidence score (-5.0)
    app.dependency_overrides[get_reranker] = lambda: FakeReranker(constant_score=-5.0)

    # Ingest document
    files = [("files", ("data.txt", b"Some random facts about geology.", "text/plain"))]
    test_client.post("/ingest", files=files)

    chat_payload = {
        "session_id": "sess-abstain-1",
        "question": "How to build a particle accelerator?",
        "top_k": 2,
    }
    resp = test_client.post("/chat", json=chat_payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["abstained"] is True
    assert data["citations"] == []
    assert "could not find reliable information" in data["answer"]


def test_chat_validation_empty_question_returns_422(test_client: TestClient) -> None:
    """Empty question string fails Pydantic schema validation (min_length=1) with HTTP 422."""
    payload = {
        "session_id": "sess-val",
        "question": "",
        "top_k": 4,
    }
    resp = test_client.post("/chat", json=payload)
    assert resp.status_code == 422


def test_chat_validation_invalid_top_k_returns_422(test_client: TestClient) -> None:
    """top_k out of bounds (ge=1, le=15) returns HTTP 422."""
    # top_k = 0 (below ge=1)
    resp1 = test_client.post("/chat", json={"session_id": "s", "question": "Valid?", "top_k": 0})
    assert resp1.status_code == 422

    # top_k = 20 (above le=15)
    resp2 = test_client.post("/chat", json={"session_id": "s", "question": "Valid?", "top_k": 20})
    assert resp2.status_code == 422


# ===========================================================================
# 4. Test Sessions & History Endpoints
# ===========================================================================


def test_sessions_lifecycle(test_client: TestClient) -> None:
    """Verify session creation, message history retrieval, and session deletion."""
    # Perform a chat to create a session
    test_client.post(
        "/chat",
        json={"session_id": "sess-lifecycle", "question": "Hello server!", "top_k": 2},
    )

    # 1. GET /sessions includes the session
    sessions_resp = test_client.get("/sessions")
    assert sessions_resp.status_code == 200
    assert "sess-lifecycle" in sessions_resp.json()

    # 2. GET /sessions/{id}/history returns 2 messages (user + assistant)
    hist_resp = test_client.get("/sessions/sess-lifecycle/history")
    assert hist_resp.status_code == 200
    history = hist_resp.json()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello server!"

    # 3. DELETE /sessions/{id} clears the session
    del_resp = test_client.delete("/sessions/sess-lifecycle")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"status": "deleted", "session_id": "sess-lifecycle"}

    # 4. GET /sessions/{id}/history is now empty
    empty_hist = test_client.get("/sessions/sess-lifecycle/history").json()
    assert empty_hist == []


# ===========================================================================
# 5. Test Documents Reset Endpoint
# ===========================================================================


def test_documents_reset(test_client: TestClient) -> None:
    """DELETE /documents clears all indexes and returns status cleared."""
    # Ingest a file first
    files = [("files", ("reset.txt", b"Document to be reset.", "text/plain"))]
    test_client.post("/ingest", files=files)

    # DELETE /documents
    del_resp = test_client.delete("/documents")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"status": "cleared"}

    # Re-ingesting the exact same file returns indexed (not skipped) because manifest was cleared
    re_resp = test_client.post("/ingest", files=files)
    assert re_resp.status_code == 200
    assert re_resp.json()[0]["status"] == "indexed"
