"""
api/main.py
===========
Phase 3: FastAPI REST Transport Layer.

Architectural guarantees:
- Pure transport layer: delegates domain logic strictly to core/.
- Never defines domain logic or chunking/retrieval implementations.
- Dependency injection with app.state lazy singletons.
- Easily testable offline via app.dependency_overrides.
- Structured JSON request & latency logging middleware.
- Global exception handler mapping unexpected errors to clean HTTP 500 JSON.

Endpoints:
  GET    /health
  POST   /ingest
  POST   /chat
  GET    /sessions
  GET    /sessions/{session_id}/history
  DELETE /sessions/{session_id}
  DELETE /documents
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.runnables import Runnable

from core.config import Settings
from core.generate import build_llm, generate_answer
from core.ingest import (
    COLLECTION_NAME,
    BaseEmbedder,
    SentenceTransformerEmbedder,
    clear_all_documents,
    ingest_bytes,
    ingest_file,
)
from core.memory import SQLiteMemoryStore
from core.retrieve import BaseReranker, CrossEncoderReranker
from core.schemas import ChatRequest, ChatResponse, IngestFileResult

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("rag_engine_api")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Lifespan — Cold-Start Sample Seeding
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan: pre-populate settings and cold-start sample seed.

    If auto_seed_samples is True, the ChromaDB collection is empty, and
    sample_docs/ exists, automatically ingest sample files so public
    visitors on Streamlit Community Cloud see a populated knowledge base
    on first boot. SHA-256 deduplication makes restarts a no-op.
    """
    # Respect DI overrides (tests set app.dependency_overrides[get_settings])
    try:
        if get_settings in application.dependency_overrides:
            settings = application.dependency_overrides[get_settings]()
        else:
            settings = Settings()
    except Exception:
        settings = Settings()
    application.state.settings = settings

    if settings.auto_seed_samples:
        sample_dir = Path("sample_docs")
        if sample_dir.exists():
            try:
                chroma_dir = Path(settings.chroma_persist_dir)
                chroma_dir.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(
                    path=settings.chroma_persist_dir,
                    settings=chromadb.config.Settings(anonymized_telemetry=False),
                )
                collection = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                if collection.count() == 0:
                    embedder = SentenceTransformerEmbedder(settings.embedding_model)
                    application.state.embedder = embedder
                    for ext in ("*.pdf", "*.docx", "*.xlsx"):
                        for filepath in sorted(sample_dir.glob(ext)):
                            result = ingest_file(filepath, settings, embedder)
                            logger.info(
                                "Auto-seeded %s → %s (%d chunks)",
                                result.filename,
                                result.status,
                                result.chunks_added,
                            )
            except Exception as exc:
                logger.warning("Cold-start sample seeding skipped: %s", exc)

    yield


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Production Conversational RAG Engine",
    description="100% zero-cost local/open-source RAG engine with hybrid retrieval and verifiable citations.",
    version="1.0.0",
    lifespan=lifespan,
)


# ===========================================================================
# Middleware & Exception Handlers
# ===========================================================================


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Response:
    """Structured JSON logging middleware recording path, status, and duration."""
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": round(duration_ms, 2),
    }
    logger.info(json.dumps(log_entry))
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map unhandled exceptions to structured JSON HTTP 500 responses."""
    logger.error(f"Unhandled error processing {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": str(exc),
        },
    )


# ===========================================================================
# Dependency Injection Providers (backed by app.state singletons)
# ===========================================================================


def get_settings(request: Request) -> Settings:
    if not hasattr(request.app.state, "settings") or request.app.state.settings is None:
        request.app.state.settings = Settings()
    return request.app.state.settings


def get_embedder(
    request: Request, settings: Settings = Depends(get_settings)
) -> BaseEmbedder:
    if not hasattr(request.app.state, "embedder") or request.app.state.embedder is None:
        request.app.state.embedder = SentenceTransformerEmbedder(settings.embedding_model)
    return request.app.state.embedder


def get_reranker(
    request: Request, settings: Settings = Depends(get_settings)
) -> BaseReranker:
    if not hasattr(request.app.state, "reranker") or request.app.state.reranker is None:
        request.app.state.reranker = CrossEncoderReranker(settings.reranker_model)
    return request.app.state.reranker


def get_memory_store(
    request: Request, settings: Settings = Depends(get_settings)
) -> SQLiteMemoryStore:
    if (
        not hasattr(request.app.state, "memory_store")
        or request.app.state.memory_store is None
    ):
        request.app.state.memory_store = SQLiteMemoryStore(settings.sqlite_db_path)
    return request.app.state.memory_store


def get_llm(
    request: Request, settings: Settings = Depends(get_settings)
) -> Runnable[Any, Any]:
    if not hasattr(request.app.state, "llm") or request.app.state.llm is None:
        request.app.state.llm = build_llm(settings)
    return request.app.state.llm


# ===========================================================================
# REST Endpoints
# ===========================================================================


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    """Healthcheck returning service status and active model configurations."""
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/ingest", response_model=list[IngestFileResult])
async def ingest_documents(
    files: list[UploadFile] = File(default=[]),
    settings: Settings = Depends(get_settings),
    embedder: BaseEmbedder = Depends(get_embedder),
) -> list[IngestFileResult]:
    """
    Ingest one or more source files into the hybrid vector & BM25 index.

    Safeguards:
      - Returns HTTP 400 if files list is empty.
      - Handles 0-byte or corrupted files gracefully per file without failing the batch.
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided for ingestion.",
        )

    results: list[IngestFileResult] = []
    for file in files:
        filename = file.filename or "unknown"
        try:
            content = await file.read()
            if len(content) == 0:
                results.append(
                    IngestFileResult(
                        filename=filename,
                        sha256_hash="",
                        status="error",
                        chunks_added=0,
                        error_detail="File is empty (0 bytes).",
                    )
                )
                continue

            result = ingest_bytes(
                filename=filename,
                data=content,
                settings=settings,
                embedder=embedder,
            )
            results.append(result)
        except Exception as exc:
            results.append(
                IngestFileResult(
                    filename=filename,
                    sha256_hash="",
                    status="error",
                    chunks_added=0,
                    error_detail=f"Ingestion failed: {exc}",
                )
            )

    return results


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    llm: Runnable[Any, Any] = Depends(get_llm),
    embedder: BaseEmbedder = Depends(get_embedder),
    reranker: BaseReranker = Depends(get_reranker),
    memory_store: SQLiteMemoryStore = Depends(get_memory_store),
) -> ChatResponse:
    """
    Execute conversational RAG: multi-turn query rewriting, hybrid retrieval,
    confidence abstention gating, and grounded answer synthesis.
    """
    return generate_answer(
        request=request,
        settings=settings,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        memory_store=memory_store,
    )


@app.get("/sessions", response_model=list[str])
def list_sessions(
    memory_store: SQLiteMemoryStore = Depends(get_memory_store),
) -> list[str]:
    """List all active conversation session IDs ordered by recency."""
    return memory_store.list_sessions()


@app.get("/sessions/{session_id}/history", response_model=list[dict[str, str]])
def get_session_history(
    session_id: str,
    memory_store: SQLiteMemoryStore = Depends(get_memory_store),
) -> list[dict[str, str]]:
    """Retrieve the chronological message history for a given session ID."""
    return memory_store.get_recent_history(session_id=session_id, max_messages=100)


@app.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    memory_store: SQLiteMemoryStore = Depends(get_memory_store),
) -> dict[str, str]:
    """Delete all stored conversation turns for a given session ID."""
    memory_store.clear_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.delete("/documents")
def clear_documents(
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    """Hard-reset the document knowledge base (ChromaDB + BM25 + SHA-256 registry)."""
    clear_all_documents(settings)
    return {"status": "cleared"}
