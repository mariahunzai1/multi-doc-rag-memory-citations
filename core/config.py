"""
core/config.py
==============
Application settings loaded via pydantic-settings BaseSettings.

Design guarantees:
- Zero .env file required — every field has a $0 / local default.
- No paid API keys needed; all ML runs locally via sentence-transformers + Ollama.
- Override any field via environment variable or a .env file at project root.
- `extra="ignore"` silently drops unknown env vars (safe for CI environments).
- `rerank_threshold` accepts negative floats because ms-marco-MiniLM-L-6-v2
  outputs raw logits (not probabilities), which are commonly negative for
  irrelevant passage pairs.

Rule compliance:
- Never imports fastapi, uvicorn, or streamlit.
- Python 3.11+ strict type annotations.
- Pydantic v2 (pydantic-settings 2.x).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the RAG engine.

    All fields default to zero-cost local values.
    Any field can be overridden by its UPPER_SNAKE_CASE environment variable
    or by a .env file placed at the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # Silently ignore unknown env vars
        case_sensitive=False,    # LLM_MODEL and llm_model both work
    )

    # ------------------------------------------------------------------
    # Embedding (local sentence-transformers — $0 cost)
    # ------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ------------------------------------------------------------------
    # Cross-Encoder Reranking (local sentence-transformers — $0 cost)
    # ------------------------------------------------------------------
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ------------------------------------------------------------------
    # LLM — OpenAI-compatible endpoint (defaults to local Ollama — $0)
    # Swap to any free-tier OpenAI-compatible endpoint via .env.
    # ------------------------------------------------------------------
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "llama3.2:3b"
    llm_temperature: float = 0.1

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ------------------------------------------------------------------
    # Hybrid Retrieval (Dense + BM25 → RRF)
    # ------------------------------------------------------------------
    dense_top_k: int = 20        # Candidates pulled from ChromaDB
    bm25_top_k: int = 20         # Candidates pulled from BM25 index
    rrf_candidates: int = 15     # Window retained after RRF fusion
    rrf_k: int = 60              # RRF constant (standard: 60)

    # ------------------------------------------------------------------
    # Cross-Encoder Reranking Threshold
    # ms-marco-MiniLM-L-6-v2 outputs raw logits — negative values are
    # normal for irrelevant pairs.  Do NOT add ge= constraints here.
    # Queries whose top rerank_score < rerank_threshold trigger abstention.
    # ------------------------------------------------------------------
    rerank_threshold: float = -1.5

    # ------------------------------------------------------------------
    # Persistent Storage
    # ------------------------------------------------------------------
    chroma_persist_dir: str = "./chroma_db"
    sqlite_db_path: str = "./rag_memory.sqlite"
    bm25_index_path: str = "./chroma_db/bm25_corpus.json"

    # ------------------------------------------------------------------
    # Cold-Start Sample Seeding (Streamlit Community Cloud)
    # ------------------------------------------------------------------
    auto_seed_samples: bool = True
