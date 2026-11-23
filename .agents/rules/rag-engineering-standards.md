# Workspace Rules: Production Multi-Document Conversational RAG Engine ($0 Local/Open-Source Stack)

## 1. Strict 4-Layer Architecture
- Enforce strict separation across four directories:
  - `core/`: Pure domain logic (multi-format parsing, chunking, hybrid retrieval, cross-encoder reranking, SQLite session memory, and grounded generation). **Never** import `fastapi`, `uvicorn`, or `streamlit` inside `core/`.
  - `api/`: FastAPI transport layer (`api/main.py`) handling request validation, dependency injection, structured JSON logging, and HTTP error mapping.
  - `eval/`: Automated evaluation harness (`eval/benchmark.py`) and golden test dataset (`eval/golden_dataset.json`).
  - `ui/`: Lightweight Streamlit presentation client (`ui/streamlit_app.py`) communicating strictly with the FastAPI backend via HTTP.

## 2. Zero-Cost ($0) Python Stack & Prohibited Imports
- Target **Python 3.11+** with strict type annotations on all functions, methods, and return values.
- Use **Pydantic v2** (`BaseModel` for domain/API contracts, `pydantic_settings.BaseSettings` for configuration).
- **Zero Paid API Dependencies:**
  - Dense embeddings must run locally via `sentence-transformers` (`BAAI/bge-small-en-v1.5` by default)—never require paid OpenAI embedding keys.
  - Cross-encoder reranking must run locally via `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2` by default).
  - LLM inference must use `langchain-openai` (`ChatOpenAI`) configured with an OpenAI-compatible `base_url` and `api_key` defaulting to local **Ollama** (`http://localhost:11434/v1`, `llama3.2:3b`, `api_key="ollama"`) so it runs at $0 cost locally or swaps seamlessly to any free-tier OpenAI-compatible endpoint via `.env`.
- Approved libraries: `fastapi`, `uvicorn`, `langchain-core`, `langchain-openai`, `chromadb`, `rank_bm25`, `sentence-transformers`, `pymupdf` (`fitz`), `python-docx`, `openpyxl`, `streamlit`, `pytest`.
- **Strictly Forbidden:** Never import from `langchain_community` or use legacy LangChain chains (`RetrievalQA`, `ConversationalRetrievalChain`). Use modern LCEL (`prompt | llm | parser`).

## 3. Production RAG Engineering Standards
- **Idempotent Ingestion:** Compute a SHA-256 hash of raw file bytes before indexing to skip duplicate uploads cleanly. Preserve exact source filename, file type, and page number or Excel sheet name in every chunk's metadata.
- **Hybrid Retrieval + Cross-Encoder Reranking:** Combine dense vector search (`ChromaDB`) and sparse lexical search (`rank_bm25`) using Reciprocal Rank Fusion ($RRF$, $k=60$), followed by a `CrossEncoder` reranking stage and a calibrated confidence threshold (allowing negative raw logits) that triggers abstention (`abstained=True`) on irrelevant queries.
- **Durable Thread Memory:** Persist multi-turn conversation history in `SQLite` keyed by `session_id` rather than volatile in-memory dictionaries.

## 4. Planning, Offline Testing & Hygiene Workflow
- **Plan Before Coding:** For every multi-file task, generate a structured Implementation Plan Artifact first and wait for user approval before writing code.
- **Zero-Cost Offline Verification:** Every module in `core/` and `api/` must have corresponding `pytest` unit tests in `tests/` with mocked LLM, embedding, and CrossEncoder calls so the full suite runs offline in seconds at $0 cost without downloading model weights or calling external servers.
- **Mandatory Test Execution:** Always execute `pytest` in the terminal after creating or modifying files and resolve all failures before finishing.
- **Secret & Repo Hygiene:** Never hardcode credentials. Manage all environment variables via `.env` and `.env.example`, and exclude `.env`, `chroma_db/`, `*.sqlite`, `*.db`, and `__pycache__/` in `.gitignore`.