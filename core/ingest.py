"""
core/ingest.py
==============
Phase 2A: Layout-Aware Multi-Format Ingestion & Idempotent Dual Indexing.

Internal layers (each depends only on layers listed above it):
  Layer 1 — Format Parsers    : _parse_pdf / _parse_docx / _parse_xlsx / _parse_txt
  Layer 2 — Chunker           : _chunk_text  (RecursiveCharacterTextSplitter)
  Layer 3 — Embedder Interface: BaseEmbedder / SentenceTransformerEmbedder
  Layer 4 — Dual Persistence  : _persist_chunks (ChromaDB + BM25 JSON)
  Layer 5 — Public API        : ingest_bytes / ingest_file / clear_all_documents

Atomic write-order guarantee
─────────────────────────────
  1. embedder.embed()          pure computation — no side effects
  2. ChromaDB.add()            vector index
  3. BM25 JSON  os.replace()   sparse index   ← atomic, before registry
  4. HashRegistry os.replace() dedup manifest ← atomic, last

If any step (2-4) raises, the HashRegistry is NOT updated, so the next
ingest attempt retries the full pipeline rather than incorrectly skipping.
If step 3 succeeds but step 4 fails (rare OS failure), the worst outcome
is a re-ingest that appends duplicate BM25 records — no data loss.

Rule compliance
───────────────
- Never imports fastapi, uvicorn, or streamlit.
- Python 3.11+ strict type annotations throughout.
- Pydantic v2 schemas consumed read-only.
- Zero langchain_community.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Literal
from uuid import uuid4

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import Settings
from core.schemas import DocumentChunk, IngestFileResult

# ---------------------------------------------------------------------------
# Constants & Canonical Tokenizer
# ---------------------------------------------------------------------------

COLLECTION_NAME = "rag_documents"
_COLLECTION_NAME = COLLECTION_NAME

def tokenize_text(text: str) -> list[str]:
    """Canonical tokenizer for BM25 indexing and querying."""
    return text.lower().split()

_EXT_TO_FILE_TYPE: dict[str, Literal["pdf", "docx", "xlsx", "txt"]] = {
    ".pdf":  "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".txt":  "txt",
}

# Matches broken hyphenated line-wraps such as "hyphen-\nated"
_HYPHEN_RE = re.compile(r"(\w+)-\n(\w+)")


# ===========================================================================
# Layer 3 — Embedder Interface
# ===========================================================================


class BaseEmbedder(ABC):
    """
    Abstract embedder interface.

    Inject a FakeEmbedder in tests to avoid any HuggingFace downloads or
    network calls — keeping the full test suite offline and at $0 cost.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Return one float vector per input text, in the same order.
        The dimensionality must match the ChromaDB collection's space.
        """
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Production embedder backed by a local sentence-transformers model.
    Downloads the model once; runs fully offline afterwards. Zero API cost.
    """

    def __init__(self, model_name: str) -> None:
        # Deferred import so unit tests that inject FakeEmbedder never
        # trigger a model download.
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()


# ===========================================================================
# SHA-256 Manifest — _HashRegistry
# ===========================================================================


class _HashRegistry:
    """
    Thin JSON-backed registry: sha256_hex → original_filename.

    Stored at <chroma_persist_dir>/hash_registry.json.
    Every mutating write uses os.replace() on a sibling .tmp file for
    atomicity — a mid-write crash leaves the previous registry intact.
    """

    def __init__(self, chroma_persist_dir: str) -> None:
        base = Path(chroma_persist_dir)
        base.mkdir(parents=True, exist_ok=True)
        self._path: Path = base / "hash_registry.json"
        self._tmp:  Path = base / "hash_registry.json.tmp"

    # ------------------------------------------------------------------
    def _load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        with self._path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    def contains(self, sha256: str) -> bool:
        return sha256 in self._load()

    # ------------------------------------------------------------------
    def register(self, sha256: str, filename: str) -> None:
        """Append entry and persist atomically."""
        registry = self._load()
        registry[sha256] = filename
        self._tmp.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        os.replace(self._tmp, self._path)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset registry to an empty dict, atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp.write_text("{}", encoding="utf-8")
        os.replace(self._tmp, self._path)


# ===========================================================================
# Layer 1 — Format Parsers
# ===========================================================================


def _fix_hyphenation(text: str) -> str:
    """
    Repair broken hyphenated line-wraps produced by PDF text extraction.

    Example:  "hyphen-\\nated"  →  "hyphenated\\n"

    The newline is preserved (line-flow structure intact) but no longer
    interrupts the word.
    """
    return _HYPHEN_RE.sub(r"\1\2\n", text)


def _parse_pdf(data: bytes) -> list[tuple[str, int]]:
    """
    Extract text from a PDF, page by page.

    Returns:
        List of (text, 1-indexed-page-number) tuples.
        Blank pages (after whitespace strip) are omitted.

    Uses fitz.open(stream=...) — no temporary files written to disk.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    blocks: list[tuple[str, int]] = []
    try:
        for page in doc:
            raw = page.get_text("text")
            text = _fix_hyphenation(raw)
            if text.strip():
                blocks.append((text, page.number + 1))
    finally:
        doc.close()
    return blocks


def _table_to_markdown(table: object) -> str:
    """
    Convert a python-docx Table to a pipe-delimited Markdown string.

    Row 0 becomes the header; a separator row is inserted after it.
    Pipe characters in cell text are escaped as \\|.
    """
    rows: list[str] = []
    for i, row in enumerate(table.rows):  # type: ignore[attr-defined]
        cells = [
            cell.text.strip().replace("|", r"\|")
            for cell in row.cells
        ]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            # Separator row
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(rows)


def _parse_docx(data: bytes) -> list[tuple[str, str]]:
    """
    Extract text from a DOCX file.

    - Collects only top-level (non-table) paragraphs to avoid double-counting
      text that appears both via doc.paragraphs and inside table cells.
    - Converts each table to a pipe-delimited Markdown block so tabular specs
      survive chunking with their structure visible to the LLM.

    Returns:
        A single-element list [ (combined_text, "Document Body") ].
        Returns [] if no content is found.
    """
    from docx import Document  # type: ignore[import-untyped]

    doc = Document(BytesIO(data))
    body_el = doc.element.body
    parts: list[str] = []

    # --- Top-level paragraphs (direct children of body, not table cells) ---
    for para in doc.paragraphs:
        if para._element.getparent() is body_el and para.text.strip():
            parts.append(para.text.strip())

    # --- Tables → Markdown ---
    for table in doc.tables:
        md = _table_to_markdown(table)
        if md.strip():
            parts.append(md)

    text = "\n\n".join(parts)
    if not text.strip():
        return []
    return [(text, "Document Body")]


def _parse_xlsx(data: bytes) -> list[tuple[str, str]]:
    """
    Extract text from an XLSX file, one block per worksheet.

    - Opens with data_only=True so formulas return computed values.
    - Row 0 is treated as a header row; empty header cells get auto-name Col_N.
    - Each subsequent data row is rendered as "Header: value\\n..." pairs.
    - Entirely empty rows and entirely empty sheets are skipped.

    Returns:
        List of (text, "Sheet: <sheet_name>") tuples, one per non-empty sheet.
    """
    from openpyxl import load_workbook  # type: ignore[import-untyped]

    wb = load_workbook(BytesIO(data), data_only=True, read_only=True)
    blocks: list[tuple[str, str]] = []

    try:
        for ws in wb.worksheets:
            all_rows = list(ws.iter_rows(values_only=True))

            if not all_rows:
                continue

            # Build header list from row 0
            raw_headers = all_rows[0]
            headers: list[str] = [
                str(h).strip() if h is not None and str(h).strip()
                else f"Col_{i + 1}"
                for i, h in enumerate(raw_headers)
            ]

            data_rows = all_rows[1:]
            if not data_rows:
                continue

            records: list[str] = []
            for row in data_rows:
                if all(cell is None for cell in row):
                    continue  # Skip entirely empty rows
                # Pad row if it has fewer cells than headers
                padded: list = list(row) + [None] * max(0, len(headers) - len(row))
                pairs: list[str] = []
                for header, cell in zip(headers, padded):
                    val = str(cell).strip() if cell is not None else ""
                    pairs.append(f"{header}: {val}")
                records.append("\n".join(pairs))

            if records:
                blocks.append(("\n\n".join(records), f"Sheet: {ws.title}"))
    finally:
        wb.close()

    return blocks


def _parse_txt(data: bytes) -> list[tuple[str, str]]:
    """
    Decode UTF-8 text, replacing any non-decodable bytes with the Unicode
    replacement character (U+FFFD).  Never raises on malformed input.
    """
    text = data.decode("utf-8", errors="replace")
    if not text.strip():
        return []
    return [(text, "Text Document")]


# ===========================================================================
# Layer 2 — Structure-Aware Chunker
# ===========================================================================


def _chunk_text(
    blocks: list[tuple[str, int | str]],
    filename: str,
    file_type: Literal["pdf", "docx", "xlsx", "txt"],
    sha256: str,
    doc_id: str,
    settings: Settings,
) -> list[DocumentChunk]:
    """
    Split each (text, page_or_sheet) block with RecursiveCharacterTextSplitter
    and produce one DocumentChunk per fragment.

    - chunk_index is a global counter across all blocks (0-based).
    - All chunks in a batch share the same ingested_at UTC timestamp.
    - Empty/whitespace-only fragments are silently dropped.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[DocumentChunk] = []
    chunk_index = 0
    now = datetime.now(timezone.utc)

    for text, page_or_sheet in blocks:
        for fragment in splitter.split_text(text):
            if not fragment.strip():
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid4()),
                    doc_id=doc_id,
                    sha256_hash=sha256,
                    content=fragment,
                    source_filename=filename,
                    file_type=file_type,
                    page_or_sheet=page_or_sheet,
                    chunk_index=chunk_index,
                    ingested_at=now,
                )
            )
            chunk_index += 1

    return chunks


# ===========================================================================
# Layer 4 — Dual Persistence helpers
# ===========================================================================


def _chunk_to_chroma_metadata(chunk: DocumentChunk) -> dict[str, str | int | float | bool]:
    """
    Convert a DocumentChunk to a ChromaDB-compatible metadata dict.

    ChromaDB metadata values must be str | int | float | bool.
    - page_or_sheet (int | str) is always cast to str.
    - ingested_at (datetime) is serialised as an ISO-8601 string.
    """
    return {
        "chunk_id":        chunk.chunk_id,
        "doc_id":          chunk.doc_id,
        "sha256_hash":     chunk.sha256_hash,
        "source_filename": chunk.source_filename,
        "file_type":       chunk.file_type,
        "page_or_sheet":   str(chunk.page_or_sheet),
        "chunk_index":     chunk.chunk_index,
        "ingested_at":     chunk.ingested_at.isoformat(),
    }


def _load_bm25_corpus(bm25_path: Path) -> list[dict]:
    if not bm25_path.exists():
        return []
    with bm25_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_bm25_corpus(bm25_path: Path, corpus: list[dict]) -> None:
    """
    Atomically persist the BM25 corpus JSON.

    Writes to a sibling .tmp file first, then os.replace() to the final
    path — guaranteeing that a mid-write failure never corrupts the
    existing corpus.
    """
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(bm25_path) + ".tmp")
    tmp_path.write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp_path, bm25_path)


def _persist_chunks(
    chunks: list[DocumentChunk],
    embedder: BaseEmbedder,
    settings: Settings,
) -> int:
    """
    Embed chunks and write to both indexes in strict write order:

      1. embedder.embed()          pure computation
      2. ChromaDB.add()            vector store
      3. BM25 JSON os.replace()    sparse index  ← atomic write here

    The caller is responsible for calling HashRegistry.register() after
    this function returns successfully, ensuring that registry update is
    the very last side effect in the pipeline.

    Returns:
        Number of chunks persisted.
    """
    texts = [c.content for c in chunks]
    vectors = embedder.embed(texts)

    # ── 2. ChromaDB ────────────────────────────────────────────────────
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[_chunk_to_chroma_metadata(c) for c in chunks],
    )

    # ── 3. BM25 JSON — atomic os.replace() ────────────────────────────
    bm25_path = Path(settings.bm25_index_path)
    corpus = _load_bm25_corpus(bm25_path)
    for chunk in chunks:
        corpus.append(
            {
                "chunk_id":        chunk.chunk_id,
                "doc_id":          chunk.doc_id,
                "sha256_hash":     chunk.sha256_hash,
                "content":         chunk.content,
                "source_filename": chunk.source_filename,
                "file_type":       chunk.file_type,
                "page_or_sheet":   str(chunk.page_or_sheet),
                "chunk_index":     chunk.chunk_index,
                "ingested_at":     chunk.ingested_at.isoformat(),
                "tokens":          tokenize_text(chunk.content),
            }
        )
    _write_bm25_corpus(bm25_path, corpus)

    return len(chunks)


# ===========================================================================
# Layer 5 — Public API
# ===========================================================================


def ingest_bytes(
    filename: str,
    data: bytes,
    settings: Settings,
    embedder: BaseEmbedder | None = None,
) -> IngestFileResult:
    """
    Ingest document bytes into the dual index (ChromaDB + BM25).

    Args:
        filename:  Original filename; the suffix determines the parser.
        data:      Raw file bytes.
        settings:  Application settings (storage paths, chunking params).
        embedder:  Optional embedder. When None, lazily constructs a
                   SentenceTransformerEmbedder from settings.embedding_model.
                   Inject a FakeEmbedder in tests to run fully offline.

    Returns:
        IngestFileResult with status "indexed", "skipped", or "error".

    Write-order guarantee (see module docstring):
        ChromaDB → BM25 JSON (atomic) → HashRegistry (atomic).
    """
    sha256 = hashlib.sha256(data).hexdigest()
    registry = _HashRegistry(settings.chroma_persist_dir)

    # ── Idempotency guard ───────────────────────────────────────────────
    if registry.contains(sha256):
        return IngestFileResult(
            filename=filename,
            sha256_hash=sha256,
            status="skipped",
            chunks_added=0,
        )

    try:
        # ── Detect file type ────────────────────────────────────────────
        suffix = Path(filename).suffix.lower()
        if suffix not in _EXT_TO_FILE_TYPE:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported extensions: {sorted(_EXT_TO_FILE_TYPE)}"
            )
        file_type = _EXT_TO_FILE_TYPE[suffix]

        # ── Parse ───────────────────────────────────────────────────────
        raw_blocks: list[tuple[str, int | str]]
        if file_type == "pdf":
            raw_blocks = _parse_pdf(data)
        elif file_type == "docx":
            raw_blocks = _parse_docx(data)
        elif file_type == "xlsx":
            raw_blocks = _parse_xlsx(data)
        else:  # txt
            raw_blocks = _parse_txt(data)

        if not raw_blocks or not any(text.strip() for text, _ in raw_blocks):
            raise ValueError(
                "No text content could be extracted from the file "
                "(it may be empty, image-only, or corrupt)."
            )

        # ── Chunk ───────────────────────────────────────────────────────
        doc_id = str(uuid4())
        effective_embedder: BaseEmbedder = (
            embedder
            if embedder is not None
            else SentenceTransformerEmbedder(settings.embedding_model)
        )
        chunks = _chunk_text(
            raw_blocks, filename, file_type, sha256, doc_id, settings
        )

        if not chunks:
            raise ValueError(
                "Chunking produced zero non-empty chunks "
                "(content may be too short relative to chunk_size)."
            )

        # ── Persist: ChromaDB → BM25 JSON (atomic) ─────────────────────
        n = _persist_chunks(chunks, effective_embedder, settings)

        # ── HashRegistry atomic write — strictly last ───────────────────
        # Only reached when both indexes have been successfully written.
        registry.register(sha256, filename)

        return IngestFileResult(
            filename=filename,
            sha256_hash=sha256,
            status="indexed",
            chunks_added=n,
        )

    except Exception as exc:
        # Per-file error isolation: one bad file must never abort a batch.
        # HashRegistry is NOT updated, so the file will be retried next time.
        return IngestFileResult(
            filename=filename,
            sha256_hash=sha256,
            status="error",
            chunks_added=0,
            error_detail=str(exc),
        )


def ingest_file(
    path: str | Path,
    settings: Settings,
    embedder: BaseEmbedder | None = None,
) -> IngestFileResult:
    """
    Convenience wrapper: read a file from disk and delegate to ingest_bytes().

    Args:
        path:      Absolute or relative path to the source file.
        settings:  Application settings.
        embedder:  Optional embedder (see ingest_bytes for details).
    """
    file_path = Path(path)
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        return IngestFileResult(
            filename=file_path.name,
            sha256_hash="",
            status="error",
            chunks_added=0,
            error_detail=f"Cannot read file: {exc}",
        )
    return ingest_bytes(
        filename=file_path.name,
        data=data,
        settings=settings,
        embedder=embedder,
    )


def clear_all_documents(settings: Settings) -> None:
    """
    Hard-reset all document indexes:
      1. Delete the ChromaDB collection (vectors + metadata).
      2. Delete the BM25 corpus JSON file (and any stale .tmp sibling).
      3. Clear the SHA-256 hash registry (atomic empty-dict write).

    SQLite conversation history is intentionally left untouched — session
    memory is independent of the document index.
    """
    # ── 1. ChromaDB collection ──────────────────────────────────────────
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass  # Collection may not exist yet — not an error

    # ── 2. BM25 corpus JSON ─────────────────────────────────────────────
    bm25_path = Path(settings.bm25_index_path)
    if bm25_path.exists():
        bm25_path.unlink()
    stale_tmp = Path(str(bm25_path) + ".tmp")
    if stale_tmp.exists():
        stale_tmp.unlink()

    # ── 3. HashRegistry — atomic clear ─────────────────────────────────
    _HashRegistry(settings.chroma_persist_dir).clear()
