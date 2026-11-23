"""
tests/test_ingest.py
====================
Offline unit tests for core/ingest.py — Phase 2A.

Design guarantees:
- 100% offline — no model downloads, no HuggingFace hub calls.
- Zero cost ($0) — FakeEmbedder returns deterministic non-zero vectors.
- ChromaDB uses isolated tmp_path directories per test run.
- 30 test cases across 6 classes.

Run with:
    pytest tests/test_ingest.py -v
"""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest

from core.config import Settings
from core.ingest import (
    BaseEmbedder,
    _HashRegistry,
    _chunk_text,
    _fix_hyphenation,
    _parse_docx,
    _parse_pdf,
    _parse_txt,
    _parse_xlsx,
    clear_all_documents,
    ingest_bytes,
)
from core.schemas import DocumentChunk


# ===========================================================================
# Fake Embedder — zero downloads, deterministic, non-zero vectors
# ===========================================================================


class FakeEmbedder(BaseEmbedder):
    """
    Test double for BaseEmbedder.

    Returns a deterministic 384-dim vector for each text: the first element
    is 1.0, the rest are 0.0.  Cosine similarity is well-defined for this
    vector (it is a valid unit vector along the first dimension).
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 383 for _ in texts]


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings whose storage paths all live inside pytest's tmp_path."""
    chroma_dir = str(tmp_path / "chroma_db")
    return Settings(
        chroma_persist_dir=chroma_dir,
        bm25_index_path=str(tmp_path / "chroma_db" / "bm25_corpus.json"),
        sqlite_db_path=str(tmp_path / "rag_memory.sqlite"),
    )


@pytest.fixture()
def minimal_pdf_bytes() -> bytes:
    """Single-page PDF with text 'Hello from PDF page one.' built in-memory."""
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 720), "Hello from PDF page one.")
    result = doc.tobytes()
    doc.close()
    return result


@pytest.fixture()
def two_page_pdf_bytes() -> bytes:
    """Two-page PDF: page 1 has text, page 2 is blank."""
    import fitz

    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 720), "Content on page one.")
    doc.new_page()  # intentionally blank
    result = doc.tobytes()
    doc.close()
    return result


@pytest.fixture()
def hyphen_pdf_bytes() -> bytes:
    """PDF whose extracted text will contain a hyphenated line-wrap."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    # Insert two separate text spans so fitz's text extractor places a
    # newline between them; we construct the hyphen manually.
    page.insert_text((72, 720), "hyphen-")
    page.insert_text((72, 735), "ated word continues here.")
    result = doc.tobytes()
    doc.close()
    return result


@pytest.fixture()
def minimal_docx_bytes() -> bytes:
    """DOCX with one paragraph and one two-column table, built in-memory."""
    from docx import Document  # type: ignore[import-untyped]

    buf = BytesIO()
    doc = Document()
    doc.add_paragraph("Hello from DOCX paragraph.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Alpha"
    table.cell(1, 1).text = "42"
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture()
def minimal_xlsx_bytes() -> bytes:
    """XLSX with one sheet ('Sales'), a header row, and two data rows."""
    from openpyxl import Workbook  # type: ignore[import-untyped]

    buf = BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Product", "Revenue"])
    ws.append(["Widget A", 1000])
    ws.append(["Widget B", 2000])
    wb.save(buf)
    return buf.getvalue()


@pytest.fixture()
def minimal_txt_bytes() -> bytes:
    return b"Hello from text file. This is sample content for testing."


# ===========================================================================
# Class 1 — TestParsers  (11 tests)
# ===========================================================================


class TestParsers:

    # --- PDF ------------------------------------------------------------------

    def test_pdf_extracts_text(self, minimal_pdf_bytes: bytes) -> None:
        """_parse_pdf returns at least one non-empty text block."""
        blocks = _parse_pdf(minimal_pdf_bytes)
        assert len(blocks) >= 1
        assert any(text.strip() for text, _ in blocks)

    def test_pdf_page_numbering_is_one_indexed(self, minimal_pdf_bytes: bytes) -> None:
        """page_or_sheet in PDF blocks is a 1-indexed integer."""
        blocks = _parse_pdf(minimal_pdf_bytes)
        assert len(blocks) >= 1
        _, page_num = blocks[0]
        assert isinstance(page_num, int)
        assert page_num >= 1

    def test_pdf_hyphen_fix_pure_function(self) -> None:
        """_fix_hyphenation removes the dash and merges the word."""
        result = _fix_hyphenation("hyphen-\nated word")
        assert "hyphen-\n" not in result
        assert "hyphenated" in result

    def test_pdf_hyphen_fix_multiple_occurrences(self) -> None:
        """_fix_hyphenation handles multiple broken words in one string."""
        result = _fix_hyphenation("first-\nword and second-\ntest.")
        assert "first-\n" not in result
        assert "second-\n" not in result
        assert "firstword" in result
        assert "secondtest" in result

    def test_pdf_blank_page_skipped(self, two_page_pdf_bytes: bytes) -> None:
        """Blank pages are excluded; only pages with text are returned."""
        blocks = _parse_pdf(two_page_pdf_bytes)
        # Page 2 is blank — must yield exactly one block
        assert len(blocks) == 1
        _, page_num = blocks[0]
        assert page_num == 1

    # --- DOCX -----------------------------------------------------------------

    def test_docx_extracts_paragraph(self, minimal_docx_bytes: bytes) -> None:
        """Paragraph text is present in the extracted block."""
        blocks = _parse_docx(minimal_docx_bytes)
        assert len(blocks) == 1
        text, loc = blocks[0]
        assert "Hello from DOCX paragraph." in text
        assert loc == "Document Body"

    def test_docx_table_markdown_format(self, minimal_docx_bytes: bytes) -> None:
        """Table renders as a pipe-delimited Markdown block."""
        blocks = _parse_docx(minimal_docx_bytes)
        text, _ = blocks[0]
        # Header row and separator row must be present
        assert "|" in text
        assert "Name" in text
        assert "Value" in text
        assert "---" in text
        # Data row
        assert "Alpha" in text
        assert "42" in text

    # --- XLSX -----------------------------------------------------------------

    def test_xlsx_header_colon_format(self, minimal_xlsx_bytes: bytes) -> None:
        """Data rows render as 'Header: value' pairs."""
        blocks = _parse_xlsx(minimal_xlsx_bytes)
        assert len(blocks) >= 1
        text, _ = blocks[0]
        assert "Product:" in text
        assert "Widget A" in text
        assert "Revenue:" in text

    def test_xlsx_sheet_name_in_metadata(self, minimal_xlsx_bytes: bytes) -> None:
        """page_or_sheet is 'Sheet: <sheet_name>'."""
        blocks = _parse_xlsx(minimal_xlsx_bytes)
        assert len(blocks) >= 1
        _, loc = blocks[0]
        assert loc == "Sheet: Sales"

    # --- TXT ------------------------------------------------------------------

    def test_txt_decodes_utf8(self, minimal_txt_bytes: bytes) -> None:
        """Plain UTF-8 text is decoded and returned correctly."""
        blocks = _parse_txt(minimal_txt_bytes)
        assert len(blocks) == 1
        text, loc = blocks[0]
        assert "Hello from text file." in text
        assert loc == "Text Document"

    def test_txt_bad_bytes_replaced_no_raise(self) -> None:
        """Non-UTF-8 bytes decode with replacement characters — never raises."""
        bad_bytes = b"Valid text \xff\xfe then more text."
        blocks = _parse_txt(bad_bytes)
        # Must return a block (not raise) and contain the valid portions
        assert len(blocks) == 1
        text, _ = blocks[0]
        assert "Valid text" in text
        assert "then more text" in text


# ===========================================================================
# Class 2 — TestChunker  (4 tests)
# ===========================================================================


class TestChunker:

    def _make_settings(self, tmp_path: Path) -> Settings:
        return Settings(
            chroma_persist_dir=str(tmp_path / "chroma_db"),
            bm25_index_path=str(tmp_path / "chroma_db" / "bm25_corpus.json"),
            sqlite_db_path=str(tmp_path / "rag_memory.sqlite"),
            chunk_size=200,
            chunk_overlap=20,
        )

    def test_chunks_produced_from_non_empty_text(self, tmp_path: Path) -> None:
        """At least one DocumentChunk is produced from non-empty content."""
        s = self._make_settings(tmp_path)
        blocks: list[tuple[str, int | str]] = [
            ("The quick brown fox jumps over the lazy dog. " * 10, 1)
        ]
        chunks = _chunk_text(blocks, "test.pdf", "pdf", "a" * 64, "doc-1", s)
        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_index_is_sequential(self, tmp_path: Path) -> None:
        """chunk_index values form a contiguous 0-based sequence."""
        s = self._make_settings(tmp_path)
        blocks: list[tuple[str, int | str]] = [
            ("Word " * 100, 1),
            ("More content " * 50, 2),
        ]
        chunks = _chunk_text(blocks, "test.pdf", "pdf", "b" * 64, "doc-2", s)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_metadata_preserved_on_every_chunk(self, tmp_path: Path) -> None:
        """Every chunk carries source_filename, file_type, and sha256_hash."""
        s = self._make_settings(tmp_path)
        sha256 = "c" * 64
        blocks: list[tuple[str, int | str]] = [("Content " * 50, 1)]
        chunks = _chunk_text(blocks, "report.pdf", "pdf", sha256, "doc-3", s)
        for chunk in chunks:
            assert chunk.source_filename == "report.pdf"
            assert chunk.file_type == "pdf"
            assert chunk.sha256_hash == sha256

    def test_chunk_content_within_size_bound(self, tmp_path: Path) -> None:
        """No chunk exceeds chunk_size + chunk_overlap characters."""
        s = self._make_settings(tmp_path)
        blocks: list[tuple[str, int | str]] = [
            ("The quick brown fox. " * 200, 1)
        ]
        chunks = _chunk_text(blocks, "big.txt", "txt", "d" * 64, "doc-4", s)
        limit = s.chunk_size + s.chunk_overlap
        for chunk in chunks:
            assert len(chunk.content) <= limit, (
                f"Chunk {chunk.chunk_index} length {len(chunk.content)} "
                f"exceeds limit {limit}"
            )


# ===========================================================================
# Class 3 — TestIngestBytes  (7 tests)
# ===========================================================================


class TestIngestBytes:

    def test_pdf_indexed_successfully(
        self,
        minimal_pdf_bytes: bytes,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        result = ingest_bytes("sample.pdf", minimal_pdf_bytes, tmp_settings, fake_embedder)
        assert result.status == "indexed"
        assert result.filename == "sample.pdf"

    def test_docx_indexed_successfully(
        self,
        minimal_docx_bytes: bytes,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        result = ingest_bytes("doc.docx", minimal_docx_bytes, tmp_settings, fake_embedder)
        assert result.status == "indexed"

    def test_xlsx_indexed_successfully(
        self,
        minimal_xlsx_bytes: bytes,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        result = ingest_bytes("data.xlsx", minimal_xlsx_bytes, tmp_settings, fake_embedder)
        assert result.status == "indexed"

    def test_txt_indexed_successfully(
        self,
        minimal_txt_bytes: bytes,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        result = ingest_bytes("notes.txt", minimal_txt_bytes, tmp_settings, fake_embedder)
        assert result.status == "indexed"

    def test_chunks_added_is_positive(
        self,
        minimal_txt_bytes: bytes,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """chunks_added > 0 after a successful first ingest."""
        result = ingest_bytes("notes.txt", minimal_txt_bytes, tmp_settings, fake_embedder)
        assert result.chunks_added > 0

    def test_bm25_corpus_json_written(
        self,
        minimal_txt_bytes: bytes,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """bm25_corpus.json exists on disk after a successful ingest."""
        ingest_bytes("notes.txt", minimal_txt_bytes, tmp_settings, fake_embedder)
        assert Path(tmp_settings.bm25_index_path).exists()

    def test_bm25_records_have_non_empty_tokens(
        self,
        minimal_txt_bytes: bytes,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """Every BM25 corpus record has a non-empty 'tokens' list."""
        ingest_bytes("notes.txt", minimal_txt_bytes, tmp_settings, fake_embedder)
        with open(tmp_settings.bm25_index_path, encoding="utf-8") as fh:
            corpus = json.load(fh)
        assert len(corpus) > 0
        for record in corpus:
            assert isinstance(record["tokens"], list)
            assert len(record["tokens"]) > 0


# ===========================================================================
# Class 4 — TestDeduplication  (3 tests)
# ===========================================================================


class TestDeduplication:

    def test_duplicate_bytes_returns_skipped(
        self,
        minimal_txt_bytes: bytes,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """Ingesting the same bytes a second time returns status='skipped'."""
        first = ingest_bytes("notes.txt", minimal_txt_bytes, tmp_settings, fake_embedder)
        assert first.status == "indexed"

        second = ingest_bytes("notes.txt", minimal_txt_bytes, tmp_settings, fake_embedder)
        assert second.status == "skipped"

    def test_duplicate_has_zero_chunks_added(
        self,
        minimal_txt_bytes: bytes,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """chunks_added is 0 on a duplicate ingest."""
        ingest_bytes("notes.txt", minimal_txt_bytes, tmp_settings, fake_embedder)
        result = ingest_bytes("notes.txt", minimal_txt_bytes, tmp_settings, fake_embedder)
        assert result.chunks_added == 0

    def test_renamed_file_same_hash_still_skipped(
        self,
        minimal_txt_bytes: bytes,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """
        Deduplication is content-based (SHA-256), not filename-based.
        Renaming a file does not bypass the duplicate check.
        """
        ingest_bytes("original.txt", minimal_txt_bytes, tmp_settings, fake_embedder)
        result = ingest_bytes("renamed.txt", minimal_txt_bytes, tmp_settings, fake_embedder)
        assert result.status == "skipped"


# ===========================================================================
# Class 5 — TestErrorIsolation  (4 tests)
# ===========================================================================


class TestErrorIsolation:

    def test_unsupported_extension_returns_error(
        self,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """An unsupported extension (e.g. .mp4) returns status='error'."""
        result = ingest_bytes("video.mp4", b"not a document", tmp_settings, fake_embedder)
        assert result.status == "error"
        assert result.error_detail is not None
        assert result.chunks_added == 0

    def test_corrupt_pdf_bytes_returns_error(
        self,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """Garbage bytes with a .pdf suffix return status='error'."""
        result = ingest_bytes("broken.pdf", b"not a pdf at all", tmp_settings, fake_embedder)
        assert result.status == "error"
        assert result.chunks_added == 0

    def test_corrupt_docx_bytes_returns_error(
        self,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """Garbage bytes with a .docx suffix return status='error'."""
        result = ingest_bytes("broken.docx", b"not a docx", tmp_settings, fake_embedder)
        assert result.status == "error"
        assert result.chunks_added == 0

    def test_error_does_not_register_hash(
        self,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """
        After a failed ingest, the SHA-256 must NOT appear in the registry.
        Verifies the write-order guarantee: registry is updated only after
        both ChromaDB and BM25 JSON writes succeed.
        """
        corrupt_data = b"not a pdf at all"
        sha256 = hashlib.sha256(corrupt_data).hexdigest()

        result = ingest_bytes("broken.pdf", corrupt_data, tmp_settings, fake_embedder)
        assert result.status == "error"

        registry = _HashRegistry(tmp_settings.chroma_persist_dir)
        assert not registry.contains(sha256), (
            "Hash must not be registered after a failed ingest — "
            "re-ingest must be retried, not silently skipped."
        )


# ===========================================================================
# Class 6 — TestClearAll  (1 test)
# ===========================================================================


class TestClearAll:

    def test_clear_resets_all_state(
        self,
        minimal_txt_bytes: bytes,
        tmp_settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """
        After clear_all_documents():
          - BM25 corpus JSON is gone.
          - Re-ingesting the same bytes returns 'indexed' (not 'skipped').
          - chunks_added > 0 on the re-ingest.
        """
        # 1. Initial ingest — establishes registry entry and BM25 file
        first = ingest_bytes("notes.txt", minimal_txt_bytes, tmp_settings, fake_embedder)
        assert first.status == "indexed"
        assert Path(tmp_settings.bm25_index_path).exists()

        # 2. Clear everything
        clear_all_documents(tmp_settings)

        # 3. BM25 corpus file must be gone
        assert not Path(tmp_settings.bm25_index_path).exists(), (
            "BM25 corpus JSON must be deleted by clear_all_documents()."
        )

        # 4. Re-ingest the same bytes → must succeed as 'indexed' again
        second = ingest_bytes("notes.txt", minimal_txt_bytes, tmp_settings, fake_embedder)
        assert second.status == "indexed", (
            "After clear_all_documents(), the same file must be re-indexed "
            "rather than returned as 'skipped'."
        )
        assert second.chunks_added > 0
