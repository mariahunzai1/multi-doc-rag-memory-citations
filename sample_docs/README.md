# Sample Documents

Drop your source documents here for ingestion.

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Text and scanned pages via PyMuPDF |
| Word Document | `.docx` | Via python-docx |
| Excel Spreadsheet | `.xlsx` | Per-sheet chunking via openpyxl |
| Plain Text | `.txt` | UTF-8 encoded |

## Ingestion Behaviour

- **Idempotent:** Each file is SHA-256 hashed before indexing. Re-uploading the same file returns `status: "skipped"` with no duplicate chunks created.
- **Metadata preserved:** Source filename, file type, and page number (PDF/DOCX) or sheet name (XLSX) are stored in every chunk.
- **De-duplication:** Only the content hash is checked — renaming a file does not bypass de-duplication.
