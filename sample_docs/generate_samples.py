"""
sample_docs/generate_samples.py
===============================
Reproducible generator for realistic multi-format sample knowledge base documents.

Generates:
  1. sample_docs/company_security_policy.pdf (2-page information security policy)
  2. sample_docs/product_specs.docx (Technical architecture narrative & table specs)
  3. sample_docs/q3_pricing_tiers.xlsx (Multi-sheet Excel workbook: Plans & AddOns)

Run with:
  python -m sample_docs.generate_samples
"""

from __future__ import annotations

from pathlib import Path


def generate_company_security_policy_pdf(output_path: Path) -> None:
    """Generate a realistic 2-page Information Security Policy PDF using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF

    doc = fitz.open()

    # --- Page 1 ---
    page1 = doc.new_page()
    page1_text = """Acme Global Information Security Policy
Document ID: SEC-POL-2026-v2.1
Classification: Confidential - Internal Only

1. Overview and Purpose
This document establishes the mandatory information security controls, encryption specifications,
and incident response requirements across all Acme Global computing systems and data repositories.

2. Data Encryption Standards
2.1 Encryption at Rest:
All customer records, production databases, vector representations, and conversational thread
histories must be encrypted at rest using AES-256 cipher specifications. Unencrypted local
storage of credentials, tokens, or raw document bytes is strictly forbidden.

2.2 Encryption in Transit:
All communications traversing public networks or internal perimeter boundaries must mandate
TLS 1.3 encryption. Deprecated protocols including SSL 3.0, TLS 1.0, and TLS 1.1 are blocked
at the API gateway level.

3. SLA and Incident Response Escalation Rules
3.1 Service Level Agreement:
Acme Global guarantees a production monthly service availability uptime SLA of 99.95%.

3.2 Incident Response Windows:
- Severity 1 (Critical Outage or Data Breach): Mandatory escalation and containment within 4 hours.
- Severity 2 (Degraded Operations): Containment window within 12 hours.
- Severity 3 (Minor Defect): Resolution target within 48 business hours.
"""
    # Insert text starting near top-left margin
    page1.insert_textbox(fitz.Rect(54, 54, 550, 750), page1_text, fontsize=10)

    # --- Page 2 ---
    page2 = doc.new_page()
    page2_text = """Acme Global Information Security Policy (Continued)
Document ID: SEC-POL-2026-v2.1
Page 2 of 2

4. Access Control and Authentication
4.1 Role-Based Access Control (RBAC):
All employee access to customer document stores and vector databases must follow the principle of
least privilege enforced via strict Role-Based Access Control (RBAC).

4.2 Multi-Factor Authentication (MFA):
Multi-Factor Authentication (MFA) is strictly mandatory for all administrative consoles,
database bastion hosts, and production API administrative accounts.

5. Audit Logging and Retention Rules
5.1 Audit Trail Mandate:
Every document ingestion event, search query, index deletion request, and permission modification
must generate an immutable, tamper-evident audit log record.

5.2 Retention Policy:
Audit logs must be retained securely in write-once-read-many (WORM) storage for a minimum of
365 days before archival or scheduled deletion.
"""
    page2.insert_textbox(fitz.Rect(54, 54, 550, 750), page2_text, fontsize=10)

    doc.save(str(output_path))
    doc.close()


def generate_product_specs_docx(output_path: Path) -> None:
    """Generate a realistic product architecture DOCX document using python-docx."""
    from docx import Document  # type: ignore[import-untyped]

    doc = Document()
    doc.add_heading("Conversational RAG Engine Product Specifications", level=1)

    p1 = doc.add_paragraph(
        "The Production Multi-Document Conversational RAG Engine delivers low-latency semantic search, "
        "hybrid retrieval, and grounded conversational question answering across diverse unstructured "
        "and tabular corporate documentation. It operates on a 100% zero-cost ($0) open-source architecture "
        "without dependency on commercial API tokens or closed models."
    )

    doc.add_heading("Key Technical Architecture Specifications", level=2)

    # Structured technical specification table
    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Specification Parameter"
    hdr_cells[1].text = "Target Engineering Value"
    hdr_cells[2].text = "Design & Validation Notes"

    specs = [
        ("Latency Budget", "< 800 ms", "End-to-end p95 response time for hybrid retrieval and generation"),
        ("Supported Formats", "PDF, DOCX, XLSX, TXT", "Layout-aware parsing with structure-preserving metadata"),
        ("Max Document Size", "50 MB", "Per individual uploaded document limit"),
        ("Memory Retention", "30 days", "SQLite multi-turn conversation thread time-to-live (TTL)"),
        ("Reranking Model", "ms-marco-MiniLM-L-6-v2", "Local cross-encoder logit scoring with negative threshold support"),
        ("Embedding Dimension", "384 float32", "BAAI/bge-small-en-v1.5 dense representations"),
    ]

    for param, val, notes in specs:
        row_cells = table.add_row().cells
        row_cells[0].text = param
        row_cells[1].text = val
        row_cells[2].text = notes

    doc.add_heading("Idempotent Ingestion & Deduplication", level=2)
    doc.add_paragraph(
        "Security isolation ensures that tenant documents are partitioned with cryptographic SHA-256 content "
        "verification. When an identical document is re-uploaded, the ingestion layer detects the duplicate "
        "hash in the local manifest and skips re-indexing, eliminating duplicate chunk storage."
    )

    doc.save(str(output_path))


def generate_q3_pricing_tiers_xlsx(output_path: Path) -> None:
    """Generate a multi-sheet pricing Excel workbook using openpyxl."""
    from openpyxl import Workbook  # type: ignore[import-untyped]

    wb = Workbook()

    # --- Sheet 1: Plans ---
    ws1 = wb.active
    ws1.title = "Plans"
    ws1.append(["Plan Name", "Monthly Cost", "Monthly Quota", "Support Level", "Max File Size"])
    ws1.append(["Starter", "$0", "50,000 queries", "Community", "10 MB"])
    ws1.append(["Pro", "$49", "500,000 queries", "Priority Email", "25 MB"])
    ws1.append(["Enterprise", "$499", "Unlimited", "24/7 Dedicated SLA", "50 MB"])

    # --- Sheet 2: AddOns ---
    ws2 = wb.create_sheet(title="AddOns")
    ws2.append(["AddOn Feature", "Monthly Cost", "Description", "Eligible Tiers"])
    ws2.append(["Extra Storage", "$10/mo", "100 GB extra vector capacity", "Starter, Pro, Enterprise"])
    ws2.append(["Custom Embeddings", "$50/mo", "Fine-tuned domain-specific embedder support", "Pro, Enterprise"])
    ws2.append(["Dedicated Ingress", "$100/mo", "Private VPC endpoint and dedicated IP allowlist", "Enterprise"])

    wb.save(str(output_path))


def generate_all_samples(target_dir: Path | None = None) -> None:
    """Generate all 3 sample files into target directory."""
    if target_dir is None:
        target_dir = Path(__file__).parent

    target_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = target_dir / "company_security_policy.pdf"
    docx_path = target_dir / "product_specs.docx"
    xlsx_path = target_dir / "q3_pricing_tiers.xlsx"

    print(f"Generating sample PDF: {pdf_path}")
    generate_company_security_policy_pdf(pdf_path)

    print(f"Generating sample DOCX: {docx_path}")
    generate_product_specs_docx(docx_path)

    print(f"Generating sample XLSX: {xlsx_path}")
    generate_q3_pricing_tiers_xlsx(xlsx_path)

    print("All sample documents generated successfully.")


if __name__ == "__main__":
    generate_all_samples()
