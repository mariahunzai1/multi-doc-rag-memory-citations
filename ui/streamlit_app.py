"""
ui/streamlit_app.py
===================
Phase 3: Streamlit Presentation Client.

Architectural guarantees:
- Communicates STRICTLY over HTTP via httpx with the FastAPI backend.
- RULE: NEVER imports anything from core/.
- Configurable BACKEND_URL via environment variable (default: http://localhost:8000).
- Safeguard: All network calls wrapped in try/except httpx.RequestError for clean error states.
- Multi-turn conversation rendering, live backend status badge, inline verifiable citations,
  and abstain warning callouts.

Run with:
  streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Production Conversational RAG",
    page_icon="🤖",
    layout="wide",
)


# ===========================================================================
# Self-Contained Cloud Launcher (auto-start backend on localhost)
# ===========================================================================


@st.cache_resource
def _ensure_backend() -> None:
    """
    Auto-launch uvicorn in a daemon thread if BACKEND_URL points to localhost
    and GET /health is unreachable.

    Runs in-process via threading.Thread (saves ~150 MB vs subprocess.Popen).
    Polls /health for up to 45 seconds so the first-boot HuggingFace
    embedding model download never triggers a premature timeout.
    """
    parsed = urlparse(BACKEND_URL)
    hostname = parsed.hostname or ""
    if hostname not in ("localhost", "127.0.0.1"):
        return  # Remote backend — do not auto-launch

    # Check if already running
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{BACKEND_URL}/health")
            if resp.status_code == 200:
                return
    except httpx.RequestError:
        pass

    # Start uvicorn in a daemon thread (in-process, no subprocess overhead)
    port = parsed.port or 8000

    def _run_server() -> None:
        import uvicorn

        uvicorn.run("api.main:app", host="127.0.0.1", port=port, log_level="info")

    thread = threading.Thread(target=_run_server, daemon=True, name="uvicorn-backend")
    thread.start()

    # Poll GET /health for up to 45 seconds (first boot downloads HF models)
    deadline = time.time() + 45.0
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{BACKEND_URL}/health")
                if resp.status_code == 200:
                    return
        except httpx.RequestError:
            pass
        time.sleep(0.5)


_ensure_backend()


# ===========================================================================
# HTTP Client Helper (Defensive error handling)
# ===========================================================================


def check_backend_health() -> dict[str, Any] | None:
    """Query GET /health with defensive error handling."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{BACKEND_URL}/health")
            if resp.status_code == 200:
                return resp.json()
    except httpx.RequestError:
        pass
    return None


def fetch_sessions() -> list[str]:
    """Fetch active session IDs from backend."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{BACKEND_URL}/sessions")
            if resp.status_code == 200:
                return resp.json()
    except httpx.RequestError:
        pass
    return []


def fetch_history(session_id: str) -> list[dict[str, str]]:
    """Fetch chronological message history for a session."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{BACKEND_URL}/sessions/{session_id}/history")
            if resp.status_code == 200:
                return resp.json()
    except httpx.RequestError:
        pass
    return []


def post_chat(session_id: str, question: str, top_k: int) -> dict[str, Any] | None:
    """Send user question to POST /chat."""
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{BACKEND_URL}/chat",
                json={
                    "session_id": session_id,
                    "question": question,
                    "top_k": top_k,
                },
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"Backend returned error {resp.status_code}: {resp.text}")
    except httpx.RequestError as exc:
        st.error(f"Cannot reach backend at {BACKEND_URL}: {exc}")
    return None


def post_ingest(uploaded_files: list[Any]) -> list[dict[str, Any]] | None:
    """Upload files to POST /ingest."""
    try:
        files_payload = [
            ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
            for f in uploaded_files
        ]
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{BACKEND_URL}/ingest", files=files_payload)
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"Ingestion failed ({resp.status_code}): {resp.text}")
    except httpx.RequestError as exc:
        st.error(f"Cannot reach backend for ingestion: {exc}")
    return None


def delete_session_backend(session_id: str) -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.delete(f"{BACKEND_URL}/sessions/{session_id}")
            return resp.status_code == 200
    except httpx.RequestError:
        return False


def delete_documents_backend() -> bool:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(f"{BACKEND_URL}/documents")
            return resp.status_code == 200
    except httpx.RequestError:
        return False


# ===========================================================================
# Session State Initialization
# ===========================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ===========================================================================
# Sidebar
# ===========================================================================

with st.sidebar:
    st.title("⚙️ Engine Control")

    # ── Live Health Status Badge ───────────────────────────────────────
    health_data = check_backend_health()
    if health_data:
        st.success(f"🟢 **Online** ({health_data.get('llm_model', 'unknown')})")
        with st.expander("Model Configuration", expanded=False):
            st.write(f"**LLM:** `{health_data.get('llm_model')}`")
            st.write(f"**Embedder:** `{health_data.get('embedding_model')}`")
    else:
        st.error(f"🔴 **Backend Offline** ({BACKEND_URL})")
        st.caption("Ensure the FastAPI backend is running via `uvicorn api.main:app`.")

    st.divider()

    # ── Document Ingestion ─────────────────────────────────────────────
    st.subheader("📁 Knowledge Base Ingestion")
    uploaded = st.file_uploader(
        "Upload Source Documents",
        type=["pdf", "docx", "xlsx", "txt"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, XLSX, or TXT documents. SHA-256 deduplication skips duplicates automatically.",
    )

    if st.button("📥 Index Documents", use_container_width=True, disabled=not uploaded):
        if uploaded:
            with st.spinner("Parsing, chunking, and indexing into ChromaDB + BM25..."):
                results = post_ingest(uploaded)
                if results:
                    indexed = sum(1 for r in results if r.get("status") == "indexed")
                    skipped = sum(1 for r in results if r.get("status") == "skipped")
                    errors = sum(1 for r in results if r.get("status") == "error")

                    st.toast(f"Done: {indexed} indexed, {skipped} skipped, {errors} errors")
                    for r in results:
                        status = r.get("status")
                        fname = r.get("filename")
                        if status == "indexed":
                            st.caption(f"✅ **{fname}**: {r.get('chunks_added', 0)} chunks added")
                        elif status == "skipped":
                            st.caption(f"⏭️ **{fname}**: Skipped (duplicate hash)")
                        else:
                            st.caption(f"❌ **{fname}**: Error: {r.get('error_detail')}")

    st.divider()

    # ── Retrieval Parameters ───────────────────────────────────────────
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider(
        "Citations (top_k)",
        min_value=1,
        max_value=15,
        value=4,
        help="Maximum qualifying passages to synthesize and cite.",
    )

    st.divider()

    # ── Session Management ─────────────────────────────────────────────
    st.subheader("💬 Sessions")
    existing_sessions = fetch_sessions()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            delete_session_backend(st.session_state.session_id)
            st.session_state.messages = []
            st.rerun()

    if existing_sessions:
        options = (
            [st.session_state.session_id] + [s for s in existing_sessions if s != st.session_state.session_id]
            if st.session_state.session_id in existing_sessions
            else [st.session_state.session_id] + existing_sessions
        )
        selected_session = st.selectbox(
            "Active Session ID",
            options=options,
            index=0,
        )
        if selected_session != st.session_state.session_id:
            st.session_state.session_id = selected_session
            # Load remote history
            history = fetch_history(selected_session)
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in history
            ]
            st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    st.divider()

    # ── Reset All Documents ────────────────────────────────────────────
    if st.button("⚠️ Reset Knowledge Base", use_container_width=True, type="secondary"):
        with st.spinner("Purging indexes..."):
            if delete_documents_backend():
                st.success("Knowledge base cleared.")
            else:
                st.error("Failed to clear knowledge base.")


# ===========================================================================
# Main Chat Area
# ===========================================================================

st.title("🤖 Multi-Document Conversational RAG Engine")
st.caption(
    "100% Zero-Cost Local Stack • Dense + BM25 + RRF Hybrid Retrieval • "
    "Cross-Encoder Logit Reranking • Verifiable Inline Citations"
)

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]:
            meta = msg["metadata"]
            if meta.get("abstained"):
                st.warning("⚠️ **Abstention**: No reliable document evidence found to answer this query.")
            st.caption(
                f"🔍 *Standalone Query:* `{meta.get('standalone_query', '')}` | "
                f"⏱️ *Latency:* `{meta.get('latency_ms', 0)} ms`"
            )
            citations = meta.get("citations", [])
            if citations:
                with st.expander(f"📚 Verifiable Citations ({len(citations)})", expanded=False):
                    for idx, cit in enumerate(citations, start=1):
                        st.markdown(
                            f"**[Source {idx}]** `{cit.get('source_filename')}` — "
                            f"Location: `{cit.get('location')}` | "
                            f"Relevance Logit: `{round(cit.get('relevance_score', 0.0), 3)}`"
                        )
                        st.info(cit.get("excerpt", ""))

# Chat input
if prompt := st.chat_input("Ask a question about your indexed documents..."):
    # Display user query immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            response_data = post_chat(
                session_id=st.session_state.session_id,
                question=prompt,
                top_k=top_k,
            )

        if response_data:
            answer = response_data.get("answer", "")
            abstained = response_data.get("abstained", False)
            standalone_query = response_data.get("standalone_query", prompt)
            citations = response_data.get("citations", [])
            latency_ms = response_data.get("latency_ms", 0.0)

            if abstained:
                st.warning("⚠️ **Abstention**: No reliable document evidence found to answer this query.")

            st.markdown(answer)

            st.caption(
                f"🔍 *Standalone Query:* `{standalone_query}` | "
                f"⏱️ *Latency:* `{latency_ms} ms`"
            )

            if citations:
                with st.expander(f"📚 Verifiable Citations ({len(citations)})", expanded=True):
                    for idx, cit in enumerate(citations, start=1):
                        st.markdown(
                            f"**[Source {idx}]** `{cit.get('source_filename')}` — "
                            f"Location: `{cit.get('location')}` | "
                            f"Relevance Logit: `{round(cit.get('relevance_score', 0.0), 3)}`"
                        )
                        st.info(cit.get("excerpt", ""))

            # Save in local state
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "metadata": {
                        "abstained": abstained,
                        "standalone_query": standalone_query,
                        "latency_ms": latency_ms,
                        "citations": citations,
                    },
                }
            )
