# ============================================================
#  Dockerfile — Production Multi-Document Conversational RAG
#  Runs the FastAPI backend. Ollama must be reachable at
#  LLM_BASE_URL (default: http://host.docker.internal:11434/v1)
# ============================================================

FROM python:3.11-slim

# Prevent .pyc files and enable stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (excludes .env, chroma_db/, *.sqlite via .dockerignore)
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Start the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
