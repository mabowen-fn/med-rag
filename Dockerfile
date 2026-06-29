# Medical RAG System - Cloud GPU Docker Deployment
# Multi-stage build for optimized image size

FROM nvidia/cuda:12.6.0-base-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-venv \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* ./

# Install Python dependencies
RUN uv sync --no-dev

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/cache data/processed data/index

# Download embedding models during build (optional, reduces startup time)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')" || true
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-reranker-v2-m3')" || true

# Expose ports (Streamlit: 8501, Ollama: 11434)
EXPOSE 8501
EXPOSE 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:11434/api/version || exit 1

# Default command (can be overridden)
CMD ["bash", "-c", "ollama serve & sleep 5 && uv run streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0"]
