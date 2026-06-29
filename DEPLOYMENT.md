# Medical RAG System - Cloud GPU Deployment Guide

This guide provides complete instructions for deploying the Medical RAG system on a cloud GPU instance (NVIDIA RTX 4090 or similar).

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or equivalent
- **RAM**: 32GB+ recommended
- **Disk**: 50GB+ free space (models + index + cache)
- **CPU**: 8+ cores recommended

### Software Requirements
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **NVIDIA Driver**: Version 535+ (CUDA 12.6 compatible)
- **Python**: 3.10+ (managed by uv)

## Quick Start (Automated)

For a fully automated setup, run:

```bash
chmod +x setup_cloud_gpu.sh
./setup_cloud_gpu.sh
```

This script will:
1. Check CUDA availability
2. Install uv package manager
3. Install PyTorch with CUDA 12.6 support
4. Install project dependencies
5. Install Ollama
6. Download Qwen3:8b model
7. Verify GPU setup

After setup completes, continue with [Building the Index](#building-the-knowledge-base-index).

## Manual Setup (Step-by-Step)

### 1. Verify NVIDIA Driver

```bash
nvidia-smi
```

Expected output should show:
- Driver Version: 535+ or higher
- CUDA Version: 12.6 or higher
- GPU: NVIDIA RTX 4090 (or similar)

If not installed, install NVIDIA drivers:

```bash
# Ubuntu
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
```

### 2. Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/bin/env
```

Verify installation:

```bash
uv --version
```

### 3. Install PyTorch with CUDA Support

```bash
uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verify CUDA availability:

```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output:
```
CUDA: True, Device: NVIDIA GeForce RTX 4090
```

### 4. Install Project Dependencies

```bash
uv sync
```

This installs all dependencies from `pyproject.toml` including:
- sentence-transformers (embedding models)
- faiss-gpu (vector search)
- llama-index (RAG framework)
- streamlit (web UI)
- ragas (evaluation)
- And other dependencies

### 5. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama service:

```bash
ollama serve > /tmp/ollama.log 2>&1 &
```

Verify Ollama is running:

```bash
curl http://localhost:11434/api/version
```

Expected output:
```json
{"version":"0.x.x"}
```

### 6. Download Qwen3:8b Model

```bash
ollama pull qwen3:8b
```

This downloads ~5GB model. Verify:

```bash
ollama list
```

Expected output:
```
NAME           ID           SIZE   MODIFIED
qwen3:8b       xxxxxxxxxxx  5.0 GB 1 minute ago
```

### 7. Download Embedding Models

The embedding models will be downloaded automatically on first use, but you can pre-download them:

```bash
uv run python -c "
from sentence_transformers import SentenceTransformer
print('Downloading BGE-M3...')
model = SentenceTransformer('BAAI/bge-m3')
print('Downloading BGE-Reranker...')
reranker = SentenceTransformer('BAAI/bge-reranker-v2-m3')
print('Done!')
"
```

## Building the Knowledge Base Index

Before running the system, you need to build the FAISS index from the medical dataset:

```bash
uv run python build_index.py
```

This will:
1. Download and process Huatuo-26M dataset (~10,000 samples by default)
2. Generate embeddings using BGE-M3
3. Build FAISS vector index
4. Build BM25 keyword index
5. Save indices to `data/index/`

**Time estimate**: 10-20 minutes on RTX 4090

**Output files**:
- `data/index/faiss.index` - Vector index
- `data/index/texts.json` - Document texts
- `data/index/metadata.json` - Document metadata

To adjust the number of samples, edit `config.py`:

```python
class DataConfig(BaseModel):
    max_samples: int = 10000  # Increase for larger index
```

## Running the Web Interface

### Start Streamlit Server

```bash
uv run streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0
```

**Flags explained**:
- `--server.port 8501` - Port to listen on
- `--server.address 0.0.0.0` - Bind to all interfaces (required for remote access)

### Access the UI

Open your browser and navigate to:

```
http://YOUR_SERVER_IP:8501
```

Replace `YOUR_SERVER_IP` with your cloud instance's public IP address.

### Firewall Configuration

If you can't access the UI, ensure port 8501 is open:

```bash
# UFW (Ubuntu)
sudo ufw allow 8501/tcp

# Or iptables
sudo iptables -A INPUT -p tcp --dport 8501 -j ACCEPT
```

Also check your cloud provider's security group / firewall rules to allow inbound traffic on port 8501.

## Running Evaluations

### Basic Evaluation (BLEU, ROUGE, BERTScore)

```bash
uv run python run_evaluation.py
```

This evaluates:
- Baseline LLM (no RAG)
- Naive RAG (simple retrieval)
- Hybrid RAG (BM25 + vector + reranking)

**Time estimate**: 30-60 minutes for 100 samples

**Output**:
- `data/evaluation_report.json` - Detailed results
- `data/evaluation_report.md` - Human-readable report

### RAGAS Evaluation (Advanced Metrics)

```bash
uv run python run_ragas_evaluation.py
```

This evaluates using RAGAS framework:
- Faithfulness (factual accuracy)
- Answer Relevancy
- Context Precision
- Context Recall
- Answer Correctness

**Time estimate**: 20-40 minutes for 20 samples

**Output**:
- `data/ragas_evaluation_results.json` - Detailed results
- `data/ragas_evaluation_report.md` - Human-readable report

### Adjust Evaluation Parameters

Edit `config.py` to change evaluation settings:

```python
class EvaluationConfig(BaseModel):
    num_samples: int = 100  # Number of test samples
    metrics: list[str] = ["recall", "accuracy", "hallucination", "response_time"]
```

## System Architecture

### Component Overview

```
User Interface (Streamlit)
         ↓
Chat Engine (MedicalChatEngine)
         ↓
    ┌────┴────┐
    ↓         ↓
Retriever   Reranker
    ↓         ↓
FAISS + BM25  BGE-Reranker
    ↓
Embedding Model (BGE-M3)
    ↓
Knowledge Base (Huatuo-26M)
```

### Key Components

1. **HybridRetriever** (`src/pipeline/hybrid_retriever.py`)
   - Combines FAISS vector search (70% weight) + BM25 keyword search (30% weight)
   - Returns top-K candidates

2. **Reranker** (`src/pipeline/reranker.py`)
   - Cross-encoder reranking using BGE-Reranker-v2-m3
   - Improves precision by reordering candidates

3. **MedicalChatEngine** (`src/pipeline/chat_engine.py`)
   - Orchestrates retrieval → reranking → generation
   - Manages conversation history
   - Tracks citations

4. **EmbeddingModel** (`src/knowledge/embedding_model.py`)
   - BGE-M3 for document/query embeddings
   - 1024-dimensional vectors

5. **Ollama LLM** (external service)
   - Qwen3:8b for response generation
   - Runs on localhost:11434

## Configuration

All configuration is in `config.py`. Key settings:

### Model Configuration

```python
class ModelConfig(BaseModel):
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "qwen3:8b"
    device: str = "cuda"  # Auto-detected
```

### Retrieval Configuration

```python
class RetrievalConfig(BaseModel):
    top_k: int = 10  # Number of documents to retrieve
    hybrid_retrieval: bool = True
    bm25_weight: float = 0.3  # BM25 weight in fusion
    vector_weight: float = 0.7  # Vector weight in fusion
    rerank_top_n: int = 5  # Top-N after reranking
```

### LLM Configuration

```python
class LLMConfig(BaseModel):
    temperature: float = 0.1  # Lower = more deterministic
    max_tokens: int = 1024
    context_window: int = 4096
```

## Performance Optimization

### GPU Memory Management

Monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

If you encounter OOM errors:
1. Reduce `max_samples` in `config.py`
2. Reduce `top_k` in RetrievalConfig
3. Use quantized models (Q4_K_M instead of Q8_0)

### Faster Index Building

For large datasets, increase batch size in `src/data/dataset_loader.py`:

```python
batch_size = 256  # Increase if you have VRAM to spare
```

### Reduce Latency

1. **Preload models**: Models are loaded on first query. To preload, run a test query after startup.

2. **Use smaller index**: Reduce `max_samples` in config for faster retrieval.

3. **Enable caching**: Streamlit caches resources automatically. For custom caching, see `src/ui/app.py`.

## Troubleshooting

### CUDA Out of Memory

**Problem**: `CUDA out of memory` error

**Solution**:
```bash
# Reduce batch size in build_index.py
# Or reduce max_samples in config.py
```

### Ollama Connection Error

**Problem**: `Connection refused` when calling Ollama

**Solution**:
```bash
# Check if Ollama is running
pgrep ollama

# If not running, start it
ollama serve > /tmp/ollama.log 2>&1 &

# Verify it's accessible
curl http://localhost:11434/api/version
```

### FAISS Index Not Found

**Problem**: `FAISS index not found` error

**Solution**:
```bash
# Build the index
uv run python build_index.py
```

### Slow Response Times

**Problem**: Queries take >10 seconds

**Diagnosis**:
```bash
# Check GPU utilization
nvidia-smi

# Check Ollama logs
tail -f /tmp/ollama.log
```

**Solutions**:
1. Ensure models are loaded on GPU (check logs)
2. Reduce `top_k` in config
3. Use smaller LLM (qwen3:3b instead of qwen3:8b)

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
# Reinstall dependencies
uv sync

# Or force reinstall
uv pip install -e .
```

## Monitoring and Logs

### Application Logs

Logs are written to `logs/` directory:

```bash
tail -f logs/med_rag_*.log
```

### Ollama Logs

```bash
tail -f /tmp/ollama.log
```

### GPU Monitoring

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# Or use nvtop (if installed)
nvtop
```

### Performance Metrics

After evaluation, check:
- `data/evaluation_report.md` - Response times, accuracy metrics
- `data/ragas_evaluation_report.md` - RAGAS metrics

## Production Deployment

### Using systemd (Recommended)

Create `/etc/systemd/system/med-rag.service`:

```ini
[Unit]
Description=Medical RAG System
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/project/med-rag
ExecStart=/usr/local/bin/uv run streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/ollama.service`:

```ini
[Unit]
Description=Ollama LLM Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start services:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama med-rag
sudo systemctl start ollama med-rag
```

Check status:

```bash
sudo systemctl status ollama med-rag
```

### Using Docker

See `Dockerfile` and `docker-compose.yml` for containerized deployment.

Build and run:

```bash
docker-compose up -d
```

## Security Considerations

### Network Security

1. **Firewall**: Only expose port 8501 to trusted IPs
2. **HTTPS**: Use reverse proxy (nginx) with SSL certificate
3. **Authentication**: Add basic auth or integrate with SSO

### Data Privacy

1. **Local Processing**: All data stays on your server
2. **No External APIs**: Ollama runs locally
3. **Encryption**: Enable disk encryption for sensitive data

### Model Security

1. **Verify Checksums**: Verify model downloads
2. **Isolate**: Run in container/VM for isolation
3. **Updates**: Regularly update Ollama and models

## Cost Optimization

### Cloud Instance Selection

**Recommended instances**:
- **AWS**: g5.xlarge (RTX 4090, ~$1.50/hr)
- **GCP**: a2-highgpu-1g (A100, ~$3.50/hr)
- **Azure**: Standard_NC4as_T4_v3 (T4, ~$0.75/hr)
- **AutoDL**: RTX 4090 instances (~¥1.5/hr)

### Cost-Saving Tips

1. **Spot Instances**: Use spot/preemptible instances (50-80% cheaper)
2. **Auto-Shutdown**: Shut down when not in use
3. **Reserved Instances**: For long-term usage (1-3 year commitment)

## Support and Resources

### Documentation

- [README.md](README.md) - Project overview
- [COMPREHENSIVE_REPORT.md](COMPREHENSIVE_REPORT.md) - Technical details
- [SOTA_IMPLEMENTATION_SUMMARY.md](SOTA_IMPLEMENTATION_SUMMARY.md) - Advanced features

### Code Structure

```
med-rag/
├── src/
│   ├── pipeline/       # RAG pipeline components
│   ├── knowledge/      # Embedding and indexing
│   ├── data/           # Dataset loading
│   ├── evaluation/     # Evaluation metrics
│   └── ui/             # Streamlit interface
├── config.py           # Configuration
├── build_index.py      # Index building script
├── run_evaluation.py   # Evaluation script
└── setup_cloud_gpu.sh  # Setup script
```

### Getting Help

1. Check logs: `logs/` and `/tmp/ollama.log`
2. Review configuration: `config.py`
3. Check GPU status: `nvidia-smi`
4. Test Ollama: `curl http://localhost:11434/api/version`

## Appendix: Complete Command Reference

### Setup Commands

```bash
# Automated setup
chmod +x setup_cloud_gpu.sh
./setup_cloud_gpu.sh

# Manual setup
uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv sync
curl -fsSL https://ollama.com/install.sh | sh
ollama serve > /tmp/ollama.log 2>&1 &
ollama pull qwen3:8b
```

### Build Commands

```bash
# Build knowledge base index
uv run python build_index.py

# Pre-download models
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

### Run Commands

```bash
# Start web interface
uv run streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0

# Interactive CLI mode
uv run python main.py
```

### Evaluation Commands

```bash
# Basic evaluation
uv run python run_evaluation.py

# RAGAS evaluation
uv run python run_ragas_evaluation.py
```

### Monitoring Commands

```bash
# GPU monitoring
watch -n 1 nvidia-smi

# Check Ollama status
curl http://localhost:11434/api/version
ollama list

# View logs
tail -f logs/med_rag_*.log
tail -f /tmp/ollama.log
```

### Service Management (systemd)

```bash
# Start services
sudo systemctl start ollama med-rag

# Stop services
sudo systemctl stop ollama med-rag

# Restart services
sudo systemctl restart ollama med-rag

# Check status
sudo systemctl status ollama med-rag

# View logs
sudo journalctl -u med-rag -f
sudo journalctl -u ollama -f
```

---

**Last Updated**: 2026-06-29  
**Version**: 1.0  
**Tested On**: Ubuntu 22.04, NVIDIA RTX 4090, CUDA 12.6
