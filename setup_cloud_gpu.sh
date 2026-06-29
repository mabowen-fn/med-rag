#!/bin/bash
# Cloud GPU Setup Script for Medical RAG System
# Run this on your cloud GPU instance (Linux with NVIDIA RTX 4090)

set -e

echo "=== Medical RAG Cloud GPU Setup ==="

# Check CUDA availability
echo "Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ NVIDIA driver detected"
else
    echo "✗ NVIDIA driver not found. Please install CUDA drivers first."
    exit 1
fi

# Check disk space
echo ""
echo "Checking disk space..."
df -h /root
AVAILABLE_GB=$(df -BG /root | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 15 ]; then
    echo "⚠ Warning: Less than 15GB available. CUDA libraries require ~10GB."
    echo "Cleaning up caches..."
    rm -rf /root/.cache/uv
    rm -rf /root/.cache/pip
    rm -rf /tmp/*
    echo "Cache cleaned. Available space:"
    df -h /root
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install PyTorch with CUDA 12.6 support
echo "Installing PyTorch with CUDA 12.6 support..."
uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install project dependencies
echo "Installing project dependencies..."
uv sync

# Install Ollama
echo ""
echo "=== Installing Ollama ==="
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed"
else
    echo "✓ Ollama already installed"
fi

# Start Ollama service in background
echo "Starting Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
    echo "✓ Ollama service started"
else
    echo "✓ Ollama service already running"
fi

# Pull Qwen3:8b model
echo ""
echo "=== Pulling Qwen3:8b model ==="
echo "This may take 5-10 minutes depending on network speed..."
ollama pull qwen3:8b
echo "✓ Model downloaded"

# Verify GPU setup
echo ""
echo "=== Verifying GPU Setup ==="
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else
    print('✗ CUDA not available - check installation')
"

# Verify Ollama
echo ""
echo "=== Verifying Ollama ==="
if ollama list | grep -q "qwen3:8b"; then
    echo "✓ Qwen3:8b model available"
else
    echo "✗ Qwen3:8b model not found"
    exit 1
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Build the knowledge base index:"
echo "     uv run python build_index.py"
echo ""
echo "  2. Start the web interface:"
echo "     uv run streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0"
echo ""
echo "  3. Access the UI at: http://YOUR_SERVER_IP:8501"
echo ""
echo "  4. Run evaluation (optional):"
echo "     uv run python run_evaluation.py"
echo "     uv run python run_ragas_evaluation.py"
