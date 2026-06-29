#!/bin/bash
# Health check script for Medical RAG System
# Checks all critical components are running and accessible

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=== Medical RAG System Health Check ==="
echo ""

# Check GPU
echo "1. Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} GPU: $GPU_INFO"
    else
        echo -e "${RED}✗${NC} nvidia-smi failed"
    fi
else
    echo -e "${RED}✗${NC} nvidia-smi not found"
fi

# Check Ollama
echo ""
echo "2. Checking Ollama..."
if pgrep -x "ollama" > /dev/null; then
    echo -e "${GREEN}✓${NC} Ollama process running"
    
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        VERSION=$(curl -s http://localhost:11434/api/version | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
        echo -e "${GREEN}✓${NC} Ollama API accessible (version: $VERSION)"
        
        if ollama list 2>/dev/null | grep -q "qwen3:8b"; then
            echo -e "${GREEN}✓${NC} Qwen3:8b model loaded"
        else
            echo -e "${RED}✗${NC} Qwen3:8b model not found"
        fi
    else
        echo -e "${RED}✗${NC} Ollama API not accessible"
    fi
else
    echo -e "${RED}✗${NC} Ollama not running"
fi

# Check Python environment
echo ""
echo "3. Checking Python environment..."
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✓${NC} uv installed"
    
    if uv run python -c "import torch" 2>/dev/null; then
        PYTORCH_VERSION=$(uv run python -c "import torch; print(torch.__version__)" 2>/dev/null)
        echo -e "${GREEN}✓${NC} PyTorch $PYTORCH_VERSION"
        
        CUDA_AVAILABLE=$(uv run python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$CUDA_AVAILABLE" = "True" ]; then
            echo -e "${GREEN}✓${NC} CUDA available"
        else
            echo -e "${YELLOW}⚠${NC} CUDA not available (CPU mode)"
        fi
    else
        echo -e "${RED}✗${NC} PyTorch not installed"
    fi
else
    echo -e "${RED}✗${NC} uv not found"
fi

# Check FAISS index
echo ""
echo "4. Checking FAISS index..."
if [ -f "data/index/faiss.index" ]; then
    INDEX_SIZE=$(du -h data/index/faiss.index | cut -f1)
    echo -e "${GREEN}✓${NC} FAISS index exists ($INDEX_SIZE)"
else
    echo -e "${RED}✗${NC} FAISS index not found"
    echo "  Run: make build"
fi

# Check Streamlit
echo ""
echo "5. Checking Streamlit..."
if pgrep -f "streamlit run" > /dev/null; then
    echo -e "${GREEN}✓${NC} Streamlit running"
    
    if curl -s http://localhost:8501/healthz > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Streamlit accessible at http://localhost:8501"
    else
        echo -e "${YELLOW}⚠${NC} Streamlit running but not accessible"
    fi
else
    echo -e "${YELLOW}⚠${NC} Streamlit not running"
    echo "  Run: make run"
fi

# Check disk space
echo ""
echo "6. Checking disk space..."
AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE" -gt 10 ]; then
    echo -e "${GREEN}✓${NC} Disk space: ${AVAILABLE}GB available"
else
    echo -e "${YELLOW}⚠${NC} Low disk space: ${AVAILABLE}GB available"
fi

# Check logs
echo ""
echo "7. Checking logs..."
if [ -d "logs" ]; then
    LOG_COUNT=$(find logs -name "*.log" -mmin -60 | wc -l)
    echo -e "${GREEN}✓${NC} $LOG_COUNT log files updated in last hour"
else
    echo -e "${YELLOW}⚠${NC} No logs directory"
fi

echo ""
echo "=== Health Check Complete ==="
