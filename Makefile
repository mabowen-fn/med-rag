.PHONY: help setup build run test clean

help:
	@echo "Medical RAG System - Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup    - Install dependencies and setup environment"
	@echo "  make build    - Build FAISS index from dataset"
	@echo "  make run      - Start Streamlit web interface"
	@echo "  make test     - Run smoke tests"
	@echo "  make eval     - Run evaluation suite"
	@echo "  make clean    - Clean cache and temporary files"
	@echo ""

setup:
	@echo "Setting up Medical RAG System..."
	chmod +x setup_cloud_gpu.sh
	./setup_cloud_gpu.sh

build:
	@echo "Building FAISS index..."
	uv run python build_index.py

run:
	@echo "Starting Streamlit server..."
	uv run streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0

test:
	@echo "Running smoke tests..."
	uv run python scripts/smoke_test.py

eval:
	@echo "Running evaluation..."
	uv run python run_evaluation.py

eval-ragas:
	@echo "Running RAGAS evaluation..."
	uv run python run_ragas_evaluation.py

clean:
	@echo "Cleaning cache and temporary files..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf logs/*
	rm -rf data/cache/*
	rm -rf data/processed/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

install-dev:
	@echo "Installing development dependencies..."
	uv sync --dev

docker-build:
	@echo "Building Docker image..."
	docker build -t med-rag:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8501:8501 --gpus all med-rag:latest
