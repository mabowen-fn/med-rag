"""Standalone script to build the FAISS index"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from main import build_index


if __name__ == "__main__":
    logger.info("Building FAISS index...")
    build_index()
    logger.info("Done!")
