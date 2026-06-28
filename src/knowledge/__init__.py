"""Knowledge module for Medical RAG System"""
from .embedding_model import EmbeddingModel
from .faiss_index import FaissIndexBuilder

__all__ = ["EmbeddingModel", "FaissIndexBuilder"]
