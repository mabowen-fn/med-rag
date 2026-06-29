"""Embedding model wrapper for Medical RAG System"""
from typing import List, Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from loguru import logger
from config import config
from tqdm import tqdm


class EmbeddingModel:
    """BGE-M3 embedding model wrapper"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or config.model.embedding_model
        self.device = device or config.model.device
        
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        
        self._model = HuggingFaceEmbedding(
            model_name=self.model_name,
            device=self.device,
            cache_folder=config.data.cache_dir,
        )
        
        logger.info("Embedding model loaded successfully")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """Embed a list of texts with progress tracking
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            show_progress: Whether to show progress bar
        """
        all_embeddings = []
        
        # Process in batches with progress bar
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding texts", total=num_batches)
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_embeddings = self._model.get_text_embedding_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        return self._model.get_query_embedding(query)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension by embedding a sample text"""
        sample_embedding = self._model.get_text_embedding("test")
        return len(sample_embedding)
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Make it callable like the LlamaIndex embedding model"""
        return self.embed_texts(texts)

