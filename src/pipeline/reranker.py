"""Reranker using BGE-Reranker-v2-m3"""
from typing import List, Dict, Any, Optional
from loguru import logger
from config import config


class Reranker:
    """Cross-encoder reranker using BGE-Reranker-v2-m3"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or config.model.reranker_model
        self.device = device or config.model.device
        
        logger.info(f"Loading reranker model: {self.model_name} on {self.device}")
        
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = None,
    ) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder"""
        top_n = top_n or config.retrieval.rerank_top_n
        
        if not documents:
            return []
        
        # Prepare pairs
        pairs = [(query, doc["text"]) for doc in documents]
        
        # Get reranker scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        reranked = []
        for doc, score in zip(documents, scores):
            reranked.append({
                **doc,
                "rerank_score": float(score),
            })
        
        # Sort by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_n]
