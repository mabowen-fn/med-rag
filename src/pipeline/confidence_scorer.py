"""Confidence scorer for Medical RAG System"""
from typing import List, Dict, Any
from loguru import logger


class ConfidenceScorer:
    """Calculate confidence scores for retrieved documents and LLM responses"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def calculate_confidence(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate overall confidence score for retrieved context"""
        if not documents:
            return {
                "confidence": 0.0,
                "is_reliable": False,
                "reason": "No documents retrieved",
            }
        
        # Calculate multiple confidence metrics
        retrieval_confidence = self._retrieval_confidence(documents)
        diversity_confidence = self._diversity_confidence(documents)
        relevance_confidence = self._relevance_confidence(query, documents)
        
        # Weighted average
        overall_confidence = (
            0.4 * retrieval_confidence +
            0.3 * diversity_confidence +
            0.3 * relevance_confidence
        )
        
        is_reliable = overall_confidence >= self.threshold
        
        return {
            "confidence": float(overall_confidence),
            "is_reliable": is_reliable,
            "retrieval_confidence": float(retrieval_confidence),
            "diversity_confidence": float(diversity_confidence),
            "relevance_confidence": float(relevance_confidence),
            "reason": "High confidence" if is_reliable else "Low confidence - consider rephrasing query",
        }
    
    def _retrieval_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """Confidence based on retrieval scores"""
        if not documents:
            return 0.0
        
        # Use rerank scores if available, otherwise use retrieval scores
        scores = [doc.get("rerank_score", doc.get("score", 0.0)) for doc in documents]
        
        # Normalize and average
        max_score = max(scores) if scores else 1.0
        if max_score > 0:
            normalized_scores = [s / max_score for s in scores]
            return sum(normalized_scores) / len(normalized_scores)
        return 0.0
    
    def _diversity_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """Confidence based on document diversity"""
        if len(documents) <= 1:
            return 0.5
        
        # Check source diversity
        sources = set()
        for doc in documents:
            source = doc.get("metadata", {}).get("source", "unknown")
            sources.add(source)
        
        # More diverse sources = higher confidence
        diversity_ratio = len(sources) / len(documents)
        return min(1.0, diversity_ratio * 1.5)  # Cap at 1.0
    
    def _relevance_confidence(self, query: str, documents: List[Dict[str, Any]]) -> float:
        """Confidence based on query-document relevance"""
        if not documents:
            return 0.0
        
        # Simple keyword overlap as proxy for relevance
        query_keywords = set(query.lower().split())
        
        relevance_scores = []
        for doc in documents:
            doc_text = doc.get("text", "").lower()
            doc_keywords = set(doc_text.split())
            
            overlap = len(query_keywords & doc_keywords)
            relevance = overlap / len(query_keywords) if query_keywords else 0.0
            relevance_scores.append(relevance)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
