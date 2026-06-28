"""Hybrid retriever combining BM25 and vector search"""
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from loguru import logger
from config import config


class HybridRetriever:
    """Hybrid retrieval combining BM25 (sparse) and FAISS (dense) retrieval"""
    
    def __init__(
        self,
        faiss_index,
        embedding_model,
        bm25_weight: float = None,
        vector_weight: float = None,
    ):
        self.faiss_index = faiss_index
        self.embedding_model = embedding_model
        self.bm25_weight = bm25_weight or config.retrieval.bm25_weight
        self.vector_weight = vector_weight or config.retrieval.vector_weight
        self.bm25 = None
        self.corpus_texts = []
        self.corpus_metadata = []
    
    def build_bm25_index(self, texts: List[str], metadata: List[Dict[str, Any]]):
        """Build BM25 index from corpus"""
        logger.info(f"Building BM25 index with {len(texts)} documents")
        
        # Tokenize for BM25
        tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_texts = texts
        self.corpus_metadata = metadata
        
        logger.info("BM25 index built successfully")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        import re
        # Split by whitespace and punctuation
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Hybrid retrieval with score fusion"""
        top_k = top_k or config.retrieval.top_k
        
        # Vector retrieval
        query_embedding = self.embedding_model.embed_query(query)
        vector_results = self.faiss_index.search(query_embedding, top_k=top_k)
        
        # BM25 retrieval
        if self.bm25 is not None:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k BM25 results
            bm25_top_indices = bm25_scores.argsort()[::-1][:top_k]
            bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
            
            bm25_results = []
            for idx in bm25_top_indices:
                bm25_results.append({
                    "score": float(bm25_scores[idx]) / bm25_max,  # Normalize
                    "metadata": self.corpus_metadata[idx],
                    "text": self.corpus_texts[idx],
                })
        else:
            bm25_results = []
        
        # Score fusion
        fused = self._fuse_scores(vector_results, bm25_results)
        
        # Sort by fused score
        fused.sort(key=lambda x: x["score"], reverse=True)
        
        return fused[:top_k]
    
    def _fuse_scores(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Fuse scores from vector and BM25 retrieval"""
        # Create lookup by metadata chunk_id
        fused_scores = {}
        
        for result in vector_results:
            key = self._result_key(result["metadata"])
            if key not in fused_scores:
                fused_scores[key] = {
                    "score": 0.0,
                    "metadata": result["metadata"],
                    "text": result.get("text", ""),
                    "retrieval_source": [],
                }
            fused_scores[key]["score"] += self.vector_weight * result["score"]
            fused_scores[key]["retrieval_source"].append("vector")
        
        for result in bm25_results:
            key = self._result_key(result["metadata"])
            if key not in fused_scores:
                fused_scores[key] = {
                    "score": 0.0,
                    "metadata": result["metadata"],
                    "text": result.get("text", ""),
                    "retrieval_source": [],
                }
            fused_scores[key]["score"] += self.bm25_weight * result["score"]
            fused_scores[key]["retrieval_source"].append("bm25")
        
        return list(fused_scores.values())
    
    def _result_key(self, metadata: Dict) -> str:
        """Generate unique key for deduplication"""
        return f"{metadata.get('source', '')}_{metadata.get('chunk_id', '')}"
