"""FAISS index builder for Medical RAG System"""
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from config import config


class FaissIndexBuilder:
    """Build and manage FAISS vector index"""
    
    def __init__(self, dimension: int, index_path: str = None):
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else config.index_dir / "faiss.index"
        self.metadata_path = self.index_path.with_suffix(".meta.json")
        
        # Detect GPU availability
        self.device = "gpu" if faiss.get_num_gpus() > 0 else "cpu"
        logger.info(f"Initializing FAISS index (dimension={dimension}, device={self.device})")
        
        # Create base index
        base_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Move to GPU if available
        if self.device == "gpu":
            self.index = faiss.index_cpu_to_all_gpus(base_index)
            logger.info(f"FAISS index using {faiss.get_num_gpus()} GPU(s)")
        else:
            self.index = base_index
        
        self.metadata = []
    
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]):
        """Add vectors to index"""
        if len(vectors) != len(metadata):
            raise ValueError("Vectors and metadata must have same length")
        
        # Convert to numpy array
        vectors_np = np.array(vectors, dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors_np)
        
        # Add to index
        self.index.add(vectors_np)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(vectors)} vectors to FAISS index")
    
    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        # Convert to numpy array
        query_np = np.array([query_vector], dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_np)
        
        # Search
        scores, indices = self.index.search(query_np, top_k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    "score": float(score),
                    "metadata": self.metadata[idx],
                })
        
        return results
    
    def save(self):
        """Save index and metadata to disk"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert GPU index to CPU for serialization
        if self.device == "gpu":
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(self.index_path))
        else:
            faiss.write_index(self.index, str(self.index_path))
        
        import json
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved FAISS index to {self.index_path}")
    
    def load(self):
        """Load index and metadata from disk"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        
        cpu_index = faiss.read_index(str(self.index_path))
        
        # Move to GPU if available
        if self.device == "gpu":
            self.index = faiss.index_cpu_to_all_gpus(cpu_index)
        else:
            self.index = cpu_index
        
        import json
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded FAISS index from {self.index_path} with {len(self.metadata)} vectors")

