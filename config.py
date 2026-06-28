"""Configuration for Medical RAG System"""
from pathlib import Path
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration"""
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "qwen3:8b"
    device: str = "cpu"  # Mac M1 CPU mode


class DataConfig(BaseModel):
    """Data configuration"""
    dataset_name: str = "Huatuo-26M"
    cache_dir: str = "./data/cache"
    processed_dir: str = "./data/processed"
    max_samples: int = 10000  # For testing


class ChunkingConfig(BaseModel):
    """Chunking configuration"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    dynamic_chunking: bool = True


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    top_k: int = 10
    hybrid_retrieval: bool = True
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    rerank_top_n: int = 5
    context_prompt: str = "Given the context information below, answer the question. If the answer is not in the context, say so.\n\nContext: {context_str}\n\nQuestion: {question}\n\nAnswer:"
    verbose: bool = False


class LLMConfig(BaseModel):
    """LLM configuration"""
    temperature: float = 0.1
    max_tokens: int = 1024
    context_window: int = 4096


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""
    metrics: list[str] = Field(default=["recall", "accuracy", "hallucination", "response_time"])
    num_samples: int = 100


class Config(BaseModel):
    """Main configuration"""
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    
    # Paths
    base_dir: Path = Path(__file__).parent
    data_dir: Path = Path("./data")
    index_dir: Path = Path("./data/index")
    log_dir: Path = Path("./logs")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
