"""Main entry point for Medical RAG System"""
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from src.data import DatasetLoader, DynamicChunker
from src.knowledge import EmbeddingModel, FaissIndexBuilder
from src.pipeline import (
    HybridRetriever,
    Reranker,
    ConfidenceScorer,
    CitationTracker,
    MedicalChatEngine,
)
from src.evaluation import RAGEvaluator


def build_index():
    """Build FAISS index from dataset"""
    logger.info("Starting index building process")
    
    # Load dataset
    loader = DatasetLoader()
    try:
        dataset = loader.load_huatuo_26m(max_samples=config.data.max_samples)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Process dataset into documents
    documents = []
    for item in dataset:
        # Extract text based on dataset structure
        if "instruction" in item and "output" in item:
            text = f"Question: {item['instruction']}\nAnswer: {item['output']}"
        elif "input" in item and "output" in item:
            text = f"Question: {item['input']}\nAnswer: {item['output']}"
        else:
            text = str(item)
        
        documents.append({
            "text": text,
            "metadata": {
                "source": "Huatuo-26M",
                "doc_id": len(documents),
            },
        })
    
    logger.info(f"Processed {len(documents)} documents")
    
    # Dynamic chunking
    chunker = DynamicChunker()
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Generated {len(chunks)} chunks")
    
    # Load embedding model
    embedding_model = EmbeddingModel()
    
    # Build FAISS index
    faiss_index = FaissIndexBuilder(dimension=embedding_model.dimension)
    
    # Embed chunks
    texts = [chunk["text"] for chunk in chunks]
    metadata = [chunk["metadata"] for chunk in chunks]
    
    logger.info("Embedding chunks...")
    embeddings = embedding_model.embed_texts(texts)
    
    # Add to index
    faiss_index.add_vectors(embeddings, metadata)
    
    # Save index
    faiss_index.save()
    
    logger.info("Index building completed")
    
    return faiss_index, texts, metadata


def create_chat_engine(faiss_index=None, texts=None, metadata=None):
    """Create and return a configured chat engine"""
    logger.info("Creating chat engine")
    
    # Load embedding model
    embedding_model = EmbeddingModel()
    
    # Load or use provided FAISS index
    if faiss_index is None:
        faiss_index = FaissIndexBuilder(dimension=embedding_model.dimension)
        try:
            faiss_index.load()
        except FileNotFoundError:
            logger.warning("FAISS index not found. Building index...")
            faiss_index, texts, metadata = build_index()
    
    # Initialize components
    retriever = HybridRetriever(faiss_index, embedding_model)
    
    # Build BM25 index if texts provided
    if texts and metadata:
        retriever.build_bm25_index(texts, metadata)
    
    reranker = Reranker()
    confidence_scorer = ConfidenceScorer()
    citation_tracker = CitationTracker()
    
    # Create chat engine
    chat_engine = MedicalChatEngine(
        retriever=retriever,
        reranker=reranker,
        confidence_scorer=confidence_scorer,
        citation_tracker=citation_tracker,
    )
    
    logger.info("Chat engine created successfully")
    return chat_engine


def run_evaluation(chat_engine, num_samples: int = None):
    """Run evaluation on test cases"""
    logger.info("Starting evaluation")
    
    num_samples = num_samples or config.evaluation.num_samples
    
    # Load test dataset
    loader = DatasetLoader()
    try:
        test_dataset = loader.load_cmeqa(split="test")
    except Exception as e:
        logger.warning(f"Failed to load CMeQA: {e}. Using sample test cases.")
        # Fallback to sample test cases
        test_dataset = [
            {
                "query": "What are the common symptoms of diabetes?",
                "ground_truth": "Common symptoms include frequent urination, increased thirst, unexplained weight loss, fatigue, and blurred vision.",
                "relevant_docs": [],
            },
            {
                "query": "How is hypertension treated?",
                "ground_truth": "Hypertension is treated with lifestyle modifications and antihypertensive medications such as ACE inhibitors, ARBs, calcium channel blockers, and diuretics.",
                "relevant_docs": [],
            },
        ]
    
    # Prepare test cases
    test_cases = []
    for item in test_dataset[:num_samples]:
        if isinstance(item, dict):
            test_cases.append({
                "query": item.get("question", item.get("query", "")),
                "ground_truth": item.get("answer", item.get("ground_truth", "")),
                "relevant_docs": item.get("relevant_docs", []),
            })
    
    # Run evaluation
    evaluator = RAGEvaluator(chat_engine)
    results = evaluator.evaluate(test_cases, output_path="data/evaluation_results.json")
    
    # Generate report
    report = evaluator.generate_report()
    print(report)
    
    return results


def main():
    """Main function"""
    logger.info("Medical RAG System starting")
    
    # Build index
    faiss_index, texts, metadata = build_index()
    
    # Create chat engine
    chat_engine = create_chat_engine(faiss_index, texts, metadata)
    
    # Interactive mode
    logger.info("Entering interactive mode. Type 'quit' to exit.")
    print("\n" + "="*60)
    print("Medical RAG System - Interactive Mode")
    print("="*60)
    print("Type your medical questions below. Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Get response
            result = chat_engine.chat(query)
            
            print(f"\nAssistant: {result['response']}\n")
            
            # Show confidence
            confidence = result['confidence']
            print(f"Confidence: {confidence['confidence']:.2%} - {confidence['reason']}")
            
            # Show citations
            if result['citations']:
                print(f"\nCitations ({len(result['citations'])} sources):")
                for citation in result['citations'][:3]:  # Show top 3
                    print(f"  [{citation['id']}] {citation['source']} (score: {citation['score']})")
            
            print("-" * 60 + "\n")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
