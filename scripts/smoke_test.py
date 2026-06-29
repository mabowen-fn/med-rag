#!/usr/bin/env python3
"""
Smoke test script for Medical RAG System
Tests basic functionality without running full evaluation
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
        
        import sentence_transformers
        logger.info(f"✓ sentence-transformers {sentence_transformers.__version__}")
        
        import faiss
        logger.info("✓ FAISS")
        
        import streamlit
        logger.info(f"✓ Streamlit {streamlit.__version__}")
        
        import ollama
        logger.info("✓ Ollama Python client")
        
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability"""
    logger.info("\nTesting CUDA...")
    
    import torch
    
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        logger.warning("✗ CUDA not available (will use CPU)")
        return False


def test_ollama():
    """Test Ollama connectivity"""
    logger.info("\nTesting Ollama...")
    
    try:
        import ollama
        client = ollama.Client()
        
        # List models
        models = client.list()
        logger.info(f"✓ Ollama connected, {len(models['models'])} models available")
        
        # Check for qwen3:8b
        model_names = [m['name'] for m in models['models']]
        if any('qwen3:8b' in name for name in model_names):
            logger.info("✓ Qwen3:8b model found")
            return True
        else:
            logger.warning("✗ Qwen3:8b model not found. Run: ollama pull qwen3:8b")
            return False
            
    except Exception as e:
        logger.error(f"✗ Ollama connection failed: {e}")
        logger.info("  Start Ollama with: ollama serve")
        return False


def test_embedding_model():
    """Test embedding model loading"""
    logger.info("\nTesting embedding model...")
    
    try:
        from src.knowledge.embedding_model import EmbeddingModel
        
        model = EmbeddingModel()
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        
        logger.info(f"✓ Embedding model loaded")
        logger.info(f"  Dimension: {len(embedding)}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Embedding model failed: {e}")
        return False


def test_index():
    """Test FAISS index loading"""
    logger.info("\nTesting FAISS index...")
    
    try:
        from src.knowledge.faiss_index import FaissIndex
        
        index = FaissIndex()
        
        if index.index is not None:
            logger.info(f"✓ FAISS index loaded")
            logger.info(f"  Total vectors: {index.index.ntotal}")
            return True
        else:
            logger.warning("✗ FAISS index not found. Run: python build_index.py")
            return False
            
    except Exception as e:
        logger.error(f"✗ FAISS index failed: {e}")
        return False


def test_retrieval():
    """Test basic retrieval"""
    logger.info("\nTesting retrieval...")
    
    try:
        from src.pipeline.hybrid_retriever import HybridRetriever
        from src.knowledge.embedding_model import EmbeddingModel
        from src.knowledge.faiss_index import FaissIndex
        
        # Load components
        embedding_model = EmbeddingModel()
        faiss_index = FaissIndex()
        
        if faiss_index.index is None:
            logger.warning("✗ Cannot test retrieval: index not loaded")
            return False
        
        retriever = HybridRetriever(faiss_index, embedding_model)
        
        # Test query
        query = "What are the symptoms of diabetes?"
        results = retriever.retrieve(query, top_k=3)
        
        logger.info(f"✓ Retrieval working")
        logger.info(f"  Retrieved {len(results)} documents")
        
        if results:
            logger.info(f"  Top score: {results[0]['score']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Retrieval failed: {e}")
        return False


def test_generation():
    """Test LLM generation"""
    logger.info("\nTesting LLM generation...")
    
    try:
        import ollama
        
        # Simple test prompt
        response = ollama.chat(
            model='qwen3:8b',
            messages=[{'role': 'user', 'content': 'Say "Hello" in one word.'}],
            options={'temperature': 0.1}
        )
        
        logger.info(f"✓ LLM generation working")
        logger.info(f"  Response: {response['message']['content'][:50]}...")
        return True
        
    except Exception as e:
        logger.error(f"✗ LLM generation failed: {e}")
        return False


def test_full_pipeline():
    """Test complete RAG pipeline"""
    logger.info("\nTesting full RAG pipeline...")
    
    try:
        from main import create_chat_engine
        
        engine = create_chat_engine()
        
        query = "What is hypertension?"
        start_time = time.time()
        
        result = engine.chat(query)
        
        elapsed = time.time() - start_time
        
        logger.info(f"✓ Full pipeline working")
        logger.info(f"  Response time: {elapsed:.2f}s")
        logger.info(f"  Confidence: {result.get('confidence', 'N/A')}")
        logger.info(f"  Citations: {len(result.get('citations', []))}")
        logger.info(f"  Response: {result['response'][:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests"""
    logger.info("=" * 60)
    logger.info("Medical RAG System - Smoke Test")
    logger.info("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Ollama", test_ollama),
        ("Embedding Model", test_embedding_model),
        ("FAISS Index", test_index),
        ("Retrieval", test_retrieval),
        ("Generation", test_generation),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Smoke Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        logger.warning(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
