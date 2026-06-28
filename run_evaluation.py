"""Standalone evaluation script for medical RAG system"""
import os
import sys
from pathlib import Path

# Set HuggingFace to offline mode to use cached models
os.environ['HF_HUB_OFFLINE'] = '1'

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.chat_engine import MedicalChatEngine
from src.evaluation.evaluator import RAGEvaluator
from main import build_index, create_chat_engine


def main():
    """Run comprehensive evaluation"""
    logger.info("=" * 80)
    logger.info("MEDICAL RAG SYSTEM - COMPREHENSIVE EVALUATION")
    logger.info("=" * 80)
    
    # Check if index exists
    index_path = Path("data/index/faiss.index")
    if not index_path.exists():
        logger.warning("FAISS index not found. Building index...")
        build_index()
    
    # Create chat engine
    logger.info("\nInitializing chat engine...")
    chat_engine = create_chat_engine()
    
    # Create evaluator
    evaluator = RAGEvaluator(chat_engine)
    
    # Load evaluation dataset
    logger.info("\nLoading evaluation dataset...")
    qa_pairs = evaluator.load_evaluation_dataset("data/medical_qa_dataset.json")
    
    # Run comparison (use 5 samples for quick demo, increase for full evaluation)
    logger.info("\nStarting method comparison...")
    logger.info("This will evaluate 3 methods on 5 samples each.")
    logger.info("Total: 15 LLM calls (may take several minutes)\n")
    
    comparison_results = evaluator.compare_methods(qa_pairs, max_samples=5)
    
    # Generate report
    logger.info("\nGenerating evaluation report...")
    report = evaluator.generate_report(
        comparison_results,
        output_path="data/evaluation_report.json"
    )
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    
    summary = report['summary']
    logger.info("\nPerformance Summary:")
    logger.info(f"{'Method':<15} {'BLEU':<10} {'ROUGE':<10} {'BERT':<10} {'Time(s)':<10} {'Halluc%':<10}")
    logger.info("-" * 65)
    
    for method in ['no_rag', 'naive_rag', 'hybrid_rag']:
        s = summary[method]
        logger.info(
            f"{method:<15} {s['avg_bleu']:<10.4f} {s['avg_rouge']:<10.4f} "
            f"{s['avg_bert_score']:<10.4f} {s['avg_response_time']:<10.2f} "
            f"{s['hallucination_rate']*100:<10.2f}"
        )
    
    logger.info("\nReports saved to:")
    logger.info("  - data/evaluation_report.json")
    logger.info("  - data/evaluation_report.md")
    
    # Calculate key improvements
    hybrid = summary['hybrid_rag']
    no_rag = summary['no_rag']
    
    logger.info("\nKey Improvements (Hybrid RAG vs No RAG):")
    bleu_improvement = (hybrid['avg_bleu'] - no_rag['avg_bleu']) / no_rag['avg_bleu'] * 100
    rouge_improvement = (hybrid['avg_rouge'] - no_rag['avg_rouge']) / no_rag['avg_rouge'] * 100
    bert_improvement = (hybrid['avg_bert_score'] - no_rag['avg_bert_score']) / no_rag['avg_bert_score'] * 100
    hallucination_reduction = (no_rag['hallucination_rate'] - hybrid['hallucination_rate']) / no_rag['hallucination_rate'] * 100
    
    logger.info(f"  BLEU Score:      {bleu_improvement:+.1f}%")
    logger.info(f"  ROUGE Score:     {rouge_improvement:+.1f}%")
    logger.info(f"  BERT Score:      {bert_improvement:+.1f}%")
    logger.info(f"  Hallucination:   {hallucination_reduction:+.1f}% reduction")
    
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
