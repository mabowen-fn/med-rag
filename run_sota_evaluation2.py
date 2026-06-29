"""
State-of-the-Art Evaluation Script
Compares baseline RAG with SOTA RAG pipeline
Based on 2025-2026 research papers
"""
import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from main import build_index
from src.pipeline import SOTAMedicalChatEngine, MedicalChatEngine
from src.evaluation.evaluator import RAGEvaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run SOTA evaluation')
    parser.add_argument('--samples', type=int, default=10, 
                       help='Number of samples to evaluate (default: 10, use 0 for all)')
    return parser.parse_args()


def run_sota_evaluation():
    """Run comprehensive SOTA evaluation"""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("STATE-OF-THE-ART MEDICAL RAG EVALUATION")
    logger.info("=" * 80)
    logger.info("Comparing Baseline RAG vs SOTA RAG Pipeline")
    logger.info("SOTA Features: Agentic Graph RAG, Multi-Source Retrieval,")
    logger.info("               Discrepancy Refinement, Citation Verification")
    logger.info("=" * 80)
    
    # Check if index exists
    index_path = Path("data/index/faiss.index")
    if not index_path.exists():
        logger.warning("FAISS index not found. Building index...")
        build_index()
    
    # Load evaluation dataset
    logger.info("\nLoading evaluation dataset...")
    expanded_path = Path("data/medical_qa_dataset_expanded.json")
    if expanded_path.exists():
        logger.info("Using expanded dataset (197 QA pairs)")
        dataset_path = str(expanded_path)
    else:
        logger.info("Using original dataset (58 QA pairs)")
        dataset_path = "data/medical_qa_dataset.json"
    
    # Load QA pairs
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    qa_pairs = data['qa_pairs']
    
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Create baseline chat engine
    logger.info("\n" + "=" * 80)
    logger.info("Initializing Baseline RAG Engine...")
    logger.info("=" * 80)
    from main import create_chat_engine
    baseline_engine = create_chat_engine()
    
    # Create SOTA chat engine
    logger.info("\n" + "=" * 80)
    logger.info("Initializing SOTA RAG Engine...")
    logger.info("=" * 80)
    sota_engine = create_sota_chat_engine()
    
    # Run evaluation
    logger.info("\n" + "=" * 80)
    logger.info("Starting Evaluation...")
    logger.info("=" * 80)
    
    # Determine number of samples
    max_samples = None if args.samples == 0 else min(args.samples, len(qa_pairs))
    logger.info(f"Evaluating on {max_samples} samples...")
    
    baseline_evaluator = RAGEvaluator(baseline_engine)
    sota_evaluator = RAGEvaluator(sota_engine)
    
    # Baseline evaluation
    logger.info("\n--- Baseline RAG Evaluation ---")
    baseline_results = baseline_evaluator.evaluate_method(qa_pairs, 'hybrid_rag', max_samples)
    
    # SOTA evaluation
    logger.info("\n--- SOTA RAG Evaluation ---")
    sota_results = sota_evaluator.evaluate_method(qa_pairs, 'hybrid_rag', max_samples)
    
    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    
    comparison = compare_results(baseline_results, sota_results)
    
    # Save detailed report
    report_path = Path("data/sota_evaluation_report.json")
    report = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": max_samples,
        "baseline_results": baseline_results,
        "sota_results": sota_results,
        "comparison": comparison
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nDetailed report saved to: {report_path}")
    
    # Print summary
    print_summary(comparison)
    
    return report


def create_sota_chat_engine():
    """Create SOTA chat engine with all advanced features"""
    from main import build_index
    from src.knowledge import EmbeddingModel, FaissIndexBuilder
    from src.pipeline import (
        HybridRetriever,
        Reranker,
        ConfidenceScorer,
        CitationTracker,
    )
    
    logger.info("Creating SOTA chat engine...")
    
    # Load models
    embedding_model = EmbeddingModel()
    faiss_index = FaissIndexBuilder(dimension=embedding_model.dimension)
    faiss_index.load()
    
    # Create retriever
    retriever = HybridRetriever(faiss_index, embedding_model)
    
    # Create reranker
    reranker = Reranker()
    
    # Create confidence scorer
    confidence_scorer = ConfidenceScorer()
    
    # Create citation tracker
    citation_tracker = CitationTracker()
    
    # Create SOTA chat engine with all features enabled
    sota_engine = SOTAMedicalChatEngine(
        retriever=retriever,
        reranker=reranker,
        confidence_scorer=confidence_scorer,
        citation_tracker=citation_tracker,
        enable_agentic_rag=True,
        enable_discrepancy_refinement=True,
        enable_citation_verification=True
    )
    
    logger.info("SOTA chat engine created with all advanced features")
    return sota_engine


def compare_results(baseline_results, sota_results):
    """Compare baseline and SOTA results"""
    
    # Calculate average metrics
    def calc_avg(results, metric_key):
        # response_time is at top level, others are in metrics dict
        if metric_key == 'response_time':
            values = [r['response_time'] for r in results]
        else:
            values = [r['metrics'][metric_key] for r in results]
        return sum(values) / len(values) if values else 0.0
    
    comparison = {}
    
    metrics = ['bleu_score', 'rouge_score', 'bert_score', 'hallucination_rate', 'response_time']
    
    for metric in metrics:
        baseline_avg = calc_avg(baseline_results, metric)
        sota_avg = calc_avg(sota_results, metric)
        
        # Calculate improvement (for hallucination_rate, lower is better)
        if metric == 'hallucination_rate':
            improvement = ((baseline_avg - sota_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
            comparison[metric] = {
                'baseline': baseline_avg,
                'sota': sota_avg,
                'improvement': improvement,
                'better': 'sota' if sota_avg < baseline_avg else 'baseline'
            }
        elif metric == 'response_time':
            # For response time, lower is better but we expect SOTA to be slower
            comparison[metric] = {
                'baseline': baseline_avg,
                'sota': sota_avg,
                'overhead': ((sota_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
            }
        else:
            # For other metrics, higher is better
            improvement = ((sota_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
            comparison[metric] = {
                'baseline': baseline_avg,
                'sota': sota_avg,
                'improvement': improvement,
                'better': 'sota' if sota_avg > baseline_avg else 'baseline'
            }
    
    # Count SOTA features used
    sota_features_used = set()
    for r in sota_results:
        if 'sota_features' in r:
            sota_features_used.update(r['sota_features'])
    
    comparison['sota_features'] = list(sota_features_used)
    
    return comparison


def print_summary(comparison):
    """Print evaluation summary"""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: Baseline RAG vs SOTA RAG")
    print("=" * 80)
    
    print("\n📊 Quality Metrics:")
    print("-" * 80)
    
    # BLEU Score
    bleu = comparison['bleu_score']
    print(f"BLEU Score:")
    print(f"  Baseline: {bleu['baseline']:.4f}")
    print(f"  SOTA:     {bleu['sota']:.4f}")
    print(f"  Change:   {bleu['improvement']:+.1f}% {'✅' if bleu['better'] == 'sota' else '❌'}")
    
    # ROUGE Score
    rouge = comparison['rouge_score']
    print(f"\nROUGE Score:")
    print(f"  Baseline: {rouge['baseline']:.4f}")
    print(f"  SOTA:     {rouge['sota']:.4f}")
    print(f"  Change:   {rouge['improvement']:+.1f}% {'✅' if rouge['better'] == 'sota' else '❌'}")
    
    # BERT Score
    bert = comparison['bert_score']
    print(f"\nBERT Score:")
    print(f"  Baseline: {bert['baseline']:.4f}")
    print(f"  SOTA:     {bert['sota']:.4f}")
    print(f"  Change:   {bert['improvement']:+.1f}% {'✅' if bert['better'] == 'sota' else '❌'}")
    
    print("\n🎯 Critical Metric:")
    print("-" * 80)
    
    # Hallucination Rate
    halluc = comparison['hallucination_rate']
    print(f"Hallucination Rate (lower is better):")
    print(f"  Baseline: {halluc['baseline']*100:.1f}%")
    print(f"  SOTA:     {halluc['sota']*100:.1f}%")
    print(f"  Reduction: {halluc['improvement']:.1f}% {'✅✅✅' if halluc['better'] == 'sota' else '❌'}")
    
    print("\n⚡ Performance:")
    print("-" * 80)
    
    # Response Time
    time = comparison['response_time']
    print(f"Response Time:")
    print(f"  Baseline: {time['baseline']:.2f}s")
    print(f"  SOTA:     {time['sota']:.2f}s")
    print(f"  Overhead: {time['overhead']:.1f}% (expected due to advanced features)")
    
    print("\n🚀 SOTA Features Activated:")
    print("-" * 80)
    features = comparison.get('sota_features', [])
    if features:
        for feature in features:
            print(f"  ✅ {feature}")
    else:
        print("  ⚠️  No SOTA features detected")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Calculate overall improvement
    quality_improvements = [
        comparison['bleu_score']['improvement'],
        comparison['rouge_score']['improvement'],
        comparison['bert_score']['improvement'],
        comparison['hallucination_rate']['improvement']  # This is already positive if reduced
    ]
    
    avg_improvement = sum(quality_improvements) / len(quality_improvements)
    
    print(f"\nAverage Quality Improvement: {avg_improvement:+.1f}%")
    
    if avg_improvement > 10:
        print("\n🎉 EXCELLENT: SOTA pipeline significantly outperforms baseline!")
    elif avg_improvement > 0:
        print("\n✅ GOOD: SOTA pipeline shows improvements over baseline.")
    else:
        print("\n⚠️  SOTA pipeline shows mixed results. Further tuning may be needed.")
    
    print("\nKey Takeaway:")
    print("The SOTA pipeline prioritizes accuracy and reliability over speed.")
    print("The hallucination rate reduction is the most critical improvement")
    print("for medical applications where factual accuracy is paramount.")
    print("=" * 80)


if __name__ == "__main__":
    run_sota_evaluation()

