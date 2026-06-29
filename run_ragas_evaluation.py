"""
Comprehensive evaluation using RAGAS framework
Compares baseline RAG vs SOTA RAG with industry-standard metrics
"""
import json
import sys
import argparse
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from main import create_chat_engine
from src.evaluation import get_ragas_evaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run RAGAS evaluation')
    parser.add_argument('--samples', type=int, default=10, 
                       help='Number of samples to evaluate (default: 10, use 0 for all)')
    return parser.parse_args()


def run_ragas_evaluation():
    """Run comprehensive RAGAS evaluation"""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("RAGAS EVALUATION: Baseline vs SOTA")
    logger.info("=" * 80)
    
    # Load test questions
    test_data_path = Path("data/medical_qa_dataset_expanded.json")
    if not test_data_path.exists():
        test_data_path = Path("data/medical_qa_dataset.json")
    
    logger.info(f"Loading test data from {test_data_path}...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = data['qa_pairs']
    
    # Determine number of samples
    num_samples = None if args.samples == 0 else min(args.samples, len(qa_pairs))
    logger.info(f"Evaluating on {num_samples} samples...")
    
    test_qa = qa_pairs[:num_samples]
    
    # Extract questions and ground truths
    questions = [qa['question'] for qa in test_qa]
    ground_truths = [qa['answer'] for qa in test_qa]
    
    # Initialize engines
    logger.info("\nInitializing baseline engine...")
    baseline_engine = create_chat_engine(use_sota=False)
    
    logger.info("\nInitializing SOTA engine...")
    sota_engine = create_chat_engine(use_sota=True)
    
    # Generate responses
    logger.info("\n" + "=" * 80)
    logger.info("Generating responses with baseline engine...")
    logger.info("=" * 80)
    
    baseline_answers = []
    baseline_contexts = []
    
    for i, question in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] {question[:80]}...")
        response = baseline_engine.chat(question)
        baseline_answers.append(response['response'])
        
        # Extract contexts from citations
        contexts = [cit.get('text', '') for cit in response.get('citations', [])]
        baseline_contexts.append(contexts)
    
    logger.info("\n" + "=" * 80)
    logger.info("Generating responses with SOTA engine...")
    logger.info("=" * 80)
    
    sota_answers = []
    sota_contexts = []
    
    for i, question in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] {question[:80]}...")
        response = sota_engine.chat(question)
        sota_answers.append(response['response'])
        
        # Extract contexts from citations
        contexts = [cit.get('text', '') for cit in response.get('citations', [])]
        sota_contexts.append(contexts)
    
    # Run RAGAS evaluation
    logger.info("\n" + "=" * 80)
    logger.info("Running RAGAS evaluation...")
    logger.info("=" * 80)
    
    # Get RAGAS classes via lazy import
    RAGASEvaluator, generate_ragas_report = get_ragas_evaluator()
    ragas_eval = RAGASEvaluator()
    
    comparison_results = ragas_eval.compare_methods(
        questions=questions,
        baseline_answers=baseline_answers,
        sota_answers=sota_answers,
        contexts=baseline_contexts,  # Use same contexts for fair comparison
        ground_truths=ground_truths
    )
    
    # Generate report
    logger.info("\n" + "=" * 80)
    logger.info("Generating RAGAS report...")
    logger.info("=" * 80)
    
    generate_ragas_report(
        comparison_results=comparison_results,
        output_path="data/ragas_evaluation_report.md"
    )
    
    # Save detailed results
    results_path = Path("data/ragas_evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ Results saved to {results_path}")
    logger.info(f"✅ Report saved to data/ragas_evaluation_report.md")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    
    improvements = comparison_results['improvements']
    
    logger.info("\nRAGAS Metrics Comparison:")
    logger.info(f"{'Metric':<25} {'Baseline':<12} {'SOTA':<12} {'Improvement':<12}")
    logger.info("-" * 61)
    
    for metric, values in improvements.items():
        logger.info(
            f"{metric:<25} {values['baseline']:<12.4f} {values['sota']:<12.4f} "
            f"{values['improvement_pct']:>+10.2f}%"
        )
    
    # Calculate overall improvement
    avg_improvement = sum(v['improvement_pct'] for v in improvements.values()) / len(improvements)
    logger.info("-" * 61)
    logger.info(f"{'Average Improvement':<25} {'':<12} {'':<12} {avg_improvement:>+10.2f}%")
    
    logger.info("\n" + "=" * 80)
    
    return comparison_results


if __name__ == "__main__":
    run_ragas_evaluation()

