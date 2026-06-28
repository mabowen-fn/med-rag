"""RAGAS-based evaluation framework for RAG systems"""
from typing import List, Dict, Any
from loguru import logger
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)


class RAGASEvaluator:
    """
    RAGAS-based evaluator for comprehensive RAG assessment.
    
    Metrics:
    - Faithfulness: Is the answer supported by retrieved context?
    - Answer Relevancy: Is the answer relevant to the question?
    - Context Precision: Are relevant documents ranked higher?
    - Context Recall: Are all relevant documents retrieved?
    - Answer Correctness: Is the answer factually correct?
    """
    
    def __init__(self):
        logger.info("Initializing RAGAS evaluator...")
        self.metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness
        }
    
    def evaluate_responses(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate RAG responses using RAGAS metrics.
        
        Args:
            questions: List of user questions
            answers: List of generated answers
            contexts: List of context lists (each context is a list of retrieved documents)
            ground_truths: Optional list of ground truth answers
        
        Returns:
            Dictionary of metric scores
        """
        logger.info(f"Evaluating {len(questions)} responses with RAGAS...")
        
        # Prepare dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        # Run evaluation
        logger.info("Running RAGAS evaluation...")
        results = evaluate(
            dataset=dataset,
            metrics=list(self.metrics.values())
        )
        
        # Convert to dictionary
        scores = {
            "faithfulness": results["faithfulness"],
            "answer_relevancy": results["answer_relevancy"],
            "context_precision": results["context_precision"],
            "context_recall": results["context_recall"],
            "answer_correctness": results["answer_correctness"] if ground_truths else None
        }
        
        logger.info("RAGAS evaluation complete:")
        for metric, score in scores.items():
            if score is not None:
                logger.info(f"  {metric}: {score:.4f}")
        
        return scores
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, float]:
        """
        Evaluate a single response.
        
        Args:
            question: User question
            answer: Generated answer
            contexts: List of retrieved documents
            ground_truth: Optional ground truth answer
        
        Returns:
            Dictionary of metric scores
        """
        return self.evaluate_responses(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None
        )
    
    def compare_methods(
        self,
        questions: List[str],
        baseline_answers: List[str],
        sota_answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare baseline vs SOTA methods using RAGAS.
        
        Args:
            questions: List of user questions
            baseline_answers: Answers from baseline RAG
            sota_answers: Answers from SOTA RAG
            contexts: List of context lists
            ground_truths: Optional ground truth answers
        
        Returns:
            Dictionary with scores for each method
        """
        logger.info("Comparing baseline vs SOTA with RAGAS...")
        
        # Evaluate baseline
        logger.info("Evaluating baseline...")
        baseline_scores = self.evaluate_responses(
            questions=questions,
            answers=baseline_answers,
            contexts=contexts,
            ground_truths=ground_truths
        )
        
        # Evaluate SOTA
        logger.info("Evaluating SOTA...")
        sota_scores = self.evaluate_responses(
            questions=questions,
            answers=sota_answers,
            contexts=contexts,
            ground_truths=ground_truths
        )
        
        # Calculate improvements
        improvements = {}
        for metric in baseline_scores:
            if baseline_scores[metric] is not None and sota_scores[metric] is not None:
                baseline_val = baseline_scores[metric]
                sota_val = sota_scores[metric]
                improvement = ((sota_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                improvements[metric] = {
                    "baseline": baseline_val,
                    "sota": sota_val,
                    "improvement_pct": improvement
                }
        
        return {
            "baseline": baseline_scores,
            "sota": sota_scores,
            "improvements": improvements
        }


def generate_ragas_report(
    comparison_results: Dict[str, Any],
    output_path: str = "data/ragas_evaluation_report.md"
):
    """
    Generate a markdown report from RAGAS comparison results.
    
    Args:
        comparison_results: Results from compare_methods
        output_path: Path to save the report
    """
    logger.info(f"Generating RAGAS report at {output_path}...")
    
    baseline = comparison_results["baseline"]
    sota = comparison_results["sota"]
    improvements = comparison_results["improvements"]
    
    report = []
    report.append("# RAGAS Evaluation Report")
    report.append("")
    report.append("## Overview")
    report.append("")
    report.append("This report compares baseline RAG vs SOTA RAG using RAGAS metrics:")
    report.append("")
    report.append("- **Faithfulness**: Is the answer supported by retrieved context?")
    report.append("- **Answer Relevancy**: Is the answer relevant to the question?")
    report.append("- **Context Precision**: Are relevant documents ranked higher?")
    report.append("- **Context Recall**: Are all relevant documents retrieved?")
    report.append("- **Answer Correctness**: Is the answer factually correct?")
    report.append("")
    
    report.append("## Results Summary")
    report.append("")
    report.append("| Metric | Baseline | SOTA | Improvement |")
    report.append("|--------|----------|------|-------------|")
    
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]:
        if metric in improvements:
            baseline_val = improvements[metric]["baseline"]
            sota_val = improvements[metric]["sota"]
            improvement = improvements[metric]["improvement_pct"]
            report.append(f"| {metric.replace('_', ' ').title()} | {baseline_val:.4f} | {sota_val:.4f} | {improvement:+.2f}% |")
    
    report.append("")
    report.append("## Detailed Analysis")
    report.append("")
    
    # Find best and worst improvements
    best_metric = max(improvements.items(), key=lambda x: x[1]["improvement_pct"])
    worst_metric = min(improvements.items(), key=lambda x: x[1]["improvement_pct"])
    
    report.append(f"### Best Improvement")
    report.append(f"- **{best_metric[0].replace('_', ' ').title()}**: {best_metric[1]['improvement_pct']:+.2f}%")
    report.append(f"  - Baseline: {best_metric[1]['baseline']:.4f}")
    report.append(f"  - SOTA: {best_metric[1]['sota']:.4f}")
    report.append("")
    
    report.append(f"### Areas for Improvement")
    report.append(f"- **{worst_metric[0].replace('_', ' ').title()}**: {worst_metric[1]['improvement_pct']:+.2f}%")
    report.append(f"  - Baseline: {worst_metric[1]['baseline']:.4f}")
    report.append(f"  - SOTA: {worst_metric[1]['sota']:.4f}")
    report.append("")
    
    report.append("## Key Insights")
    report.append("")
    
    # Generate insights based on scores
    if sota.get("faithfulness", 0) > 0.8:
        report.append("✅ **High Faithfulness**: SOTA pipeline effectively grounds answers in retrieved context.")
    elif sota.get("faithfulness", 0) > 0.6:
        report.append("⚠️ **Moderate Faithfulness**: SOTA pipeline shows improvement but could better ground answers.")
    else:
        report.append("❌ **Low Faithfulness**: SOTA pipeline struggles to ground answers in context.")
    
    if sota.get("answer_relevancy", 0) > 0.8:
        report.append("✅ **High Answer Relevancy**: SOTA answers are highly relevant to questions.")
    elif sota.get("answer_relevancy", 0) > 0.6:
        report.append("⚠️ **Moderate Answer Relevancy**: SOTA answers are somewhat relevant.")
    else:
        report.append("❌ **Low Answer Relevancy**: SOTA answers may not address the questions well.")
    
    if sota.get("context_precision", 0) > 0.8:
        report.append("✅ **High Context Precision**: Relevant documents are ranked highly.")
    elif sota.get("context_precision", 0) > 0.6:
        report.append("⚠️ **Moderate Context Precision**: Document ranking could be improved.")
    else:
        report.append("❌ **Low Context Precision**: Relevant documents are not ranked highly enough.")
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("*Report generated by RAGAS evaluation framework*")
    
    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    logger.info(f"RAGAS report saved to {output_path}")
