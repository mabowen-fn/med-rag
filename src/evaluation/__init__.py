"""Evaluation module for Medical RAG System"""
from .metrics import EvaluationMetrics
from .evaluator import RAGEvaluator

__all__ = ["EvaluationMetrics", "RAGEvaluator"]


def get_ragas_evaluator():
    """Lazy import for RAGAS evaluator (only needed for RAGAS evaluation)"""
    from .ragas_evaluator import RAGASEvaluator, generate_ragas_report
    return RAGASEvaluator, generate_ragas_report
