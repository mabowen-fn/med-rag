# Evaluation Results Summary

## Basic Evaluation: No RAG vs Naive RAG vs Hybrid RAG

| Method | BLEU | ROUGE | BERTScore | Hallucination Rate | Response Time |
|--------|------|-------|-----------|-------------------|---------------|
| no_rag | 0.0175 | 0.0436 | 0.0376 | 80.0% | 5.26s |
| naive_rag | 0.0122 | 0.0320 | 0.0294 | 30.7% | 8.28s |
| hybrid_rag | 0.0111 | 0.0316 | 0.0280 | 30.0% | 7.08s |

## SOTA Comparison: Baseline vs SOTA

| Metric | Baseline | SOTA | Improvement |
|--------|----------|------|-------------|
| bleu_score | 0.0387 | 0.0079 | -79.6% |
| rouge_score | 0.0941 | 0.0401 | -57.4% |
| bert_score | 0.0867 | 0.0305 | -64.8% |
| hallucination_rate | 30.0% | 15.8% | 47.3% reduction |

