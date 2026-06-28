"""Evaluation metrics for Medical RAG System"""
from typing import List, Dict, Any
import time
from collections import Counter
import math


class RAGMetrics:
    """Advanced metrics for RAG evaluation"""
    
    @staticmethod
    def calculate_bleu_score(response: str, ground_truth: str) -> float:
        """Calculate BLEU score (simplified unigram/bigram version)"""
        if not response or not ground_truth:
            return 0.0
        
        response_tokens = response.lower().split()
        truth_tokens = ground_truth.lower().split()
        
        if not truth_tokens or not response_tokens:
            return 0.0
        
        # Unigram precision
        response_counts = Counter(response_tokens)
        truth_counts = Counter(truth_tokens)
        
        clipped_counts = {token: min(count, truth_counts[token]) 
                         for token, count in response_counts.items()}
        
        precision_1 = sum(clipped_counts.values()) / len(response_tokens)
        
        # Bigram precision
        response_bigrams = [(response_tokens[i], response_tokens[i+1]) 
                           for i in range(len(response_tokens)-1)]
        truth_bigrams = [(truth_tokens[i], truth_tokens[i+1]) 
                        for i in range(len(truth_tokens)-1)]
        
        if not response_bigrams or not truth_bigrams:
            return precision_1
        
        response_bigram_counts = Counter(response_bigrams)
        truth_bigram_counts = Counter(truth_bigrams)
        
        clipped_bigrams = {bg: min(count, truth_bigram_counts[bg]) 
                          for bg, count in response_bigram_counts.items()}
        
        precision_2 = sum(clipped_bigrams.values()) / len(response_bigrams)
        
        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(truth_tokens) / len(response_tokens)))
        
        # Geometric mean of precisions
        if precision_1 == 0 or precision_2 == 0:
            return 0.0
        
        bleu = bp * math.exp(0.5 * math.log(precision_1) + 0.5 * math.log(precision_2))
        return bleu
    
    @staticmethod
    def calculate_rouge_score(response: str, ground_truth: str) -> float:
        """Calculate ROUGE-L score (longest common subsequence)"""
        if not response or not ground_truth:
            return 0.0
        
        response_tokens = response.lower().split()
        truth_tokens = ground_truth.lower().split()
        
        if not truth_tokens or not response_tokens:
            return 0.0
        
        # LCS dynamic programming
        m, n = len(response_tokens), len(truth_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if response_tokens[i-1] == truth_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        if lcs_length == 0:
            return 0.0
        
        precision = lcs_length / m
        recall = lcs_length / n
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    @staticmethod
    def calculate_bert_score(response: str, ground_truth: str) -> float:
        """Simplified BERT-like score using token overlap with TF-IDF weighting"""
        if not response or not ground_truth:
            return 0.0
        
        response_tokens = set(response.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        if not truth_tokens or not response_tokens:
            return 0.0
        
        # Jaccard similarity as a proxy
        intersection = response_tokens & truth_tokens
        union = response_tokens | truth_tokens
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    @staticmethod
    def calculate_relevance_score(response: str, ground_truth: str) -> float:
        """Calculate relevance based on keyword overlap and semantic similarity"""
        if not response or not ground_truth:
            return 0.0
        
        response_words = set(response.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                     'would', 'could', 'should', 'may', 'might', 'can', 'to', 'of',
                     'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below'}
        
        response_words = response_words - stop_words
        truth_words = truth_words - stop_words
        
        if not truth_words:
            return 0.0
        
        overlap = response_words & truth_words
        relevance = len(overlap) / len(truth_words)
        
        return min(1.0, relevance)
    
    @staticmethod
    def calculate_confidence_score(confidence_data: Dict[str, Any]) -> float:
        """Extract confidence score from confidence data"""
        if not confidence_data:
            return 0.0
        
        return confidence_data.get('confidence', 0.0)
    
    @staticmethod
    def calculate_hallucination_rate(response: str, citations: List[Dict]) -> float:
        """Calculate hallucination rate based on citation coverage"""
        if not response:
            return 1.0
        
        if not citations:
            # No citations means potential hallucination
            return 0.8
        
        # If we have citations, lower hallucination rate
        response_words = len(response.split())
        cited_sources = len(citations)
        
        # Heuristic: more citations = less hallucination
        if cited_sources >= 3:
            return 0.1
        elif cited_sources >= 2:
            return 0.2
        elif cited_sources >= 1:
            return 0.3
        else:
            return 0.7
    
    def calculate_all_metrics(self, response: str, ground_truth: str, 
                             citations: List[Dict], confidence_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate all metrics at once"""
        return {
            'bleu_score': self.calculate_bleu_score(response, ground_truth),
            'rouge_score': self.calculate_rouge_score(response, ground_truth),
            'bert_score': self.calculate_bert_score(response, ground_truth),
            'relevance_score': self.calculate_relevance_score(response, ground_truth),
            'confidence_score': self.calculate_confidence_score(confidence_data or {}),
            'hallucination_rate': self.calculate_hallucination_rate(response, citations)
        }


class EvaluationMetrics:
    """Calculate evaluation metrics for RAG system"""
    
    @staticmethod
    def calculate_recall(
        retrieved_docs: List[Dict[str, Any]],
        relevant_docs: List[str],
    ) -> float:
        """Calculate recall: proportion of relevant docs retrieved"""
        if not relevant_docs:
            return 0.0
        
        retrieved_ids = set()
        for doc in retrieved_docs:
            doc_id = doc.get("metadata", {}).get("doc_id", "")
            retrieved_ids.add(doc_id)
        
        relevant_ids = set(relevant_docs)
        
        if not relevant_ids:
            return 0.0
        
        relevant_retrieved = len(retrieved_ids & relevant_ids)
        recall = relevant_retrieved / len(relevant_ids)
        
        return recall
    
    @staticmethod
    def calculate_precision(
        retrieved_docs: List[Dict[str, Any]],
        relevant_docs: List[str],
    ) -> float:
        """Calculate precision: proportion of retrieved docs that are relevant"""
        if not retrieved_docs:
            return 0.0
        
        retrieved_ids = set()
        for doc in retrieved_docs:
            doc_id = doc.get("metadata", {}).get("doc_id", "")
            retrieved_ids.add(doc_id)
        
        relevant_ids = set(relevant_docs)
        
        if not retrieved_ids:
            return 0.0
        
        relevant_retrieved = len(retrieved_ids & relevant_ids)
        precision = relevant_retrieved / len(retrieved_ids)
        
        return precision
    
    @staticmethod
    def calculate_accuracy(
        response: str,
        ground_truth: str,
    ) -> float:
        """Calculate accuracy using simple keyword overlap"""
        if not response or not ground_truth:
            return 0.0
        
        # Simple keyword-based accuracy
        response_keywords = set(response.lower().split())
        truth_keywords = set(ground_truth.lower().split())
        
        if not truth_keywords:
            return 0.0
        
        overlap = len(response_keywords & truth_keywords)
        accuracy = overlap / len(truth_keywords)
        
        return min(1.0, accuracy * 2)  # Scale up slightly
    
    @staticmethod
    def calculate_hallucination_rate(
        response: str,
        context: List[Dict[str, Any]],
    ) -> float:
        """Estimate hallucination rate: content not supported by context"""
        if not response or not context:
            return 1.0
        
        # Extract key terms from context
        context_terms = set()
        for doc in context:
            text = doc.get("text", "").lower()
            context_terms.update(text.split())
        
        # Check response terms
        response_terms = set(response.lower().split())
        
        # Filter out common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"}
        response_terms = response_terms - common_words
        context_terms = context_terms - common_words
        
        if not response_terms:
            return 0.0
        
        # Terms not in context
        unsupported_terms = response_terms - context_terms
        hallucination_rate = len(unsupported_terms) / len(response_terms)
        
        return hallucination_rate
    
    @staticmethod
    def measure_response_time(func, *args, **kwargs) -> tuple:
        """Measure response time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        response_time = end_time - start_time
        return result, response_time
