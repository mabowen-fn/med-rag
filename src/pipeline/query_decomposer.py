"""Query decomposition for complex medical questions"""
from typing import List, Dict, Any
from loguru import logger
import re


class QueryDecomposer:
    """
    Decompose complex medical questions into simpler sub-queries.
    
    This improves retrieval for multi-part questions like:
    "What are the symptoms, causes, and treatments for diabetes?"
    
    Becomes:
    - "What are the symptoms of diabetes?"
    - "What causes diabetes?"
    - "How is diabetes treated?"
    """
    
    def __init__(self, llm):
        self.llm = llm
        logger.info("Query decomposer initialized")
    
    def decompose(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-queries.
        
        Args:
            query: Original complex query
        
        Returns:
            List of simpler sub-queries
        """
        # Check if query is complex
        if not self._is_complex(query):
            return [query]
        
        logger.info(f"Decomposing complex query: {query[:100]}...")
        
        # Use LLM to decompose
        decompose_prompt = f"""Break down this complex medical question into 2-4 simpler sub-questions that can be answered independently:

Original question: {query}

Requirements:
- Each sub-question should be self-contained and answerable on its own
- Sub-questions should cover different aspects of the original question
- Keep sub-questions focused and specific
- Use clear, simple language

Return ONLY the sub-questions, one per line, without numbering or bullets."""
        
        try:
            response = self.llm.complete(decompose_prompt)
            sub_queries = self._parse_sub_queries(str(response))
            
            # Validate sub-queries
            if len(sub_queries) < 2:
                logger.warning("Decomposition produced fewer than 2 sub-queries, using original")
                return [query]
            
            logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]
    
    def _is_complex(self, query: str) -> bool:
        """
        Determine if a query is complex enough to need decomposition.
        
        Heuristics:
        - Contains multiple question words (what, how, why, when, where)
        - Contains conjunctions (and, or, but)
        - Contains multiple medical terms
        - Longer than average (>100 chars)
        """
        query_lower = query.lower()
        
        # Count question words
        question_words = ["what", "how", "why", "when", "where", "which"]
        question_count = sum(1 for word in question_words if word in query_lower)
        
        # Count conjunctions
        conjunctions = [" and ", " or ", " but ", " as well as ", " in addition "]
        conjunction_count = sum(1 for conj in conjunctions if conj in query_lower)
        
        # Check length
        is_long = len(query) > 100
        
        # Check for multiple clauses
        has_multiple_clauses = query.count(",") >= 2 or query.count("?") >= 2
        
        # Complex if meets multiple criteria
        complexity_score = (
            (question_count >= 2) +
            (conjunction_count >= 1) +
            is_long +
            has_multiple_clauses
        )
        
        return complexity_score >= 2
    
    def _parse_sub_queries(self, response: str) -> List[str]:
        """Parse sub-queries from LLM response"""
        # Split by lines
        lines = response.strip().split('\n')
        
        sub_queries = []
        for line in lines:
            line = line.strip()
            
            # Remove numbering/bullets
            line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•]\s*', '', line)
            
            # Skip empty lines
            if not line:
                continue
            
            # Add if it looks like a question
            if line.endswith('?') or len(line) > 10:
                sub_queries.append(line)
        
        return sub_queries
    
    def synthesize_answers(
        self,
        original_query: str,
        sub_queries: List[str],
        sub_answers: List[str]
    ) -> str:
        """
        Synthesize answers from sub-queries into a coherent response.
        
        Args:
            original_query: Original complex query
            sub_queries: List of sub-queries
            sub_answers: List of answers to sub-queries
        
        Returns:
            Synthesized answer
        """
        if len(sub_queries) <= 1:
            return sub_answers[0] if sub_answers else ""
        
        logger.info("Synthesizing answers from sub-queries...")
        
        # Build synthesis prompt
        qa_pairs = []
        for q, a in zip(sub_queries, sub_answers):
            qa_pairs.append(f"Q: {q}\nA: {a}")
        
        qa_text = "\n\n".join(qa_pairs)
        
        synthesis_prompt = f"""Combine these answers into a coherent response to the original question.

Original question: {original_query}

Sub-questions and answers:
{qa_text}

Requirements:
- Create a single, well-organized response
- Maintain all important information from the sub-answers
- Use clear transitions between different aspects
- Keep the response focused and avoid repetition
- Use appropriate medical terminology

Synthesized answer:"""
        
        try:
            response = self.llm.complete(synthesis_prompt)
            synthesized = str(response).strip()
            logger.info("Answer synthesis complete")
            return synthesized
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            # Fallback: concatenate answers
            return "\n\n".join(sub_answers)
