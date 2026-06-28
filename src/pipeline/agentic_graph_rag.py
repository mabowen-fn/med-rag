"""
Agentic Graph RAG with Self-Correcting Retrieve-Evaluate-Refine Loop
Based on: Frontiers in Medicine (2025) - "A self-correcting Agentic Graph RAG for clinical decision support"
Key Innovation: Achieved 94% faithfulness, 92% recall using iterative refinement
"""
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import json


class AgenticGraphRAG:
    """
    Self-correcting Agentic Graph RAG system.
    
    Implements a "retrieve-evaluate-refine" loop where agents:
    1. Generate graph search strategies
    2. Semantically validate retrieved context
    3. Assess context completeness
    4. Iteratively optimize until reliable information is obtained
    """
    
    def __init__(self, llm, retriever, reranker, max_iterations: int = 3):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.max_iterations = max_iterations
        logger.info("Agentic Graph RAG initialized with self-correcting loop")
    
    def retrieve_with_refinement(self, query: str) -> Dict[str, Any]:
        """
        Main retrieval loop with self-correction.
        
        Process:
        1. Initial retrieval
        2. Evaluate context quality
        3. If insufficient, refine query and retrieve again
        4. Repeat until context is adequate or max iterations reached
        """
        logger.info(f"Starting agentic retrieval for: {query[:100]}...")
        
        current_query = query
        all_retrieved_docs = []
        iteration_history = []
        
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Step 1: Retrieve documents
            retrieved_docs = self.retriever.retrieve(current_query)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Step 2: Rerank if available
            if retrieved_docs and self.reranker:
                retrieved_docs = self.reranker.rerank(current_query, retrieved_docs)
            
            # Step 3: Evaluate context quality
            evaluation = self._evaluate_context(query, retrieved_docs)
            
            iteration_history.append({
                "iteration": iteration + 1,
                "query": current_query,
                "docs_retrieved": len(retrieved_docs),
                "evaluation": evaluation
            })
            
            # Step 4: Check if context is sufficient
            if evaluation["is_sufficient"]:
                logger.info(f"Context sufficient at iteration {iteration + 1}")
                all_retrieved_docs.extend(retrieved_docs)
                break
            
            # Step 5: If not sufficient, refine query
            if iteration < self.max_iterations - 1:
                logger.info(f"Context insufficient (score: {evaluation['completeness_score']:.2f}), refining query...")
                current_query = self._refine_query(query, retrieved_docs, evaluation)
                logger.info(f"Refined query: {current_query[:100]}...")
            
            all_retrieved_docs.extend(retrieved_docs)
        
        # Remove duplicates based on doc_id
        unique_docs = self._deduplicate_docs(all_retrieved_docs)
        
        return {
            "documents": unique_docs,
            "iterations": iteration_history,
            "final_query": current_query,
            "total_docs": len(unique_docs)
        }
    
    def _evaluate_context(self, query: str, documents: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate if retrieved context is sufficient to answer the query.
        
        Criteria:
        - Relevance: How relevant are the documents to the query?
        - Completeness: Do the documents cover all aspects of the query?
        - Consistency: Are there contradictions in the retrieved information?
        """
        if not documents:
            return {
                "is_sufficient": False,
                "relevance_score": 0.0,
                "completeness_score": 0.0,
                "consistency_score": 0.0,
                "reason": "No documents retrieved"
            }
        
        # Build context text
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):  # Evaluate top 5
            context_parts.append(f"[{i}] {doc.get('text', '')}")
        context_text = "\n\n".join(context_parts)
        
        # Use LLM to evaluate context
        eval_prompt = f"""Evaluate if the following context is sufficient to answer the medical question.

Question: {query}

Retrieved Context:
{context_text}

Evaluate on these criteria (score 0-10):
1. Relevance: How relevant is the context to the question?
2. Completeness: Does the context cover all aspects needed to answer?
3. Consistency: Is the information consistent and non-contradictory?

Provide your evaluation in JSON format:
{{
  "relevance_score": <0-10>,
  "completeness_score": <0-10>,
  "consistency_score": <0-10>,
  "is_sufficient": <true/false>,
  "missing_information": ["list", "of", "missing", "aspects"],
  "reason": "brief explanation"
}}"""
        
        try:
            response = self.llm.complete(eval_prompt)
            response_text = str(response)
            
            # Parse JSON from response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                evaluation = json.loads(json_str)
                
                # Normalize scores to 0-1
                evaluation["relevance_score"] = evaluation.get("relevance_score", 5) / 10.0
                evaluation["completeness_score"] = evaluation.get("completeness_score", 5) / 10.0
                evaluation["consistency_score"] = evaluation.get("consistency_score", 5) / 10.0
                
                # Determine if sufficient (threshold: 0.7)
                avg_score = (
                    evaluation["relevance_score"] +
                    evaluation["completeness_score"] +
                    evaluation["consistency_score"]
                ) / 3.0
                
                evaluation["is_sufficient"] = avg_score >= 0.7 and evaluation.get("is_sufficient", False)
                
                return evaluation
            else:
                # Fallback evaluation
                return {
                    "is_sufficient": len(documents) >= 3,
                    "relevance_score": 0.6,
                    "completeness_score": 0.5,
                    "consistency_score": 0.7,
                    "reason": "Evaluation parsing failed, using heuristic"
                }
        except Exception as e:
            logger.error(f"Context evaluation failed: {e}")
            return {
                "is_sufficient": len(documents) >= 3,
                "relevance_score": 0.6,
                "completeness_score": 0.5,
                "consistency_score": 0.7,
                "reason": f"Evaluation error: {str(e)}"
            }
    
    def _refine_query(self, original_query: str, documents: List[Dict], evaluation: Dict) -> str:
        """
        Refine the query based on evaluation feedback.
        
        Uses Medical Gap Analysis to identify missing information
        and generate a more targeted query.
        """
        missing_info = evaluation.get("missing_information", [])
        missing_text = ", ".join(missing_info) if missing_info else "more specific medical details"
        
        refine_prompt = f"""You are a medical information retrieval expert. The current search did not retrieve sufficient information.

Original Question: {original_query}

What's Missing: {missing_text}

Current Evaluation:
- Relevance: {evaluation.get('relevance_score', 0.5):.2f}
- Completeness: {evaluation.get('completeness_score', 0.5):.2f}
- Consistency: {evaluation.get('consistency_score', 0.5):.2f}

Generate a refined search query that will retrieve more relevant and complete medical information.
The refined query should:
1. Include specific medical terminology
2. Target the missing aspects
3. Be more precise than the original query

Refined Query:"""
        
        try:
            response = self.llm.complete(refine_prompt)
            refined_query = str(response).strip()
            
            # Clean up the query (remove any extra text)
            if len(refined_query) > 200:
                refined_query = refined_query[:200]
            
            return refined_query if refined_query else original_query
        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            return original_query
    
    def _deduplicate_docs(self, documents: List[Dict]) -> List[Dict]:
        """Remove duplicate documents based on doc_id"""
        seen_ids = set()
        unique_docs = []
        
        for doc in documents:
            doc_id = doc.get("metadata", {}).get("doc_id")
            if doc_id is not None and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
            elif doc_id is None:
                # If no doc_id, use text hash
                text_hash = hash(doc.get("text", ""))
                if text_hash not in seen_ids:
                    seen_ids.add(text_hash)
                    unique_docs.append(doc)
        
        return unique_docs
