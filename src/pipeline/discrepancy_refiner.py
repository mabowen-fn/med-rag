"""
Discrepancy-Aware Answer Refinement Module
Based on: MEGA-RAG (Frontiers in Public Health, 2025)
Key Innovation: Detects and resolves discrepancies between retrieved evidence
Achieved: 40%+ hallucination reduction through evidence verification
"""
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import json


class DiscrepancyRefiner:
    """
    Discrepancy-aware answer refinement module.
    
    Process:
    1. Detect discrepancies between retrieved documents
    2. Verify evidence consistency
    3. Refine answer based on verified evidence
    4. Flag unresolved discrepancies
    """
    
    def __init__(self, llm):
        self.llm = llm
        logger.info("Discrepancy refiner initialized")
    
    def refine_answer(
        self,
        query: str,
        initial_answer: str,
        retrieved_docs: List[Dict],
        citations: List[Dict]
    ) -> Dict[str, Any]:
        """
        Refine answer by detecting and resolving discrepancies.
        
        Args:
            query: User question
            initial_answer: Initial LLM-generated answer
            retrieved_docs: Retrieved documents
            citations: Citation information
        
        Returns:
            Refined answer with discrepancy analysis
        """
        logger.info("Starting discrepancy-aware refinement...")
        
        # Step 1: Detect discrepancies in retrieved evidence
        discrepancies = self._detect_discrepancies(query, retrieved_docs)
        logger.info(f"Detected {len(discrepancies)} discrepancies")
        
        # Step 2: Verify evidence consistency
        verification = self._verify_evidence(query, initial_answer, retrieved_docs)
        logger.info(f"Evidence verification score: {verification['consistency_score']:.2f}")
        
        # Step 3: Refine answer based on verified evidence
        if discrepancies or verification['consistency_score'] < 0.7:
            logger.info("Discrepancies detected, refining answer...")
            refined_answer = self._refine_with_discrepancies(
                query, initial_answer, retrieved_docs, discrepancies, verification
            )
        else:
            logger.info("No significant discrepancies, keeping initial answer")
            refined_answer = initial_answer
        
        # Step 4: Generate discrepancy report
        discrepancy_report = {
            "discrepancies_detected": len(discrepancies),
            "discrepancies": discrepancies,
            "consistency_score": verification['consistency_score'],
            "evidence_coverage": verification['evidence_coverage'],
            "refined": len(discrepancies) > 0 or verification['consistency_score'] < 0.7
        }
        
        return {
            "answer": refined_answer,
            "discrepancy_report": discrepancy_report,
            "citations": citations
        }
    
    def _detect_discrepancies(
        self,
        query: str,
        documents: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Detect discrepancies between retrieved documents.
        
        Discrepancies include:
        - Contradictory information
        - Inconsistent facts
        - Conflicting recommendations
        """
        if len(documents) < 2:
            return []
        
        # Build context from top documents
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):
            context_parts.append(f"[{i}] {doc.get('text', '')}")
        context_text = "\n\n".join(context_parts)
        
        # Use LLM to detect discrepancies
        detect_prompt = f"""Analyze the following medical documents for discrepancies.

Question: {query}

Documents:
{context_text}

Identify any discrepancies, contradictions, or inconsistencies between the documents.
Consider:
- Conflicting medical facts
- Contradictory treatment recommendations
- Inconsistent statistics or data
- Opposing conclusions

For each discrepancy found, provide:
- Type: contradiction | inconsistency | conflict
- Description: What conflicts
- Documents involved: Which documents [1], [2], etc.
- Severity: high | medium | low

Provide results in JSON format:
{{
  "discrepancies": [
    {{
      "type": "contradiction",
      "description": "Document [1] states X, but Document [2] states Y",
      "documents_involved": [1, 2],
      "severity": "high"
    }}
  ]
}}

If no discrepancies found, return: {{"discrepancies": []}}"""
        
        try:
            response = self.llm.complete(detect_prompt)
            response_text = str(response)
            
            # Parse JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                result = json.loads(json_str)
                return result.get("discrepancies", [])
            else:
                return []
        except Exception as e:
            logger.error(f"Discrepancy detection failed: {e}")
            return []
    
    def _verify_evidence(
        self,
        query: str,
        answer: str,
        documents: List[Dict]
    ) -> Dict[str, Any]:
        """
        Verify that the answer is supported by retrieved evidence.
        
        Checks:
        - Evidence coverage: What % of answer claims are supported?
        - Consistency: Are claims consistent with evidence?
        - Hallucination risk: Are there unsupported claims?
        """
        # Build context
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):
            context_parts.append(f"[{i}] {doc.get('text', '')}")
        context_text = "\n\n".join(context_parts)
        
        verify_prompt = f"""Verify if the answer is supported by the retrieved evidence.

Question: {query}

Retrieved Evidence:
{context_text}

Generated Answer:
{answer}

Evaluate:
1. Evidence Coverage: What percentage of claims in the answer are supported by the evidence? (0-100%)
2. Consistency: Are the claims consistent with the evidence? (score 0-10)
3. Hallucination Risk: Are there claims not supported by evidence? (score 0-10, higher = more risk)

Provide results in JSON format:
{{
  "evidence_coverage": <0-100>,
  "consistency_score": <0-10>,
  "hallucination_risk": <0-10>,
  "unsupported_claims": ["list of claims not supported by evidence"],
  "supported_claims": ["list of claims supported by evidence"]
}}"""
        
        try:
            response = self.llm.complete(verify_prompt)
            response_text = str(response)
            
            # Parse JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                result = json.loads(json_str)
                
                # Normalize scores
                result["evidence_coverage"] = result.get("evidence_coverage", 50) / 100.0
                result["consistency_score"] = result.get("consistency_score", 5) / 10.0
                result["hallucination_risk"] = result.get("hallucination_risk", 5) / 10.0
                
                return result
            else:
                return {
                    "evidence_coverage": 0.5,
                    "consistency_score": 0.5,
                    "hallucination_risk": 0.5,
                    "unsupported_claims": [],
                    "supported_claims": []
                }
        except Exception as e:
            logger.error(f"Evidence verification failed: {e}")
            return {
                "evidence_coverage": 0.5,
                "consistency_score": 0.5,
                "hallucination_risk": 0.5,
                "unsupported_claims": [],
                "supported_claims": []
            }
    
    def _refine_with_discrepancies(
        self,
        query: str,
        initial_answer: str,
        documents: List[Dict],
        discrepancies: List[Dict],
        verification: Dict
    ) -> str:
        """
        Refine answer by addressing detected discrepancies.
        
        Strategy:
        1. Acknowledge discrepancies
        2. Prefer higher-quality evidence
        3. Flag uncertain claims
        4. Add appropriate caveats
        """
        # Build context
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):
            context_parts.append(f"[{i}] {doc.get('text', '')}")
        context_text = "\n\n".join(context_parts)
        
        # Format discrepancies
        discrepancy_text = ""
        if discrepancies:
            discrepancy_text = "\n\nDetected Discrepancies:\n"
            for i, disc in enumerate(discrepancies, 1):
                discrepancy_text += f"{i}. [{disc.get('severity', 'medium').upper()}] {disc.get('description', 'Unknown')}\n"
        
        # Format unsupported claims
        unsupported_text = ""
        if verification.get("unsupported_claims"):
            unsupported_text = "\n\nUnsupported Claims in Initial Answer:\n"
            for claim in verification["unsupported_claims"]:
                unsupported_text += f"- {claim}\n"
        
        refine_prompt = f"""You are a medical AI assistant. Refine the answer to address discrepancies and ensure factual accuracy.

Question: {query}

Retrieved Evidence:
{context_text}
{discrepancy_text}
{unsupported_text}

Initial Answer:
{initial_answer}

Refinement Guidelines:
1. Address all detected discrepancies by acknowledging conflicting information
2. Remove or flag unsupported claims
3. Prefer information from higher-quality sources (if identifiable)
4. Add appropriate caveats for uncertain or conflicting information
5. Maintain citation markers [1], [2], etc.
6. If evidence is insufficient, explicitly state limitations
7. Do NOT fabricate information to fill gaps

Provide the refined answer:"""
        
        try:
            response = self.llm.complete(refine_prompt)
            refined_answer = str(response).strip()
            return refined_answer
        except Exception as e:
            logger.error(f"Answer refinement failed: {e}")
            return initial_answer
