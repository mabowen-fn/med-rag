"""
Citation-Aware Reasoning and Evidence Verification
Based on: MedTrust-RAG (arXiv 2025)
Key Innovation: Enforces citation-aware reasoning where all content must be grounded in retrieved documents
Achieved: 2.7% accuracy improvement on LLaMA3.1-8B, 2.4% on Qwen3-8B
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from loguru import logger
import re


class CitationVerifier:
    """
    Citation-aware reasoning and evidence verification system.
    
    Ensures:
    1. All claims are explicitly grounded in retrieved documents
    2. Citations accurately reference supporting evidence
    3. Unsupported claims are flagged or removed
    4. Citation integrity is maintained
    """
    
    def __init__(self, llm):
        self.llm = llm
        logger.info("Citation verifier initialized")
    
    def verify_and_enforce_citations(
        self,
        query: str,
        response: str,
        retrieved_docs: List[Dict]
    ) -> Dict[str, Any]:
        """
        Verify citations and enforce citation-aware reasoning.
        
        Process:
        1. Extract all claims from response
        2. Extract all citations
        3. Verify each claim is supported by its cited source
        4. Flag unsupported claims
        5. Generate verified response with proper citations
        """
        logger.info("Starting citation verification...")
        
        # Step 1: Extract claims and citations
        claims = self._extract_claims(response)
        citations = self._extract_citations(response)
        
        logger.info(f"Extracted {len(claims)} claims and {len(citations)} citations")
        
        # Step 2: Verify each claim against cited evidence
        verification_results = []
        for claim in claims:
            verification = self._verify_claim(claim, citations, retrieved_docs)
            verification_results.append(verification)
        
        # Step 3: Calculate citation metrics
        supported_claims = [v for v in verification_results if v["supported"]]
        unsupported_claims = [v for v in verification_results if not v["supported"]]
        
        citation_accuracy = len(supported_claims) / len(claims) if claims else 1.0
        
        logger.info(f"Citation accuracy: {citation_accuracy:.2%} ({len(supported_claims)}/{len(claims)} claims supported)")
        
        # Step 4: Generate verified response
        if unsupported_claims:
            logger.warning(f"Found {len(unsupported_claims)} unsupported claims, generating verified response...")
            verified_response = self._generate_verified_response(
                query, response, verification_results, retrieved_docs
            )
        else:
            verified_response = response
        
        return {
            "verified_response": verified_response,
            "citation_accuracy": citation_accuracy,
            "total_claims": len(claims),
            "supported_claims": len(supported_claims),
            "unsupported_claims": len(unsupported_claims),
            "verification_details": verification_results
        }
    
    def _extract_claims(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract factual claims from response.
        
        Each claim is a statement that should be supported by evidence.
        """
        # Use LLM to extract claims
        extract_prompt = f"""Extract all factual claims from the following medical response.

Response:
{response}

For each claim, identify:
1. The claim text
2. The citation marker (e.g., [1], [2]) if present
3. The type of claim (fact, statistic, recommendation, definition)

Return as JSON array:
[
  {{
    "claim": "text of the claim",
    "citation": "[1]" or null,
    "type": "fact|statistic|recommendation|definition"
  }}
]"""
        
        try:
            result = self.llm.complete(extract_prompt)
            result_text = str(result)
            
            # Parse JSON array
            import json
            start = result_text.find('[')
            end = result_text.rfind(']') + 1
            
            if start >= 0 and end > start:
                json_str = result_text[start:end]
                claims = json.loads(json_str)
                return claims
            else:
                return []
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return []
    
    def _extract_citations(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract citation markers from response.
        
        Citations are markers like [1], [2], etc.
        """
        # Find all citation markers
        citation_pattern = r'\[(\d+)\]'
        matches = re.finditer(citation_pattern, response)
        
        citations = []
        for match in matches:
            citation_num = int(match.group(1))
            citations.append({
                "citation_num": citation_num,
                "marker": match.group(0),
                "position": match.start()
            })
        
        return citations
    
    def _verify_claim(
        self,
        claim: Dict[str, Any],
        citations: List[Dict],
        retrieved_docs: List[Dict]
    ) -> Dict[str, Any]:
        """
        Verify if a claim is supported by its cited evidence.
        
        Process:
        1. Find the citation for this claim
        2. Get the cited document
        3. Check if the document supports the claim
        """
        claim_text = claim.get("claim", "")
        citation_marker = claim.get("citation")
        
        # If no citation, claim is unsupported
        if not citation_marker:
            return {
                "claim": claim_text,
                "supported": False,
                "reason": "No citation provided",
                "citation": None,
                "evidence": None
            }
        
        # Extract citation number
        citation_match = re.search(r'\[(\d+)\]', citation_marker)
        if not citation_match:
            return {
                "claim": claim_text,
                "supported": False,
                "reason": "Invalid citation format",
                "citation": citation_marker,
                "evidence": None
            }
        
        citation_num = int(citation_match.group(1))
        
        # Get cited document
        if citation_num > len(retrieved_docs):
            return {
                "claim": claim_text,
                "supported": False,
                "reason": f"Citation [{citation_num}] references non-existent document",
                "citation": citation_marker,
                "evidence": None
            }
        
        cited_doc = retrieved_docs[citation_num - 1]
        evidence_text = cited_doc.get("text", "")
        
        # Verify claim against evidence
        verify_prompt = f"""Verify if the claim is supported by the evidence.

Claim: {claim_text}

Evidence (from document [{citation_num}]):
{evidence_text}

Does the evidence support the claim?
- Yes: The evidence clearly supports the claim
- Partially: The evidence partially supports but is incomplete
- No: The evidence contradicts or does not support the claim

Respond with JSON:
{{
  "supported": true|false,
  "support_level": "full|partial|none",
  "reason": "brief explanation"
}}"""
        
        try:
            result = self.llm.complete(verify_prompt)
            result_text = str(result)
            
            # Parse JSON
            import json
            start = result_text.find('{')
            end = result_text.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = result_text[start:end]
                verification = json.loads(json_str)
                
                return {
                    "claim": claim_text,
                    "supported": verification.get("supported", False),
                    "support_level": verification.get("support_level", "none"),
                    "reason": verification.get("reason", "Unknown"),
                    "citation": citation_marker,
                    "evidence": evidence_text[:200] + "..." if len(evidence_text) > 200 else evidence_text
                }
            else:
                return {
                    "claim": claim_text,
                    "supported": False,
                    "reason": "Verification parsing failed",
                    "citation": citation_marker,
                    "evidence": evidence_text[:200]
                }
        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            return {
                "claim": claim_text,
                "supported": False,
                "reason": f"Verification error: {str(e)}",
                "citation": citation_marker,
                "evidence": evidence_text[:200]
            }
    
    def _generate_verified_response(
        self,
        query: str,
        original_response: str,
        verification_results: List[Dict],
        retrieved_docs: List[Dict]
    ) -> str:
        """
        Generate a verified response with only supported claims.
        
        Strategy:
        1. Keep only supported claims
        2. Remove or flag unsupported claims
        3. Add appropriate caveats
        4. Maintain citation integrity
        """
        # Separate supported and unsupported claims
        supported = [v for v in verification_results if v["supported"]]
        unsupported = [v for v in verification_results if not v["supported"]]
        
        # Build verified response
        verify_prompt = f"""You are a medical AI assistant committed to factual accuracy. Revise the response to include only claims that are supported by the cited evidence.

Question: {query}

Original Response:
{original_response}

Claim Verification Results:

Supported Claims (keep these):
"""
        
        for i, claim in enumerate(supported, 1):
            verify_prompt += f"{i}. {claim['claim']} (Citation: {claim['citation']})\n"
        
        verify_prompt += "\nUnsupported Claims (remove or flag these):\n"
        for i, claim in enumerate(unsupported, 1):
            verify_prompt += f"{i}. {claim['claim']} - Reason: {claim['reason']}\n"
        
        verify_prompt += """
Guidelines:
1. Keep all supported claims with their citations
2. Remove unsupported claims entirely, OR
3. If an unsupported claim is important, flag it with "Evidence insufficient to confirm: ..."
4. Do NOT add new claims without citations
5. Maintain the original structure and flow where possible
6. Ensure all remaining claims have valid citations

Provide the revised response:"""
        
        try:
            result = self.llm.complete(verify_prompt)
            verified_response = str(result).strip()
            return verified_response
        except Exception as e:
            logger.error(f"Verified response generation failed: {e}")
            return original_response
    
    def enforce_citation_requirements(
        self,
        query: str,
        context: str,
        initial_response: str
    ) -> str:
        """
        Enforce citation-aware reasoning during generation.
        
        This is used as a post-processing step to ensure all claims have citations.
        """
        enforce_prompt = f"""You are a medical AI assistant. Ensure every factual claim in the response has a proper citation.

Question: {query}

Retrieved Context:
{context}

Response to Review:
{initial_response}

Requirements:
1. Every factual claim MUST have a citation [1], [2], etc.
2. Citations must reference the retrieved context
3. If a claim cannot be supported by the context, remove it or flag it as uncertain
4. Use the format: "Statement [citation_number]"
5. Do not make claims without evidence

Provide the revised response with proper citations:"""
        
        try:
            result = self.llm.complete(enforce_prompt)
            revised_response = str(result).strip()
            return revised_response
        except Exception as e:
            logger.error(f"Citation enforcement failed: {e}")
            return initial_response
