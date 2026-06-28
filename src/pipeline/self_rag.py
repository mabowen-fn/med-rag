"""Self-RAG implementation for medical chat engine"""
from typing import Dict, Any, List
from loguru import logger


class SelfRAG:
    """Self-Reflective RAG for improved medical responses"""
    
    def __init__(self, llm):
        self.llm = llm
        logger.info("Self-RAG module initialized")
    
    def critique_response(self, question: str, response: str, context: str) -> Dict[str, Any]:
        """LLM critiques its own response"""
        critique_prompt = f"""You are a medical AI critic. Evaluate this response critically.

Question: {question}

Retrieved Context:
{context}

Response to Critique:
{response}

Evaluate on these criteria:
1. Factual Accuracy: Are all medical facts correct?
2. Completeness: Does it cover all important aspects?
3. Source Support: Is the response supported by the context?
4. Safety: Does it avoid harmful advice?
5. Clarity: Is it clear and understandable?

Provide:
- Overall score (0-10)
- Specific issues found
- Suggestions for improvement
- Whether the response needs revision (yes/no)

Format as JSON:
{{
  "score": <0-10>,
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1", "suggestion2"],
  "needs_revision": <true/false>
}}"""
        
        try:
            critique = self.llm.complete(critique_prompt)
            critique_text = str(critique)
            
            # Try to parse JSON from response
            import json
            # Find JSON in response
            start = critique_text.find('{')
            end = critique_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = critique_text[start:end]
                critique_data = json.loads(json_str)
                return critique_data
            else:
                # Fallback: extract score from text
                score = 5
                if "score:" in critique_text.lower():
                    import re
                    match = re.search(r'score[:\s]+(\d+)', critique_text.lower())
                    if match:
                        score = int(match.group(1))
                
                return {
                    "score": score,
                    "issues": [],
                    "suggestions": [],
                    "needs_revision": score < 7
                }
        except Exception as e:
            logger.error(f"Failed to critique response: {e}")
            return {
                "score": 5,
                "issues": ["Critique failed"],
                "suggestions": [],
                "needs_revision": True
            }
    
    def improve_response(self, question: str, response: str, critique: Dict, context: str) -> str:
        """Improve response based on critique"""
        if not critique.get("needs_revision", False):
            return response
        
        improve_prompt = f"""You are a medical AI assistant. Improve this response based on the critique.

Question: {question}

Original Response:
{response}

Critique:
- Score: {critique.get('score', 'N/A')}/10
- Issues: {', '.join(critique.get('issues', []))}
- Suggestions: {', '.join(critique.get('suggestions', []))}

Retrieved Context:
{context}

Provide an improved response that:
1. Addresses all identified issues
2. Incorporates the suggestions
3. Remains factually accurate
4. Is well-supported by the context
5. Maintains a professional tone

Improved Response:"""
        
        try:
            improved = self.llm.complete(improve_prompt)
            return str(improved)
        except Exception as e:
            logger.error(f"Failed to improve response: {e}")
            return response
    
    def self_reflect(self, question: str, response: str, context: str) -> Dict[str, Any]:
        """Complete self-reflection pipeline"""
        logger.info("Starting self-reflection...")
        
        # Step 1: Critique
        critique = self.critique_response(question, response, context)
        logger.info(f"Critique score: {critique.get('score', 'N/A')}/10")
        
        # Step 2: Improve if needed
        if critique.get("needs_revision", False):
            logger.info("Response needs revision, improving...")
            improved_response = self.improve_response(question, response, critique, context)
            
            # Step 3: Re-critique improved response
            new_critique = self.critique_response(question, improved_response, context)
            logger.info(f"Improved critique score: {new_critique.get('score', 'N/A')}/10")
            
            # Use improved response if score is better
            if new_critique.get('score', 0) > critique.get('score', 0):
                return {
                    "response": improved_response,
                    "critique": new_critique,
                    "improved": True
                }
        
        return {
            "response": response,
            "critique": critique,
            "improved": False
        }
