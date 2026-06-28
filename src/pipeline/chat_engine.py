"""Chat engine for medical RAG system - manual pipeline orchestration"""
from typing import List, Dict, Any, Optional
from loguru import logger
from config import config


class MedicalChatEngine:
    """Medical chat engine with multi-turn dialogue support.
    
    Manually orchestrates the RAG pipeline:
    1. Hybrid Retrieval (BM25 + Vector)
    2. Reranking (BGE-Reranker-v2-m3)
    3. Confidence Scoring
    4. Citation Tracking
    5. LLM Generation with context
    6. Confidence-based routing (safety)
    7. Optional Self-RAG (self-reflection)
    """
    
    # Confidence thresholds for routing
    CONFIDENCE_HIGH = 0.7
    CONFIDENCE_MEDIUM = 0.5
    
    def __init__(self, retriever, reranker, confidence_scorer, citation_tracker, enable_self_rag=False):
        self.retriever = retriever
        self.reranker = reranker
        self.confidence_scorer = confidence_scorer
        self.citation_tracker = citation_tracker
        self.enable_self_rag = enable_self_rag
        
        # Initialize Ollama LLM
        from llama_index.llms.ollama import Ollama
        self.llm = Ollama(
            model=config.model.llm_model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            context_window=config.llm.context_window,
            request_timeout=600.0,
        )
        
        # Initialize Self-RAG if enabled
        if enable_self_rag:
            from src.pipeline.self_rag import SelfRAG
            self.self_rag = SelfRAG(self.llm)
            logger.info("Self-RAG enabled")
        
        # Multi-turn conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info("Medical chat engine initialized")
    
    def chat(self, query: str, use_rag: bool = True) -> Dict[str, Any]:
        """Process a chat query through the full RAG pipeline with confidence-based routing"""
        logger.info(f"Processing query: {query[:100]}...")
        
        if not use_rag:
            response_text = self._generate_direct(query)
            return {
                "response": response_text,
                "citations": [],
                "confidence": {"confidence": 0.0, "is_reliable": False, "reason": "RAG disabled"},
                "routed": "no_rag",
            }
        
        # Step 1: Hybrid Retrieval (BM25 + Vector)
        retrieved_docs = self.retriever.retrieve(query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents via hybrid retrieval")
        
        # Step 2: Reranking
        if retrieved_docs and self.reranker is not None:
            reranked_docs = self.reranker.rerank(query, retrieved_docs)
            logger.info(f"Reranked to top {len(reranked_docs)} documents")
        else:
            reranked_docs = retrieved_docs
        
        # Step 3: Citation Tracking
        citations = self.citation_tracker.track_citations(reranked_docs)
        
        # Step 4: Confidence Scoring
        confidence = self.confidence_scorer.calculate_confidence(query, reranked_docs)
        
        # Step 5: Confidence-based routing (safety mechanism)
        routing_decision = self._route_by_confidence(confidence)
        logger.info(f"Confidence routing: {routing_decision['level']}")
        
        if routing_decision['level'] == 'low':
            # Low confidence: escalate to disclaimer
            response_text = routing_decision['disclaimer']
            routed = "low_confidence_escalated"
        else:
            # Medium/High confidence: generate response
            context_text = self._build_context(reranked_docs)
            response_text = self._generate_with_context(query, context_text, confidence)
            routed = "rag_generated"
            
            # Step 6: Optional Self-RAG for high-quality responses
            if self.enable_self_rag and routing_decision['level'] == 'high':
                self_rag_result = self.self_rag.self_reflect(query, response_text, context_text)
                if self_rag_result.get("improved", False):
                    response_text = self_rag_result["response"]
                    routed = "self_rag_improved"
                    logger.info("Response improved via Self-RAG")
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Keep history manageable (last 6 turns)
        if len(self.conversation_history) > 12:
            self.conversation_history = self.conversation_history[-12:]
        
        return {
            "response": response_text,
            "citations": self.citation_tracker.format_citations_for_display(),
            "confidence": confidence,
            "routed": routed,
        }
    
    def _route_by_confidence(self, confidence: Dict[str, Any]) -> Dict[str, Any]:
        """Route response based on confidence level (safety mechanism)"""
        score = confidence.get("confidence", 0.0)
        
        if score >= self.CONFIDENCE_HIGH:
            return {
                "level": "high",
                "action": "generate",
                "disclaimer": None,
            }
        elif score >= self.CONFIDENCE_MEDIUM:
            return {
                "level": "medium",
                "action": "generate_with_warning",
                "disclaimer": None,
            }
        else:
            return {
                "level": "low",
                "action": "escalate",
                "disclaimer": (
                    "I'm not confident enough to provide a reliable answer to this question. "
                    "Medical information requires high accuracy, and I don't have sufficient "
                    "evidence to respond safely.\n\n"
                    "**Please consult a qualified healthcare professional** who can provide "
                    "personalized medical advice based on your specific situation.\n\n"
                    "If this is a medical emergency, please call your local emergency services immediately."
                ),
            }
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents"""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            source = doc.get("metadata", {}).get("source", "Unknown")
            context_parts.append(f"[{i}] (Source: {source})\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _generate_with_context(
        self, query: str, context: str, confidence: Dict[str, Any]
    ) -> str:
        """Generate response using LLM with retrieved context"""
        
        system_prompt = """You are a professional medical AI assistant. Follow these rules:
1. Base your answers on the provided context when available
2. Cite sources using [1], [2], etc. notation
3. If confidence is low or context is insufficient, acknowledge uncertainty and recommend consulting healthcare professionals
4. Never provide specific medical diagnoses or prescriptions
5. Maintain a professional, empathetic tone
6. Respond in the same language as the user's question"""
        
        prompt = f"{system_prompt}\n\n"
        
        # Add context
        if context:
            prompt += f"=== Retrieved Context ===\n{context}\n\n"
        
        # Add low-confidence warning
        if not confidence.get("is_reliable", False):
            prompt += "WARNING: Low confidence in retrieved context. Acknowledge uncertainty and recommend professional consultation.\n\n"
        
        # Add conversation history (last 3 turns)
        recent_history = self.conversation_history[-6:]
        if recent_history:
            prompt += "=== Conversation History ===\n"
            for turn in recent_history:
                role = "User" if turn["role"] == "user" else "Assistant"
                prompt += f"{role}: {turn['content']}\n"
            prompt += "\n"
        
        # Current query
        prompt += f"=== Current Question ===\nUser: {query}\nAssistant:"
        
        response = self.llm.complete(prompt)
        return str(response)
    
    def _generate_direct(self, query: str) -> str:
        """Generate response without RAG (direct LLM)"""
        prompt = f"You are a helpful assistant. Answer the following question:\n\nQuestion: {query}\nAnswer:"
        response = self.llm.complete(prompt)
        return str(response)
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history
