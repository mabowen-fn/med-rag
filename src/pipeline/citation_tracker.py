"""Citation tracker for Medical RAG System"""
from typing import List, Dict, Any
import re


class CitationTracker:
    """Track and format citations from retrieved documents"""
    
    def __init__(self):
        self.citations = []
    
    def track_citations(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Track citations from retrieved documents"""
        self.citations = []
        
        for idx, doc in enumerate(documents, start=1):
            citation = {
                "id": idx,
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
                "score": doc.get("rerank_score", doc.get("score", 0.0)),
                "source": self._extract_source(doc.get("metadata", {})),
            }
            self.citations.append(citation)
        
        return self.citations
    
    def _extract_source(self, metadata: Dict[str, Any]) -> str:
        """Extract source information from metadata"""
        source_parts = []
        
        if "source" in metadata:
            source_parts.append(metadata["source"])
        
        if "chunk_type" in metadata:
            source_parts.append(f"({metadata['chunk_type']})")
        
        if "chunk_id" in metadata:
            source_parts.append(f"chunk_{metadata['chunk_id']}")
        
        return " ".join(source_parts) if source_parts else "Unknown source"
    
    def format_citations_for_prompt(self) -> str:
        """Format citations for inclusion in LLM prompt"""
        if not self.citations:
            return ""
        
        citation_text = "\n\nRetrieved Context:\n"
        for citation in self.citations:
            citation_text += f"\n[{citation['id']}] {citation['source']}\n"
            citation_text += f"{citation['text']}\n"
            citation_text += f"Score: {citation['score']:.3f}\n"
        
        return citation_text
    
    def format_citations_for_display(self) -> List[Dict[str, Any]]:
        """Format citations for UI display"""
        formatted = []
        
        for citation in self.citations:
            formatted.append({
                "id": citation["id"],
                "source": citation["source"],
                "text_preview": citation["text"][:200] + "..." if len(citation["text"]) > 200 else citation["text"],
                "score": f"{citation['score']:.3f}",
                "metadata": citation["metadata"],
            })
        
        return formatted
    
    def extract_citations_from_response(self, response: str) -> List[int]:
        """Extract citation IDs from LLM response"""
        # Look for patterns like [1], [2], etc.
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, response)
        return [int(m) for m in matches]
