"""Dynamic chunking for Medical RAG System"""
import re
from typing import List, Dict, Any
from loguru import logger
from config import config


class DynamicChunker:
    """Dynamic chunking strategy for medical documents"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        self.dynamic_chunking = config.chunking.dynamic_chunking
        
        # Medical-specific patterns for smart chunking
        self.medical_patterns = [
            r'(?i)(?:question|问题|问)[:\s]',
            r'(?i)(?:answer|回答|答)[:\s]',
            r'(?i)(?:diagnosis|诊断)[:\s]',
            r'(?i)(?:treatment|治疗)[:\s]',
            r'(?i)(?:symptom|症状)[:\s]',
            r'(?i)(?:patient|患者)[:\s]',
        ]
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk documents with dynamic strategy"""
        if self.dynamic_chunking:
            return self._dynamic_chunk(documents)
        else:
            return self._fixed_chunk(documents)
    
    def _dynamic_chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dynamic chunking based on content structure"""
        logger.info("Using dynamic chunking strategy")
        chunks = []
        
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            # Detect document structure
            if self._is_qa_format(text):
                doc_chunks = self._chunk_qa_format(text, metadata)
            elif self._is_medical_report(text):
                doc_chunks = self._chunk_medical_report(text, metadata)
            else:
                doc_chunks = self._chunk_general_text(text, metadata)
            
            chunks.extend(doc_chunks)
        
        logger.info(f"Generated {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _fixed_chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fixed-size chunking"""
        logger.info("Using fixed chunking strategy")
        chunks = []
        
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            doc_chunks = self._chunk_by_size(text, metadata)
            chunks.extend(doc_chunks)
        
        logger.info(f"Generated {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _is_qa_format(self, text: str) -> bool:
        """Check if text is in QA format"""
        qa_patterns = [
            r'(?i)(question|问题|问)[:\s]',
            r'(?i)(answer|回答|答)[:\s]',
        ]
        return any(re.search(pattern, text) for pattern in qa_patterns)
    
    def _is_medical_report(self, text: str) -> bool:
        """Check if text is a medical report"""
        report_patterns = [
            r'(?i)(diagnosis|诊断)[:\s]',
            r'(?i)(treatment|治疗)[:\s]',
            r'(?i)(patient|患者)[:\s]',
            r'(?i)(symptom|症状)[:\s]',
        ]
        return sum(1 for pattern in report_patterns if re.search(pattern, text)) >= 2
    
    def _chunk_qa_format(self, text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """Chunk QA format documents"""
        chunks = []
        
        # Split by QA pairs
        qa_pattern = r'(?i)(question|问题|问)[:\s](.*?)(?=(?:question|问题|问)[:\s]|$)'
        qa_matches = re.findall(qa_pattern, text, re.DOTALL)
        
        if qa_matches:
            for i, (_, qa_text) in enumerate(qa_matches):
                if len(qa_text.strip()) > 0:
                    chunk = {
                        "text": qa_text.strip(),
                        "metadata": {**metadata, "chunk_type": "qa", "chunk_id": i},
                    }
                    chunks.append(chunk)
        else:
            # Fallback to size-based chunking
            chunks = self._chunk_by_size(text, metadata)
        
        return chunks
    
    def _chunk_medical_report(self, text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """Chunk medical report by sections"""
        chunks = []
        
        # Split by medical sections
        section_pattern = r'(?i)((?:diagnosis|诊断|treatment|治疗|symptom|症状|patient|患者)[:\s][^\n]*(?:\n|$))'
        sections = re.split(section_pattern, text)
        
        current_chunk = ""
        for section in sections:
            if len(current_chunk) + len(section) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {**metadata, "chunk_type": "medical_section"},
                    })
                current_chunk = section
            else:
                current_chunk += section
        
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {**metadata, "chunk_type": "medical_section"},
            })
        
        return chunks if chunks else self._chunk_by_size(text, metadata)
    
    def _chunk_general_text(self, text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """Chunk general text with overlap"""
        return self._chunk_by_size(text, metadata)
    
    def _chunk_by_size(self, text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """Chunk text by fixed size with overlap"""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            if len(chunk_text.strip()) > 0:
                chunks.append({
                    "text": chunk_text.strip(),
                    "metadata": {
                        **metadata,
                        "chunk_id": chunk_id,
                        "start_char": start,
                        "end_char": end,
                    },
                })
                chunk_id += 1
            
            start = end - self.chunk_overlap
        
        return chunks
