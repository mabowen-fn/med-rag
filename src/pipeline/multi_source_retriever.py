"""
Multi-Source Evidence Retrieval System
Based on: MEGA-RAG (Frontiers in Public Health, 2025)
Key Innovation: Combines dense retrieval, keyword retrieval, and knowledge graphs
Achieved: 40%+ hallucination reduction
"""
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import networkx as nx
from collections import defaultdict


class MultiSourceRetriever:
    """
    Multi-source evidence retrieval combining:
    1. Dense retrieval (vector similarity)
    2. Keyword retrieval (BM25)
    3. Knowledge graph retrieval (entity relationships)
    """
    
    def __init__(self, dense_retriever, bm25_retriever, knowledge_graph=None):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.knowledge_graph = knowledge_graph
        logger.info("Multi-source retriever initialized (dense + keyword + KG)")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve from multiple sources and fuse results.
        
        Strategy:
        1. Retrieve from each source independently
        2. Apply Reciprocal Rank Fusion (RRF)
        3. Deduplicate and return top-k results
        """
        logger.info(f"Multi-source retrieval for: {query[:100]}...")
        
        # Source 1: Dense retrieval (semantic similarity)
        dense_results = self._dense_retrieve(query, top_k=20)
        logger.info(f"Dense retrieval: {len(dense_results)} documents")
        
        # Source 2: Keyword retrieval (BM25)
        keyword_results = self._keyword_retrieve(query, top_k=20)
        logger.info(f"Keyword retrieval: {len(keyword_results)} documents")
        
        # Source 3: Knowledge graph retrieval (if available)
        kg_results = []
        if self.knowledge_graph:
            kg_results = self._kg_retrieve(query, top_k=10)
            logger.info(f"Knowledge graph retrieval: {len(kg_results)} documents")
        
        # Fuse results using Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            [dense_results, keyword_results, kg_results],
            weights=[0.4, 0.4, 0.2]  # Dense and keyword weighted higher
        )
        
        # Deduplicate
        unique_results = self._deduplicate(fused_results)
        
        # Return top-k
        final_results = unique_results[:top_k]
        logger.info(f"Fused and deduplicated to {len(final_results)} documents")
        
        return final_results
    
    def _dense_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using dense vector similarity"""
        try:
            results = self.dense_retriever.retrieve(query)
            # Tag source
            for doc in results:
                doc["retrieval_source"] = "dense"
            return results[:top_k]
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []
    
    def _keyword_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using BM25 keyword matching"""
        try:
            results = self.bm25_retriever.retrieve(query)
            # Tag source
            for doc in results:
                doc["retrieval_source"] = "keyword"
            return results[:top_k]
        except Exception as e:
            logger.error(f"Keyword retrieval failed: {e}")
            return []
    
    def _kg_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """
        Retrieve using knowledge graph.
        
        Process:
        1. Extract medical entities from query
        2. Find related entities in knowledge graph
        3. Retrieve documents associated with those entities
        """
        try:
            # Extract entities from query
            entities = self._extract_entities(query)
            logger.info(f"Extracted entities: {entities}")
            
            if not entities:
                return []
            
            # Find related entities in KG
            related_entities = set()
            for entity in entities:
                if entity in self.knowledge_graph:
                    # Get neighbors (1-hop)
                    neighbors = list(self.knowledge_graph.neighbors(entity))
                    related_entities.update(neighbors)
                    
                    # Get 2-hop neighbors for broader context
                    for neighbor in neighbors:
                        if neighbor in self.knowledge_graph:
                            second_hop = list(self.knowledge_graph.neighbors(neighbor))
                            related_entities.update(second_hop[:3])  # Limit to avoid explosion
            
            logger.info(f"Found {len(related_entities)} related entities in KG")
            
            # Retrieve documents for related entities
            kg_documents = []
            for entity in related_entities:
                docs = self.knowledge_graph.nodes[entity].get("documents", [])
                for doc in docs:
                    doc["retrieval_source"] = "knowledge_graph"
                    doc["entity"] = entity
                    kg_documents.append(doc)
            
            # Rank by entity relevance
            kg_documents.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return kg_documents[:top_k]
        except Exception as e:
            logger.error(f"Knowledge graph retrieval failed: {e}")
            return []
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract medical entities from query.
        
        Simple implementation using keyword matching.
        In production, use NER models like scispaCy or PubMedBERT.
        """
        # Common medical terms (simplified)
        medical_terms = {
            "diabetes": ["diabetes", "diabetic", "blood sugar", "glucose", "insulin"],
            "hypertension": ["hypertension", "high blood pressure", "blood pressure"],
            "heart": ["heart", "cardiac", "cardiovascular", "myocardial"],
            "cancer": ["cancer", "tumor", "oncology", "malignant"],
            "infection": ["infection", "bacteria", "virus", "pathogen"],
            "symptom": ["symptom", "sign", "manifestation", "presentation"],
            "treatment": ["treatment", "therapy", "medication", "drug"],
            "diagnosis": ["diagnosis", "diagnostic", "test", "screening"]
        }
        
        query_lower = query.lower()
        extracted = []
        
        for category, terms in medical_terms.items():
            for term in terms:
                if term in query_lower:
                    extracted.append(category)
                    break
        
        return extracted
    
    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Dict]],
        weights: List[float],
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) to combine multiple retrieval results.
        
        Formula: score(d) = sum(weight_i / (k + rank_i(d)))
        
        Args:
            result_lists: List of result lists from different retrievers
            weights: Weight for each retriever
            k: Smoothing parameter (default 60)
        """
        # Calculate RRF scores
        doc_scores = defaultdict(float)
        doc_map = {}
        
        for results, weight in zip(result_lists, weights):
            for rank, doc in enumerate(results):
                doc_id = doc.get("metadata", {}).get("doc_id", id(doc))
                
                # RRF score
                rrf_score = weight / (k + rank + 1)
                doc_scores[doc_id] += rrf_score
                
                # Store document (first occurrence)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
        
        # Sort by RRF score
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Build final results
        fused_results = []
        for doc_id in sorted_doc_ids:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = doc_scores[doc_id]
            fused_results.append(doc)
        
        return fused_results
    
    def _deduplicate(self, documents: List[Dict]) -> List[Dict]:
        """Remove duplicate documents"""
        seen_ids = set()
        unique_docs = []
        
        for doc in documents:
            doc_id = doc.get("metadata", {}).get("doc_id", id(doc))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        
        return unique_docs


class MedicalKnowledgeGraph:
    """
    Medical Knowledge Graph for entity-based retrieval.
    
    Based on: MedGraphRAG (ACL 2025)
    Stores medical entities and their relationships.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        logger.info("Medical knowledge graph initialized")
    
    def add_entity(self, entity: str, entity_type: str, documents: List[Dict] = None):
        """Add a medical entity to the graph"""
        self.graph.add_node(
            entity,
            entity_type=entity_type,
            documents=documents or []
        )
    
    def add_relationship(self, source: str, target: str, relationship: str):
        """Add a relationship between entities"""
        self.graph.add_edge(source, target, relationship=relationship)
    
    def get_related_entities(self, entity: str, hops: int = 2) -> List[str]:
        """Get entities within N hops"""
        if entity not in self.graph:
            return []
        
        related = set()
        current_level = {entity}
        
        for _ in range(hops):
            next_level = set()
            for node in current_level:
                if node in self.graph:
                    # Successors (outgoing edges)
                    next_level.update(self.graph.successors(node))
                    # Predecessors (incoming edges)
                    next_level.update(self.graph.predecessors(node))
            
            related.update(next_level)
            current_level = next_level
        
        return list(related - {entity})
    
    def get_documents_for_entity(self, entity: str) -> List[Dict]:
        """Get documents associated with an entity"""
        if entity in self.graph:
            return self.graph.nodes[entity].get("documents", [])
        return []
    
    def build_from_dataset(self, qa_pairs: List[Dict]):
        """
        Build knowledge graph from QA dataset.
        
        Extract entities and relationships from medical Q&A pairs.
        """
        logger.info(f"Building knowledge graph from {len(qa_pairs)} QA pairs...")
        
        for qa_pair in qa_pairs:
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")
            category = qa_pair.get("category", "general")
            
            # Extract entities (simplified - use keyword matching)
            entities = self._extract_entities_from_text(question + " " + answer)
            
            # Add entities
            for entity in entities:
                self.add_entity(
                    entity["text"],
                    entity["type"],
                    documents=[qa_pair]
                )
            
            # Add relationships (simplified - connect entities in same QA pair)
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    self.add_relationship(
                        entity1["text"],
                        entity2["text"],
                        "co_occurs_with"
                    )
        
        logger.info(f"Knowledge graph built: {len(self.graph.nodes)} entities, {len(self.graph.edges)} relationships")
    
    def _extract_entities_from_text(self, text: str) -> List[Dict]:
        """
        Extract medical entities from text.
        
        Simplified implementation using keyword matching.
        In production, use NER models.
        """
        # Medical entity patterns
        entity_patterns = {
            "condition": [
                "diabetes", "hypertension", "heart attack", "stroke",
                "cancer", "pneumonia", "asthma", "arthritis"
            ],
            "symptom": [
                "pain", "fever", "cough", "fatigue", "nausea",
                "headache", "dizziness", "shortness of breath"
            ],
            "treatment": [
                "medication", "surgery", "therapy", "exercise",
                "diet", "insulin", "antibiotics", "chemotherapy"
            ],
            "body_part": [
                "heart", "lung", "brain", "liver", "kidney",
                "blood", "bone", "muscle", "nerve"
            ]
        }
        
        text_lower = text.lower()
        entities = []
        
        for entity_type, terms in entity_patterns.items():
            for term in terms:
                if term in text_lower:
                    entities.append({
                        "text": term,
                        "type": entity_type
                    })
        
        return entities
