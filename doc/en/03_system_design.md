# Chapter 3: System Design

## 3.1 System Architecture Design

The system adopts a layered architecture design, decomposing the complex RAG process into multiple independent and composable module layers. The overall architecture is divided from top to bottom into the user interface layer, application logic layer, retrieval processing layer, knowledge representation layer, and data storage layer.

The user interface layer is built on the Streamlit framework, providing a web-based interactive chat interface. This layer is responsible for receiving user input, displaying system responses, showing confidence scores and citation information, and providing system configuration options. The interface design follows principles of simplicity and intuitiveness, enabling users to easily interact with the system and understand its working status.

The application logic layer is the core coordination layer of the system, primarily implemented by the MedicalChatEngine class. This layer orchestrates the entire RAG process, including calling retrieval modules to obtain relevant documents, invoking reranking modules to optimize retrieval results, calculating confidence scores, executing safety routing decisions, calling large language models to generate responses, and managing multi-turn dialogue history. The application logic layer interacts with various lower-level modules through clear interfaces, achieving flexible process control and loose component coupling.

The retrieval processing layer implements hybrid retrieval and reranking functions. This layer contains two main components: the HybridRetriever and the Reranker. The hybrid retriever simultaneously performs vector semantic retrieval and BM25 keyword retrieval, merging the two retrieval paths through weighted fusion strategies. The reranker uses cross-encoder models to fine-rank the fused candidate documents, improving final retrieval quality.

The knowledge representation layer manages the index structure of the medical knowledge base. This layer maintains two indexes: a FAISS vector index for semantic retrieval and a BM25 index for keyword retrieval. The vector index stores 1024-dimensional embedding representations of documents, supporting fast similarity calculations; the BM25 index stores document term frequency statistics, supporting precise keyword matching. Together, these two indexes form the system's knowledge foundation.

The data storage layer manages raw medical datasets and processed intermediate data. The system uses medical question-answering datasets such as Huatuo-26M and CMeQA as knowledge sources, implementing data loading and preprocessing through the DatasetLoader class. The DynamicChunker splits long documents into text chunks suitable for retrieval while preserving metadata information for citation traceability.

## 3.2 System Workflow Design

The system's core workflow is divided into two phases: offline index construction and online question answering.

The offline index construction phase first loads medical datasets, cleaning and formatting the data. Then it uses the BGE-M3 embedding model to convert text chunks into vector representations, building the FAISS vector index. Simultaneously, the system performs tokenization on the text to build the BM25 keyword index. After index construction is complete, the indexes are persisted to disk for online query use.

The online question answering workflow proceeds as follows: users submit medical questions through the web interface, and the system first performs query preprocessing upon receiving the query. Then it executes two parallel retrieval paths: vector retrieval encodes the query as a vector and searches for similar documents in the FAISS index, while BM25 retrieval calculates keyword matching scores between the query and documents. The two retrieval results are merged through weighted fusion strategies, with vector retrieval weighted at 0.7 and BM25 retrieval weighted at 0.3. The fused candidate documents are fine-ranked by the BGE-Reranker to obtain final retrieval results.

The system calculates confidence scores for retrieval results, comprehensively evaluating based on factors such as retrieval quality, context completeness, and generation consistency. Based on confidence levels, the system executes safety routing decisions: high confidence (≥0.7) directly generates responses, medium confidence (0.5-0.7) generates responses with warnings, and low confidence (<0.5) refuses generation and recommends consulting professional medical personnel.

For medium to high confidence queries, the system constructs retrieved documents into context, combines user queries and dialogue history, constructs complete prompts through prompt engineering, and calls the Qwen3:8b large language model to generate responses. During generation, the system requires the model to answer based on retrieved context, use citation annotation format to mark information sources, and acknowledge uncertainty when context is insufficient.

After generation is complete, the system extracts citation information from responses and formats citation lists for user viewing. The final response, confidence scores, citation lists, and routing decision results are returned together to the user interface for display.

Multi-turn dialogue management runs through the entire process, with the system maintaining the most recent 6 turns of dialogue history and passing it as context information to the large language model during response generation, enabling the model to understand dialogue coherence and context dependencies.

## 3.3 Core Module Design

### 3.3.1 Hybrid Retriever (HybridRetriever)

The hybrid retriever is the system's core retrieval component, implementing fusion strategies for vector retrieval and BM25 retrieval. This module receives query strings, separately calls vector retrieval and BM25 retrieval to obtain candidate documents, then merges results through weighted fusion strategies.

The vector retrieval portion calls the embedding model to encode queries as 1024-dimensional vectors, performs similarity search in the FAISS index, and returns the Top-K most similar documents along with their similarity scores. The BM25 retrieval portion tokenizes queries, calculates BM25 scores between the query and each document, and returns the Top-K highest-scoring documents.

The fusion strategy uses weighted summation. For documents appearing in both retrieval results, their vector retrieval scores and BM25 scores are multiplied by corresponding weights and summed to obtain fusion scores. For documents appearing in only one retrieval result, only that path's weighted score is calculated. Finally, documents are sorted by fusion scores and the Top-K results are returned.

### 3.3.2 Reranker

The reranker uses the BGE-Reranker-v2-m3 cross-encoder model to fine-rank hybrid retrieval results. Unlike bi-encoder models, cross-encoders simultaneously encode queries and documents, capturing finer-grained semantic relationships.

This module receives query strings and candidate document lists, calculating relevance scores for each query-document pair. The cross-encoder concatenates queries and documents, inputting them into the Transformer model, capturing semantic matching degrees through attention mechanisms. After reranking, the system re-sorts documents by new relevance scores, returning the Top-N most relevant documents.

Reranking significantly improves retrieval quality, particularly when handling synonyms, abbreviations, and complex expressions of medical terminology, where cross-encoders can more accurately judge semantic relevance.

### 3.3.3 Confidence Scorer (ConfidenceScorer)

The confidence scorer evaluates retrieval result quality, providing decision basis for safety routing. This module comprehensively considers multiple factors to calculate confidence scores: average relevance scores of retrieved documents, quantity and distribution of retrieval results, semantic consistency between queries and retrieval results, etc.

Confidence score calculation uses weighted combination, primarily considering retrieval quality (mean and variance of retrieval scores) and context completeness (whether retrieval results sufficiently cover key query concepts). Confidence scores are normalized to the 0-1 interval for subsequent routing decisions.

### 3.3.4 Citation Tracker (CitationTracker)

The citation tracker records and manages source information of retrieved documents, supporting citation traceability for responses. This module assigns unique citation numbers to each retrieved document, recording document sources, relevance, and score information.

During response generation, the system passes citation numbers and document information to the large language model, requiring the model to use citation annotation format (such as [1], [2]) in responses to mark information sources. After generation, the system extracts citation annotations from responses, associates them with original citation information, and generates formatted citation lists for user viewing.

### 3.3.5 Chat Engine (MedicalChatEngine)

The chat engine is the system's main control module, responsible for orchestrating the entire RAG process. This module coordinates various steps including retrieval, reranking, confidence scoring, citation tracking, and response generation, implementing complete question-answering workflows.

The chat engine maintains dialogue history, supporting multi-turn dialogue. During response generation, the engine passes dialogue history as context information to the large language model, enabling the model to understand dialogue coherence. The engine also implements safety routing mechanisms, deciding whether to generate responses based on confidence scores, refusing to answer in low-confidence scenarios and recommending users consult professional medical personnel.

## 3.4 Database and Vector Store Design

The system uses two index structures to store medical knowledge: FAISS vector index and BM25 keyword index.

The FAISS vector index stores 1024-dimensional embedding representations of documents. The index uses FlatL2 distance measurement, supporting precise similarity search. The index structure contains two parts: vector data and metadata. Vector data is used for similarity calculations, while metadata stores document text content, source information, and chunk identifiers. The index supports batch addition and persistent storage, facilitating offline construction and online loading.

The BM25 index stores document term frequency statistics. The index is implemented using the BM25Okapi algorithm, supporting fast keyword matching queries. The index structure contains vocabulary, document frequency, and inverse document frequency information, used to calculate relevance scores between queries and documents.

These two indexes together form the system's hybrid retrieval foundation, with the vector index capturing semantic similarity and the BM25 index capturing keyword matching, complementing each other to improve retrieval effectiveness.

## 3.5 Interface Design

The system builds a web interface based on the Streamlit framework, adopting a simple and intuitive chat interaction design. The interface is primarily divided into three areas: the left sidebar for system configuration and status display, the central main area for chat interaction, and the right panel for displaying citation information (optional).

The left sidebar displays system status information, including current running mode (baseline mode or SOTA mode), retrieval configuration parameters, and system performance metrics. Users can adjust retrieval weights, confidence thresholds, and other parameters through sidebar controls, switching between different processing modes.

The central main area is the chat interface, using standard conversational interaction design. The user input box is located at the bottom, with dialogue history extending upward. System responses are displayed as message bubbles, containing response content, confidence scores, and citation annotations. Confidence scores are intuitively displayed through color coding: green indicates high confidence, yellow indicates medium confidence, and red indicates low confidence.

Citation information is annotated in responses using numbered format (such as [1], [2]), and users can click to view detailed citation lists. Citation lists display document sources, relevance scores, and text snippets, supporting users in verifying response accuracy and reliability.

The interface design follows the special requirements of medical applications, displaying clear warning information in low-confidence scenarios, reminding users that responses may be unreliable and recommending consultation with professional medical personnel. This design ensures that while providing convenience, the system does not mislead users into making incorrect medical decisions.
