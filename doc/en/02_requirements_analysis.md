# Chapter 2: Requirements Analysis

## 2.1 Functional Requirements

This system targets the medical vertical industry, with the core function of providing medical question-answering services based on retrieval-augmented generation. The system needs to receive users' natural language medical questions, retrieve relevant information from the medical knowledge base, combine the retrieved context to generate accurate and reliable answers, and provide complete citation traceability and confidence evaluation.

In terms of data processing, the system needs to support loading and preprocessing of medical datasets. The system uses medical question-answering datasets such as Huatuo-26M and CMeQA as knowledge sources, requiring automated dataset loading, cleaning, and formatting. Data needs to undergo dynamic chunking processing, splitting long documents into text chunks suitable for retrieval and generation while maintaining semantic integrity. The chunking strategy needs to support configurable chunk sizes and overlap regions to balance retrieval precision and context coherence.

In terms of knowledge representation, the system needs to build a dual-index structure. On one hand, using the BGE-M3 embedding model to convert text chunks into 1024-dimensional vector representations, building a FAISS vector index to support semantic retrieval; on the other hand, building a BM25 keyword index to support exact matching. The dual-index structure can simultaneously capture semantic similarity and keyword matching, improving the comprehensiveness and accuracy of retrieval.

In terms of retrieval process, the system needs to implement hybrid retrieval strategies. When users ask questions, the system simultaneously performs vector retrieval and BM25 retrieval, then merges the two retrieval results through weighted fusion strategies. The vector retrieval weight is set to 0.7 and BM25 retrieval weight to 0.3, a configuration optimized based on medical domain characteristics. After retrieval is complete, the system uses BGE-Reranker-v2-m3 cross-encoder to rerank candidate documents, further improving retrieval quality.

In terms of generation process, the system needs to implement confidence-based safety routing mechanisms. The system calculates confidence scores for retrieval results and determines subsequent processing strategies based on confidence levels. When confidence is above 0.7, the system directly generates answers; when confidence is between 0.5 and 0.7, the system generates answers but appends warning information; when confidence is below 0.5, the system refuses to generate answers and recommends users consult professional medical personnel. This mechanism ensures system safety in high-uncertainty scenarios.

In terms of citation traceability, the system needs to implement complete citation tracking mechanisms. The system records the source, relevance, and score information of each retrieved document, passing citation information to users when generating answers. Users can trace back to original documents through citation numbers to verify the accuracy and reliability of answers. Citation information includes document source, retrieval scores, and relevance descriptions, helping users evaluate information credibility.

In terms of user interaction, the system needs to provide a web interface supporting multi-turn dialogue. Users can ask questions consecutively, with the system maintaining dialogue history and considering context information when generating answers. The interface needs to display answer content, confidence scores, citation lists, and system status, providing intuitive interactive experience. The system also needs to support configuration adjustments, allowing users to switch between different processing modes (baseline mode or SOTA mode) and adjust retrieval parameters.

In terms of evaluation, the system needs to implement comprehensive performance evaluation functions. The system supports multiple evaluation metrics, including retrieval accuracy (Recall@K, Precision@K, MRR), generation quality (BLEU, ROUGE-L, BERTScore), hallucination rate, and response time. The system can automatically generate evaluation reports, comparing performance differences across different configurations and modes, providing data support for system optimization.

## 2.2 Non-Functional Requirements

In terms of performance, the system needs to meet real-time interaction requirements. End-to-end response time should be controlled within 10 seconds, with retrieval phase not exceeding 2 seconds and generation phase not exceeding 8 seconds. The system needs to support knowledge base scale of at least 10,000 documents, maintaining acceptable retrieval latency at this scale. The system needs to support both CPU and GPU operating modes, with significantly improved performance under GPU acceleration.

In terms of accuracy, the system needs to control hallucination rate below 30%, achieving significant reduction compared to baseline LLM's 80% hallucination rate. Retrieval accuracy (Recall@10) should reach above 85%, ensuring relevant information can be effectively retrieved. Generated answer faithfulness should reach above 90%, ensuring answer content remains consistent with retrieved context.

In terms of reliability, the system needs comprehensive error handling mechanisms. When component loading fails, the system should provide degradation solutions rather than crashing directly. When retrieval result quality is poor, the system should identify and refuse to generate low-quality answers through confidence mechanisms. The system needs to record detailed log information for problem diagnosis and performance analysis.

In terms of scalability, the system needs modular design, with components interacting through clear interfaces. This design allows the system to conveniently replace or upgrade specific components, such as replacing embedding models, adjusting retrieval strategies, or integrating new evaluation metrics. System configuration needs centralized management, supporting parameter adjustment through configuration files without code modification.

In terms of deployment, the system needs to support multiple hardware environments. In development environments, the system should run on CPU devices such as Mac M1 for convenient development and testing. In production environments, the system should run on GPU devices such as NVIDIA RTX 4090, fully utilizing GPU acceleration to improve performance. The system needs to provide automated deployment scripts, simplifying environment configuration and dependency installation processes.

In terms of security, the system adopts local deployment strategy, with all data processing and model inference completed locally, without involving external API calls, ensuring privacy and security of medical data. The system avoids generating potentially erroneous answers in low-confidence scenarios through confidence routing mechanisms, reducing medical risks.

## 2.3 Performance Metrics

System performance evaluation covers multiple dimensions. Retrieval performance metrics include Recall@K (proportion of relevant documents retrieved), Precision@K (proportion of relevant documents in retrieval results), and MRR (reciprocal of first relevant document rank), measuring retrieval system effectiveness.

Generation quality metrics include BLEU (similarity based on n-gram overlap), ROUGE-L (similarity based on longest common subsequence), and BERTScore (scoring based on semantic similarity), measuring the match between generated answers and reference answers.

Hallucination rate is a key metric for medical question-answering systems, measuring the proportion of generated content that cannot be supported by retrieved context. The system goal is to reduce hallucination rate from baseline 80% to below 30%, significantly reducing false information generation through retrieval augmentation and citation verification.

Response time metrics measure system real-time performance, including retrieval time, reranking time, generation time, and end-to-end total time. The system needs to minimize latency while ensuring quality, providing smooth user experience.

Confidence scores reflect the system's evaluation of answer reliability, calculated comprehensively based on multiple factors including retrieval quality, context completeness, and generation consistency. Confidence is used not only for safety routing but also as an important performance evaluation metric, with high-confidence answers expected to have higher accuracy.
