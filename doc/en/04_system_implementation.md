# Chapter 4: System Implementation

## 4.1 Technology Stack and Development Environment

The system is developed based on Python 3.11, using the modern dependency management tool uv for package management. The selection of core frameworks and libraries follows the principles of performance priority, mature ecosystem, and easy deployment.

The deep learning framework uses PyTorch 2.12.1 as the runtime foundation for embedding models and reranking models. PyTorch's dynamic computation graph and rich GPU optimization support provide good performance guarantees for model inference. The system supports both CPU and GPU operating modes, selecting the optimal computing device through automatic device detection mechanisms.

The vector retrieval library uses FAISS (Facebook AI Similarity Search), a widely-used high-performance vector similarity search library in industry. FAISS supports multiple index types and distance metrics. This system uses FlatL2 index to implement precise L2 distance calculation, providing a good balance of retrieval accuracy and speed on small to medium-scale datasets.

The embedding model uses BAAI/bge-m3, a multilingual, multi-granularity embedding model developed by the Beijing Academy of Artificial Intelligence. The bge-m3 supports 1024-dimensional vector output, demonstrating excellent performance in Chinese medical text semantic understanding. The model is loaded through the sentence-transformers library, supporting batch encoding and GPU acceleration.

The reranking model uses BAAI/bge-reranker-v2-m3, a cross-encoder reranking model paired with bge-m3. This model captures finer-grained semantic matching relationships by simultaneously encoding queries and documents, significantly improving accuracy in the retrieval refinement stage.

The large language model uses Alibaba Cloud's Qwen3:8b, deployed locally through the Ollama framework. Qwen3 is the latest version of the Tongyi Qianwen series, with the 8b parameter scale maintaining good performance while having relatively low hardware requirements, suitable for running on consumer-grade GPUs. Ollama provides a simple API interface and efficient model management mechanism, supporting fast model loading and inference.

The web interface framework uses Streamlit, a Python framework designed specifically for data applications and machine learning demonstrations. Streamlit provides rich interactive components and real-time refresh mechanisms, enabling rapid construction of functional web interfaces.

The logging system uses Loguru, a simple and efficient Python logging library. Loguru provides rich log formatting and output options, supporting dual output to console and files, facilitating system debugging and runtime monitoring.

## 4.2 Data Processing and Index Construction

The system's data processing flow begins with loading medical datasets. The system supports multiple data sources, including the Huatuo-26M medical question-answering dataset and the CMeQA medical examination question-answering dataset. The DatasetLoader class implements a unified data loading interface, supporting dataset download from Hugging Face Hub or loading from local files.

After data loading, the system uses DynamicChunker for dynamic chunking processing. The chunking strategy considers the characteristics of medical text: medical terminology is usually longer, and medical concepts require complete context for accurate understanding. The chunker uses a sentence boundary-based chunking strategy, ensuring no truncation occurs in the middle of sentences. Each text chunk is controlled at approximately 512 tokens in size, with 50 tokens of overlap between chunks, balancing retrieval granularity and context coherence.

After chunking is complete, the system builds a dual-index structure. The vector index construction process is as follows: first, use the BGE-M3 embedding model to encode all text chunks into 1024-dimensional vectors, then use FAISS to build a FlatL2 index. The embedding process supports batch processing, fully utilizing GPU acceleration. After index construction is complete, vectors and metadata (including text content, source information, and chunk identifiers) are persisted to disk together.

The BM25 index construction process is relatively simple: the system tokenizes text chunks, counts term frequencies and document frequencies, calculates inverse document frequency, and builds the BM25 index. The BM25 index is also persisted for fast loading.

The index construction process is executed through the build_index.py script, supporting incremental updates and full rebuilds. During construction, the system outputs detailed progress information, including the number of processed documents, number of generated text chunks, and estimated remaining time, facilitating user monitoring of construction progress.

## 4.3 Hybrid Retrieval Implementation

Hybrid retrieval is the system's core retrieval strategy, implemented through the HybridRetriever class. This class encapsulates the complete process of vector retrieval and BM25 retrieval, providing a unified retrieval interface.

Vector retrieval implementation is based on the FAISS library. When receiving a query, the system first uses the BGE-M3 model to encode the query as a 1024-dimensional vector, then performs L2 distance search in the FAISS index, returning the Top-K most similar documents and their similarity scores. Similarity scores are normalized, converted to values in the 0-1 interval for subsequent fusion.

BM25 retrieval implementation is based on the rank_bm25 library. The system tokenizes the query, then calculates BM25 scores between the query and each document. The BM25 algorithm considers term frequency (TF), inverse document frequency (IDF), and document length normalization, effectively identifying keyword-matched documents. Retrieval results are also normalized to the 0-1 interval.

The fusion strategy uses weighted summation. The system sets weights for vector retrieval and BM25 retrieval separately, with default configuration of vector retrieval weight 0.7 and BM25 retrieval weight 0.3. This configuration is optimized based on medical domain characteristics: medical questions usually involve complex semantic understanding, and vector retrieval can capture semantic similarity; simultaneously, medical text contains大量 professional terminology, and BM25 retrieval can precisely match these terms. The weighted fusion strategy can leverage the advantages of both retrieval methods simultaneously, improving overall retrieval effectiveness.

During the fusion process, the system first performs deduplication processing on the two retrieval results. For documents appearing in both results, the system performs weighted summation of their scores from both paths; for documents appearing in only one result, the system only calculates that path's weighted score. Finally, all documents are sorted by fusion scores, returning the Top-K results.

## 4.4 Reranking Implementation

The reranking module is implemented through the Reranker class, using the BGE-Reranker-v2-m3 cross-encoder model to refine hybrid retrieval results.

The working principle of cross-encoders differs from bi-encoders. Bi-encoders (such as BGE-M3) encode queries and documents separately, calculating relevance through vector similarity; cross-encoders concatenate queries and documents and input them into the model together, capturing fine-grained semantic relationships through self-attention mechanisms. This approach can more accurately judge query-document relevance, with obvious advantages when handling synonyms, abbreviations, and complex expressions of medical terminology.

The reranking implementation process is as follows: the system receives the candidate document list returned by hybrid retrieval, calling the cross-encoder for each query-document pair to calculate relevance scores. The cross-encoder outputs a scalar score indicating the matching degree between query and document. The system re-sorts all documents by the new scores, returning the Top-N most relevant documents.

Reranking computational overhead is relatively large, as it requires calling the model separately for each query-document pair. To balance performance and effectiveness, the system typically performs reranking after hybrid retrieval returns Top-20 candidate documents, finally returning Top-5 results for response generation. This "coarse ranking + fine ranking" two-stage strategy ensures retrieval quality while controlling computational costs.

## 4.5 Confidence Evaluation and Safety Routing

Confidence evaluation is an important guarantee for system safety, implemented through the ConfidenceScorer class. This module comprehensively considers multiple factors to calculate confidence scores, providing decision basis for safety routing.

Confidence score calculation primarily considers the following factors: the average relevance score of retrieved documents reflects the overall quality of retrieval results; the variance of retrieval scores reflects the consistency of retrieval results; the number of retrieval results reflects the knowledge base's coverage of relevant information; the semantic consistency between queries and retrieval results reflects the matching degree between retrieval results and queries.

The system uses weighted combination to calculate the final confidence score. Retrieval quality (mean and variance) occupies larger weight, as high-quality retrieval results are the foundation for generating accurate responses. Context completeness also occupies important weight, ensuring retrieval results sufficiently cover key query concepts.

After confidence scores are normalized to the 0-1 interval, the system executes safety routing decisions based on preset thresholds. Threshold settings consider the specificity of medical applications: high confidence threshold is set to 0.7, medium confidence threshold is set to 0.5.

When confidence ≥0.7, the system considers retrieval result quality is good and can directly generate responses. When confidence is between 0.5-0.7, the system considers retrieval results have certain uncertainty, generating responses but appending warning information to remind users that responses may not be accurate enough. When confidence <0.5, the system considers retrieval result quality is insufficient to support reliable responses, refusing to generate and recommending users consult professional medical personnel.

This safety routing mechanism ensures the system does not generate responses that might mislead users in high-uncertainty scenarios, reducing medical risks.

## 4.6 Citation Traceability Implementation

Citation traceability is an important guarantee for system credibility, implemented through the CitationTracker class. This module records source information of retrieved documents, supporting citation annotation and traceability verification for responses.

The citation tracking implementation process is as follows: in the retrieval stage, the system assigns unique citation numbers (such as [1], [2], [3]) to each retrieved document and records document metadata information, including source dataset, original document identifier, chunk position, and retrieval score.

In the generation stage, the system passes citation numbers and document information together to the large language model, requiring the model to use citation annotation format in responses through prompt engineering. For example, the model-generated response might contain "Typical symptoms of diabetes include polydipsia, polyuria, and weight loss [1], these symptoms are caused by osmotic diuresis due to elevated blood glucose [2]."

In the post-response processing stage, the system extracts citation annotations from responses, associates them with original citation information, and generates formatted citation lists. Citation lists display detailed information for each citation, including document source, relevance score, and text snippets, supporting users in verifying response accuracy and reliability.

The citation traceability mechanism not only improves system transparency but also provides users with the ability to verify information. In medical applications, this traceability is particularly important, as users can trace response sources and judge information authority and timeliness.

## 4.7 Large Language Model Integration

The system integrates the Qwen3:8b large language model through the Ollama framework. Ollama provides a simple Python API, supporting model loading, inference, and management.

The model loading process is implemented through Ollama's LLM class. The system configures the model's temperature parameter to 0.1, ensuring stability and consistency of generation results; configures max_tokens to 1024, controlling generation length; configures context_window to 4096, ensuring sufficient context space.

Prompt engineering is a key link in LLM integration. The system designs structured prompt templates, containing the following parts: system role definition (professional medical AI assistant), behavioral rules (answer based on context, annotate citations, acknowledge uncertainty), retrieved context information, dialogue history, and current user query.

The system role definition clarifies the model's identity and behavioral guidelines, requiring the model to play a professional medical assistant and follow medical ethical standards. Behavioral rules include: prioritize answering based on retrieved context, use citation annotation format to mark information sources, acknowledge uncertainty and recommend consulting professional medical personnel when context is insufficient, avoid providing specific diagnostic and prescription recommendations, and maintain a professional and empathetic tone.

The context information section contains retrieved relevant documents, each with citation numbers and source information. The dialogue history section contains recent dialogue records, helping the model understand dialogue coherence. The current query section contains the user's latest question.

Through carefully designed prompt templates, the system can guide the large language model to generate responses that meet medical application requirements: accurate, reliable, traceable, and safe.

## 4.8 Multi-turn Dialogue Management

The system implements multi-turn dialogue management through the MedicalChatEngine class. This class maintains dialogue history records, considering context dependencies when generating responses.

Dialogue history is stored using a list structure, with each element containing role (user or assistant) and content. The system retains the most recent 6 turns of dialogue (12 records), providing sufficient context information while avoiding token consumption and noise interference caused by overly long history.

When generating responses, the system passes dialogue history as context information to the large language model. The model can understand dialogue coherence, handling coreference resolution and context dependency issues. For example, when a user asks "What is the treatment for this disease," the model can understand "this disease" refers to the disease discussed earlier.

Multi-turn dialogue management also involves state maintenance. The system needs to maintain understanding of the current topic during dialogue, handling topic switching and backtracking. Although the current implementation uses a simple sliding window strategy, it can meet requirements in most medical question-answering scenarios.

## 4.9 Web Interface Implementation

The system builds a web interface based on the Streamlit framework, implemented through src/ui/app.py. The interface design follows principles of simplicity and intuitiveness, providing complete chat interaction functions.

The interface layout is divided into two main areas: sidebar and main area. The sidebar is located on the left side of the page, displaying system status and configuration options. The main area is located in the center of the page, used for chat interaction.

The sidebar contains the following functional modules: system status display (current running mode, retrieval configuration, performance metrics), parameter adjustment controls (retrieval weights, confidence thresholds, Top-K values), mode switching toggle (baseline mode/SOTA mode), and system operation buttons (reset dialogue, clear cache).

The main area uses standard chat interface design. The top displays system title and introduction, the middle is the dialogue history area, and the bottom is the user input box. System responses are displayed as message bubbles, containing response content, confidence scores, and citation annotations. Confidence scores are intuitively displayed through color coding: green (≥0.7) indicates high confidence, yellow (0.5-0.7) indicates medium confidence, and red (<0.5) indicates low confidence.

Citation information is annotated in numbered format in responses, and users can expand to view detailed citation lists. Citation lists display document sources, relevance scores, and text snippets, supporting users in verifying response accuracy.

The interface also implements error handling and user prompting functions. When the system encounters abnormalities, the interface displays friendly error messages, avoiding exposure of technical details. In low-confidence scenarios, the interface displays clear warning information, reminding users that responses may be unreliable.

## 4.10 Evaluation System Implementation

The system implements a comprehensive evaluation framework, supporting multiple evaluation metrics and automated evaluation processes through the RAGEvaluator class.

Retrieval evaluation metrics include Recall@K, Precision@K, and MRR (Mean Reciprocal Rank). Recall@K measures the proportion of relevant documents retrieved among all relevant documents, reflecting retrieval completeness; Precision@K measures the proportion of relevant documents in retrieval results, reflecting retrieval accuracy; MRR measures the ranking position of the first relevant document, reflecting retrieval response speed.

Generation quality evaluation metrics include BLEU, ROUGE-L, and BERTScore. BLEU calculates similarity between generated text and reference answers based on n-gram overlap; ROUGE-L calculates similarity based on longest common subsequence; BERTScore calculates scores based on semantic similarity of pre-trained language models. These metrics measure generation quality from different angles, providing comprehensive evaluation perspectives.

Hallucination rate evaluation is a key metric for medical question-answering systems. The system identifies content that cannot be supported by context by comparing generated responses and retrieved context, calculating hallucination rate. This metric directly reflects system reliability.

Response time evaluation records time consumption at various stages, including retrieval time, reranking time, generation time, and end-to-end total time. These metrics reflect system real-time performance.

The evaluation process is executed through the run_evaluation.py script. The system loads test datasets, executes complete question-answering processes for each test case, collects evaluation metrics, and generates evaluation reports. Evaluation reports are output in JSON and Markdown formats, containing detailed metric statistics and comparative analysis.

The system also supports comparative evaluation, able to compare performance differences between baseline mode and SOTA mode, compare effects of different configuration parameters, providing data support for system optimization.
