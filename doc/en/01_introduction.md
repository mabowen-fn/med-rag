# Chapter 1: Introduction

## 1.1 Project Background and Significance

With the rapid advancement of artificial intelligence technology, Large Language Models (LLMs) have achieved breakthrough progress in the field of natural language processing. However, general-purpose LLMs face significant challenges when applied to vertical industries such as healthcare. The medical domain demands extremely high standards for information accuracy, reliability, and traceability, where any errors or hallucinations can lead to serious consequences. Traditional LLMs suffer from core issues including delayed knowledge updates, susceptibility to hallucinations, and lack of citation traceability, problems that are particularly pronounced in medical scenarios.

Retrieval-Augmented Generation (RAG) technology provides an effective approach to address these challenges. By retrieving relevant knowledge before generating responses and using the retrieved context as the basis for generation, RAG significantly improves both the accuracy and timeliness of responses. This project targets the medical vertical industry, designing and implementing a medical question-answering system based on RAG technology, aiming to solve the knowledge acquisition challenges in the healthcare domain.

The core significance of this project lies in several key aspects. First, it improves the accuracy and comprehensiveness of medical knowledge retrieval through hybrid retrieval strategies combining BM25 keyword retrieval with vector semantic retrieval. Second, it introduces reranking models (Cross-Encoder Reranker) to optimize retrieval result quality. Third, it implements confidence evaluation and citation traceability mechanisms to enhance system credibility and transparency. Fourth, it employs a locally deployed LLM (Qwen3:8b) to ensure data privacy and security. These technical features enable the system to provide reliable, traceable medical knowledge services to healthcare practitioners, patients, and medical students.

## 1.2 Related Technologies

### 1.2.1 Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation is a technical framework that combines information retrieval with text generation. Its core concept involves retrieving relevant documents from external knowledge bases before the LLM generates responses, then feeding the retrieval results as context to the model, thereby guiding the model to generate responses based on factual knowledge. RAG technology effectively mitigates the knowledge hallucination problem in LLMs while supporting dynamic knowledge updates without requiring model retraining.

### 1.2.2 Vector Embeddings and Semantic Retrieval

Vector embedding technology converts text into high-dimensional vector representations, making semantically similar texts closer in vector space. This project employs the BAAI/bge-m3 embedding model, which supports multiple languages and granularities, demonstrating excellent performance in medical text semantic understanding. Based on vector embeddings, the system uses FAISS (Facebook AI Similarity Search) to build efficient vector indices, enabling fast semantic similarity retrieval.

### 1.2.3 BM25 Keyword Retrieval

BM25 is a classic information retrieval algorithm that calculates query-document relevance based on term frequency (TF) and inverse document frequency (IDF). Compared to semantic retrieval, BM25 has advantages in exact matching of medical terminology, drug names, and disease codes. This project implements BM25Okapi, combining BM25 retrieval results with vector retrieval results through weighted fusion strategies to achieve hybrid retrieval.

### 1.2.4 Cross-Encoder Reranking

Reranking is the fine-ranking stage after retrieval, using more complex models to rescore candidate documents. This project uses the BAAI/bge-reranker-v2-m3 cross-encoder model, which simultaneously encodes queries and documents, capturing fine-grained semantic relationships between them, achieving higher accuracy than bi-encoder models. Reranking significantly improves retrieval result quality, particularly in specialized medical domains.

### 1.2.5 Large Language Models

This project employs Qwen3:8b as the generation model, deployed locally through the Ollama framework. Qwen3 is part of Alibaba Cloud's Tongyi Qianwen series, with the 8b version maintaining good performance while having relatively low hardware requirements, making it suitable for running on consumer-grade GPUs. Local deployment ensures privacy and security of medical data, avoiding data leakage risks.

## 1.3 Project Content and Objectives

This project aims to build an intelligent question-answering system for the medical vertical industry, completing the following main work:

**Requirements Analysis**: Clarifying the core requirements for medical question-answering scenarios, including performance metrics such as accuracy, reliability, traceability, and response speed, and analyzing the usage needs of different user groups including healthcare practitioners, patients, and medical students.

**System Design**: Designing a complete RAG system architecture, including data processing layers, knowledge representation layers, retrieval layers, reranking layers, generation layers, and user interface layers. The focus is on designing hybrid retrieval strategies, confidence evaluation mechanisms, citation traceability systems, and multi-turn dialogue management.

**Core Implementation**: Implementing a hybrid retriever based on BM25 and vector retrieval, integrating cross-encoder reranking models, developing confidence scoring and routing mechanisms, building citation tracking and verification systems, and implementing multi-turn dialogue management.

**Optimization and Innovation**: Introducing dynamic chunking strategies to optimize document segmentation, implementing confidence-based safety routing mechanisms, developing citation traceability and verification functions, and exploring advanced technologies such as Self-RAG.

**Experimental Evaluation**: Establishing a comprehensive evaluation system, including retrieval accuracy (Recall@K, Precision@K), generation quality (BLEU, ROUGE-L, BERTScore), hallucination rate, response time, and other metrics, validating the effectiveness of various technical components through comparative experiments.

The project objective is to build a runnable, reproducible, and evaluable medical RAG system, providing technical references and practical experience for AI applications in vertical industries.
