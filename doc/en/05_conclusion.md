# Chapter 5: Conclusion

## 5.1 Project Achievement Summary

This project successfully implemented a Retrieval-Augmented Generation (RAG) question-answering system for the medical vertical industry, completing the full development process from requirements analysis, system design, core implementation to experimental evaluation. The system addresses the strict requirements of the medical domain for information accuracy, reliability, and traceability, building a complete technical solution.

In terms of technical implementation, the project successfully integrated multiple key technologies. The hybrid retrieval strategy combines vector semantic retrieval with BM25 keyword retrieval, fully leveraging the advantages of both retrieval methods through weighted fusion mechanisms, achieving complementarity in two dimensions: medical professional terminology matching and semantic understanding. The introduction of cross-encoder reranking models significantly improved retrieval precision, effectively solving the matching problems of medical terminology synonyms, abbreviations, and complex expressions through fine-grained query-document interaction modeling. The confidence evaluation and safety routing mechanisms provide important safety guarantees for the system, refusing to generate responses in high-uncertainty scenarios and guiding users to consult professional medical personnel through multi-level threshold control, reducing medical risks. The citation traceability function achieves response traceability, allowing users to verify the authority and timeliness of information sources, enhancing system transparency and credibility.

In terms of system performance, through optimizing retrieval strategies and prompt engineering, the system effectively controls hallucination rate, achieving significant improvements compared to baseline large language models. The multi-turn dialogue management function supports contextually coherent interactive experiences, and the dynamic chunking strategy achieves a good balance between retrieval granularity and context completeness. The web interface provides intuitive user interaction, supporting configuration adjustments and mode switching to meet different usage scenario requirements.

In terms of engineering practice, the project adopts modular design, with components interacting through clear interfaces, facilitating maintenance and extension. The centralized configuration management mechanism supports flexible parameter adjustment, optimizing system performance without code modification. The comprehensive logging and evaluation framework provides strong support for system debugging and performance analysis.

## 5.2 Technical Innovations

The technical innovations of this project are mainly reflected in the following aspects:

First, the two-stage strategy of hybrid retrieval and reranking. By combining coarse-grained hybrid retrieval with fine-grained cross-encoder reranking, retrieval quality is significantly improved while ensuring retrieval efficiency. This hierarchical retrieval architecture demonstrates obvious advantages in the medical professional domain, effectively handling complex matching requirements for medical terminology.

Second, the confidence-based safety routing mechanism. Addressing the specificity of medical applications, the system introduces multi-level confidence evaluation and safety routing strategies, proactively refusing to generate responses when retrieval quality is insufficient, avoiding the risk of hallucinations in knowledge blind spots of large language models. This design philosophy of "knowing what you know and knowing what you don't know" has important value in high-risk fields such as healthcare.

Third, the complete citation traceability system. Through citation number tracking, context annotation, and source information display, the system establishes a complete traceability chain from retrieved documents to generated responses, allowing users to verify the source of each key information point, enhancing system transparency and credibility.

Fourth, prompt engineering optimization for the medical domain. Through carefully designed system role definitions, behavioral rules, and context organization methods, the system guides large language models to generate responses that conform to medical ethics and professional standards, achieving a good balance between accuracy, reliability, and safety.

## 5.3 Challenges and Solutions

During project implementation, the team faced multiple technical challenges and found effective solutions through in-depth analysis and repeated experimentation.

The first challenge was retrieval quality. Medical text contains大量 professional terminology and complex expressions, making it difficult for a single retrieval method to comprehensively capture relevant information. By introducing a hybrid retrieval strategy, combining the generalization ability of vector semantic retrieval with the precise matching ability of BM25 keyword retrieval, this problem was effectively solved. The introduction of cross-encoder reranking further improved retrieval precision, and through experiments verifying the impact of different configuration parameters on retrieval effectiveness, the optimal weight configuration was found.

The second challenge was hallucination control. Large language models are prone to generating false information when lacking sufficient context, which is unacceptable in the medical field. Through the confidence evaluation mechanism, the system can identify scenarios with insufficient retrieval quality and proactively refuse to generate responses. Simultaneously, through prompt engineering strengthening the model's dependence on context, requiring the model to acknowledge uncertainty when information is insufficient, the hallucination rate was effectively reduced.

The third challenge was balancing system performance and quality. Although cross-encoder reranking improved retrieval quality, it increased computational overhead. Through the "coarse ranking + fine ranking" two-stage strategy, reranking is performed after hybrid retrieval returns Top-20 candidate documents, finally returning Top-5 results, controlling computational costs while ensuring retrieval quality. The dynamic chunking strategy also found a balance point between retrieval granularity and context completeness.

The fourth challenge was deployment environment adaptation. The system needs to support both CPU and GPU operating modes, adapting to different hardware environments. Through PyTorch's automatic device detection mechanism and conditional loading strategy, the system can automatically select the optimal operating mode based on available hardware, ensuring functional completeness in CPU environments and fully utilizing hardware acceleration to improve performance in GPU environments.

## 5.4 Experience and Reflections

Through the development practice of this project, the team accumulated valuable engineering experience and also identified areas for improvement.

In terms of technology selection, choosing mature open-source components and frameworks reduced development difficulty but also brought some limitations. For example, FAISS's performance on large-scale datasets needs further optimization, and approximate nearest neighbor search algorithms could be considered. The selection of embedding models and reranking models was based on performance evaluations at the time, and with the continuous emergence of new models, more advanced models can be explored in the future to improve system performance.

In terms of system design, modular design brought good maintainability and scalability, but interface design between modules still needs further optimization. Some modules have high coupling, requiring modification of multiple modules when extending functionality, increasing development costs. In the future, clearer layered architecture and dependency injection mechanisms can be adopted to further reduce coupling between modules.

In terms of evaluation system, the project established a comprehensive evaluation framework covering multiple dimensions including retrieval quality, generation quality, and system performance. However, the scale and diversity of evaluation datasets still have room for improvement, and more standard datasets from the medical field can be introduced in the future for more comprehensive performance evaluation. Simultaneously, manual evaluation and user research can be introduced to evaluate system usability and satisfaction from actual usage perspectives.

In terms of user experience, the web interface provides basic interaction functions, but there is still room for improvement in visualization and interaction details. For example, visualization of the retrieval process can be added to help users understand system working principles; richer configuration options can be provided to meet personalized needs of different users; response speed and loading performance can be optimized to improve user experience.

## 5.5 Future Outlook

The completion of this project provides a feasible technical path for AI applications in the medical vertical field, but there are still many directions worth exploring.

In terms of technical optimization, more advanced retrieval techniques can be introduced, such as query expansion, pseudo-relevance feedback, and document reranking algorithms, to further improve retrieval quality. The combination of knowledge graphs and large language models can be explored, enhancing system reasoning capabilities through structured knowledge. More powerful embedding models and reranking models can be introduced to improve semantic understanding and matching precision. Vector index structures can be optimized to support larger-scale knowledge bases and more efficient retrieval.

In terms of function expansion, multi-modal information processing can be supported, including understanding and generation of non-text information such as medical images and medical record charts. Personalized recommendation mechanisms can be introduced to provide customized responses based on users' historical queries and preferences. Multi-language interaction can be supported to meet the needs of internationalized medical scenarios. Collaboration mechanisms can be introduced to support medical experts in reviewing and correcting system responses, establishing a human-machine collaborative knowledge update mechanism.

In terms of application expansion, the system can be deployed to actual medical institutions for clinical validation and effectiveness evaluation. It can be expanded to more medical sub-fields, such as drug development, medical education, and health management. It can be integrated with other medical information systems to build a complete intelligent medical ecosystem.

In terms of safety and ethics, continuous attention needs to be paid to ethical issues in medical AI, ensuring system use complies with medical ethical standards and data privacy requirements. More comprehensive risk control mechanisms need to be established to prevent system abuse or misuse. Transparent accountability mechanisms need to be established to clarify responsibility boundaries between system developers, deployers, and users.

Overall, this project has laid a solid foundation for RAG technology applications in the medical vertical field. Through systematic design, implementation, and evaluation, it has verified the feasibility and effectiveness of retrieval-augmented generation technology in medical question-answering scenarios. In the future, with continuous technological advancement and in-depth application expansion, intelligent medical question-answering systems will play an increasingly important role in improving medical service quality, promoting medical knowledge popularization, and reducing medical costs.
