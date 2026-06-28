"""Streamlit UI for Medical RAG System"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline import MedicalChatEngine, SOTAMedicalChatEngine
from src.pipeline.hybrid_retriever import HybridRetriever
from src.pipeline.reranker import Reranker
from src.pipeline.confidence_scorer import ConfidenceScorer
from src.pipeline.citation_tracker import CitationTracker
from src.knowledge import FaissIndexBuilder, EmbeddingModel
from config import config


@st.cache_resource
def load_chat_engine(use_sota=False):
    """Load chat engine with caching"""
    with st.spinner("Loading models... This may take a few minutes on first run."):
        # Load embedding model
        embedding_model = EmbeddingModel()
        
        # Load FAISS index
        faiss_index = FaissIndexBuilder(dimension=embedding_model.dimension)
        try:
            faiss_index.load()
        except FileNotFoundError:
            st.warning("FAISS index not found. Please build the index first.")
            return None
        
        # Initialize components
        retriever = HybridRetriever(faiss_index, embedding_model)
        reranker = Reranker()
        confidence_scorer = ConfidenceScorer()
        citation_tracker = CitationTracker()
        
        # Create chat engine
        if use_sota:
            chat_engine = SOTAMedicalChatEngine(
                retriever=retriever,
                reranker=reranker,
                confidence_scorer=confidence_scorer,
                citation_tracker=citation_tracker,
                enable_agentic_rag=True,
                enable_discrepancy_refinement=True,
                enable_citation_verification=True,
            )
        else:
            chat_engine = MedicalChatEngine(
                retriever=retriever,
                reranker=reranker,
                confidence_scorer=confidence_scorer,
                citation_tracker=citation_tracker,
            )
        
        return chat_engine


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Medical RAG System",
        page_icon="🏥",
        layout="wide",
    )
    
    st.title("🏥 Medical RAG System")
    st.markdown("**Retrieval-Augmented Generation for Medical QA**")
    st.markdown("*Yunnan University Graduate Course Project*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # SOTA mode toggle
        use_sota = st.checkbox(
            "🚀 Enable SOTA Pipeline", 
            value=False,
            help="Enable advanced features: Agentic Graph RAG, Discrepancy Refinement, Citation Verification"
        )
        
        use_rag = st.checkbox("Use RAG", value=True)
        show_confidence = st.checkbox("Show Confidence Score", value=True)
        show_citations = st.checkbox("Show Citations", value=True)
        
        if st.button("🔄 Reset Conversation"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        # Display mode information
        if use_sota:
            st.success("**SOTA Mode Active**")
            st.markdown("""
            **Advanced Features:**
            - ✅ Agentic Graph RAG
            - ✅ Multi-Source Retrieval
            - ✅ Discrepancy Refinement
            - ✅ Citation Verification
            """)
        else:
            st.info("**Baseline Mode**")
            st.markdown("""
            Standard RAG pipeline with:
            - Hybrid retrieval
            - Reranking
            - Confidence scoring
            """)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This system uses:
        - **BGE-M3** for embeddings
        - **BGE-Reranker-v2-m3** for reranking
        - **Qwen3:8b** via Ollama for generation
        - **FAISS** for vector search
        - **BM25** for sparse retrieval
        """)
    
    # Load chat engine with appropriate mode
    chat_engine = load_chat_engine(use_sota=use_sota)
    
    if chat_engine is None:
        st.error("Failed to load chat engine. Please check your configuration.")
        return
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show confidence score
                if show_confidence and message["role"] == "assistant" and "confidence" in message:
                    confidence = message["confidence"]
                    with st.expander("📊 Confidence Score"):
                        st.metric("Overall Confidence", f"{confidence['confidence']:.2%}")
                        st.progress(confidence["confidence"])
                        st.write(f"**Status:** {confidence['reason']}")
                
                # Show citations
                if show_citations and message["role"] == "assistant" and "citations" in message:
                    citations = message["citations"]
                    if citations:
                        with st.expander(f"📚 Citations ({len(citations)} sources)"):
                            for citation in citations:
                                st.markdown(f"**[{citation['id']}] {citation['source']}**")
                                st.markdown(f"*Score: {citation['score']}*")
                                st.markdown(citation['text_preview'])
                                st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a medical question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Generate response
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_engine.chat(prompt, use_rag=use_rag)
                    
                    st.markdown(response["response"])
                    
                    # Show confidence
                    if show_confidence:
                        confidence = response["confidence"]
                        with st.expander("📊 Confidence Score"):
                            st.metric("Overall Confidence", f"{confidence['confidence']:.2%}")
                            st.progress(confidence["confidence"])
                            st.write(f"**Status:** {confidence['reason']}")
                    
                    # Show citations
                    if show_citations and response["citations"]:
                        citations = response["citations"]
                        with st.expander(f"📚 Citations ({len(citations)} sources)"):
                            for citation in citations:
                                st.markdown(f"**[{citation['id']}] {citation['source']}**")
                                st.markdown(f"*Score: {citation['score']}*")
                                st.markdown(citation['text_preview'])
                                st.markdown("---")
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["response"],
            "confidence": response["confidence"],
            "citations": response["citations"],
        })


if __name__ == "__main__":
    main()
