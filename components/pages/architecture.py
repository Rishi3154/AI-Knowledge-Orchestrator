import streamlit as st

from app.services.workspace import Workspace


def render_architecture():

    # ==========================================================
    # Header
    # ==========================================================
    workspace = Workspace()
    st.title("🧠 AI Pipeline Architecture")

    st.caption(
        "End-to-End Retrieval-Augmented Generation (RAG) system "
        "built using Local LLMs, Hybrid Retrieval, and Persistent Knowledge Storage."
    )

    # ==========================================================
    # System Overview
    # ==========================================================

    st.subheader("📊 System Overview")

    stats = workspace.stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📄 Documents",
            stats["documents"],
            help="Number of indexed PDF documents",
        )

    with col2:
        st.metric(
            "🧩 Chunks",
            stats["chunks"],
            help="Total indexed document chunks",
        )

    with col3:
        st.metric(
            "🤖 Local LLM",
            "Llama 3.2",
            help="Running locally through Ollama",
        )

    with col4:
        st.metric(
            "🔍 Retrieval",
            "Hybrid",
            help="Vector Search + BM25",
        )

    st.divider()

    # ==========================================================
    # Architecture Diagram
    # ==========================================================

    st.subheader("🏗 System Architecture")

    st.image(
        "components/assets/new.png",
        use_container_width=True,
    )

    st.caption(
        "The architecture separates the offline document indexing pipeline "
        "from the online question-answering pipeline to enable efficient and "
        "accurate Retrieval-Augmented Generation (RAG)."
    )

    st.divider()

    # ==========================================================
    # Component Explorer
    # ==========================================================

    st.subheader("📦 Component Explorer")

    with st.expander("📄 PDF Loader", expanded=False):

        st.markdown("""
### Purpose

Extract textual content from uploaded PDF documents.

### Technology

- PyMuPDF

### Output

Raw document text ready for chunking.
""")

    with st.expander("✂️ Document Chunker"):

        st.markdown("""
### Purpose

Split large documents into overlapping chunks suitable for semantic retrieval.

### Configuration

- Chunk Size: **500**
- Chunk Overlap: **100**

### Why?

Overlapping chunks preserve context across chunk boundaries, improving retrieval quality.
""")

    with st.expander("🧠 Embedding Model"):

        st.markdown("""
### Model

- nomic-embed-text

### Purpose

Transform every chunk into a dense vector representation that captures semantic meaning.

### Output

Embeddings stored inside ChromaDB.
""")

    with st.expander("🗄️ ChromaDB"):

        st.markdown("""
### Purpose

Persistent vector database used to store:

- Dense embeddings
- Chunk metadata
- Source references

Provides efficient semantic similarity search.
""")

    with st.expander("🔍 Hybrid Retrieval"):

        st.markdown("""
### Retrieval Strategy

1. Dense Vector Search
2. BM25 Keyword Search
3. Reciprocal Rank Fusion (RRF)

### Why Hybrid?

Combining semantic and lexical retrieval improves recall and ranking quality compared to either approach alone.
""")

    with st.expander("📝 Prompt Builder"):

        st.markdown("""
### Purpose

Construct the final prompt using:

- System instructions
- Retrieved context
- User question

This ensures grounded and context-aware responses.
""")

    with st.expander("🤖 Ollama LLM"):

        st.markdown("""
### Current Model

- Llama 3.2:latest

### Features

- Local inference
- Streaming responses
- Privacy-first execution
""")

    with st.expander("📚 Response Generation"):

        st.markdown("""
### Final Output

Each response includes:

- Grounded answer
- Source citations
- Streaming generation
- Context-aware reasoning
""")

    st.divider()

    # ==========================================================
    # Design Principles
    # ==========================================================

    st.subheader("🏛 Design Principles")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("🧠 Local AI")
        st.success("📦 Modular Design")

    with col2:
        st.success("🔍 Hybrid Retrieval")
        st.success("💾 Persistent Storage")

    with col3:
        st.success("📚 Grounded Responses")
        st.success("⚡ Streaming Generation")

    st.divider()

    # ==========================================================
    # Footer
    # ==========================================================

    st.info(
        "This architecture follows a production-inspired Retrieval-Augmented Generation (RAG) "
        "design, separating document ingestion from query-time retrieval and response generation "
        "to deliver accurate, explainable, and context-aware answers."
    )