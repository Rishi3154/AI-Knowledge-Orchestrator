import streamlit as st


def render_about():

    st.title("🤖 AI Knowledge Orchestrator")

    st.caption(
        "A fully local Retrieval-Augmented Generation (RAG) application "
        "for querying PDF documents using Large Language Models with "
        "grounded responses and source attribution."
    )

    st.divider()

    # ----------------------------------------------------------
    # Features
    # ----------------------------------------------------------

    st.subheader("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.success("🧠 Local LLM Inference")
        st.write(
            "Run AI models entirely offline using Ollama for privacy "
            "and fast local inference."
        )

        st.success("📄 PDF Knowledge Base")
        st.write(
            "Upload and index multiple PDF documents into a persistent "
            "knowledge base."
        )

        st.success("🔍 Hybrid Retrieval")
        st.write(
            "Combine semantic vector search with BM25 keyword retrieval "
            "using Reciprocal Rank Fusion."
        )

    with col2:
        st.success("📚 Source Attribution")
        st.write(
            "Every response is grounded using retrieved document chunks "
            "with source references."
        )

        st.success("⚡ Streaming Responses")
        st.write(
            "Receive answers in real time with streamed LLM generation."
        )

        st.success("🗂 Workspace Management")
        st.write(
            "View indexed documents, monitor statistics, and manage "
            "your knowledge base."
        )

    st.divider()

    # ----------------------------------------------------------
    # Tech Stack
    # ----------------------------------------------------------

    st.subheader("🛠 Technology Stack")

    st.table(
        {
            "Layer": [
                "Programming Language",
                "Frontend",
                "Large Language Model",
                "Embedding Model",
                "Vector Database",
                "PDF Processing",
                "Retrieval",
                "Ranking",
            ],
            "Technology": [
                "Python",
                "Streamlit",
                "Ollama",
                "nomic-embed-text",
                "ChromaDB",
                "PyMuPDF",
                "Hybrid (Vector + BM25)",
                "Reciprocal Rank Fusion (RRF)",
            ],
        }
    )

    st.divider()

    # ----------------------------------------------------------
    # Project Highlights
    # ----------------------------------------------------------

    st.subheader("🚀 Project Highlights")

    st.markdown(
        """
- ✅ Fully Local AI Pipeline
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Persistent Knowledge Base
- ✅ Hybrid Retrieval Strategy
- ✅ Streaming Chat Interface
- ✅ Modular Software Architecture
- ✅ Source-Grounded Responses
- ✅ Production-Inspired Design
"""
    )

    st.divider()

    # ----------------------------------------------------------
    # Project Structure
    # ----------------------------------------------------------

    st.subheader("📂 Project Structure")

    st.code(
        """
app/
│
├── core/
├── embeddings/
├── indexing/
├── ingestion/
├── llm/
├── memory/
├── retrieval/
├── services/
├── utils/
├── vectorstore/
│
│
components/
│
├── assets/
├── chat/
├── layout/
├── pages/
├── sidebar/
├── chat.py
├── sidebar_panel.py
├── styles.css
├── theme.py
├── visualizer.py
│
│
data/
│
├── index/
├── upload/
│
│
app.py
chat.py
index.py
requirements.txt

""",
        language="text",
    )

    st.divider()

    # ----------------------------------------------------------
    # Future Improvements
    # ----------------------------------------------------------

    st.subheader("🛣 Future Roadmap")

    st.markdown(
        """
- OCR support for scanned PDF documents
- Metadata-based filtering and search
- Docker deployment
- Conversation memory
- Multi-document comparison
- Cloud model support
"""
    )