import streamlit as st

from app.rag_pipeline import RAGPipeline
from components.layout.hero import render_hero


# =====================================================
# RAG
# =====================================================

def get_rag():
    return RAGPipeline()


# =====================================================
# Session State
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# Sources
# =====================================================

def render_sources(sources):

    st.markdown("### 📄 Sources")

    sorted_sources = sorted(
        sources,
        key=lambda chunk: (
            chunk.metadata["source"],
            chunk.metadata["page"],
        ),
    )

    seen = set()

    for source in sorted_sources:

        key = (
            source.metadata["source"],
            source.metadata["page"],
        )

        if key in seen:
            continue

        seen.add(key)

        st.markdown(
            f"""
<div class="source-card">

<b>📄 {source.metadata["source"]}</b>

<br>

Page {source.metadata["page"]}

</div>
""",
            unsafe_allow_html=True,
        )


# =====================================================
# Metrics
# =====================================================

def render_metrics(latency, chunks):

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Latency",
            f"{latency:.2f}s",
        )

    with col2:
        st.metric(
            "Retrieved Chunks",
            chunks,
        )


# =====================================================
# Chat Page
# =====================================================

def render_chat():

    rag = get_rag()

    if rag.retriever is None:

        st.markdown(
            """
    <div class="hero">

        <div class="hero-icon">📄</div>

        <div class="hero-heading">
            No Knowledge Base
        </div>

        <div class="hero-subheading">
            Upload a PDF to start chatting.
        </div>

    </div>
        """,
            unsafe_allow_html=True,
        )

        return

    if len(st.session_state.messages) == 0:
        render_hero()

    # ==============================================
    # Previous Messages
    # ==============================================

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            if message["role"] == "user":

                st.markdown(
                    f"""
                <div class="user-message">

                {message["content"]}

                </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(message["content"])

                render_sources(message["sources"])

                render_metrics(
                    message["latency"],
                    message["chunks"],
                )

    # ==============================================
    # Custom ChatGPT Style Input
    # ==============================================

    col1, col2 = st.columns([18, 1])

    with col1:

        prompt = st.text_input(
            "",
            placeholder="Ask anything about your documents...",
            label_visibility="collapsed",
            key="chat_prompt",
        )

    with col2:

        send = st.button(
            "➜",
            use_container_width=True,
        )

    if not send or not prompt.strip():
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(
            f"""
            <div class="user-message">

                {prompt}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==============================================
    # Assistant
    # ==============================================

    with st.chat_message("assistant"):

        placeholder = st.empty()

        answer = ""

        # ==============================================
        # Stream Response
        # ==============================================

        for token in rag.stream(prompt):

            answer += token

            placeholder.markdown(answer)

        response = rag.last_response

        st.write("")

        render_sources(
            response.sources
        )

        st.write("")

        render_metrics(
            response.latency,
            response.retrieved_chunks,
        )

    # ==============================================
    # Save Conversation
    # ==============================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": response.sources,
            "latency": response.latency,
            "chunks": response.retrieved_chunks,
        }
    )

    # Clear input after sending
    st.session_state.chat_prompt = ""