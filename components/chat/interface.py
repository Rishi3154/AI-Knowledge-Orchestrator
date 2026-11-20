import re
from html import escape

import streamlit as st

from app.llm.citations import cited_sources
from app.rag_pipeline import RAGPipeline
from components.layout.hero import render_hero
from app.llm.llm import MISSING_KEY_MESSAGE, has_api_key
from components.sidebar.model_picker import selected_llm


# =====================================================
# RAG
# =====================================================

@st.cache_resource
def get_rag():
    """
    Build the pipeline once and reuse it across Streamlit reruns.

    Without the cache, every click reloaded all chunks, rebuilt the
    BM25 index and reopened ChromaDB. Call rag.reload() after the
    knowledge base changes.
    """
    return RAGPipeline()


def render_user_message(text):
    """
    Render a user message in the styled bubble.

    The text is escaped so anything that looks like HTML is shown
    as plain text instead of being rendered by the browser.
    """

    safe = escape(text).replace("\n", "<br>")

    st.markdown(
        f'<div class="user-message">{safe}</div>',
        unsafe_allow_html=True,
    )


# =====================================================
# Session State
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# Sources
# =====================================================

def render_sources(answer, sources):
    """
    List the sources the answer cites, numbered to match its [n]
    markers; each one opens to show the passage itself. If it cites nothing, the retrieved passages are listed
    under a different heading so they aren't mistaken for evidence.
    """

    if not sources:
        return

    cited = cited_sources(answer, sources)

    if cited:
        st.markdown("### 📄 Sources")
    else:
        st.markdown("### 🔍 Passages searched (none cited)")
        cited = list(enumerate(sources, start=1))

    for number, source in cited:

        # Opening a source shows the exact passage the model was
        # given, so a claim can be checked against the document.
        with st.expander(
            f"\\[{number}\\] 📄 {escape_markdown(source.location)}"
        ):
            st.text(source.content)


def escape_markdown(text):
    """
    Stop characters in a file name (such as _ or *) from being
    read as Markdown formatting.
    """

    return re.sub(r"([\\`*_{}\[\]()#+\-.!|~>$:])", r"\\\1", text)


def render_search_query(question, search_query):
    """
    Tell the user when a follow-up was reworded before searching.
    """

    if search_query and search_query != question:
        st.caption(f"Searched for: {search_query}")


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
    if "messages" not in st.session_state:
         st.session_state.messages = []
    rag = get_rag()

    if rag.retriever is None:

        st.write("📄 No Knowledge Base")
        st.write("Upload a document to start chatting.")

        return

    # The hero lives in a placeholder so it can be removed as soon
    # as the first question is asked.
    hero = st.empty()

    if len(st.session_state.messages) == 0:
        with hero.container():
            render_hero()

    # The conversation is drawn inside this container so new messages
    # appear above the input box, not below it.
    conversation = st.container()

    # ==============================================
    # Previous Messages
    # ==============================================

    with conversation:

        question = ""

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                if message["role"] == "user":

                    question = message["content"]

                    render_user_message(question)

                else:

                    render_search_query(
                        question,
                        message.get("search_query", ""),
                    )

                    st.markdown(message["content"])

                    render_sources(
                        message["content"],
                        message["sources"],
                    )

                    render_metrics(
                        message["latency"],
                        message["chunks"],
                    )

    # ==============================================
    # Chat Input (Enter sends, box clears itself)
    # ==============================================

    # Documents can be indexed without a key, but answering needs one.
    if not has_api_key():

        st.warning(MISSING_KEY_MESSAGE)

        st.chat_input(
            "Add an Anthropic API key to ask questions",
            disabled=True,
        )

        return

    prompt = st.chat_input(
        "Ask anything about your documents...",
    )

    if not prompt or not prompt.strip():
        return

    hero.empty()

    # Earlier turns, so a follow-up question can be understood.
    history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.messages
    ]

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with conversation:

        with st.chat_message("user"):

            render_user_message(prompt)

        # ==============================================
        # Assistant
        # ==============================================

        with st.chat_message("assistant"):

            query_note = st.empty()

            placeholder = st.empty()

            answer = ""

            stream = rag.stream(
                prompt,
                history,
                llm=selected_llm(),
            )

            try:

                for token in stream:

                    answer += token

                    placeholder.markdown(answer)

            except Exception as error:

                # Drop the unanswered question so the history never
                # holds a user message without a reply.
                st.session_state.messages.pop()

                placeholder.empty()

                st.error(
                    "Couldn't generate an answer. Check that the "
                    "Anthropic API is reachable and ANTHROPIC_API_KEY "
                    "is valid, then try "
                    f"again.\n\n{error}"
                )

                return

            response = stream.response

            with query_note:
                render_search_query(
                    prompt,
                    response.search_query,
                )

            st.write("")

            render_sources(
                answer,
                response.sources,
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
            "search_query": response.search_query,
            "latency": response.latency,
            "chunks": response.retrieved_chunks,
        }
    )
