import streamlit as st


def render_sidebar():

    with st.sidebar:

        st.title("📄 Documents")

        st.file_uploader(
            "Upload PDF",
            type=["pdf"]
        )

        st.divider()

        st.subheader("Statistics")

        st.metric("Retriever", "Hybrid")

        st.metric("Model", "Llama 3.2")
        