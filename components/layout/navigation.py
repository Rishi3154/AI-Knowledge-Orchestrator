import streamlit as st

from components.chat.interface import get_rag
from components.sidebar.upload import render_upload
from components.sidebar.render_workspace import render_workspace


def render_navigation():

    rag = get_rag()

    with st.container(border=True):

        render_upload(rag)

        st.divider()

        render_workspace(rag)

        st.divider()

        st.subheader("🔍 Retriever")
        st.info("Hybrid Search")

        st.divider()

        st.subheader("🤖 Model")
        st.success("Llama 3.2")