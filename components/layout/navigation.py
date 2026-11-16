import streamlit as st

from app.core.config import USE_LOCAL_MODELS

from components.chat.interface import get_rag
from components.sidebar.upload import render_upload
from components.sidebar.render_workspace import render_workspace
from components.sidebar.model_picker import render_model_picker


def render_navigation():

    rag = get_rag()

    with st.container(border=True):

        render_upload(rag)

        if not USE_LOCAL_MODELS:
            st.caption(
                "Shared workspace: documents uploaded here are "
                "visible to everyone using this app."
            )

        st.divider()

        render_workspace(rag)

        st.divider()

        st.subheader("🔍 Retriever")
        st.info("Hybrid Search")

        st.divider()

        render_model_picker()

        # Filled in by app.py once the page has rendered, so the
        # download always includes the newest answer.
        return st.container()