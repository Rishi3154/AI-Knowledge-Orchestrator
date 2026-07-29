import streamlit as st
from app.services.knowledge_base import KnowledgeBase
from app.services.workspace import Workspace


def render_workspace(rag):

    workspace = Workspace()

    stats = workspace.stats()

    st.subheader("📚 Workspace")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Documents",
            stats["documents"],
        )

    with col2:
        st.metric(
            "Chunks",
            stats["chunks"],
        )

    st.divider()

    documents = workspace.documents()

    if not documents:

        st.caption("No PDFs uploaded")

        return


    kb = KnowledgeBase()

    for document in documents:

        col1, col2 = st.columns([9, 1.5])

        with col1:
          st.write(f"📄 {document}")

        with col2:
            if st.button(
                "",
                icon=":material/delete:",
                key=f"delete_{document}",
                use_container_width=True,
            ):
                kb.delete_document(document)
                rag.reload()
                st.rerun()