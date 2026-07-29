from app.services.upload_services import UploadService
from app.indexing.document_ingester import DocumentIngestor
import streamlit as st

def render_upload(rag):

    st.subheader("📤 Upload PDF")

    uploaded = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        return

    if st.button(
        "Index Document",
        use_container_width=True,
    ):

        with st.spinner("Indexing document..."):

            upload_service = UploadService()

            pdf_path = upload_service.save(uploaded)

            ingestor = DocumentIngestor()

            response = ingestor.ingest(pdf_path)

            if response.indexed:

                rag.reload()

                st.success(
                    f"✅ Indexed {response.document}"
                )

                st.rerun()

            else:

                st.warning(
                    "Document already exists or contains no extractable text."
                )