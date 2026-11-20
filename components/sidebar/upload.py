from app.services.upload_services import UploadService
from app.indexing.document_ingester import DocumentIngestor
from app.services.knowledge_base import KnowledgeBase
import streamlit as st

from app.core.config import SUPPORTED_EXTENSIONS

def render_upload(rag):

    st.subheader("📤 Upload Document")

    uploaded = st.file_uploader(
        "Choose a document",
        type=[
            extension.lstrip(".")
            for extension in SUPPORTED_EXTENSIONS
        ],
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

            # Same name, different content: drop the old version so
            # the new one replaces it instead of being skipped.
            replaced = upload_service.is_new_version(uploaded)

            if replaced:
                KnowledgeBase().delete_document(
                    upload_service.destination(uploaded).name
                )

                rag.reload()

            file_path = upload_service.save(uploaded)

            try:

                ingestor = DocumentIngestor()

                response = ingestor.ingest(file_path)

            except Exception as error:

                # Nothing was indexed, so don't leave the file behind.
                if file_path.name not in rag.documents:
                    file_path.unlink(missing_ok=True)

                st.error(
                    "Indexing failed. The embedding model downloads "
                    "on first use, so check the internet connection "
                    f"and try again.\n\n{error}"
                )

                return

            if response.indexed:

                rag.reload()

                st.success(
                    f"✅ Indexed {response.document}"
                )

                st.rerun()

            elif response.reason == "no_text":

                # Nothing usable was stored, so don't keep the file.
                file_path.unlink(missing_ok=True)

                st.warning(
                    "No text could be read from this file. If it is "
                    "a PDF, it is probably a scan or made of images, "
                    "which isn't supported yet."
                )

            else:

                st.info(
                    f"{response.document} is already indexed."
                )