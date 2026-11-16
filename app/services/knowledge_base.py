"""
knowledge_base.py

Knowledge base management.
"""

from app.core.config import UPLOAD_DIR

from app.vectorstore.vector_store import VectorStore


class KnowledgeBase:

    def __init__(self):

        self.vector_store = VectorStore()

    def delete_document(
        self,
        document: str,
    ):

        # Delete chunks and vectors
        self.vector_store.delete_document(document)

        # Delete the uploaded file
        uploaded = UPLOAD_DIR / document

        if uploaded.exists():
            uploaded.unlink()
