"""
knowledge_base.py

Knowledge base management.
"""

from app.core.config import (
    CHUNK_INDEX_FILE,
    UPLOAD_DIR,
)

from app.indexing.chunk_index import ChunkIndex
from app.vectorstore.vector_store import VectorStore


class KnowledgeBase:

    def __init__(self):

        self.chunk_index = ChunkIndex(CHUNK_INDEX_FILE)
        self.vector_store = VectorStore()

    def delete_document(
        self,
        document: str,
    ):

        # Delete vectors
        self.vector_store.delete_document(document)

        # Delete chunks
        self.chunk_index.delete_document(document)

        # Delete PDF
        pdf = UPLOAD_DIR / document

        if pdf.exists():
            pdf.unlink()