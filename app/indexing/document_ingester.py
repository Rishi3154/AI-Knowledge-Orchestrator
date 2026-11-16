"""
document_ingester.py

End-to-end document ingestion pipeline.
"""

from pathlib import Path

from app.ingestion.document_loader import load_document
from app.ingestion.chunker import DocumentChunker
from app.embeddings.embedder import create_embedder
from app.vectorstore.vector_store import VectorStore

from app.core.models import IngestionResponse


class DocumentIngestor:

    def __init__(self):

        self.chunker = DocumentChunker()
        self.embedder = create_embedder()
        self.vector_store = VectorStore()

    def ingest(
        self,
        path: Path,
    ) -> IngestionResponse:

        # -------------------------
        # Already Indexed?
        # -------------------------

        if self.vector_store.has_document(path.name):

            return IngestionResponse(
                indexed=False,
                pages=0,
                chunks=0,
                document=path.name,
                reason="duplicate",
            )

        # -------------------------
        # Load Document
        # -------------------------

        documents = load_document(path)

        if not documents:

            return IngestionResponse(
                indexed=False,
                pages=0,
                chunks=0,
                document=path.name,
                reason="no_text",
            )

        # -------------------------
        # Chunk
        # -------------------------

        chunks = self.chunker.split(documents)

        # -------------------------
        # Create Embeddings
        # -------------------------

        embeddings = self.embedder.embed_many(
            [chunk.content for chunk in chunks]
        )

        # -------------------------
        # Store in ChromaDB
        # -------------------------

        try:

            self.vector_store.add_chunks(
                chunks,
                embeddings,
            )

        except Exception:

            # Don't leave half a document behind: it would count as
            # already indexed and could never be completed.
            self.vector_store.delete_document(path.name)

            raise

        return IngestionResponse(
            indexed=True,
            pages=len(documents),
            chunks=len(chunks),
            document=path.name,
        )
