"""
document_ingester.py

End-to-end document ingestion pipeline.
"""

from pathlib import Path

from app.ingestion.document_loader import PDFLoader
from app.ingestion.chunker import DocumentChunker
from app.embeddings.embedder import create_embedder
from app.indexing.chunk_index import ChunkIndex
from app.vectorstore.vector_store import VectorStore

from app.core.models import IngestionResponse
from app.core.config import (
    CHUNK_INDEX_FILE
)


class DocumentIngestor:

    def __init__(self):

        self.loader = PDFLoader
        self.chunker = DocumentChunker()
        self.embedder = create_embedder()
        self.vector_store = VectorStore()
        self.chunk_index = ChunkIndex(CHUNK_INDEX_FILE)

    def ingest(
        self,
        pdf_path: Path,
    ) -> IngestionResponse:

        # -------------------------
        # Load PDF
        # -------------------------

        documents = self.loader(pdf_path).load()

        if not documents:

            return IngestionResponse(
                indexed=False,
                pages=0,
                chunks=0,
                document=pdf_path.name,
            )

        # -------------------------
        # Chunk
        # -------------------------

        chunks = self.chunker.split(documents)

        # -------------------------
        # Existing Chunks
        # -------------------------

        existing_chunks = self.chunk_index.load()

        existing = {
            (
                chunk.metadata["source"],
                chunk.metadata["page"],
                chunk.content,
            )
            for chunk in existing_chunks
        }

        new_chunks = []

        for chunk in chunks:

            key = (
                chunk.metadata["source"],
                chunk.metadata["page"],
                chunk.content,
            )

            if key not in existing:
                new_chunks.append(chunk)

        if not new_chunks:

            return IngestionResponse(
                indexed=False,
                pages=len(documents),
                chunks=0,
                document=pdf_path.name,
            )

        # -------------------------
        # Create Embeddings
        # -------------------------

        embeddings = self.embedder.embed_many(
            [chunk.content for chunk in new_chunks]
        )

        # -------------------------
        # Store in ChromaDB
        # -------------------------

        self.vector_store.add_chunks(
            new_chunks,
            embeddings,
        )

        # -------------------------
        # Update Chunk Index
        # -------------------------

        existing_chunks.extend(new_chunks)

        self.chunk_index.save(existing_chunks)

        return IngestionResponse(
            indexed=True,
            pages=len(documents),
            chunks=len(new_chunks),
            document=pdf_path.name,
        )