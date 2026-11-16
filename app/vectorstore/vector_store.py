"""
vector_store.py

Stores chunks and their embeddings inside ChromaDB.

This is the only place chunks are persisted. The keyword index and
the workspace statistics are both built from what is stored here,
so they can never disagree with the vectors.
"""

import chromadb

from app.core.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
)
from app.core.models import Chunk


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME
        )

    def add_chunks(
        self,
        chunks,
        embeddings,
    ):

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            ids.append(chunk.chunk_id)

            documents.append(chunk.content)

            metadatas.append(chunk.metadata)

        # ChromaDB rejects a single write above its batch limit,
        # which a long book can exceed.
        batch_size = self.client.get_max_batch_size()

        for start in range(0, len(ids), batch_size):

            end = start + batch_size

            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

    def count(self):

        return self.collection.count()

    def all_chunks(self) -> list[Chunk]:
        """
        Every stored chunk, without its embedding.
        """

        stored = self.collection.get(
            include=["documents", "metadatas"],
        )

        return [
            Chunk(
                chunk_id=chunk_id,
                content=content,
                metadata=metadata,
            )
            for chunk_id, content, metadata in zip(
                stored["ids"],
                stored["documents"],
                stored["metadatas"],
            )
        ]

    def has_document(self, document: str) -> bool:

        found = self.collection.get(
            where={
                "source": document,
            },
            limit=1,
            include=[],
        )

        return bool(found["ids"])

    def delete_collection(self):

        try:

            self.client.delete_collection(
                COLLECTION_NAME
            )

        except Exception:

            pass

        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME
        )

    def delete_document(self,document: str,):
        """
        Delete all vectors belonging to a document.
        """

        self.collection.delete(
            where={
                "source": document,
            }
        )