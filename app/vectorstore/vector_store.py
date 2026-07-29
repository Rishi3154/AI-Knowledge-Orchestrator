"""
vector_store.py

Stores embeddings inside ChromaDB.
"""

import chromadb

from app.core.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
)


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

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self):

        return self.collection.count()

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