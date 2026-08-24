"""
retriever.py

Semantic retrieval from ChromaDB.
"""

from typing import List

from app.core.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    TOP_K,
)
from app.core.models import SearchResult
from app.embeddings.embedder import create_embedder

import chromadb


class Retriever:
    """
    Retrieves the most relevant chunks for a user query.
    """

    def __init__(self) -> None:

        self.embedder = create_embedder()

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR)
        )

        self.collection = self.client.get_collection(
            COLLECTION_NAME
        )

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[SearchResult]:
        """
        Retrieve the most relevant chunks.
        """

        embedding = self.embedder.embed(query)

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):

            retrieved.append(
                SearchResult(
                    content=document,
                    metadata=metadata,
                    distance=float(distance),
                )
            )

        retrieved.sort(key=lambda x: x.distance)

        return retrieved