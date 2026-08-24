"""
vector_retriever.py

Semantic retrieval using ChromaDB.
"""

from typing import List

import chromadb

from app.core.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    TOP_K,
)
from app.core.models import SearchResult
from app.embeddings.embedder import create_embedder


class VectorRetriever:
    """
    Retrieves relevant chunks using vector similarity search.
    """

    def __init__(self):

        self.embedder = create_embedder()

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR)
        )

        self.collection = self.client.get_or_create_collection(
        COLLECTION_NAME)

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[SearchResult]:

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

        return retrieved