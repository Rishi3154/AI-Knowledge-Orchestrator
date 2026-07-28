"""
retriever.py

Retrieves the most relevant chunks from ChromaDB.
"""

from typing import List

import chromadb
import ollama

from app.core.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    MIN_RELEVANCE_DISTANCE,
)

from app.core.models import SearchResult


class Retriever:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR)
        )

        self.collection = self.client.get_collection(
            COLLECTION_NAME
        )

    def embed_query(
        self,
        query: str,
    ) -> List[float]:

        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=query,
        )

        return response["embedding"]

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[SearchResult]:

        embedding = self.embed_query(query)

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        results = self.collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
)

        print(results)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved = []

        for doc, meta, distance in zip(
            docs,
            metas,
            distances,
        ):

            if distance > MIN_RELEVANCE_DISTANCE:
                continue

            retrieved.append(
                SearchResult(
                    content=doc,
                    metadata=meta,
                    distance=distance,
                )
            )

        retrieved.sort(
            key=lambda x: x.distance
        )

        return retrieved