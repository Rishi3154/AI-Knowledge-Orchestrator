"""
bm25_retriever.py

Keyword-based retrieval using BM25.
"""

from typing import List

from rank_bm25 import BM25Okapi

from app.core.models import Chunk, SearchResult
from app.utils.text import tokenize


class BM25Retriever:

    def __init__(self):

        self.documents: List[Chunk] = []
        self.bm25 = None

    def build(
        self,
        chunks: List[Chunk],
    ) -> None:

        self.documents = chunks

        corpus = [
            chunk.content.split()
            for chunk in chunks
            ]

        if not corpus:
            self.bm25 = None
            self.chunks = []
            return

        self.bm25 = BM25Okapi(corpus)
        self.chunks = chunks

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:

        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(
            tokenize(query)
        )

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []

        for chunk, score in ranked[:top_k]:

            results.append(
                SearchResult(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    distance=-float(score),
                )
            )

        return results