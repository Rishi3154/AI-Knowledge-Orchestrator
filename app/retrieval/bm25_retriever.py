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

        # Documents and queries must be tokenized the same way,
        # otherwise "Exam." in a chunk never matches "exam" in a query.
        corpus = [
            tokenize(chunk.content)
            for chunk in chunks
        ]

        if not corpus:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(corpus)

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

            # A score of zero means no query term matched. Returning
            # those would feed arbitrary chunks into the fusion.
            if score <= 0:
                break

            results.append(
                SearchResult(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    distance=-float(score),
                    chunk_id=chunk.chunk_id,
                )
            )

        return results