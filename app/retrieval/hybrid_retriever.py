from collections import defaultdict
from typing import List

from app.core.models import SearchResult
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:

    def __init__(self, bm25: BM25Retriever):

        self.vector = VectorRetriever()
        self.bm25 = bm25

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[SearchResult]:

        # Retrieve more candidates than we finally return
        vector_results = self.vector.search(query, top_k=5)
        bm25_results = self.bm25.search(query, top_k=5)

        fused_scores = defaultdict(float)
        result_lookup = {}

        K = 60

        # Vector rankings
        for rank, result in enumerate(vector_results, start=1):
            fused_scores[result.content] += 1 / (K + rank)
            result_lookup[result.content] = result

        # BM25 rankings
        for rank, result in enumerate(bm25_results, start=1):
            fused_scores[result.content] += 1 / (K + rank)

            if result.content not in result_lookup:
                result_lookup[result.content] = result

        ranked = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            result_lookup[content]
            for content, _ in ranked[:top_k]
        ]
        

        