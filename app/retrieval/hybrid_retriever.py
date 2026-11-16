from collections import defaultdict
from typing import List

from app.core.config import CANDIDATE_K, TOP_K
from app.core.models import SearchResult
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import rerank


class HybridRetriever:

    def __init__(self, bm25: BM25Retriever):

        self.vector = VectorRetriever()
        self.bm25 = bm25

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[SearchResult]:

        # Retrieve more candidates than we finally return
        vector_results = self.vector.search(query, top_k=CANDIDATE_K)
        bm25_results = self.bm25.search(query, top_k=CANDIDATE_K)

        # Fusion narrows the pool; the reranker picks the final few.
        candidates = self.fuse(
            [vector_results, bm25_results],
            CANDIDATE_K,
        )

        return rerank(query, candidates, top_k)

    @staticmethod
    def fuse(
        rankings: List[List[SearchResult]],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion over several ranked lists.
        """

        fused_scores = defaultdict(float)
        result_lookup = {}

        K = 60

        for results in rankings:

            for rank, result in enumerate(results, start=1):

                # Identify chunks by id, not text: the same passage
                # can legitimately appear on different pages.
                key = result.chunk_id or result.content

                fused_scores[key] += 1 / (K + rank)

                result_lookup.setdefault(key, result)

        ranked = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            result_lookup[key]
            for key, _ in ranked[:top_k]
        ]
