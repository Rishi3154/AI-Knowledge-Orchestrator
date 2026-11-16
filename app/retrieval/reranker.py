"""
reranker.py

Re-scores retrieved chunks against the question with a cross-encoder.

The retrievers rank chunks by embedding similarity and keyword
overlap. A cross-encoder reads the question and a chunk together,
which is slower but a much better judge of whether the chunk
actually answers the question.
"""

import logging
from functools import lru_cache
from typing import List

from app.core.config import MODEL_DIR, RERANK_MODEL
from app.core.models import SearchResult

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_ranker():
    """
    Load the model once per process. Returns None if it is
    unavailable (package missing, or no network on first run).
    """

    try:

        from flashrank import Ranker

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        return Ranker(
            model_name=RERANK_MODEL,
            cache_dir=str(MODEL_DIR),
        )

    except Exception:

        logger.exception(
            "Reranker unavailable; using the fused ranking as is."
        )

        return None


def rerank(
    query: str,
    results: List[SearchResult],
    top_k: int,
) -> List[SearchResult]:
    """
    Return the top_k results, best match first.

    Falls back to the incoming order if the model cannot be loaded,
    so retrieval keeps working without it.
    """

    ranker = _load_ranker()

    if ranker is None or not results:
        return results[:top_k]

    from flashrank import RerankRequest

    scored = ranker.rerank(
        RerankRequest(
            query=query,
            passages=[
                {"id": index, "text": result.content}
                for index, result in enumerate(results)
            ],
        )
    )

    return [
        results[passage["id"]]
        for passage in scored[:top_k]
    ]
