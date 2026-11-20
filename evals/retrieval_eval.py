"""
retrieval_eval.py

Measures how often retrieval finds the page that answers a question.

Each entry in questions.json names a question, the document that
answers it and the page(s) where the answer is. For every stage of
the retrieval pipeline this reports:

  hit rate  share of questions with a correct page in the top TOP_K
  MRR       mean of 1 / rank of the first correct page (0 if absent)

Run it before and after changing chunking, retrieval or reranking
settings to see whether the change helped:

    python -m evals.retrieval_eval

Runs offline once the embedding model has been downloaded; it
needs no API key. Questions about documents that are not
currently indexed are skipped.
"""

import json
from pathlib import Path

from app.core.config import CANDIDATE_K, TOP_K
from app.rag_pipeline import RAGPipeline
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import rerank

QUESTIONS_FILE = Path(__file__).parent / "questions.json"


def first_correct_rank(results, source, pages):
    """
    1-based position of the first result from a correct page.
    """

    for rank, result in enumerate(results[:TOP_K], start=1):

        if (
            result.metadata["source"] == source
            and result.metadata.get("page") in pages
        ):
            return rank

    return None


def main():

    rag = RAGPipeline()

    questions = [
        item
        for item in json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
        if item["source"] in rag.documents
    ]

    if not questions:
        print("None of the documents in questions.json are indexed.")
        return

    hybrid = rag.retriever

    stages = {
        "vector only": [],
        "keyword only": [],
        "fused": [],
        "fused + reranked": [],
    }

    misses = []

    for item in questions:

        query = item["question"]

        vector = hybrid.vector.search(query, top_k=CANDIDATE_K)
        keyword = hybrid.bm25.search(query, top_k=CANDIDATE_K)
        fused = HybridRetriever.fuse([vector, keyword], CANDIDATE_K)
        final = rerank(query, fused, TOP_K)

        ranked = {
            "vector only": vector,
            "keyword only": keyword,
            "fused": fused,
            "fused + reranked": final,
        }

        for stage, results in ranked.items():

            stages[stage].append(
                first_correct_rank(results, item["source"], item["pages"])
            )

        if stages["fused + reranked"][-1] is None:
            misses.append(query)

    total = len(questions)

    print(f"{total} questions, top {TOP_K} of {CANDIDATE_K} candidates\n")
    print(f"{'stage':<20}{'hit rate':>10}{'MRR':>8}")

    for stage, ranks in stages.items():

        hits = sum(rank is not None for rank in ranks)
        mrr = sum(1 / rank for rank in ranks if rank) / total

        print(f"{stage:<20}{hits / total:>9.0%}{mrr:>8.2f}")

    if misses:

        print("\nMissed by the full pipeline:")

        for query in misses:
            print(f"  - {query}")


if __name__ == "__main__":
    main()
