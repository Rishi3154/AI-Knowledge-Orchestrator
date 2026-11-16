from app.core.models import Chunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.utils.text import tokenize

from tests.conftest import result


def chunk(chunk_id, content):
    return Chunk(chunk_id, content, {"source": "doc.pdf", "page": 1})


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Exam 1, Oct. 12!") == ["exam", "1", "oct", "12"]


def test_bm25_ranks_the_matching_chunk_first():

    bm25 = BM25Retriever()

    bm25.build([
        chunk("a", "The exam is in October."),
        chunk("b", "Parents should praise effort."),
        chunk("c", "Office hours are on Monday."),
    ])

    results = bm25.search("When is the Exam?")

    assert results[0].chunk_id == "a"


def test_bm25_returns_nothing_when_no_term_matches():

    bm25 = BM25Retriever()

    bm25.build([
        chunk("a", "The exam is in October."),
        chunk("b", "Parents should praise effort."),
        chunk("c", "Office hours are on Monday."),
    ])

    assert bm25.search("zebra quokka") == []


def test_bm25_with_no_chunks_returns_nothing():

    bm25 = BM25Retriever()

    bm25.build([])

    assert bm25.search("anything") == []


def test_fusion_prefers_chunks_both_retrievers_agree_on():

    a, b, c = result("a"), result("b"), result("c")

    fused = HybridRetriever.fuse([[a, b], [c, b]], top_k=3)

    assert fused[0].chunk_id == "b"


def test_fusion_keeps_identical_text_from_different_chunks():

    first = result("a", content="same text", page=1)
    second = result("b", content="same text", page=9)

    fused = HybridRetriever.fuse([[first, second]], top_k=5)

    assert [r.chunk_id for r in fused] == ["a", "b"]


def test_fusion_respects_top_k():

    results = [result(str(n)) for n in range(10)]

    assert len(HybridRetriever.fuse([results], top_k=4)) == 4
