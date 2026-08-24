"""
End-to-end Retrieval-Augmented Generation pipeline.
"""

from time import perf_counter

from app.core.models import ChatResponse
from app.core.config import CHUNK_INDEX_FILE

from app.llm.llm import create_llm
from app.llm.prompt_builder import PromptBuilder

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.bm25_retriever import BM25Retriever

from app.indexing.chunk_index import ChunkIndex


class RAGPipeline:

    def __init__(self):

        # Create the correct LLM provider
        self.llm = create_llm()

        self.chunk_index = ChunkIndex(
            CHUNK_INDEX_FILE
        )

        self.retriever = None

        self.reload()

    def reload(self):
        """
        Reload the retrieval pipeline after indexing new documents.
        """

        chunks = self.chunk_index.load()

        if not chunks:
            self.retriever = None
            return

        bm25 = BM25Retriever()
        bm25.build(chunks)

        self.retriever = HybridRetriever(bm25)

    def ask(self, question: str) -> ChatResponse:

        if self.retriever is None:
            return ChatResponse(
                answer="Please upload and index a PDF first.",
                sources=[],
                retrieved_chunks=0,
                latency=0.0,
            )

        start = perf_counter()

        chunks = self.retriever.search(question)

        prompt = PromptBuilder.build(
            question=question,
            chunks=chunks,
        )

        answer = self.llm.generate(prompt)

        latency = perf_counter() - start

        return ChatResponse(
            answer=answer,
            sources=chunks,
            retrieved_chunks=len(chunks),
            latency=latency,
        )

    def stream(self, question: str):

        if self.retriever is None:

            self.last_response = ChatResponse(
                answer="Please upload and index a PDF first.",
                sources=[],
                retrieved_chunks=0,
                latency=0.0,
            )

            yield "Please upload and index a PDF first."

            return

        start = perf_counter()

        chunks = self.retriever.search(question)

        prompt = PromptBuilder.build(
            question=question,
            chunks=chunks,
        )

        answer = ""

        for token in self.llm.stream(prompt):
            answer += token
            yield token

        latency = perf_counter() - start

        self.last_response = ChatResponse(
            answer=answer,
            sources=chunks,
            retrieved_chunks=len(chunks),
            latency=latency,
        )