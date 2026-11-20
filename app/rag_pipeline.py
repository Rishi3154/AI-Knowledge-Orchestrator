"""
End-to-end Retrieval-Augmented Generation pipeline.
"""

from time import perf_counter

from app.core.models import ChatResponse
from app.core.config import HISTORY_MESSAGES

from app.llm.llm import create_llm
from app.llm.prompt_builder import PromptBuilder

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.bm25_retriever import BM25Retriever

from app.vectorstore.vector_store import VectorStore


def standalone_question(llm, question: str, history) -> str:
    """
    Resolve a follow-up ("what about the second one?") into a question
    that makes sense on its own, using the recent conversation.
    """

    if not history:
        return question

    rewritten = llm.generate(
        PromptBuilder.build_rewrite(
            question=question,
            history=history[-HISTORY_MESSAGES:],
        )
    ).strip().strip('"')

    # A blank or runaway reply means the rewrite failed; the original
    # question is the safer thing to search for.
    if not rewritten or len(rewritten) > max(300, 3 * len(question)):
        return question

    return rewritten


class RAGPipeline:

    def __init__(self):

        # Created on first use, so the app can start (and index
        # documents) before an API key is configured.
        self.llm = None

        self.vector_store = VectorStore()

        self.retriever = None

        # Kept in memory for the sidebar and architecture page.
        self.documents: list[str] = []
        self.chunk_count = 0

        self.reload()

    def default_llm(self):

        if self.llm is None:
            self.llm = create_llm()

        return self.llm

    def reload(self):
        """
        Reload the retrieval pipeline after indexing new documents.
        """

        chunks = self.vector_store.all_chunks()

        self.documents = sorted(
            {chunk.metadata["source"] for chunk in chunks}
        )

        self.chunk_count = len(chunks)

        if not chunks:
            self.retriever = None
            return

        bm25 = BM25Retriever()
        bm25.build(chunks)

        self.retriever = HybridRetriever(bm25)

    def ask(self, question: str, history=None, llm=None) -> ChatResponse:

        llm = llm or self.default_llm()

        retriever = self.retriever

        if retriever is None:
            return ChatResponse(
                answer="Please upload and index a document first.",
                sources=[],
                retrieved_chunks=0,
                latency=0.0,
            )

        start = perf_counter()

        search_query = standalone_question(
            llm,
            question,
            history,
        )

        chunks = retriever.search(search_query)

        prompt = PromptBuilder.build(
            question=search_query,
            chunks=chunks,
        )

        answer = llm.generate(prompt)

        latency = perf_counter() - start

        return ChatResponse(
            answer=answer,
            sources=chunks,
            retrieved_chunks=len(chunks),
            latency=latency,
            search_query=search_query,
        )

    def stream(self, question: str, history=None, llm=None) -> "AnswerStream":
        """
        Start a streamed answer.

        `history` is the earlier conversation as a list of
        {"role", "content"} dicts; it is used to resolve follow-ups.
        `llm` overrides the default model for this answer only.

        Iterate the result to receive tokens; once it is exhausted,
        its `response` holds the final ChatResponse. The state lives
        on the returned object, not on the pipeline, because the
        pipeline is shared between every browser session.
        """

        return AnswerStream(
            retriever=self.retriever,
            llm=llm or self.default_llm(),
            question=question,
            history=history,
        )


class AnswerStream:
    """
    One in-flight answer: yields tokens, then exposes the response.
    """

    def __init__(self, retriever, llm, question: str, history=None):

        self.retriever = retriever
        self.llm = llm
        self.question = question
        self.history = history
        self.response: ChatResponse | None = None

    def __iter__(self):

        if self.retriever is None:

            answer = "Please upload and index a document first."

            self.response = ChatResponse(
                answer=answer,
                sources=[],
                retrieved_chunks=0,
                latency=0.0,
            )

            yield answer

            return

        start = perf_counter()

        search_query = standalone_question(
            self.llm,
            self.question,
            self.history,
        )

        chunks = self.retriever.search(search_query)

        prompt = PromptBuilder.build(
            question=search_query,
            chunks=chunks,
        )

        answer = ""

        for token in self.llm.stream(prompt):
            answer += token
            yield token

        self.response = ChatResponse(
            answer=answer,
            sources=chunks,
            retrieved_chunks=len(chunks),
            latency=perf_counter() - start,
            search_query=search_query,
        )
