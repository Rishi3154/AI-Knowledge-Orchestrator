"""
Application Data Models
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(slots=True)
class Document:
    """
    Represents a single page extracted from a document.
    """

    content: str

    metadata: Dict


@dataclass(slots=True)
class Chunk:
    """
    Represents a chunk generated from a document.
    """

    chunk_id: str

    content: str

    metadata: Dict


@dataclass(slots=True)
class SearchResult:
    """
    Represents one retrieved chunk.
    """

    content: str

    metadata: Dict

    distance: float

    chunk_id: str = ""

    @property
    def location(self) -> str:
        """
        Where the chunk came from, e.g. "notes.pdf, page 3". Only
        PDFs have page numbers.
        """

        source = self.metadata.get("source", "Unknown")
        page = self.metadata.get("page")

        if page is None:
            return str(source)

        return f"{source}, page {page}"


@dataclass(slots=True)
class ChatResponse:
    """
    Final response returned by the RAG pipeline.
    """

    answer: str

    sources: List[SearchResult]

    retrieved_chunks: int

    latency: float

    # The question actually searched for. Differs from the user's
    # wording when a follow-up was rewritten using the chat history.
    search_query: str = ""


@dataclass
class IngestionResponse:
    indexed: bool
    pages: int
    chunks: int
    document: str

    # Pages read (1 for files without pages) and chunks stored.
    # `reason` says why nothing was indexed: "duplicate" or "no_text".
    reason: str = ""