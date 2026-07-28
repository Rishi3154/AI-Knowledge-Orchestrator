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


@dataclass(slots=True)
class ChatResponse:
    """
    Final response returned by the RAG pipeline.
    """

    answer: str

    sources: List[SearchResult]

    retrieved_chunks: int

    latency: float