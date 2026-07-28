"""
Document Chunker

Splits documents into overlapping chunks.
"""

from typing import List
import uuid

from app.core.models import Document, Chunk
from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunker:
    """
    Creates overlapping chunks.
    """

    def split(self, documents: List[Document]) -> List[Chunk]:

        chunks = []

        step = CHUNK_SIZE - CHUNK_OVERLAP

        for document in documents:

            text = document.content

            start = 0

            while start < len(text):

                end = start + CHUNK_SIZE

                chunk_text = text[start:end]

                if chunk_text.strip():

                    chunks.append(
                        Chunk(
                            chunk_id=str(uuid.uuid4()),
                            content=chunk_text,
                            metadata=document.metadata.copy(),
                        )
                    )

                start += step

        return chunks