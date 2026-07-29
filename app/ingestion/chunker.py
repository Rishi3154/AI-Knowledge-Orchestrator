"""
chunker.py

Splits documents into overlapping chunks while preserving metadata.
"""

from typing import List
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.models import Document, Chunk


class DocumentChunker:
    """
    Splits Document objects into Chunk objects.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def split(self, documents: List[Document]) -> List[Chunk]:

        chunks: List[Chunk] = []

        for document in documents:

            texts = self.splitter.split_text(document.content)

            for text in texts:

                chunks.append(
                    Chunk(
                        chunk_id=str(uuid4()),
                        content=text,
                        metadata=document.metadata.copy(),
                    )
                )

        return chunks