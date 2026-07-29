"""
chunk_index.py

Save and load document chunks.
"""

from pathlib import Path
import json

from app.core.models import Chunk


class ChunkIndex:

    def __init__(self, index_path: Path):

        self.index_path = index_path

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(self, chunks: list[Chunk]) -> None:
        """
        Save chunks to JSON.
        """

        data = []

        for chunk in chunks:

            data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
            )

        with open(
            self.index_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

    def load(self) -> list[Chunk]:
        """
        Load chunks from JSON.
        """

        if not self.index_path.exists():
            return []

        with open(
            self.index_path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        chunks = []

        for item in data:

            chunks.append(
                Chunk(
                    chunk_id=item["chunk_id"],
                    content=item["content"],
                    metadata=item["metadata"],
                )
            )

        return chunks

    def clear(self):

        if self.index_path.exists():
            self.index_path.unlink()


    def count(self) -> int:

        return len(self.load())


    def documents(self):

        docs = set()

        for chunk in self.load():
            docs.add(chunk.metadata["source"])

        return sorted(docs)     


    def delete_document(self, document: str,) -> int:
        """
        Delete all chunks belonging to a document.
        Returns:
        Number of deleted chunks.
        """

        chunks = self.load()

        remaining = [
            chunk
            for chunk in chunks
            if chunk.metadata["source"] != document
            ]

        deleted = len(chunks) - len(remaining)

        self.save(remaining)

        return deleted