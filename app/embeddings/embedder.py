"""
embedder.py

Generates embeddings using Ollama.
"""

from typing import List

import ollama

from app.core.config import EMBEDDING_MODEL


class OllamaEmbedder:
    """
    Wrapper around Ollama embedding model.
    """

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        """

        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text,
        )

        return response["embedding"]

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        """

        embeddings = []

        total = len(texts)

        for index, text in enumerate(texts, start=1):

            print(f"Embedding {index}/{total}", end="\r")

            embeddings.append(self.embed(text))

        print()

        return embeddings