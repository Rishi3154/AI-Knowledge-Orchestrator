"""
embedder.py

Shared embedding interface for indexing and retrieval.
"""

from ollama import Client


class OllamaEmbedder:
    def __init__(self, model: str):
        self.model = model
        self.client = Client()

    def embed(self, text: str) -> list[float]:
        response = self.client.embed(
            model=self.model,
            input=text,
        )

        return response.embeddings[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(
            model=self.model,
            input=texts,
        )

        return response.embeddings