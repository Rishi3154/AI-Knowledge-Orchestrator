"""
Embedding providers for indexing and retrieval.
"""

from typing import Protocol

from ollama import Client as OllamaClient
from openai import OpenAI

from app.core.config import (
    USE_LOCAL_MODELS,
    EMBEDDING_MODEL,
    CLOUD_EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


class Embedder(Protocol):
    """
    Common interface for all embedding providers.
    """

    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        ...


class OllamaEmbedder:
    """
    Local embedding provider using Ollama.
    """

    def __init__(self, model: str):

        self.model = model
        self.client = OllamaClient()

    def embed(
        self,
        text: str,
    ) -> list[float]:

        response = self.client.embed(
            model=self.model,
            input=text,
        )

        return response.embeddings[0]

    def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        response = self.client.embed(
            model=self.model,
            input=texts,
        )

        return response.embeddings


class OpenAIEmbedder:
    """
    Cloud embedding provider for deployed environments.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
    ):

        self.client = OpenAI(
            api_key=api_key,
        )

        self.model = model

    def embed(
        self,
        text: str,
    ) -> list[float]:

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding

    def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [
            item.embedding
            for item in response.data
        ]


def create_embedder() -> Embedder:
    """
    Create the correct embedding provider depending
    on the current environment.
    """

    if USE_LOCAL_MODELS:

        return OllamaEmbedder(
            EMBEDDING_MODEL
        )

    if not OPENAI_API_KEY:

        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

    return OpenAIEmbedder(
        api_key=OPENAI_API_KEY,
        model=CLOUD_EMBEDDING_MODEL,
    )