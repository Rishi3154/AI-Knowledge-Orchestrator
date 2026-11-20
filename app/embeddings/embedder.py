"""
Embedding model for indexing and retrieval.

Runs all-MiniLM-L6-v2 inside the app through ChromaDB's bundled
ONNX model, so it works anywhere Python runs: no API key, and no
Ollama or other separate install.
"""

from functools import lru_cache, partial
from typing import Protocol

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from tqdm import tqdm

# Model downloads normally draw a progress bar on stderr. Some hosts,
# Streamlit Community Cloud among them, close stderr, and every write
# to it then fails with "[Errno 32] Broken pipe", aborting the
# download. Downloads therefore run with the progress bar off.
quiet_tqdm = partial(tqdm, disable=True)


class Embedder(Protocol):
    """
    Common interface for embedding models.
    """

    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        ...


class LocalEmbedder:
    """
    all-MiniLM-L6-v2, run in-process with ONNX Runtime.
    """

    # Chunks embedded per call; keeps memory flat for long books.
    BATCH_SIZE = 64

    def __init__(self):

        # One instance for the whole process, so the model is
        # loaded once rather than for every batch.
        self.model = ONNXMiniLM_L6_V2()
        self.model.tqdm = quiet_tqdm

    def embed(
        self,
        text: str,
    ) -> list[float]:

        return self.embed_many([text])[0]

    def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.BATCH_SIZE):

            batch = texts[start:start + self.BATCH_SIZE]

            embeddings.extend(
                [float(value) for value in vector]
                for vector in self.model(batch)
            )

        return embeddings


@lru_cache(maxsize=1)
def create_embedder() -> Embedder:
    """
    The embedding model, loaded once per process.
    """

    return LocalEmbedder()
