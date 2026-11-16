"""
Shared fixtures.

Nothing here talks to Ollama, OpenAI or the real knowledge base:
storage is redirected to a temporary directory and the embedder
is replaced with a deterministic fake.
"""

import fitz
import pytest

from app.core.models import SearchResult


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    """
    Point ChromaDB and the upload folder at a temporary directory.
    """

    import app.services.knowledge_base as knowledge_base
    import app.services.upload_services as upload_services
    import app.vectorstore.vector_store as vector_store

    uploads = tmp_path / "uploads"
    uploads.mkdir()

    monkeypatch.setattr(vector_store, "CHROMA_DB_DIR", tmp_path / "chroma")
    monkeypatch.setattr(knowledge_base, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(upload_services, "UPLOAD_DIR", uploads)

    return uploads


class FakeEmbedder:
    """
    Maps each text to a small fixed-size vector without a model.
    """

    def embed(self, text):

        vector = [0.0] * 8

        for index, byte in enumerate(text.encode("utf-8")):
            vector[index % 8] += byte

        return vector

    def embed_many(self, texts):

        return [self.embed(text) for text in texts]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def make_pdf():
    """
    Write a one-page PDF containing the given text (or none).
    """

    def _make(path, text):

        document = fitz.open()
        page = document.new_page()

        if text:
            page.insert_text((72, 72), text)

        document.save(path)
        document.close()

        return path

    return _make


def result(chunk_id, content="text", source="doc.pdf", page=1):

    return SearchResult(
        content=content,
        metadata={"source": source, "page": page},
        distance=0.0,
        chunk_id=chunk_id,
    )
