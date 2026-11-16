import importlib

import pytest

from app.core.models import Chunk


def chunks_for(source, count):

    return [
        Chunk(f"{source}-{n}", f"{source} text {n}", {"source": source, "page": n + 1})
        for n in range(count)
    ]


# ---------------- vector store ----------------

def test_store_round_trips_chunks(temp_store, fake_embedder):

    from app.vectorstore.vector_store import VectorStore

    store = VectorStore()
    chunks = chunks_for("a.pdf", 3)

    store.add_chunks(chunks, fake_embedder.embed_many([c.content for c in chunks]))

    stored = sorted(store.all_chunks(), key=lambda c: c.chunk_id)

    assert [c.chunk_id for c in stored] == ["a.pdf-0", "a.pdf-1", "a.pdf-2"]
    assert stored[0].content == "a.pdf text 0"
    assert stored[0].metadata == {"source": "a.pdf", "page": 1}


def test_delete_removes_only_that_document(temp_store, fake_embedder):

    from app.vectorstore.vector_store import VectorStore

    store = VectorStore()

    for source in ("a.pdf", "b.pdf"):
        chunks = chunks_for(source, 2)
        store.add_chunks(chunks, fake_embedder.embed_many([c.content for c in chunks]))

    store.delete_document("a.pdf")

    assert not store.has_document("a.pdf")
    assert store.has_document("b.pdf")
    assert store.count() == 2


# ---------------- ingestion ----------------

@pytest.fixture
def ingestor(temp_store, fake_embedder):

    from app.indexing.document_ingester import DocumentIngestor

    ingestor = DocumentIngestor()
    ingestor.embedder = fake_embedder

    return ingestor


def test_ingest_indexes_a_text_pdf(ingestor, temp_store, make_pdf):

    pdf = make_pdf(temp_store / "notes.pdf", "Exam 1 is on October 12.")

    response = ingestor.ingest(pdf)

    assert response.indexed
    assert response.pages == 1
    assert ingestor.vector_store.has_document("notes.pdf")


def test_ingesting_twice_reports_a_duplicate(ingestor, temp_store, make_pdf):

    pdf = make_pdf(temp_store / "notes.pdf", "Exam 1 is on October 12.")

    ingestor.ingest(pdf)
    response = ingestor.ingest(pdf)

    assert not response.indexed
    assert response.reason == "duplicate"
    assert ingestor.vector_store.count() == 1


def test_pdf_without_text_reports_no_text(ingestor, temp_store, make_pdf):

    pdf = make_pdf(temp_store / "scan.pdf", "")

    response = ingestor.ingest(pdf)

    assert not response.indexed
    assert response.reason == "no_text"


def test_failed_store_leaves_no_partial_document(ingestor, temp_store, make_pdf):

    pdf = make_pdf(temp_store / "notes.pdf", "Exam 1 is on October 12.")

    real_add = ingestor.vector_store.collection.add

    def add_then_fail(**kwargs):
        real_add(**kwargs)
        raise RuntimeError("disk full")

    ingestor.vector_store.collection.add = add_then_fail

    with pytest.raises(RuntimeError):
        ingestor.ingest(pdf)

    assert not ingestor.vector_store.has_document("notes.pdf")


def test_deleting_a_document_removes_chunks_and_file(ingestor, temp_store, make_pdf):

    from app.services.knowledge_base import KnowledgeBase

    pdf = make_pdf(temp_store / "notes.pdf", "Exam 1 is on October 12.")

    ingestor.ingest(pdf)

    KnowledgeBase().delete_document("notes.pdf")

    assert not ingestor.vector_store.has_document("notes.pdf")
    assert not pdf.exists()


# ---------------- uploads ----------------

class Upload:

    def __init__(self, name, data):
        self.name = name
        self.data = data

    def getvalue(self):
        return self.data


def test_same_name_with_new_content_is_a_new_version(temp_store):

    from app.services.upload_services import UploadService

    service = UploadService()

    service.save(Upload("doc.pdf", b"v1"))

    assert not service.is_new_version(Upload("doc.pdf", b"v1"))
    assert service.is_new_version(Upload("doc.pdf", b"v2"))
    assert not service.is_new_version(Upload("other.pdf", b"v1"))


def test_upload_cannot_escape_the_upload_folder(temp_store):

    from app.services.upload_services import UploadService

    destination = UploadService().destination(Upload("../../evil.pdf", b""))

    assert destination.parent == temp_store


# ---------------- config ----------------

@pytest.fixture
def config(monkeypatch):
    """
    Reload the config under different settings, then restore it.
    """

    import app.core.config as config

    def load(use_local):
        monkeypatch.setenv("USE_LOCAL_MODELS", use_local)
        return importlib.reload(config)

    yield load

    monkeypatch.undo()
    importlib.reload(config)


def test_local_model_keeps_the_original_collection(config):

    assert config("true").COLLECTION_NAME == "knowledge_base"


def test_cloud_model_gets_its_own_collection(config):

    cloud = config("false")

    assert cloud.COLLECTION_NAME == "knowledge_base_text-embedding-3-small"
    assert cloud.ACTIVE_LLM_MODEL == cloud.CLOUD_LLM_MODEL
