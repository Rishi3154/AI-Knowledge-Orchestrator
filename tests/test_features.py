from datetime import datetime

import pytest
from docx import Document as WordDocument

from app.core.models import SearchResult
from app.ingestion.document_loader import load_document
from app.llm import llm as llm_module
from app.llm.prompt_builder import PromptBuilder
from app.rag_pipeline import RAGPipeline
from components.chat.export import build_transcript

from tests.conftest import result
from tests.test_llm import FakeLLM, FakeRetriever


# ---------------- loaders ----------------

def test_pdf_pages_carry_page_numbers(tmp_path, make_pdf):

    documents = load_document(make_pdf(tmp_path / "a.pdf", "Hello PDF"))

    assert documents[0].metadata == {"source": "a.pdf", "page": 1}
    assert "Hello PDF" in documents[0].content


@pytest.mark.parametrize("name", ["notes.txt", "README.md", "NOTES.TXT"])
def test_text_files_load_without_a_page(tmp_path, name):

    path = tmp_path / name
    path.write_text("Exam 1 is on October 12.", encoding="utf-8")

    documents = load_document(path)

    assert documents[0].content == "Exam 1 is on October 12."
    assert documents[0].metadata == {"source": name}


def test_text_file_with_bad_encoding_still_loads(tmp_path):

    path = tmp_path / "latin.txt"
    path.write_bytes("caf\xe9 menu".encode("latin-1"))

    assert "menu" in load_document(path)[0].content


def test_docx_includes_paragraphs_and_tables(tmp_path):

    word = WordDocument()
    word.add_paragraph("Grading policy")
    table = word.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Exam 1"
    table.rows[0].cells[1].text = "50%"

    path = tmp_path / "syllabus.docx"
    word.save(str(path))

    content = load_document(path)[0].content

    assert "Grading policy" in content
    assert "Exam 1 | 50%" in content


def test_empty_file_loads_nothing(tmp_path):

    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")

    assert load_document(path) == []


def test_unsupported_type_is_rejected(tmp_path):

    path = tmp_path / "sheet.xlsx"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="Unsupported"):
        load_document(path)


def test_text_file_can_be_ingested_and_found(temp_store, fake_embedder):

    from app.indexing.document_ingester import DocumentIngestor

    path = temp_store / "notes.txt"
    path.write_text("Exam 1 is on October 12.", encoding="utf-8")

    ingestor = DocumentIngestor()
    ingestor.embedder = fake_embedder

    assert ingestor.ingest(path).indexed
    assert ingestor.vector_store.all_chunks()[0].metadata == {"source": "notes.txt"}


# ---------------- locations ----------------

def test_location_omits_the_page_when_there_is_none():

    paged = result("a", source="a.pdf", page=3)
    unpaged = SearchResult("text", {"source": "notes.txt"}, 0.0, "b")

    assert paged.location == "a.pdf, page 3"
    assert unpaged.location == "notes.txt"
    assert "[1] notes.txt\ntext" in PromptBuilder.build("q", [unpaged])


# ---------------- model choice ----------------

def test_opus_is_the_default_and_every_model_is_claude():

    models = llm_module.available_models()

    assert models[0] == "claude-opus-5-5"
    assert all(model.startswith("claude-") for model in models)


def test_create_llm_builds_claude_or_asks_for_its_key(monkeypatch):

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    assert llm_module.has_api_key()
    assert llm_module.create_llm().model == "claude-opus-5-5"

    chosen = llm_module.create_llm("claude-haiku-5-5")

    assert isinstance(chosen, llm_module.ClaudeLLM)
    assert chosen.model == "claude-haiku-5-5"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert not llm_module.has_api_key()

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        llm_module.create_llm()


def test_key_added_after_startup_is_picked_up(monkeypatch):

    # Streamlit Cloud sets the variable when Secrets are saved, after
    # the app's modules were imported.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert not llm_module.has_api_key()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "added-later")

    assert llm_module.has_api_key()
    assert llm_module.create_llm().client.api_key == "added-later"


def test_pipeline_starts_without_a_key(monkeypatch, temp_store):

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    import app.rag_pipeline as rag_pipeline

    monkeypatch.setattr(rag_pipeline, "create_llm", llm_module.create_llm)

    pipeline = RAGPipeline()

    assert pipeline.llm is None


def test_stream_uses_the_llm_passed_for_that_answer():

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.retriever = FakeRetriever([result("a")])
    pipeline.llm = FakeLLM(tokens=["default"])

    chosen = FakeLLM(tokens=["chosen"])

    assert "".join(pipeline.stream("q", llm=chosen)) == "chosen"
    assert "".join(pipeline.stream("q")) == "default"


# ---------------- export ----------------

def test_transcript_lists_questions_answers_and_cited_sources():

    messages = [
        {"role": "user", "content": "When is Exam 1?"},
        {
            "role": "assistant",
            "content": "October 12 [2].",
            "sources": [
                result("a", source="a.pdf", page=1),
                result("b", source="syllabus.pdf", page=2),
            ],
        },
    ]

    transcript = build_transcript(messages, datetime(2026, 11, 11, 9, 30))

    assert "Exported 2026-11-11 09:30" in transcript
    assert "## You\n\nWhen is Exam 1?" in transcript
    assert "## Assistant\n\nOctober 12 [2]." in transcript
    assert "- [2] syllabus.pdf, page 2" in transcript
    assert "a.pdf" not in transcript
