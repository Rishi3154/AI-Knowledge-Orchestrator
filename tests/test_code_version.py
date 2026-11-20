import os

import pytest

import app.core.code_version as code_version_module


@pytest.fixture
def project(tmp_path, monkeypatch):

    (tmp_path / "app").mkdir()
    (tmp_path / "components").mkdir()
    (tmp_path / "app.py").write_text("main")
    (tmp_path / "app" / "config.py").write_text("COLLECTION = 'old'")

    monkeypatch.setattr(code_version_module, "PROJECT_ROOT", tmp_path)

    return tmp_path


def test_version_is_stable_when_nothing_changes(project):

    assert code_version_module.code_version() == code_version_module.code_version()


def test_editing_a_file_changes_the_version(project):

    before = code_version_module.code_version()

    path = project / "app" / "config.py"
    path.write_text("COLLECTION = 'new'")
    os.utime(path, ns=(1, 1))

    assert code_version_module.code_version() != before


def test_adding_a_file_changes_the_version(project):

    before = code_version_module.code_version()

    (project / "components" / "page.py").write_text("x = 1")

    assert code_version_module.code_version() != before


def test_code_update_builds_a_new_pipeline(monkeypatch):
    """
    The situation on Streamlit Cloud: code changes under a running
    app. The cached pipeline from the old code must be replaced.
    """

    import components.chat.interface as interface

    built = []

    class FakePipeline:
        def __init__(self):
            built.append(self)

    monkeypatch.setattr(interface, "RAGPipeline", FakePipeline)
    interface.build_rag.clear()

    version = {"value": "v1"}
    monkeypatch.setattr(interface, "code_version", lambda: version["value"])

    first = interface.get_rag()

    assert interface.get_rag() is first

    version["value"] = "v2"

    assert interface.get_rag() is not first
    assert len(built) == 2

    interface.build_rag.clear()
