"""
Document Loaders

Turn an uploaded file into Document objects ready for chunking.

PDFs produce one Document per page, with the page number in the
metadata. Word, text and Markdown files have no fixed pages, so
they produce a single Document without one.
"""

from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from docx import Document as WordDocument

from app.core.config import SUPPORTED_EXTENSIONS
from app.core.models import Document


class PDFLoader:
    """
    Loads a PDF file page by page.
    """

    def __init__(self, pdf_path: Path):

        self.pdf_path = pdf_path

    def load(self) -> List[Document]:
        """
        Load every page from the PDF.

        Returns
        -------
        List[Document]
        """

        documents = []

        pdf = fitz.open(self.pdf_path)

        for page_number, page in enumerate(pdf):

            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append(
                Document(
                    content=text,
                    metadata={
                        "source": self.pdf_path.name,
                        "page": page_number + 1,
                    },
                )
            )

        pdf.close()

        # An empty list means the PDF has no text layer (it is
        # probably scanned); the caller reports that to the user.
        return documents


class TextLoader:
    """
    Loads a plain text or Markdown file.
    """

    def __init__(self, path: Path):

        self.path = path

    def load(self) -> List[Document]:

        # Not every text file is UTF-8; a few odd characters are
        # better than refusing the whole file.
        text = self.path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        return _single_document(self.path, text)


class DocxLoader:
    """
    Loads the paragraphs and tables of a Word document.
    """

    def __init__(self, path: Path):

        self.path = path

    def load(self) -> List[Document]:

        word = WordDocument(str(self.path))

        parts = [
            paragraph.text
            for paragraph in word.paragraphs
        ]

        for table in word.tables:

            for row in table.rows:

                parts.append(
                    " | ".join(cell.text for cell in row.cells)
                )

        return _single_document(self.path, "\n".join(parts))


def _single_document(path: Path, text: str) -> List[Document]:

    text = text.strip()

    if not text:
        return []

    return [
        Document(
            content=text,
            metadata={
                "source": path.name,
            },
        )
    ]


LOADERS = {
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_document(path: Path) -> List[Document]:
    """
    Load a file with the loader that matches its extension.
    """

    loader = LOADERS.get(path.suffix.lower())

    if loader is None:

        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported: "
            + ", ".join(SUPPORTED_EXTENSIONS)
        )

    return loader(path).load()
