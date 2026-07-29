"""
PDF Document Loader

Loads PDF files using PyMuPDF and converts each page
into a Document object.
"""

from pathlib import Path
from typing import List

import fitz  # PyMuPDF

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

        if not documents:

            print(f"No extractable text found in '{self.pdf_path.name}'. "
        "This PDF may be scanned or image-based. OCR support will be added in a future version."
        )

        return documents