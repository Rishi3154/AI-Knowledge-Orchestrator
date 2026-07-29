"""
upload_service.py

Handles saving uploaded PDFs.
"""

from pathlib import Path

from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.core.config import UPLOAD_DIR


class UploadService:

    def save(
        self,
        uploaded_file: UploadedFile,
    ) -> Path:

        destination = UPLOAD_DIR / uploaded_file.name

        if not destination.exists():

            with open(destination, "wb") as f:
                f.write(uploaded_file.getbuffer())

        return destination