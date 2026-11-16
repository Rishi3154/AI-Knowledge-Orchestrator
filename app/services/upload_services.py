"""
upload_service.py

Handles saving uploaded documents.
"""

from pathlib import Path

from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.core.config import UPLOAD_DIR


class UploadService:

    def destination(
        self,
        uploaded_file: UploadedFile,
    ) -> Path:

        # Keep only the file name so an upload can never be written
        # outside the upload directory.
        return UPLOAD_DIR / Path(uploaded_file.name).name

    def is_new_version(
        self,
        uploaded_file: UploadedFile,
    ) -> bool:
        """
        True when a different file is already saved under this name.
        """

        destination = self.destination(uploaded_file)

        return (
            destination.exists()
            and destination.read_bytes() != uploaded_file.getvalue()
        )

    def save(
        self,
        uploaded_file: UploadedFile,
    ) -> Path:

        destination = self.destination(uploaded_file)

        destination.write_bytes(uploaded_file.getvalue())

        return destination
