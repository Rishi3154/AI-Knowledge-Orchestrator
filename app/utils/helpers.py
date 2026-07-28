"""
Common utility functions.
"""

from pathlib import Path
from typing import List

from app.core.config import SUPPORTED_EXTENSIONS


def discover_documents(directory: Path) -> List[Path]:
    """
    Discover all supported documents inside a directory.

    Parameters
    ----------
    directory : Path
        Directory containing PDFs.

    Returns
    -------
    List[Path]
        List of discovered files.
    """

    documents = []

    for extension in SUPPORTED_EXTENSIONS:
        documents.extend(directory.rglob(f"*{extension}"))

    return sorted(documents)


def ensure_directory(path: Path) -> None:
    """
    Create directory if it doesn't already exist.
    """

    path.mkdir(parents=True, exist_ok=True)