"""
text.py

Utility functions for text processing.
"""

import re
from typing import List


def tokenize(text: str) -> List[str]:
    """
    Normalize text into tokens for BM25.
    """

    text = text.lower()

    return re.findall(r"\b[a-zA-Z0-9]+\b", text)