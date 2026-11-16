"""
citations.py

Matches the [n] markers in an answer to the retrieved sources.
"""

import re
from typing import List, Tuple

from app.core.models import SearchResult


def cited_sources(
    answer: str,
    sources: List[SearchResult],
) -> List[Tuple[int, SearchResult]]:
    """
    Return (number, source) for each source the answer cites.
    """

    cited = {
        int(number)
        for number in re.findall(r"\[(\d+)\]", answer)
    }

    return [
        (number, source)
        for number, source in enumerate(sources, start=1)
        if number in cited
    ]
