"""
workspace.py

Workspace service.
"""

from app.core.config import CHUNK_INDEX_FILE
from app.indexing.chunk_index import ChunkIndex


class Workspace:

    def __init__(self):

        self.chunk_index = ChunkIndex(CHUNK_INDEX_FILE)

    def documents(self):

        return self.chunk_index.documents()

    def stats(self):

        return {
            "documents": len(self.documents()),
            "chunks": self.chunk_index.count(),
        }