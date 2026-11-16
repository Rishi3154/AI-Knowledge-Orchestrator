"""
workspace.py

Workspace service.
"""


class Workspace:
    """
    What is currently in the knowledge base.

    Reads from the chunks the pipeline already holds in memory, so
    drawing the sidebar costs no disk access.
    """

    def __init__(self, rag):

        self.rag = rag

    def documents(self):

        return self.rag.documents

    def stats(self):

        return {
            "documents": len(self.rag.documents),
            "chunks": self.rag.chunk_count,
        }
