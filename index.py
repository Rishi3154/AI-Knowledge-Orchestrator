"""
index.py

Indexes all PDFs inside data/raw into ChromaDB.
"""

from app.core.config import RAW_DATA_DIR
from app.utils.helpers import discover_documents
from app.ingestion.document_loader import PDFLoader
from app.ingestion.chunker import DocumentChunker
from app.embeddings.embedder import OllamaEmbedder
from app.vectorstore.vector_store import VectorStore


def main():

    print("=" * 60)
    print("AI Knowledge Orchestrator - Document Indexer")
    print("=" * 60)

    # ----------------------------------------------------
    # Discover PDFs
    # ----------------------------------------------------

    pdf_files = discover_documents(RAW_DATA_DIR)

    if not pdf_files:
        print("\nNo PDF files found in data/raw/")
        return

    print(f"\nFound {len(pdf_files)} PDF(s)\n")

    # ----------------------------------------------------
    # Initialize Components
    # ----------------------------------------------------

    chunker = DocumentChunker()
    embedder = OllamaEmbedder()
    vector_store = VectorStore()

    # Uncomment this if you want a fresh database every run.
    # vector_store.delete_collection()

    total_documents = 0
    total_chunks = 0

    # ----------------------------------------------------
    # Process PDFs
    # ----------------------------------------------------

    for pdf in pdf_files:

        print(f"Processing: {pdf.name}")

        loader = PDFLoader(pdf)

        documents = loader.load()

        chunks = chunker.split(documents)

        embeddings = embedder.embed_many(
            [chunk.content for chunk in chunks]
        )

        vector_store.add_chunks(
            chunks,
            embeddings,
        )

        total_documents += len(documents)
        total_chunks += len(chunks)

        print(
            f"   ✓ Pages: {len(documents)} | Chunks: {len(chunks)}\n"
        )

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------

    print("=" * 60)

    print("Indexing Complete\n")

    print(f"Documents : {total_documents}")
    print(f"Chunks    : {total_chunks}")
    print(f"Vectors   : {vector_store.count()}")

    print("=" * 60)


if __name__ == "__main__":
    main()