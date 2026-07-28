"""
Application Configuration

Central location for all project-wide constants and paths.
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# ==========================================================
# Models
# ==========================================================

EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "llama3.2"

# ==========================================================
# Chunking
# ==========================================================

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100

# ==========================================================
# Retrieval
# ==========================================================

TOP_K = 5

MIN_RELEVANCE_DISTANCE = 1.5

# ==========================================================
# ChromaDB
# ==========================================================

COLLECTION_NAME = "knowledge_base"

# ==========================================================
# Supported File Types
# ==========================================================

SUPPORTED_EXTENSIONS = [".pdf"]