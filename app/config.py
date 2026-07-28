"""
Project Configuration
"""

from pathlib import Path

# ===========================
# Project Paths
# ===========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# ===========================
# Models
# ===========================

EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "llama3.2"

# ===========================
# Chunking
# ===========================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ===========================
# Retrieval
# ===========================

TOP_K = 5

# ===========================
# ChromaDB
# ===========================

COLLECTION_NAME = "knowledge_base"

# ===========================
# Supported Files
# ===========================

SUPPORTED_EXTENSIONS = [".pdf"]