"""
Application Configuration

Central location for all project-wide constants and paths.
"""
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Read settings from a local .env file if there is one (see
# .env.example). Variables already set in the environment win.
load_dotenv()

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# ==========================================================
# Models
# ==========================================================

# Local Ollama models
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:latest"

# Cloud models
CLOUD_EMBEDDING_MODEL = "text-embedding-3-small"
CLOUD_LLM_MODEL = "gpt-4o-mini"

# ==========================================================
# Providers
# ==========================================================

USE_LOCAL_MODELS = os.getenv(
    "USE_LOCAL_MODELS",
    "true",
).lower() == "true"

# ==========================================================
# API Keys
# ==========================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

# ==========================================================
# Active Models
# ==========================================================

ACTIVE_EMBEDDING_MODEL = (
    EMBEDDING_MODEL
    if USE_LOCAL_MODELS
    else CLOUD_EMBEDDING_MODEL
)

ACTIVE_LLM_MODEL = (
    LLM_MODEL
    if USE_LOCAL_MODELS
    else CLOUD_LLM_MODEL
)

# Vectors from different embedding models have different sizes and
# cannot share a collection, so each model gets its own. The original
# local model keeps the original name so an index built before this
# change still loads.
_LEGACY_EMBEDDING_MODEL = "nomic-embed-text"

_INDEX_SUFFIX = (
    ""
    if ACTIVE_EMBEDDING_MODEL == _LEGACY_EMBEDDING_MODEL
    else "_" + re.sub(r"[^a-zA-Z0-9]+", "-", ACTIVE_EMBEDDING_MODEL).strip("-")
)

# ==========================================================
# Chunking
# ==========================================================

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100

# ==========================================================
# Retrieval
# ==========================================================

# Chunks passed to the LLM.
TOP_K = 6

# Chunks fetched from each retriever before fusion. A wide pool lets
# a chunk that only one retriever ranks well still make the final cut.
CANDIDATE_K = 20

# Cross-encoder that re-scores the fused candidates against the
# question. Downloaded to MODEL_DIR on first use (about 34 MB).
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

MODEL_DIR = DATA_DIR / "models"

# Previous messages used to resolve follow-up questions.
HISTORY_MESSAGES = 6

# ==========================================================
# ChromaDB
# ==========================================================

COLLECTION_NAME = f"knowledge_base{_INDEX_SUFFIX}"

# ==========================================================
# Uploads
# ==========================================================

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]

UPLOAD_DIR = DATA_DIR / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)