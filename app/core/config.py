"""
Application Configuration

Central location for all project-wide constants and paths.
"""
import os
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

# Answers come from Claude through the Anthropic API. The first
# entry is the default; the others can be picked in the sidebar.
CLAUDE_MODELS = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-5-5": "Claude Haiku 5.5",
}

DEFAULT_CLAUDE_MODEL = next(iter(CLAUDE_MODELS))

ANTHROPIC_API_KEY = os.getenv(
    "ANTHROPIC_API_KEY"
)

# Anthropic has no embedding model, so search uses a small model
# that runs inside the app (through ChromaDB and ONNX Runtime). It
# needs no API key or separate install; the first use downloads it
# (about 80 MB).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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

# Named after the embedding model: vectors from different models
# can't share a collection.
COLLECTION_NAME = "knowledge_base_minilm"

# ==========================================================
# Uploads
# ==========================================================

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]

UPLOAD_DIR = DATA_DIR / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)