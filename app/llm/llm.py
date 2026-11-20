"""
Chat model used to answer questions: Claude, through the Anthropic API.
"""

from typing import Protocol

from app.core.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODELS,
    DEFAULT_CLAUDE_MODEL,
)
from app.llm.claude_llm import ClaudeLLM


class LLM(Protocol):
    """
    What the pipeline needs from a chat model.
    """

    def generate(self, prompt: str) -> str:
        ...

    def stream(self, prompt: str):
        ...


MISSING_KEY_MESSAGE = (
    "ANTHROPIC_API_KEY is not configured. Add it to a .env file "
    "locally, or to the app's Secrets on Streamlit Cloud."
)


def has_api_key() -> bool:

    return bool(ANTHROPIC_API_KEY)


def available_models() -> list[str]:
    """
    Claude models that can be picked, the default first.
    """

    return list(CLAUDE_MODELS)


def create_llm(model: str | None = None) -> LLM:
    """
    A Claude client for the given model (the default if omitted).
    """

    if not ANTHROPIC_API_KEY:
        raise ValueError(MISSING_KEY_MESSAGE)

    return ClaudeLLM(
        api_key=ANTHROPIC_API_KEY,
        model=model or DEFAULT_CLAUDE_MODEL,
    )
