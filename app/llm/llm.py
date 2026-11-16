"""
LLM providers for local and cloud inference.
"""

from typing import Protocol

from ollama import Client as OllamaClient
from openai import OpenAI

from app.core.config import (
    USE_LOCAL_MODELS,
    LLM_MODEL,
    CLOUD_LLM_MODEL,
    OPENAI_API_KEY,
)


class LLM(Protocol):
    """
    Common interface for all LLM providers.
    """

    def generate(self, prompt: str) -> str:
        ...

    def stream(self, prompt: str):
        ...


class OllamaLLM:
    """
    Local Ollama chat model.
    """

    def __init__(self, model: str):
        self.model = model
        self.client = OllamaClient()

    def generate(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.message.content.strip()

    def stream(self, prompt: str):
        stream = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
        )

        for chunk in stream:
            if chunk.message and chunk.message.content:
                yield chunk.message.content


class CloudLLM:
    """
    Cloud-based LLM for deployment.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
    ):
        self.client = OpenAI(
            api_key=api_key,
        )
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        content = response.choices[0].message.content

        return content.strip() if content else ""

    def stream(self, prompt: str):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content

            if content:
                yield content


OLLAMA = "ollama"
OPENAI = "openai"


def default_model() -> tuple[str, str]:
    """
    (provider, model) used unless the user picks another.
    """

    if USE_LOCAL_MODELS:
        return OLLAMA, LLM_MODEL

    return OPENAI, CLOUD_LLM_MODEL


def available_models() -> list[tuple[str, str]]:
    """
    Every (provider, model) that can answer questions right now,
    with the default first.

    Only the chat model can be switched freely. The embedding model
    is fixed by the configuration, because the stored index was
    built with it.
    """

    models = [default_model()]

    try:

        # A short timeout keeps the page responsive when Ollama
        # isn't running (for example on a cloud deployment).
        installed = OllamaClient(timeout=2).list().models

    except Exception:

        installed = []

    for entry in installed:

        # Embedding models can't chat.
        family = (entry.details.family or "") if entry.details else ""

        if "bert" in family or "embed" in entry.model:
            continue

        models.append((OLLAMA, entry.model))

    if OPENAI_API_KEY:
        models.append((OPENAI, CLOUD_LLM_MODEL))

    # Drop repeats while keeping the order.
    return list(dict.fromkeys(models))


def create_llm(
    provider: str | None = None,
    model: str | None = None,
) -> LLM:
    """
    Create an LLM. With no arguments, the default for the
    current environment.
    """

    if provider is None:
        provider, model = default_model()

    if provider == OLLAMA:
        return OllamaLLM(
            model=model or LLM_MODEL,
        )

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Add it to your environment or Streamlit Secrets."
        )

    return CloudLLM(
        api_key=OPENAI_API_KEY,
        model=model or CLOUD_LLM_MODEL,
    )
