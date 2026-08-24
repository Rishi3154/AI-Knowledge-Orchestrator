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


def create_llm() -> LLM:
    """
    Create the correct LLM provider for the
    current environment.
    """

    if USE_LOCAL_MODELS:
        return OllamaLLM(
            model=LLM_MODEL,
        )

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Add it to your environment or Streamlit Secrets."
        )

    return CloudLLM(
        api_key=OPENAI_API_KEY,
        model=CLOUD_LLM_MODEL,
    )