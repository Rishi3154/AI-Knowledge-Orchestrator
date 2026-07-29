"""
llm.py

Wrapper around Ollama chat models.
"""

from ollama import Client


class OllamaLLM:
    """
    Simple wrapper for Ollama chat models.
    """

    def __init__(self, model: str):
        self.model = model
        self.client = Client()

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