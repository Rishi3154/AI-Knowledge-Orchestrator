"""
prompt_builder.py

Builds prompts for Retrieval-Augmented Generation (RAG).
"""

from typing import List

from app.core.models import SearchResult


class PromptBuilder:
    """
    Creates prompts by combining retrieved context with the user's question.
    """

    @staticmethod
    def build(
        question: str,
        chunks: List[SearchResult],
    ) -> str:

        context = "\n\n".join(
            chunk.content for chunk in chunks
        )

        prompt = f"""
You are an AI assistant answering questions using ONLY the provided context.

Instructions:
- Answer only using the context below.
- If the answer is not present, say:
  "I couldn't find that information in the provided documents."
- Do not make up facts.
- Be concise but informative.

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""

        return prompt.strip()