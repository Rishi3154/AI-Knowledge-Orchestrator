"""
prompt_builder.py

Builds prompts for Retrieval-Augmented Generation (RAG).
"""

from typing import Dict, List

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

        # Number every chunk so the model can cite it and the UI can
        # show which sources an answer actually relied on.
        context = "\n\n".join(
            f"[{number}] {chunk.location}\n"
            f"{chunk.content}"
            for number, chunk in enumerate(chunks, start=1)
        )

        prompt = f"""
You are an AI assistant answering questions using ONLY the provided context.

Instructions:
- Answer only using the numbered sources below.
- After each fact, cite the number of the source it came from
  in square brackets, for example [2] or [1][3].
- If the answer is not present, say:
  "I couldn't find that information in the provided documents."
- Do not make up facts.
- Be concise but informative.

Sources:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""

        return prompt.strip()

    @staticmethod
    def build_rewrite(
        question: str,
        history: List[Dict],
    ) -> str:
        """
        Prompt that turns a follow-up into a standalone question.
        """

        conversation = "\n".join(
            f"{message['role'].capitalize()}: "
            f"{message['content'][:500]}"
            for message in history
        )

        prompt = f"""
Rewrite the follow-up question so it can be understood without the
conversation. Replace pronouns and references such as "it", "that"
or "the second one" with what they refer to. If the question already
stands on its own, return it unchanged.

Reply with the rewritten question only. Do not answer it.

Conversation:
--------------------
{conversation}
--------------------

Follow-up question:
{question}

Standalone question:
"""

        return prompt.strip()
