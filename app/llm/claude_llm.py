"""
Claude (Anthropic) chat model.
"""

import anthropic

# If Claude's safety classifiers decline a request, the API re-runs
# it on a fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

REFUSAL_MESSAGE = "The model declined to answer this request."


class ClaudeLLM:
    """
    Cloud model served through the Anthropic API.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
    ):
        self.client = anthropic.Anthropic(
            api_key=api_key,
        )
        self.model = model

    def _request(self, prompt: str, max_tokens: int) -> dict:
        return dict(
            model=self.model,
            max_tokens=max_tokens,
            output_config={"effort": "medium"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

    def generate(self, prompt: str) -> str:
        response = self.client.beta.messages.create(
            **self._request(prompt, max_tokens=16000),
        )

        # A refusal has no usable text; callers treat a blank
        # reply as "no answer".
        if response.stop_reason == "refusal":
            return ""

        return "".join(
            block.text
            for block in response.content
            if block.type == "text"
        ).strip()

    def stream(self, prompt: str):
        with self.client.beta.messages.stream(
            **self._request(prompt, max_tokens=64000),
        ) as stream:

            answered = False

            for text in stream.text_stream:
                answered = True
                yield text

            final = stream.get_final_message()

        if final.stop_reason == "refusal" and not answered:
            yield REFUSAL_MESSAGE
