from typing import Iterable

import openai
from openai.types.chat import ChatCompletionToolUnionParam
from openai.types.shared.chat_model import ChatModel

from src.constants import LLM_BACKEND_ENDPOINT
from src.llm.history import ChatHistory
from src.utils import StatefulGenerator


class LLMClient:
    _client: openai.OpenAI = None  # pyright: ignore

    def use(self, *, url: str, api_key: str | None = None):
        self._client = openai.OpenAI(base_url=url, api_key=api_key or "")
        return self

    def get(self) -> openai.OpenAI:
        return self._client

    def stream(
        self,
        *,
        model: ChatModel | str,
        chat_history: ChatHistory,
        tools: Iterable[ChatCompletionToolUnionParam] = [],
    ):
        def gen(chat_history: ChatHistory):
            with self._client.chat.completions.stream(
                model=model,
                messages=chat_history,  # pyright: ignore
                tools=tools,
            ) as stream:
                for event in stream:
                    if event.type == "chunk":
                        content = event.chunk.choices[0].delta.content
                        if content:
                            yield content

                completion = stream.get_final_completion()
                return completion.choices[0].message

        return StatefulGenerator(gen(chat_history))


llm_client = LLMClient().use(url=LLM_BACKEND_ENDPOINT)
