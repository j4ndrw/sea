from typing import Any, Callable, Iterable

import openai
from openai.types.chat import ChatCompletionToolUnionParam, ParsedChatCompletionMessage
from openai.types.shared.chat_model import ChatModel

from src.constants import LLM_BACKEND_ENDPOINT
from src.llm.history import ChatHistory
from src.utils import StatefulGenerator


class Stream(
    StatefulGenerator[
        tuple[str, None] | tuple[None, str], ParsedChatCompletionMessage[None]
    ]
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(
        self,
        *,
        on_content_token: Callable[[str], None] = lambda _: None,
        on_tool_call_token: Callable[[str], None] = lambda _: None,
        on_parsed_tool_call: Callable[[tuple[str, str, dict[str, Any]]], None],
        on_generation_finish: Callable[[], None],
    ) -> ParsedChatCompletionMessage[None]:
        for content_token, tool_call_token in self:
            if content_token:
                on_content_token(content_token)
            if tool_call_token:
                on_tool_call_token(tool_call_token)

        llm_response = self.ret

        if llm_response.tool_calls is not None and len(llm_response.tool_calls) > 0:
            for tool_call in llm_response.tool_calls:
                args: dict[str, Any] = tool_call.function.parsed_arguments or {}  # pyright: ignore
                on_parsed_tool_call((tool_call.id, tool_call.function.name, args))

        on_generation_finish()

        return llm_response


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
                temperature=0.6,
                top_p=0.95,
                max_tokens=2048
            ) as stream:
                for event in stream:
                    if event.type == "chunk":
                        content = event.chunk.choices[0].delta.content
                        if content:
                            yield content, None

                        tool_calls = event.chunk.choices[0].delta.tool_calls
                        if tool_calls:
                            for tool_call in tool_calls:
                                if tool_call.function is not None:
                                    if tool_call.function.name is not None:
                                        yield (
                                            None,
                                            f"Executing tool call: {tool_call.function.name} ",
                                        )
                                    if tool_call.function.arguments is not None:
                                        yield None, tool_call.function.arguments

                completion = stream.get_final_completion()
                return completion.choices[0].message

        return Stream(gen(chat_history))


llm_client = LLMClient().use(url=LLM_BACKEND_ENDPOINT, api_key="placeholder")
