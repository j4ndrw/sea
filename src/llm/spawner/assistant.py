from typing import Callable, Literal
from openai.types.chat import ChatCompletionToolUnionParam, ParsedChatCompletionMessage
from src.llm.history import ChatHistory
from src.llm.client import LLMClient
from src.llm.session.actor.assistant import AssistantActor
from src.llm.utils import LLMGenerationConfig


def spawn_assistant_actor(
    *,
    llm_client: LLMClient,
    config: LLMGenerationConfig,
    tools_factory: Callable[[], list[ChatCompletionToolUnionParam]] = lambda: [],
    before_stream: Callable[[], None] = lambda: None,
    turns_allowed: int | Literal["unlimited"] = "unlimited",
):
    def response_factory(
        scoped_chat_history: ChatHistory,
    ) -> ParsedChatCompletionMessage[None]:
        before_stream()
        stream = llm_client.stream(
            model=config.model, chat_history=scoped_chat_history, tools=tools_factory()
        )
        response = stream.process(
            on_content_token=config.on_content_token,
            on_tool_call_token=config.on_tool_call_token,
            on_generation_finish=config.on_generation_finish,
        )
        return response

    return AssistantActor.with_stream(
        response_factory=response_factory,
        turns_allowed=turns_allowed,
    )
