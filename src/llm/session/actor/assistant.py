from dataclasses import dataclass, field
from typing import Callable, Literal

from openai.types.chat import (
    ChatCompletionMessageToolCallUnionParam,
    ParsedChatCompletionMessage,
)

from src.llm.history import ChatHistory
from src.llm.session.actor.actor import Actor
from src.llm.session.message.assistant import AssistantMessage


@dataclass
class AssistantActor(Actor):
    message: AssistantMessage | None = field(default=None)
    role: Literal["assistant"] = field(default="assistant")
    response_factory: (
        Callable[[ChatHistory], ParsedChatCompletionMessage[None]] | None
    ) = field(default=None)

    @staticmethod
    def with_message(
        content: str,
        *,
        turns_allowed: int | Literal["unlimited"] = "unlimited",
        tool_calls: list[ChatCompletionMessageToolCallUnionParam],
    ) -> "AssistantActor":
        return AssistantActor(
            turns_allowed=turns_allowed,
            message=AssistantMessage(content=content, tool_calls=tool_calls),
        )

    @staticmethod
    def with_stream(
        response_factory: Callable[[ChatHistory], ParsedChatCompletionMessage[None]],
        *,
        turns_allowed: int | Literal["unlimited"] = "unlimited",
    ) -> "AssistantActor":
        actor = AssistantActor(turns_allowed=turns_allowed)
        actor.response_factory = response_factory
        return actor

    def invoke(self, history: ChatHistory):
        if self.message:
            return self.message

        assert self.response_factory is not None

        response = self.response_factory(history)
        return AssistantMessage(
            content=response.content or "",
            tool_calls=response.tool_calls or [],  # pyright: ignore
        )
