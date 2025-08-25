from dataclasses import dataclass, field
from typing import Literal

from openai.types.chat import ChatCompletionMessageToolCallUnionParam


@dataclass
class AssistantMessage:
    tool_calls: list[ChatCompletionMessageToolCallUnionParam]
    content: str
    role: Literal["assistant"] = field(default="assistant")

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
        }
