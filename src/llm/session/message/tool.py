from dataclasses import dataclass, field
from typing import Literal

from src.llm.evolution import ToolCallResult


@dataclass
class ToolMessage:
    id: str
    result: ToolCallResult
    tool: str
    role: Literal["tool"] = field(default="tool")

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.result.serialized(),
            "tool_call_id": self.id,
        }
