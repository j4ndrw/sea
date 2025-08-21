import json
from dataclasses import asdict
from typing import Any, Callable

from src.llm.evolution import ToolCallResult


class ChatHistory(list[dict[str, Any]]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_user_message(self, prompt: str):
        self.append({"role": "user", "content": prompt})
        return self

    def add_assistant_message(
        self, content: str, *, handle_content: Callable[[str], None] = lambda _: None
    ):
        self.append({"role": "assistant", "content": content})
        handle_content(content)
        return self

    def add_system_message(self, prompt: str):
        self.insert(0, {"role": "system", "content": prompt})
        return self

    def add_tool_call_results(self, tool_call_results: dict[str, ToolCallResult]):
        for tool_id, result in tool_call_results.items():
            self.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps(asdict(result)),
                }
            )
        return self

    def reset(self):
        self.clear()
        return self
