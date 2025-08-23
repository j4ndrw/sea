from datetime import datetime
import json
from dataclasses import asdict
from typing import Any
import uuid

from src.llm.evolution import ToolCallResult


class ChatHistory(list[dict[str, Any]]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = f"{int(datetime.now().timestamp())}"

    def add_user_message(self, prompt: str):
        self.append({"role": "user", "content": prompt})
        return self

    def inject_tool_call_result(
        self, *, tool_name: str, content: str, mark_as_not_injected: bool = False
    ):
        if not mark_as_not_injected:
            tool_name = f"injected-{tool_name}-DO-NOT-CALL-YOURSELF-THIS-IS-AUTOMATED"
        tool_id = str(uuid.uuid4())

        self.add_tool_call_results(
            {
                f"{tool_name}-{tool_id}": ToolCallResult(
                    tool=tool_name, success=True, error=None, result=content
                )
            }
        )
        return self

    def upsert_system_message(self, prompt: str):
        if len(self) == 0 or self[0]["role"] != "system":
            self.insert(0, {"role": "system", "content": prompt})
            return self

        self[0] = {"role": "system", "content": prompt}
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
