from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import uuid

from src.llm.evolution import ToolCallResult
from src.llm.session.actor.actor import Actor
from src.llm.session.message.tool import ToolMessage


@dataclass
class ToolActor(Actor):
    message: ToolMessage | None = field(default=None)
    role: Literal["tool"] = field(default="tool")
    handler: tuple[str, str, Callable[[], ToolCallResult]] | None = field(default=None)

    @staticmethod
    def with_message(
        *,
        turns_allowed: int | Literal["unlimited"] = "unlimited",
        id: str,
        tool: str,
        result: ToolCallResult,
    ) -> "ToolActor":
        return ToolActor(
            turns_allowed=turns_allowed,
            message=ToolMessage(id=id, tool=tool, result=result),
        )

    @staticmethod
    def from_handler(
        *,
        turns_allowed: int | Literal["unlimited"] = "unlimited",
        id: str,
        tool: str,
        handler: Callable[[], ToolCallResult],
    ) -> "ToolActor":
        return ToolActor(
            turns_allowed=turns_allowed,
            handler=(id, tool, handler),
        )

    @staticmethod
    def injected(
        *,
        turns_allowed: int | Literal["unlimited"] = "unlimited",
        tool: str,
        result: Any,
    ) -> "ToolActor":
        return ToolActor(
            turns_allowed=turns_allowed,
            message=ToolMessage(
                id=str(uuid.uuid4()),
                tool=f"injected-{tool}-DO-NOT-CALL-YOURSELF-THIS-IS-AUTOMATED",
                result=ToolCallResult(success=True, error=None, result=result),
            ),
        )

    @staticmethod
    def from_injected_handler(
        *,
        turns_allowed: int | Literal["unlimited"] = "unlimited",
        tool: str,
        handler: Callable[[], Any],
    ) -> "ToolActor":
        return ToolActor(
            turns_allowed=turns_allowed,
            handler=(
                str(uuid.uuid4()),
                f"injected-{tool}-DO-NOT-CALL-YOURSELF-THIS-IS-AUTOMATED",
                lambda: ToolCallResult(success=True, error=None, result=handler()),
            ),
        )

    def invoke(self):
        if self.message:
            return self.message

        assert self.handler is not None
        tool_id, tool, handle = self.handler
        return ToolMessage(id=tool_id, tool=tool, result=handle())
