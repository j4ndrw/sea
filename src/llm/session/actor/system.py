from dataclasses import dataclass, field
from typing import Literal

from src.llm.session.actor.actor import Actor
from src.llm.session.message.system import SystemMessage


@dataclass
class SystemActor(Actor):
    message: SystemMessage | None = field(default=None)
    role: Literal["system"] = field(default="system")

    @staticmethod
    def with_message(
        content: str,
        *,
        turns_allowed: int | Literal["unlimited"] = "unlimited",
    ) -> "SystemActor":
        return SystemActor(turns_allowed=turns_allowed, message=SystemMessage(content))

    def invoke(self):
        assert self.message is not None
        return self.message
