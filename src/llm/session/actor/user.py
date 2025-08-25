from dataclasses import dataclass, field
from typing import Literal

from src.llm.session.actor.actor import Actor
from src.llm.session.message.user import UserMessage


@dataclass
class UserActor(Actor):
    message: UserMessage | None = field(default=None)
    role: Literal["user"] = field(default="user")

    @staticmethod
    def with_message(
        content: str,
        *,
        turns_allowed: int | Literal["unlimited"] = "unlimited",
    ) -> "UserActor":
        return UserActor(
            turns_allowed=turns_allowed, message=UserMessage(content=content)
        )

    @staticmethod
    def with_interactive_message(
        *,
        prefix=">>> ",
        turns_allowed: int | Literal["unlimited"] = "unlimited",
    ) -> "UserActor":
        while True:
            content = input(prefix)
            if not content:
                print("You must type something to chat with the LLM!")
                continue

            return UserActor(
                turns_allowed=turns_allowed, message=UserMessage(content=content)
            )

    def invoke(self):
        assert self.message is not None
        return self.message
