from dataclasses import dataclass, field
from typing import Literal


@dataclass
class UserMessage:
    content: str
    role: Literal["user"] = field(default="user")

    def to_dict(self):
        return {"role": self.role, "content": self.content}
