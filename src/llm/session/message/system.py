from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SystemMessage:
    content: str
    role: Literal["system"] = field(default="system")

    def to_dict(self):
        return {"role": self.role, "content": self.content}
