import uuid
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Actor:
    turns_allowed: int | Literal["unlimited"] = field(default="unlimited")

    def __post_init__(self):
        self.id = str(uuid.uuid4())
        self.turns_taken: int = 0
