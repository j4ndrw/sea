from datetime import datetime
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal

from src.llm.evolution import ToolCallResult
from src.llm.session.actor.assistant import AssistantActor
from src.llm.session.actor.tool import ToolActor
from src.llm.session.actor.actor import Actor
from src.llm.session.actor.system import SystemActor
from src.llm.history import ChatHistory


@dataclass
class SessionState:
    chat_histories: list[ChatHistory] = field(default_factory=lambda: [])
    scoped_chat_history: ChatHistory = field(default_factory=lambda: ChatHistory())
    evicted_actors: list[str] = field(default_factory=lambda: [])
    injected_tool_actors: list[ToolActor] = field(default_factory=lambda: [])
    injected_assistant_actors: list[AssistantActor] = field(default_factory=lambda: [])
    injected_system_actors: list[SystemActor] = field(default_factory=lambda: [])
    tool_call_handlers: list[tuple[str, Callable[[ToolCallResult], None]]] = field(
        default_factory=lambda: []
    )
    created_at: str = field(
        default_factory=lambda: str(
            datetime.now().isoformat().replace(" ", "").replace(":", "-")
        )
    )
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    actors: list[Actor | Literal["end-round"]] = field(default_factory=lambda: [])

    def handle_tool_call_result(
        self, tool: str, handler: Callable[[ToolCallResult], None]
    ):
        self.tool_call_handlers.append((tool, handler))
