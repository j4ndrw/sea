from datetime import datetime
from dataclasses import dataclass
from typing import Callable

from openai.types.chat import ChatCompletionMessageFunctionToolCall

from src.llm.session.operations.actor import ActorOperations
from src.llm.session.state import SessionState
from src.llm.session.actor.assistant import AssistantActor
from src.llm.session.actor.tool import ToolActor
from src.llm.session.actor.actor import Actor
from src.llm.session.actor.user import UserActor
from src.llm.session.actor.system import SystemActor
from src.vector_db.client import knowledge_base_client


@dataclass
class TurnOperations:
    state: SessionState

    def __post_init__(self):
        self.actor_ops = ActorOperations(self.state)

    def on_turn_start(self, actor: Actor):
        if actor.turns_allowed == "unlimited":
            return

        if actor.turns_taken < actor.turns_allowed:
            return

        self.actor_ops.evict(actor)

    def static_actor(self, actor: SystemActor | UserActor):
        if isinstance(actor, SystemActor):
            self.state.scoped_chat_history.upsert_system_message(actor.invoke().content)
        else:
            self.state.scoped_chat_history.append(actor.invoke().to_dict())

    def assistant_actor(
        self,
        actor: AssistantActor,
        *,
        tool_actor_spawner: Callable[[ChatCompletionMessageFunctionToolCall], ToolActor]
        | None,
    ):
        message = actor.invoke(self.state.scoped_chat_history)

        self.state.scoped_chat_history.append(message.to_dict())
        if len(message.tool_calls) == 0:
            self.state.actors.append("end-round")
            return

        if tool_actor_spawner is None:
            self.state.actors.append("end-round")
            return

        self.state.actors.extend(
            [
                tool_actor_spawner(tool_call)
                for tool_call in message.tool_calls
                if isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
            ]
        )

    def tool_actor(self, actor: ToolActor):
        message = actor.invoke()
        self.state.scoped_chat_history.append(message.to_dict())
        for tool, handler in self.state.tool_call_handlers:
            if message.tool == tool:
                handler(message.result)

    def on_turn_end(self, actor: Actor):
        actor.turns_taken += 1

        collection = knowledge_base_client.get_or_create_collection(
            f"chat-history__session-{self.state.session_id}__created-at-{self.state.created_at}"
        )
        collection.upsert(
            ids=[
                f"{message['role']}__{datetime.now().isoformat().replace(' ', '').replace(':', '-')}"
                for message in self.state.scoped_chat_history
            ],
            documents=[
                f"{message['role']}: {message['content']}"
                for message in self.state.scoped_chat_history
            ],
        )
