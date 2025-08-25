from dataclasses import dataclass, field
from typing import Callable, Literal

from openai.types.chat import ChatCompletionMessageFunctionToolCall

from src.llm.session.operations.round import InteractiveRoundOperations, RoundOperations
from src.llm.session.operations.actor import ActorOperations
from src.llm.session.operations.injection import InjectionOperations
from src.llm.session.operations.operations import SessionOperations
from src.llm.session.operations.turn import TurnOperations
from src.llm.session.state import SessionState
from src.llm.session.actor.assistant import AssistantActor
from src.llm.session.actor.tool import ToolActor
from src.llm.session.actor.actor import Actor
from src.llm.session.actor.user import UserActor
from src.llm.session.actor.system import SystemActor
from src.utils import distinct_by


@dataclass
class Session:
    looped: bool
    static_actors: list[Actor]
    main_assistant_actor: AssistantActor
    tool_actor_spawner: (
        Callable[[ChatCompletionMessageFunctionToolCall], ToolActor] | None
    ) = field(default_factory=lambda: None)

    def __post_init__(self):
        self.state = SessionState()
        self.ops = SessionOperations(
            turn=TurnOperations(state=self.state),
            injection=InjectionOperations(state=self.state),
            actor=ActorOperations(state=self.state),
            round=RoundOperations(state=self.state),
        )

    def start(self):
        assert len(self.static_actors) > 0, (
            "Cannot start chat session with no user prompt and no system prompt..."
        )
        while True:
            self.ops.round.on_start(static_actors=self.static_actors)

            self.state.actors = distinct_by(
                lambda actor: actor.id,
                self.static_actors
                + self.state.injected_system_actors
                + self.state.injected_tool_actors
                + self.state.injected_assistant_actors
            )
            processed: list[str] = []

            while True:
                actor: Actor | Literal['end-round'] | None = None
                for _actor in self.state.actors:
                    if _actor == 'end-round' or _actor.id not in processed:
                        actor = _actor
                        break

                if actor == 'end-round':
                    break

                if actor is not None:
                    self.act(actor)
                    processed.append(actor.id)

                if len(self.state.actors) > 0 and len(processed) == len(self.state.actors):
                    self.act(self.main_assistant_actor)

            self.ops.round.on_end()

            if not self.looped:
                break

    def act(self, actor: Actor):
        self.ops.turn.on_turn_start(actor)
        if not self.ops.actor.is_evicted(actor):
            if isinstance(actor, (SystemActor, UserActor)):
                self.ops.turn.static_actor(actor)
            if isinstance(actor, AssistantActor):
                self.ops.turn.assistant_actor(
                    actor, tool_actor_spawner=self.tool_actor_spawner
                )
            if isinstance(actor, ToolActor):
                self.ops.turn.tool_actor(actor)
        self.ops.turn.on_turn_end(actor)

@dataclass
class InteractiveSession(Session):
    def __post_init__(self):
        self.state = SessionState()
        self.ops = SessionOperations(
            turn=TurnOperations(state=self.state),
            injection=InjectionOperations(state=self.state),
            actor=ActorOperations(state=self.state),
            round=InteractiveRoundOperations(state=self.state),
        )
