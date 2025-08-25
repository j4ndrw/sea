from dataclasses import dataclass

from src.llm.session.actor.user import UserActor
from src.llm.session.operations.actor import ActorOperations
from src.llm.session.state import SessionState
from src.llm.session.actor.actor import Actor


@dataclass
class RoundOperations:
    state: SessionState

    def __post_init__(self):
        self.actor_ops = ActorOperations(self.state)

    def on_start(self, *, static_actors: list[Actor]):
        self.actor_ops.enroll_in_new_round(static_actors=static_actors)

    def on_end(self):
        self.state.chat_histories.append(self.state.scoped_chat_history)
        self.state.scoped_chat_history.clear()
        self.state.actors.clear()

@dataclass
class InteractiveRoundOperations(RoundOperations):
    def on_start(self, *, static_actors: list[Actor]):
        super().on_start(static_actors=static_actors)
        static_actors.append(
            UserActor.with_interactive_message(turns_allowed="unlimited")
        )
