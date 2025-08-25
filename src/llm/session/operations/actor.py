from dataclasses import dataclass
from src.llm.session.state import SessionState
from src.llm.session.actor.actor import Actor


@dataclass
class ActorOperations:
    state: SessionState

    def evict(self, actor: Actor):
        self.state.evicted_actors.append(actor.id)

    def is_evicted(self, actor: Actor):
        return actor.id in self.state.evicted_actors

    def enroll_in_new_round(self, *, static_actors: list[Actor]):
        self.state.actors.clear()
        self.state.evicted_actors.clear()
        for actor in (
            static_actors
            + self.state.injected_system_actors
            + self.state.injected_assistant_actors
            + self.state.injected_tool_actors
        ):
            actor.turns_taken = 0
