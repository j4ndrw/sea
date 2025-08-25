from dataclasses import dataclass

from src.llm.session.state import SessionState
from src.llm.session.actor.assistant import AssistantActor
from src.llm.session.actor.tool import ToolActor
from src.llm.session.actor.system import SystemActor


@dataclass
class InjectionOperations:
    state: SessionState

    def inject_tool(self, actor: ToolActor, *, deferred: bool = False):
        if deferred:
            self.state.actors.append(actor)
        else:
            self.state.injected_tool_actors.append(actor)

    def inject_assistant(self, actor: AssistantActor, *, deferred: bool = False):
        if deferred:
            self.state.actors.append(actor)
        else:
            self.state.injected_assistant_actors.append(actor)

    def inject_system_prompt(self, prompt: str, *, deferred: bool = False):
        actor = SystemActor.with_message(content=prompt)
        if deferred:
            self.state.actors.append(actor)
        else:
            self.state.injected_system_actors.append(actor)
