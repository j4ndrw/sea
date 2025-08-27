from dataclasses import dataclass, field
from typing import Callable, Literal

from openai.types.shared.chat_model import ChatModel


SemanticRouterTarget = (
    Literal["conversational"] | Literal["search"] | Literal["agentic"]
)
SEMANTIC_ROUTER_TARGETS: list[SemanticRouterTarget] = [
    "conversational",
    "search",
    "agentic",
]


@dataclass
class LLMGenerationConfig:
    model: ChatModel | str
    on_content_token: Callable[[str], None] = field(default=lambda _: None)
    on_tool_call_token: Callable[[str], None] = field(default=lambda _: None)
    on_generation_finish: Callable[[], None] = field(default=lambda: None)

    def with_model(self, model: ChatModel | str):
        return LLMGenerationConfig(model=model, on_content_token=self.on_content_token, on_tool_call_token=self.on_tool_call_token, on_generation_finish=self.on_generation_finish)
