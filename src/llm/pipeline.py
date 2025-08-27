from dataclasses import dataclass, field

from src.llm.spawner.assistant import spawn_assistant_actor
from src.llm.session.actor.tool import ToolActor
from src.llm.session.session import Session
from src.llm.utils import (
    LLMGenerationConfig,
    SemanticRouterTarget,
    SEMANTIC_ROUTER_TARGETS,
)
from src.constants import (
    AGENTIC_SYSTEM_PROMPT,
    CONVERSATIONAL_SYSTEM_PROMPT,
    GENERALIST_LLM,
    PRIMITIVE_TOOLS_DIR,
    SEARCH_SYSTEM_PROMPT,
    SYSTEM_REMINDERS,
)
from src.llm.client import LLMClient
from src.llm.evolution import ToolCallResult, get_tools_from


@dataclass
class SeaConfig:
    llm_client: LLMClient
    session: Session
    short_term_memory: int = field(default=5)


class SeaPipeline:
    def __init__(self, *, config: SeaConfig):
        self.config = config

    def with_available_agents_injection(self, *, deferred: bool = False):
        from src.llm.tools import get_available_agents

        def injection():
            agents = get_available_agents.invoke()
            print(f"[INJECTION] [AVAILABLE AGENTS] {agents}")
            return agents

        self.config.session.ops.injection.inject_tool(
            ToolActor.from_injected_handler(
                tool=get_available_agents.invoke.__name__,
                handler=injection,
            ),
            deferred=deferred
        )
        return self

    def with_available_knowledge_base_collections_injection(self, *, deferred: bool = False):
        from src.llm.tools import get_available_collections_in_knowledge_base

        def injection():
            available_collections = get_available_collections_in_knowledge_base.invoke()
            print(
                f"[INJECTION] [AVAILABLE KNOWLEDGE BASE COLLECTIONS] {available_collections}"
            )
            return available_collections

        self.config.session.ops.injection.inject_tool(
            ToolActor.from_injected_handler(
                tool=get_available_collections_in_knowledge_base.invoke.__name__,
                handler=injection,
            ),
            deferred=deferred
        )
        return self

    def force_llm_to_think(self, *, deferred: bool = False):
        self.config.session.ops.injection.inject_tool(
            ToolActor.injected(
                tool="reminder", result=SYSTEM_REMINDERS, turns_allowed="unlimited"
            ),
            deferred=deferred
        )
        return self

    def with_semantic_router(self, *, config: LLMGenerationConfig):
        from src.llm.tools import categorize_prompt

        def categorize_prompt_handler(tool_call_result: ToolCallResult):
            category: SemanticRouterTarget | None = None

            if tool_call_result.error is not None:
                raise Exception(tool_call_result.error)
            else:
                category = tool_call_result.result

            if category not in SEMANTIC_ROUTER_TARGETS:
                return

            self.with_available_knowledge_base_collections_injection(deferred=True)
            match category:
                case "search":
                    print("PASSING PROMPT TO SEARCH LLM")
                    from src.llm.tools import (
                        dump_knowledge_base_collection,
                        query_knowledge_base,
                        search_for_information_on_the_web,
                    )

                    self.config.session.ops.injection.inject_system_prompt(
                        SEARCH_SYSTEM_PROMPT(), deferred=True
                    )
                    self.config.session.main_assistant_actor = spawn_assistant_actor(
                        llm_client=self.config.llm_client,
                        config=config.with_model(GENERALIST_LLM),
                        tools_factory=lambda: [
                            dump_knowledge_base_collection.spec,
                            query_knowledge_base.spec,
                            search_for_information_on_the_web.spec,
                        ],
                    )

                case "agentic":
                    print("PASSING PROMPT TO AGENTIC LLM")
                    self.config.session.ops.injection.inject_system_prompt(
                        AGENTIC_SYSTEM_PROMPT()
                    )
                    self.with_available_agents_injection(deferred=True)
                    self.force_llm_to_think(deferred=True)
                    self.config.session.main_assistant_actor = spawn_assistant_actor(
                        llm_client=self.config.llm_client,
                        config=config.with_model(GENERALIST_LLM),
                        tools_factory=lambda: get_tools_from(
                            dir=PRIMITIVE_TOOLS_DIR,
                            module_name="tools",
                            evolved=False,
                        ),
                    )

                case "conversational":
                    print("PASSING PROMPT TO CONVERSATIONAL LLM")
                    from src.llm.tools import (
                        dump_knowledge_base_collection,
                        query_knowledge_base,
                        add_to_knowledge_base,
                        update_data_in_knowledge_base,
                        forget_data_from_knowledge_base,
                    )

                    self.config.session.ops.injection.inject_system_prompt(
                        CONVERSATIONAL_SYSTEM_PROMPT(),
                        deferred=True,
                    )
                    self.config.session.main_assistant_actor = spawn_assistant_actor(
                        llm_client=self.config.llm_client,
                        config=config.with_model(GENERALIST_LLM),
                        tools_factory=lambda: [
                            dump_knowledge_base_collection.spec,
                            query_knowledge_base.spec,
                            add_to_knowledge_base.spec,
                            update_data_in_knowledge_base.spec,
                            forget_data_from_knowledge_base.spec,
                        ],
                    )

        self.config.session.ops.injection.inject_assistant(
            spawn_assistant_actor(
                before_stream=lambda: print("[RUNNING SEMANTIC ROUTER]"),
                turns_allowed=1,
                llm_client=self.config.llm_client,
                config=config,
                tools_factory=lambda: [categorize_prompt.spec],
            )
        )
        self.config.session.state.handle_tool_call_result(
            categorize_prompt.invoke.__name__,
            categorize_prompt_handler,
        )

        return self

    def with_short_term_memory_summary(self):
        from src.llm.tools import summarize

        def injection():
            system_message_indexes: list[int] = []
            full_chat_history = [
                message
                for history in self.config.session.state.chat_histories
                for message in history
            ]
            for idx, message in enumerate(full_chat_history):
                if message["role"] == "system":
                    if len(system_message_indexes) > self.config.short_term_memory:
                        system_message_indexes.pop(0)
                    system_message_indexes.append(idx)

            if len(system_message_indexes) == 0:
                return

            start = system_message_indexes[0]
            end = system_message_indexes[-1]
            windowed_context = (
                full_chat_history[start:end]
                if end != start
                else full_chat_history[start:]
            )
            return summarize.invoke(
                "\n".join(
                    [
                        f"{message['role']} said: {message['content']}"
                        for message in windowed_context
                        if "content" in message
                        and message["content"]
                        and message["role"] != "system"
                    ]
                )
            )

        self.config.session.ops.injection.inject_tool(
            ToolActor.from_injected_handler(
                tool="summary_of_ongoing_conversation",
                handler=injection,
            )
        )
        return self

    def run(self):
        self.config.session.start()
        return self.config.session.state.chat_histories
