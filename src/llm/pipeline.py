from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy
import json
import traceback
from typing import Any, Callable, Literal
import uuid

from openai.types.chat import ChatCompletionToolUnionParam

from src.llm.utils import (
    LLMGenerationConfig,
    SemanticRouterTarget,
    SEMANTIC_ROUTER_TARGETS,
)
from src.constants import (
    AGENTIC_SYSTEM_PROMPT,
    CONVERSATIONAL_SYSTEM_PROMPT,
    PRIMITIVE_TOOLS_DIR,
    SEARCH_SYSTEM_PROMPT,
    SYSTEM_REMINDERS,
)
from src.llm.client import LLMClient
from src.llm.evolution import ToolCallResult, get_tools_from, tool_registry
from src.llm.history import ChatHistory
from src.vector_db.client import knowledge_base_client


@dataclass
class Pipe:
    active: bool = field(default=True)
    implementation: Callable[[], None] | None = field(default=None)

    def invoke(self, once: bool = False):
        if not self.implementation or not self.active:
            return

        self.implementation()
        if once:
            self.disable()

    def disable(self):
        self.active = False

    def activate(self):
        self.active = True


@dataclass
class SeaPipes:
    use_system_message: Pipe = field(default_factory=lambda: Pipe())
    use_user_message: Pipe = field(default_factory=lambda: Pipe())
    use_available_agents_injection: Pipe = field(default_factory=lambda: Pipe())
    use_available_knowledge_base_collections_injection: Pipe = field(
        default_factory=lambda: Pipe()
    )
    force_llm_to_think: Pipe = field(default_factory=lambda: Pipe())
    use_semantic_router: Pipe = field(default_factory=lambda: Pipe())
    use_tools: Pipe = field(default_factory=lambda: Pipe())
    generate: Pipe = field(default_factory=lambda: Pipe())


@dataclass
class SeaState:
    prompt: str | None = field(default=None)
    system_prompt: str | None = field(default=None)
    llm_response: dict[str, Any] | None = field(default=None)
    full_chat_history: ChatHistory = field(default_factory=lambda: ChatHistory())
    scoped_chat_history: ChatHistory = field(default_factory=lambda: ChatHistory())
    tools_to_pass: list[ChatCompletionToolUnionParam] = field(
        default_factory=lambda: []
    )
    invoked_tools: list[tuple[str, str, dict[str, Any]]] = field(
        default_factory=lambda: []
    )

    def soft_reset(self, *, ctx_type: Literal['full'] | Literal['fragmented']):
        match ctx_type:
            case 'fragmented':
                self.scoped_chat_history.clear()
                self.invoked_tools = []
            case 'full':
                self.invoked_tools = []

    def hard_reset(self, *, ctx_type: Literal['full'] | Literal['fragmented']):
        self.soft_reset(ctx_type=ctx_type)

        self.prompt = None
        self.system_prompt = None
        self.llm_response = None

@dataclass
class SeaConfig:
    llm_client: LLMClient
    short_term_memory: int = field(default=5)
    use_fragmented_context: bool = field(default=False)


class SeaPipeline:
    def __init__(self, *, config: SeaConfig):
        self.config = config

        self.pipes = SeaPipes()
        self.state = SeaState()

        self.run_timestamp: str | None = None
        self.run_uuid: str | None = None

    def with_system_message(
        self,
        *,
        system_prompt: str,
    ):
        def fn():
            self.state.system_prompt = system_prompt

            assert self.run_timestamp is not None
            assert self.run_uuid is not None

            collection = knowledge_base_client.get_or_create_collection(
                f"chat-history__{self.state.scoped_chat_history.id}"
            )
            collection.upsert(
                ids=[f"{self.run_timestamp}__{self.run_uuid}"],
                documents=[f"SYSTEM PROMPT: {self.state.system_prompt}"],
            )

            self.state.scoped_chat_history.upsert_system_message(
                self.state.system_prompt
            )

        self.pipes.use_system_message.implementation = fn
        return self

    def with_user_prompt(
        self, *, prompt_factory: Callable[[], str] = lambda: input(">>> ")
    ):
        def fn():
            if self.state.prompt is not None:
                return

            self.state.prompt = prompt_factory()

            assert self.run_timestamp is not None
            assert self.run_uuid is not None

            collection = knowledge_base_client.get_or_create_collection(
                f"chat-history__{self.state.scoped_chat_history.id}"
            )
            collection.upsert(
                ids=[f"{self.run_timestamp}__{self.run_uuid}"],
                documents=[f"USER PROMPT: {self.state.prompt}"],
            )

            self.state.scoped_chat_history.add_user_message(self.state.prompt)

        self.pipes.use_user_message.implementation = fn
        return self

    def with_available_agents_injection(self):
        from src.llm.tools import get_available_agents

        def fn():
            agents = get_available_agents.invoke()
            print(f"[INJECTION] [AVAILABLE AGENTS] {agents}")
            self.state.scoped_chat_history.inject_tool_call_result(
                tool_name=get_available_agents.invoke.__name__,
                content=json.dumps(agents),
            )

        self.pipes.use_available_agents_injection.implementation = fn
        return self

    def with_available_knowledge_base_collections_injection(self):
        from src.llm.tools import get_available_collections_in_knowledge_base

        def fn():
            available_collections = get_available_collections_in_knowledge_base.invoke()
            print(
                f"[INJECTION] [AVAILABLE KNOWLEDGE BASE COLLECTIONS] {available_collections}"
            )
            self.state.scoped_chat_history.inject_tool_call_result(
                tool_name=get_available_collections_in_knowledge_base.invoke.__name__,
                content=json.dumps(available_collections),
            )

        self.pipes.use_available_knowledge_base_collections_injection.implementation = (
            fn
        )
        return self

    def force_llm_to_think(self):
        def fn():
            self.state.scoped_chat_history.inject_tool_call_result(
                tool_name="reminder", content=SYSTEM_REMINDERS
            )

        self.pipes.force_llm_to_think.implementation = fn
        return self

    def with_semantic_router(self, *, config: LLMGenerationConfig):
        def fn():
            from src.llm.tools import categorize_prompt

            stream = self.config.llm_client.stream(
                model=config.model,
                chat_history=self.state.scoped_chat_history,
                tools=[categorize_prompt.spec],
            )
            tool_calls: list[tuple[str, str, dict[str, Any]]] = []
            stream.process(
                on_tool_call_token=config.on_tool_call_token,
                on_generation_finish=config.on_generation_finish,
                on_parsed_tool_call=tool_calls.append,
            )
            if len(tool_calls) == 0:
                return

            category: SemanticRouterTarget | None = None
            try:
                for tool_id, tool_name, args in tool_calls:
                    if tool_name == "categorize_prompt" and "category" in args:
                        tool_call_result = (
                            self._run_tools(tools=[(tool_id, tool_name, args)]) or {}
                        )
                        tool_call_result = tool_call_result[tool_id]
                        if tool_call_result.error is not None:
                            raise Exception(tool_call_result.error)

                        category = tool_call_result.result
                        assert category in SEMANTIC_ROUTER_TARGETS
                        break
            except AssertionError:
                return

            if category is None:
                return

            match category:
                case "search":
                    print("PASSING PROMPT TO SEARCH LLM")
                    from src.llm.tools import (
                        query_knowledge_base,
                        search_for_information_on_the_web,
                    )

                    self.with_system_message(
                        system_prompt=SEARCH_SYSTEM_PROMPT(),
                    )
                    self.with_available_knowledge_base_collections_injection()
                    self.with_tools(
                        tools_factory=lambda: [
                            query_knowledge_base.spec,
                            search_for_information_on_the_web.spec,
                        ]
                    )

                case "agentic":
                    print("PASSING PROMPT TO AGENTIC LLM")
                    self.with_system_message(
                        system_prompt=AGENTIC_SYSTEM_PROMPT(),
                    )
                    self.with_available_knowledge_base_collections_injection()
                    self.with_available_agents_injection()
                    self.force_llm_to_think()
                    self.with_tools(
                        tools_factory=lambda: get_tools_from(
                            dir=PRIMITIVE_TOOLS_DIR, module_name="tools", evolved=False
                        )
                    )

                case "conversational":
                    print("PASSING PROMPT TO CONVERSATIONAL LLM")
                    from src.llm.tools import (
                        query_knowledge_base,
                        add_to_knowledge_base,
                        update_data_in_knowledge_base,
                        forget_data_from_knowledge_base,
                    )

                    self.with_available_knowledge_base_collections_injection()
                    self.with_system_message(
                        system_prompt=CONVERSATIONAL_SYSTEM_PROMPT(),
                    )
                    self.with_tools(
                        tools_factory=lambda: [
                            query_knowledge_base.spec,
                            add_to_knowledge_base.spec,
                            update_data_in_knowledge_base.spec,
                            forget_data_from_knowledge_base.spec,
                        ]
                    )

        self.pipes.use_semantic_router.implementation = fn
        return self

    def with_tools(
        self, *, tools_factory: Callable[[], list[ChatCompletionToolUnionParam]]
    ):
        def fn():
            self.state.tools_to_pass = tools_factory()

        self.pipes.use_tools.implementation = fn
        return self

    def generate(self, *, config: LLMGenerationConfig):
        def fn():
            stream = self.config.llm_client.stream(
                model=config.model,
                chat_history=self.state.scoped_chat_history,
                tools=self.state.tools_to_pass,
            )
            response = stream.process(
                on_content_token=config.on_content_token,
                on_tool_call_token=config.on_tool_call_token,
                on_generation_finish=config.on_generation_finish,
                on_parsed_tool_call=self.state.invoked_tools.append,
            )
            self.state.llm_response = response.to_dict()

            assert self.run_timestamp is not None
            assert self.run_uuid is not None

            collection = knowledge_base_client.get_or_create_collection(
                f"chat-history__{self.state.scoped_chat_history.id}"
            )
            if response.content:
                collection.upsert(
                    ids=[f"{self.run_timestamp}__{self.run_uuid}"],
                    documents=[f"SEA ASSISTANT RESPONSE: {response.content}"],
                )

            if response.tool_calls is not None and len(response.tool_calls) > 0:
                formatted_tool_calls = [
                    f"{tool_call.function.name}({tool_call.function.arguments})" + "\n"
                    for tool_call in response.tool_calls
                ]
                collection.upsert(
                    ids=[f"{self.run_timestamp}__{self.run_uuid}"],
                    documents=[
                        f"SEA ASSISTANT TOOL CALLS:\n{'    - '.join(formatted_tool_calls)}"
                    ],
                )

            if self.state.llm_response is not None:
                self.state.scoped_chat_history.append(self.state.llm_response)

        self.pipes.generate.implementation = fn
        return self

    def _inject_short_term_memory_summary(self):
        from src.llm.tools import summarize

        system_message_indexes: list[int] = []
        for idx, message in enumerate(self.state.full_chat_history):
            if message["role"] == "system":
                if len(system_message_indexes) > self.config.short_term_memory:
                    system_message_indexes.pop(0)
                system_message_indexes.append(idx)

        if len(system_message_indexes) == 0:
            return

        start = system_message_indexes[0]
        end = system_message_indexes[-1]
        windowed_context = (
            self.state.full_chat_history[start:end]
            if end != start
            else self.state.full_chat_history[start:]
        )
        self.state.scoped_chat_history.inject_tool_call_result(
            tool_name="summary_of_ongoing_conversation",
            content=json.dumps(
                summarize.invoke(
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
            ),
        )

    def _human_in_the_loop(self, *, tool_name: str) -> tuple[bool, str | None]:
        prompt = input(
            f"[Allow Sea to run the `{tool_name}` tool with those arguments]\n[y/N/<reason_for_refusal>] >>> "
        )

        if prompt.lower() == "y":
            return True, None

        return False, None if prompt == "N" else prompt

    def _run_tools(self, *, tools: list[tuple[str, str, dict[str, Any]]]):
        if len(tools) == 0:
            return

        tool_call_results: dict[str, ToolCallResult] = {}

        for tool_id, tool_name, args in tools:
            tool = tool_registry.get(tool_name)
            if tool is None:
                tool_call_results[tool_id] = ToolCallResult(
                    tool=tool_name,
                    success=False,
                    error="Tool does not exist in registry",
                    result=None,
                )
                continue

            try:
                allowed, message = (
                    self._human_in_the_loop(tool_name=tool_name)
                    if tool.requires_hitl
                    else (True, None)
                )
                if allowed:
                    ret = tool.invoke(**args)
                    tool_call_results[tool_id] = ToolCallResult(
                        tool=tool_name, success=True, error=None, result=ret
                    )
                else:
                    tool_call_results[tool_id] = ToolCallResult(
                        tool=tool_name,
                        success=False,
                        error=f"[REFUSAL FROM USER] {message or 'I cannot allow you to proceed with this'}",
                        result=None,
                    )
            except Exception:
                tb = traceback.format_exc()
                tool_call_results[tool_id] = ToolCallResult(
                    tool=tool_name, success=False, error=str(tb), result=None
                )

        self.state.scoped_chat_history.add_tool_call_results(tool_call_results)
        return tool_call_results

    def _run_with_fragmented_context(self):
        self.pipes.use_system_message.activate()
        self.pipes.use_user_message.activate()
        self.pipes.use_tools.activate()
        self.pipes.use_semantic_router.activate()
        self.pipes.use_available_agents_injection.activate()
        self.pipes.use_available_knowledge_base_collections_injection.activate()
        self.pipes.force_llm_to_think.activate()
        self.pipes.generate.activate()

        check_if_semantic_router_is_used = (
            lambda: self.pipes.use_semantic_router.active
            and self.pipes.use_semantic_router.implementation is not None
        )

        first_pass = True

        while True:
            last_scoped_chat_history = ChatHistory()
            last_scoped_chat_history.extend(self.state.scoped_chat_history)

            if not first_pass:
                self.state.soft_reset(ctx_type='fragmented')

            self.pipes.use_system_message.invoke()
            self.pipes.use_user_message.invoke()
            self.pipes.use_tools.invoke()

            if not first_pass:
                self._inject_short_term_memory_summary()

            if check_if_semantic_router_is_used():
                self.pipes.use_semantic_router.invoke(once=True)
                self.pipes.use_system_message.invoke()
                self.pipes.use_tools.invoke()
            else:
                self.pipes.use_available_knowledge_base_collections_injection.invoke()
                self.pipes.use_available_agents_injection.invoke()
                self.pipes.force_llm_to_think.invoke()

            self.pipes.generate.invoke()

            if len(self.state.invoked_tools) > 0:
                self._run_tools(tools=self.state.invoked_tools)

            first_pass = False
            if len(self.state.invoked_tools) == 0:
                break

        saved_scoped_history = deepcopy(self.state.scoped_chat_history)
        self.state.full_chat_history.extend(saved_scoped_history)

        self.state.hard_reset(ctx_type='fragmented')

        self.run_timestamp = None
        self.run_uuid = None

        return saved_scoped_history

    def _run_with_full_context(self):
        self.pipes.use_system_message.activate()
        self.pipes.use_user_message.activate()
        self.pipes.use_tools.activate()
        self.pipes.use_semantic_router.activate()
        self.pipes.use_available_agents_injection.activate()
        self.pipes.use_available_knowledge_base_collections_injection.activate()
        self.pipes.force_llm_to_think.activate()
        self.pipes.generate.activate()

        first_pass = True

        check_if_semantic_router_is_used = (
            lambda: self.pipes.use_semantic_router.active
            and self.pipes.use_semantic_router.implementation is not None
        )

        while True:
            if not first_pass:
                self.state.soft_reset(ctx_type='full')

            self.pipes.use_system_message.invoke()
            self.pipes.use_user_message.invoke(once=True)
            self.pipes.use_tools.invoke()

            if check_if_semantic_router_is_used():
                self.pipes.use_semantic_router.invoke(once=True)
                self.pipes.use_system_message.invoke()
                self.pipes.use_tools.invoke()
            else:
                self.pipes.use_available_knowledge_base_collections_injection.invoke()
                self.pipes.use_available_agents_injection.invoke()
                self.pipes.force_llm_to_think.invoke()

            self.pipes.generate.invoke()
            self._run_tools(tools=self.state.invoked_tools)

            first_pass = False
            if len(self.state.invoked_tools) == 0:
                break

        saved_scoped_history = deepcopy(self.state.scoped_chat_history)

        self.state.hard_reset(ctx_type='full')

        self.run_timestamp = None
        self.run_uuid = None

        return saved_scoped_history

    def run(self, *, debug: bool = False):
        self.run_timestamp = str(int(datetime.now().timestamp() * 1000))
        self.run_uuid = str(uuid.uuid4())

        saved_scoped_history = (
            self._run_with_fragmented_context()
            if self.config.use_fragmented_context
            else self._run_with_full_context()
        )

        if debug:
            print(saved_scoped_history)
        return saved_scoped_history

    def chat_loop(self, *, debug: bool = False) -> list[ChatHistory]:
        while True:
            self.run(debug=debug)
