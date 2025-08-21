from textwrap import dedent
import traceback
from typing import Any, Callable

from openai.types.chat import ChatCompletionFunctionToolParam

from src.constants import PRIMITIVE_TOOLS_DIR
from src.llm.client import LLMClient
from src.llm.evolution import ToolCallResult, get_tools_from, tool_registry
from src.llm.history import ChatHistory


class SeaPipeline:
    def __init__(self, *, llm_client: LLMClient, chat_history: ChatHistory):
        self.llm_client = llm_client
        self.chat_history = chat_history

        self.pipes: list[tuple[str, Callable[[], None], bool]] = []
        self.tools_to_call: list[tuple[str, str, dict[str, Any]]] = []
        self.tool_call_results: dict[str, ToolCallResult] = {}

    def use(self, *, llm_client: LLMClient, chat_history: ChatHistory):
        self.llm_client = llm_client
        self.chat_history = chat_history

    def with_system_message(self, *, system_prompt: str):
        def fn():
            if len(self.chat_history) > 0 and self.chat_history[0]["role"] == "system":
                return

            self.chat_history.add_system_message(system_prompt)

        self.pipes.append((self.with_system_message.__name__, fn, True))
        return self

    def prompt_user(self, *, prompt_factory: Callable[[], str] = lambda: input(">>> ")):
        def fn():
            prompt = prompt_factory()
            self.chat_history.add_user_message(prompt)

        self.pipes.append((self.prompt_user.__name__, fn, True))
        return self

    def force_llm_to_think(self):
        def fn():
            message = dedent("""
                I need to ask myself:
                    a. Is this a task that requires searching for something on the web or not?
                    b. Is this a task that I can rely on my knowledge base for (i.e. saving info, or retrieving info that I previously searched, or saved, like the user's birthday, or why the sky is blue)?
                    c. Is this a task that I can fulfill with an existing agent? Maybe I need to first see what agents I have available.
                    d. Is this a task that requires the creation of a new agent?
                    e. Is this just a regular conversation with the user?
                    f. Am I stuck? Do I have to consult my knowledge base to see if I can find a solution to get un-stuck?
            """)
            self.chat_history.add_assistant_message(message)

        self.pipes.append((self.force_llm_to_think.__name__, fn, True))
        return self

    def generate_llm_answer(
        self,
        *,
        model: str,
        on_token: Callable[[str], None] = lambda _: None,
        on_generation_finish: Callable[[], None] = lambda: None,
        tools: list[ChatCompletionFunctionToolParam] = [],
    ):
        def fn():
            stream = self.llm_client.stream(
                model=model,
                chat_history=self.chat_history,
                tools=tools,
            )
            for token in stream:
                on_token(token)

            llm_response = stream.ret
            if llm_response.tool_calls is not None and len(llm_response.tool_calls) > 0:
                print("Sea wants to use the following tools:")
                for tool_call in llm_response.tool_calls:
                    args: dict[str, Any] = tool_call.function.parsed_arguments or {}  # pyright: ignore
                    print(f"\t - {tool_call.function.name} | Args: [{[*args.items()]}]")
                    self.tools_to_call.append(
                        (
                            tool_call.id,
                            tool_call.function.name,
                            args,
                        )
                    )
            on_generation_finish()
            self.chat_history.append(llm_response.to_dict())

        self.pipes.append((self.generate_llm_answer.__name__, fn, False))
        return self

    def chat_loop(self):
        while True:
            self.run()

    def _human_in_the_loop(self, *, tool_name: str) -> tuple[bool, str | None]:
        prompt = input(
            f"[Allow Sea to run the `{tool_name}` tool with those arguments]\n[y/N/<reason_for_refusal>] >>> "
        )

        if prompt.lower() == "y":
            return True, None

        return False, None if prompt == "N" else prompt

    def _run_tools(self):
        for tool_id, tool_name, args in self.tools_to_call:
            tool = tool_registry.get(tool_name)
            if tool is None:
                self.tool_call_results[tool_id] = ToolCallResult(
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
                    self.tool_call_results[tool_id] = ToolCallResult(
                        tool=tool_name, success=True, error=None, result=ret
                    )
                else:
                    self.tool_call_results[tool_id] = ToolCallResult(
                        tool=tool_name,
                        success=False,
                        error=f"[REFUSAL FROM USER] {message or 'I cannot allow you to proceed with this'}",
                        result=None,
                    )
            except Exception:
                tb = traceback.format_exc()
                self.tool_call_results[tool_id] = ToolCallResult(
                    tool=tool_name, success=False, error=str(tb), result=None
                )

    def run(self):
        for _, pipe, is_single_run in self.pipes:
            if is_single_run:
                pipe()

        running = True
        while running:
            self.tools_to_call = []
            self.tool_call_results = {}

            for _, pipe, is_single_run in self.pipes:
                if not is_single_run:
                    pipe()

            running = len(self.tools_to_call) > 0

            if len(self.tools_to_call) > 0:
                self._run_tools()

            self.chat_history.add_tool_call_results(self.tool_call_results)
