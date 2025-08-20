import traceback
from typing import Any, Callable

from openai.types.chat import ChatCompletionFunctionToolParam

from src.llm.client import LLMClient
from src.llm.history import ChatHistory
from src.llm.evolution import ToolCallResult, tool_registry


class Pipeline:
    llm_client: LLMClient
    chat_history: ChatHistory

    pipes: list[tuple[str, Callable[[], None]]]
    tools_to_call: list[tuple[str, str, dict[str, Any]]]
    tool_call_results: dict[str, ToolCallResult]

    def __init__(self, *, llm_client: LLMClient, chat_history: ChatHistory):
        self.pipes = []
        self.tools_to_call = []
        self.tool_call_results = {}
        self.use(llm_client=llm_client, chat_history=chat_history)

    def use(self, *, llm_client: LLMClient, chat_history: ChatHistory):
        self.llm_client = llm_client
        self.chat_history = chat_history

    def with_system_message(self, *, system_prompt: str):
        def fn():
            if len(self.chat_history) > 0 and self.chat_history[0]["role"] == "system":
                return

            self.chat_history.add_system_message(system_prompt)

        self.pipes.append(("with_system_message", fn))
        return self

    def prompt_user(self, *, prompt_factory: Callable[[], str] = lambda: input(">>> ")):
        def fn():
            prompt = prompt_factory()
            self.chat_history.add_user_message(prompt)

        self.pipes.append(("prompt_user", fn))
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
            from src.llm.tools import (
                add_to_knowledge_base,
                dispatch_agent,
                fix_agent,
                modify_agent,
                query_knowledge_base,
                register_agent,
                retrieve_agent_implementation,
                summarize,
                web_search,
            )

            stream = self.llm_client.stream(
                model=model,
                chat_history=self.chat_history,
                tools=tools
                or [
                    web_search.spec,
                    query_knowledge_base.spec,
                    add_to_knowledge_base.spec,
                    summarize.spec,
                    retrieve_agent_implementation.spec,
                    register_agent.spec,
                    modify_agent.spec,
                    fix_agent.spec,
                    dispatch_agent.spec,
                ],
            )
            for token in stream:
                on_token(token)

            llm_response = stream.ret
            if llm_response.tool_calls is not None and len(llm_response.tool_calls) > 0:
                print(f"Sea wants to use the following tools:")
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

        self.pipes.append(("generate_llm_answer", fn))
        return self

    def chat_loop(self):
        while True:
            self.run()

    def _run_tools(self):
        for tool_id, tool, args in self.tools_to_call:
            fn = tool_registry.get(tool)
            if fn is None:
                self.tool_call_results[tool_id] = ToolCallResult(
                    tool=tool,
                    success=False,
                    error="Tool does not exist in registry",
                    result=None,
                )
            else:
                try:
                    ret = fn(**args)
                    self.tool_call_results[tool_id] = ToolCallResult(
                        tool=tool, success=True, error=None, result=ret
                    )
                except Exception:
                    tb = traceback.format_exc()
                    self.tool_call_results[tool_id] = ToolCallResult(
                        tool=tool, success=False, error=str(tb), result=None
                    )

    def _human_in_the_loop(self) -> tuple[bool, str | None]:
        prompt = input(
            "[Allow Sea to run those tool calls?]\n[y/N/<reason_for_refusal>] >>> "
        )

        if prompt.lower() == "y":
            return True, None

        return False, None if prompt == "N" else prompt

    def run(self):
        for pipe_name, pipe in self.pipes:
            if pipe_name == "prompt_user":
                pipe()
                break

        while True:
            for pipe_name, pipe in self.pipes:
                if pipe_name != "prompt_user":
                    pipe()

            if len(self.tools_to_call) == 0:
                break

            # allowed, message = self._human_in_the_loop()
            # if allowed:
            self._run_tools()
            self.chat_history.add_tool_call_results(self.tool_call_results)
            # else:
            #     self.chat_history.add_user_message(
            #         message or "I cannot allow you to proceed with this."
            #     )

            self.tools_to_call = []
            self.tool_call_results = {}
