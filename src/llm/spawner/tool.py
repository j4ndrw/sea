import traceback
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from src.llm.spawner.utils import human_in_the_loop
from src.llm.session.actor.tool import ToolActor
from src.llm.evolution import ToolCallResult, tool_registry


def create_tool_actor_spawner():
    def handle(tool_call: ChatCompletionMessageFunctionToolCall):
        args: dict[str, Any] = tool_call.function.parsed_arguments or {}  # pyright: ignore
        tool = tool_registry.get(tool_call.function.name)
        if tool is None:
            return ToolActor.with_message(
                id=tool_call.id,
                tool=tool_call.function.name,
                result=ToolCallResult(
                    success=False,
                    error="Tool does not exist in registry",
                    result=None,
                ),
            )

        def call_tool():
            try:
                allowed, message = (
                    human_in_the_loop(tool_name=tool_call.function.name)
                    if tool.requires_hitl
                    else (True, None)
                )
                if allowed:
                    return ToolCallResult(
                        success=True, error=None, result=tool.invoke(**args)
                    )

                return ToolCallResult(
                    success=False,
                    error=f"[REFUSAL FROM USER] {message or 'I cannot allow you to proceed with this'}",
                    result=None,
                )

            except Exception:
                tb = traceback.format_exc()
                return ToolCallResult(success=False, error=str(tb), result=None)

        return ToolActor.from_handler(
            turns_allowed=1,
            id=tool_call.id,
            tool=tool_call.function.name,
            handler=call_tool,
        )

    return handle
