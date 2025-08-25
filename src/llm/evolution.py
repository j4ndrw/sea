import importlib.util
import inspect
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import Any, Callable, Literal

import pydantic
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionToolUnionParam,
)
from openai.types.chat.chat_completion_tool_param import FunctionDefinition


@dataclass
class Tool:
    spec: ChatCompletionToolUnionParam
    invoke: Callable
    requires_hitl: bool = field(default=False)
    standalone: bool = field(default=False)


tool_registry: dict[str, Tool] = {}


@dataclass
class ToolCallResult:
    success: bool
    error: str | None
    result: Any

    def serialized(self):
        return json.dumps(asdict(self))


# Credit: https://github.com/ollama/ollama-python/blob/main/ollama/_utils.py#L13
def _parse_docstring(doc_string: str | None) -> dict[str, str]:
    parsed_docstring = defaultdict(str)
    if not doc_string:
        return parsed_docstring

    key = str(hash(doc_string))
    for line in doc_string.splitlines():
        lowered_line = line.lower().strip()
        if lowered_line.startswith("args:"):
            key = "args"
        elif lowered_line.startswith(("returns:", "yields:", "raises:")):
            key = "_"

        else:
            parsed_docstring[key] += f"{line.strip()}\n"

    last_key = None
    for line in parsed_docstring["args"].splitlines():
        line = line.strip()
        if ":" in line:
            # Split the line on either:
            # 1. A parenthetical expression like (integer) - captured in group 1
            # 2. A colon :
            # Followed by optional whitespace. Only split on first occurrence.
            parts = re.split(r"(?:\(([^)]*)\)|:)\s*", line, maxsplit=1)

            arg_name = parts[0].strip()
            last_key = arg_name

            # Get the description - will be in parts[1] if parenthetical or parts[-1] if after colon
            arg_description = parts[-1].strip()
            if len(parts) > 2 and parts[1]:  # Has parenthetical content
                arg_description = parts[-1].split(":", 1)[-1].strip()

            parsed_docstring[last_key] = arg_description

        elif last_key and line:
            parsed_docstring[last_key] += " " + line

    return parsed_docstring


# Credit: https://github.com/ollama/ollama-python/blob/main/ollama/_utils.py#L56
def convert_function_to_tool(func: Callable) -> ChatCompletionFunctionToolParam:
    doc_string_hash = str(hash(inspect.getdoc(func)))
    parsed_docstring = _parse_docstring(inspect.getdoc(func))
    schema = type(
        func.__name__,
        (pydantic.BaseModel,),
        {
            "__annotations__": {
                k: v.annotation if v.annotation != inspect._empty else str
                for k, v in inspect.signature(func).parameters.items()
            },
            "__signature__": inspect.signature(func),
            "__doc__": parsed_docstring[doc_string_hash],
        },
    ).model_json_schema()

    for k, v in schema.get("properties", {}).items():
        # If type is missing, the default is string
        types = (
            {t.get("type", "string") for t in v.get("anyOf")}
            if "anyOf" in v
            else {v.get("type", "string")}
        )
        if "null" in types:
            schema["required"].remove(k)
            types.discard("null")

        schema["properties"][k] = {
            "description": parsed_docstring[k],
            "type": ", ".join(types),
        }
    schema["additionalProperties"] = False

    return ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name=func.__name__,
            description=schema.get("description", ""),
            parameters=dict(**schema),
            strict=True,
        ),
    )


@dataclass
class Description:
    details: str
    args: list[tuple[str, str]] = field(default_factory=lambda: [])
    returns: list[tuple[str, str]] = field(default_factory=lambda: [])


def update_function_docstring(
    *,
    kind: Literal["tool"] | Literal["resource"],
    description: Description | None = None,
):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if description is None:
            return wrapper

        wrapper.__doc__ = f"\n    Function type: {kind.capitalize()}"
        wrapper.__doc__ = "\n    ---"
        wrapper.__doc__ = f"{description.details}"
        if len(description.args) > 0:
            wrapper.__doc__ += "\n\n    Args: \n        "
            wrapper.__doc__ += "\n        ".join(
                [*map(lambda arg: f"{arg[0]}: {arg[1]}", description.args)]
            )
        if len(description.returns) > 0:
            wrapper.__doc__ += "\n\n    Returns: \n        "
            wrapper.__doc__ += "\n        ".join(
                [*map(lambda ret: f"{ret[0]}: {ret[1]}", description.returns)]
            )

        return wrapper

    return decorator


def tool(
    details: str,
    *,
    args: list[tuple[str, str]] = [],
    returns: list[tuple[str, str]] = [],
    requires_hitl: bool = False,
    standalone: bool = False,
):
    def decorator(func: Callable):
        @update_function_docstring(
            kind="tool",
            description=Description(details=details, args=args, returns=returns),
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        tool = Tool(
            spec=convert_function_to_tool(wrapper),
            invoke=wrapper,
            requires_hitl=requires_hitl,
            standalone=standalone,
        )
        tool_registry[wrapper.__name__] = tool
        return tool

    return decorator


def get_tools_from(*, dir: str, module_name: str, evolved: bool):
    module_path = os.path.join(dir, f"{module_name}.py")
    package_name = os.path.relpath(dir).replace(os.path.sep, ".")

    module_name = f"{package_name}.{module_name}" if package_name else module_name

    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tools: list[tuple[str, Tool]] = [
        *filter(
            lambda t: not t[1].standalone,
            inspect.getmembers(
                module,
                lambda m: isinstance(m, Tool),
            ),
        )
    ]
    tool_specs = [tool.spec for (_, tool) in tools]
    for tool_name, tool in tools:
        tool_registry[tool_name] = Tool(
            spec=tool.spec,
            invoke=tool.invoke,
            requires_hitl=True if evolved else tool.requires_hitl,
        )
    return tool_specs
