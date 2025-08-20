import importlib.util
import os
from inspect import getmembers
from textwrap import dedent
from typing import Any

from openai.types.chat import ChatCompletionFunctionToolParam

from src.constants import EVOLVED_AGENT_DIR, LLM
from src.llm.client import llm_client
from src.llm.evolution import Tool, tool
from src.llm.history import ChatHistory
from src.llm.pipeline import Pipeline


@tool(
    "Tool used to search for a given query on the web",
    args=[
        ("query", "The query to perform on the web"),
        ("max_results", "Max search results. Defaults to 10."),
    ],
    returns=[
        (
            "list[dict[str, str]]",
            "A list of objects of shape {url: <URL>, content: <CONTENT>, title: <TITLE>}",
        )
    ],
)
def web_search(query: str, max_results=10) -> list[dict[str, str]]:
    return [
        {
            "url": "https://example.com",
            "title": "Example Title",
            "content": "Example content",
        }
    ]


@tool(
    "Tool used to add a particular piece of info to the knowledge base",
    args=[("info", "The info to add to the knowledge base")],
)
def add_to_knowledge_base(info: str) -> None:
    pass


@tool(
    "Tool used to look for information related to the query in the knowledge base",
    args=[
        ("query", "The query to use in the knowledge base search"),
        ("max_results", "Max search results. Defaults to 10."),
    ],
    returns=[("list[str]", "List of results from the knowledge base")],
)
def query_knowledge_base(query: str, max_results=10):
    return ["Example item in knowledge base"]

@tool(
    "Tool used to get all available agents in the collection",
    returns=[
        (
            "list[str]",
            "The agents available. Always call this before starting to work on a task, if appropriate",
        )
    ],
)
def get_available_agents() -> list[str]:
    if not os.path.exists(EVOLVED_AGENT_DIR):
        return []

    agent_scripts = os.listdir(EVOLVED_AGENT_DIR)
    return [script[:-3] for script in agent_scripts if script.endswith(".py")]

@tool(
    "Tool used to summarize a piece of text into a shorter piece of text",
    args=[("text", "The text to summarize")],
    returns=[("str", "The summarized text")],
)
def summarize(text: str) -> str:
    return "Summarized text: [...]"


@tool(
    "Tool used to retrieve a given agent's implementation",
    args=[("agent", "The agent's whose implementation we want to retrieve")],
    returns=[
        (
            "str",
            "The implementation of the agent.",
        )
    ],
)
def retrieve_agent_implementation(agent: str) -> str | None:
    if not os.path.exists(EVOLVED_AGENT_DIR):
        raise FileNotFoundError(f"This {agent} agent file does not exist.")

    agent_scripts = os.listdir(EVOLVED_AGENT_DIR)
    if f"{agent}.py" not in agent_scripts:
        raise FileNotFoundError(f"This {agent} agent file does not exist.")

    implementation = ""
    with open(os.path.join(EVOLVED_AGENT_DIR, f"{agent}.py"), "r") as f:
        implementation = f.read()

    return implementation


@tool(
    """
    Tool used to register an agent.
    IMPORTANT: Always call `@tool` with a description, the arguments the function takes and the returns of the function!!!
    Something like
    @tool(
        "Some description",
        args=[(...)],
        returns=[(...)],
    )
    def my_tool(...):
        ...
    """,
    args=[
        (
            "agent_name",
            "The name of the agent you want to register (make sure it's a snake_case string)",
        ),
        (
            "implementation",
            "The python code that contains the tools you want the agent to use.",
        ),
    ],
)
def register_agent(agent_name: str, implementation: str) -> None:
    os.makedirs(EVOLVED_AGENT_DIR, exist_ok=True)

    with open(os.path.join(EVOLVED_AGENT_DIR, f"{agent_name}.py"), "w") as f:
        f.write(implementation)


@tool(
    "Tool used to fix an agent if the task fails",
    args=[
        ("agent_to_fix", "The name of the agent you want to fix"),
        (
            "fixed_implementation",
            "The python code that should replace the faulty implementation.",
        ),
    ],
)
def fix_agent(
    agent_to_fix: str,
    fixed_implementation: str,
) -> None:
    if not os.path.exists(EVOLVED_AGENT_DIR):
        raise FileNotFoundError(f"This {agent_to_fix} agent file does not exist.")

    agent_scripts = os.listdir(EVOLVED_AGENT_DIR)
    if f"{agent_to_fix}.py" not in agent_scripts:
        raise FileNotFoundError(f"This {agent_to_fix} agent file does not exist.")

    with open(os.path.join(EVOLVED_AGENT_DIR, f"{agent_to_fix}.py"), "w") as f:
        f.write(fixed_implementation)


@tool(
    "Tool used to modify an agent",
    args=[
        ("agent_to_modify", "The name of the agent you want to modify"),
        (
            "new_implementation",
            "The python code that should replace the current implementation.",
        ),
    ],
)
def modify_agent(agent_to_modify: str, new_implementation: str) -> None:
    if not os.path.exists(EVOLVED_AGENT_DIR):
        raise FileNotFoundError(f"This {agent_to_modify} agent file does not exist.")

    agent_scripts = os.listdir(EVOLVED_AGENT_DIR)
    if f"{agent_to_modify}.py" not in agent_scripts:
        raise FileNotFoundError(f"This {agent_to_modify} agent file does not exist.")

    with open(os.path.join(EVOLVED_AGENT_DIR, f"{agent_to_modify}.py"), "w") as f:
        f.write(new_implementation)


@tool(
    "Tool used to ask an agent to perform a task for you",
    args=[
        (
            "original_request",
            "The request from the user. Can be rephrased so that it makes sense for the agent.",
        ),
        (
            "context",
            'Context regarding things relevant to the user\'s request. E.g. "I did so and so and eventually created you so you take care of this task for me"',
        ),
        (
            "agent_to_dispatch",
            "The agent that is responsible with fulfilling the task",
        ),
    ],
    returns=[
        (
            "list[str]",
            "A list of things the agent said and did while performing the task",
        )
    ],
)
def dispatch_agent(
    original_request: str,
    context: str,
    agent_to_dispatch: str,
) -> list[str]:
    if not os.path.exists(EVOLVED_AGENT_DIR):
        return []

    agent_scripts = os.listdir(EVOLVED_AGENT_DIR)
    if f"{agent_to_dispatch}.py" not in agent_scripts:
        return []

    agent_module_path = os.path.join(EVOLVED_AGENT_DIR, f"{agent_to_dispatch}.py")
    agent_package_name = os.path.relpath(
        EVOLVED_AGENT_DIR
    ).replace(os.path.sep, ".")
    full_agent_module_name = (
        f"{agent_package_name}.{agent_to_dispatch}"
        if agent_package_name
        else agent_to_dispatch
    )
    spec = importlib.util.spec_from_file_location(
        full_agent_module_name, agent_module_path
    )
    tools: list[ChatCompletionFunctionToolParam] = []
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tools.extend(
            [tool.spec for (_, tool) in getmembers(module, lambda m: isinstance(m, Tool))]  # pyright: ignore
        )

    chat_history = ChatHistory()

    pipeline = Pipeline(llm_client=llm_client, chat_history=chat_history)
    pipeline = pipeline.with_system_message(
        system_prompt=dedent(f"""
        You are the `{agent_to_dispatch}`.
        You are tasked to take care of the following request from the user: `{original_request}`.

        Here's some context from the master agent that reach for your help:
        {context}
    """)
    )
    pipeline = pipeline.generate_llm_answer(
        model=LLM,
        tools=tools,
    )
    pipeline = pipeline.run()

    return [message["content"] for message in chat_history if "content" in message and message["content"] is not None]
