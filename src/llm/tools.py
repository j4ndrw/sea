import json
import os
import uuid
from textwrap import dedent

from src.constants import EVOLVED_AGENT_DIR, LLM
from src.llm.client import llm_client
from src.llm.evolution import get_tools_from, tool
from src.llm.history import ChatHistory
from src.llm.pipeline import SeaPipeline
from src.vector_db.client import knowledge_base_client


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
    "Tool used to check available collections in knowledge base. Useful to know the category of information you have available to you.",
    returns=[
        (
            "list[str]",
            "The collections in the knowledge base, available to you.",
        ),
    ],
)
def get_available_collections_in_knowledge_base() -> list[str]:
    collections = knowledge_base_client.list_collections()
    return [collection.name for collection in collections]


@tool(
    "Tool used to add a particular piece of info to the knowledge base",
    args=[
        (
            "collection",
            dedent("""
                The collection to store data in.
                For example, you may have a `user_specifics` where you store their birthday, or their name, you may have a `things_i_learned` where you store things you learned during your interactions with the user, etc...
                NOTE: This is also really useful for storing data that you found in a web search. For example, you would use your "things_from_the_web" collection or something to store the data there...
            """),
        ),
        ("info", "The info to add to the knowledge base"),
    ],
)
def add_to_knowledge_base(collection: str, info: str) -> None:
    c = knowledge_base_client.get_or_create_collection(name=collection)
    c.upsert(ids=[str(uuid.uuid4())], documents=[info])


@tool(
    "Tool used to update a particular piece of info to the knowledge base",
    args=[
        (
            "collection",
            dedent("""
                The collection to store data in.
                For example, you may have a `user_specifics` where you store their birthday, or their name, you may have a `things_i_learned` where you store things you learned during your interactions with the user, etc...
                NOTE: This is also really useful for storing data that you found in a web search. For example, you would use your "things_from_the_web" collection or something to store the data there...
            """),
        ),
        (
            "queries",
            "The queries to use in the knowledge base search, so that we can get the entries and update them. Please use multiple queries for a better search result.",
        ),
        ("replacement", "The info to replace the data with."),
    ],
)
def update_data_in_knowledge_base(
    collection: str, queries: list[str], replacement: str
) -> None:
    c = knowledge_base_client.get_or_create_collection(name=collection)
    query_results = c.query(query_texts=queries, include=["documents"])
    doc_id = query_results["ids"][0][0]
    c.upsert(ids=[doc_id], documents=[replacement])


@tool(
    "Tool used to forget a particular piece of info to the knowledge base",
    args=[
        (
            "collection",
            dedent("""
                The collection where data to forget lives in.
                For example, you may have a `user_specifics` where you store their birthday, or their name, you may have a `things_i_learned` where you store things you learned during your interactions with the user, etc...
                NOTE: This is also really useful for storing data that you found in a web search. For example, you would use your "things_from_the_web" collection or something to store the data there...
            """),
        ),
        (
            "queries",
            "The queries to use in the knowledge base search, so that we can get the entries and delete them. Please use multiple queries for a better search result.",
        ),
    ],
)
def forget_data_from_knowledge_base(collection: str, queries: list[str]) -> None:
    c = knowledge_base_client.get_or_create_collection(name=collection)
    query_results = c.query(query_texts=queries, include=["documents"])
    ids = [ids for ids_cluster in query_results["ids"] for ids in ids_cluster]
    c.delete(ids=ids)


@tool(
    "Tool used to look for information related to the query in the knowledge base",
    args=[
        (
            "collection",
            dedent("""
                The collection to look for data in.
                For example, you may have a `user_specifics` where you store their birthday, or their name, you may have a `things_i_learned` where you store things you learned during your interactions with the user, etc...
                NOTE: This is really useful for getting data that you previously searched for, in a summarized manner. For example, you would use your "things_from_the_web" collection or something...
            """),
        ),
        (
            "queries",
            "The queries to use in the knowledge base search. Try to use multiple queries for better results.",
        ),
        ("max_results", "Max search results. Defaults to 10."),
    ],
    returns=[("list[str]", "List of results from the knowledge base")],
)
def query_knowledge_base(collection: str, queries: list[str], max_results: int = 10):
    c = knowledge_base_client.get_collection(collection)
    query_results = c.query(
        query_texts=queries, n_results=max_results, include=["documents"]
    )

    doc_cluster = query_results["documents"] or []
    if len(doc_cluster) == 0:
        return []

    return [doc for docs in doc_cluster for doc in docs]


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
    chat_history = ChatHistory()
    pipeline = SeaPipeline(llm_client=llm_client, chat_history=chat_history)
    pipeline = pipeline.with_system_message(
        system_prompt=dedent(f"""
        You are the `summarizer` agent.
        You only exist to summarize text so that it's easy to comprehend in a concise form.
        Please keep the text 2 paragraphs at most.

        The text is:
        {text}
    """)
    )
    pipeline = pipeline.generate_llm_answer(model=LLM)
    return chat_history[-1]["content"]

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
        raise FileNotFoundError(f"This `{agent}` agent file does not exist.")

    agent_scripts = os.listdir(EVOLVED_AGENT_DIR)
    if f"{agent}.py" not in agent_scripts:
        raise FileNotFoundError(f"This `{agent}` agent file does not exist.")

    implementation = ""
    with open(os.path.join(EVOLVED_AGENT_DIR, f"{agent}.py"), "r") as f:
        implementation = f.read()

    return json.dumps(implementation)


@tool(
    """
    Tool used to register an agent.
    An agent is just a file containing a bunch of functions.
    Those functions are also called tools.
    IMPORTANT: Always call `@tool` with a description, the arguments the function takes and the returns of the function!!!
    <example>
        # Creating an agent named "calculator"
        from src.llm.evolution import tool

        @tool(
            "A tool that adds multiple numbers"),
            args=[("numbers", "The numbers to add")],
            returns=[("int", "The sum of the given numbers")]
        )
        def calculator__add(numbers: list[int]) -> int:
            return sum(numbers)

        @tool(
            "A tool that subtracts multiple numbers"),
            args=[("numbers", "The numbers to subtract")],
            returns=[("int", "The difference of the given numbers")]
        )
        def calculator__subtract(numbers: list[int]) -> int:
            ret = numbers[0]
            for number in numbers:
                ret -= number
            return ret

        # Notice how those helper functions don't use the @tool decorator
        def helper1(numbers: list[int]) -> int:
            ...

        # Notice how those helper functions don't use the @tool decorator
        def helper2() -> int:
            ...

        @tool(
            "A tool that performs a complex calculation on multiple numbers"),
            args=[("numbers", "The numbers to perform the calculation on")],
            returns=[("int", "The result of the calculation")]
        )
        def calculator__complex_calculation(numbers: list[int]) -> int:
            return helper1(*numbers) + helper2()
    </example>
    <notes>
        1. Always fit your solution in a single file.
        2. Always decorate your tools with the `@tool` decorator from `src.llm.evolution`
        3. Never decorate helper functions with the `@tool` decorator
        4. Never implement tools as generators or anything complex - follow the KISS (keep it simple, stupid) principle
    </notes>
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
        raise FileNotFoundError(f"This `{agent_to_fix}` agent file does not exist.")

    agent_scripts = os.listdir(EVOLVED_AGENT_DIR)
    if f"{agent_to_fix}.py" not in agent_scripts:
        raise FileNotFoundError(f"This `{agent_to_fix}` agent file does not exist.")

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
        raise FileNotFoundError(f"This `{agent_to_modify}` agent file does not exist.")

    agent_scripts = os.listdir(EVOLVED_AGENT_DIR)
    if f"{agent_to_modify}.py" not in agent_scripts:
        raise FileNotFoundError(f"This `{agent_to_modify}` agent file does not exist.")

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
            "A summary of how executing the agent went.",
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

    tools = get_tools_from(
        dir=EVOLVED_AGENT_DIR, module_name=agent_to_dispatch, evolved=True
    )

    if len(tools) == 0:
        return [
            f"Agent `{agent_to_dispatch}` has no tools in its collection. I need to check the implementation of agent `{agent_to_dispatch}` and debug..."
        ]

    pipeline = SeaPipeline(llm_client=llm_client, chat_history=ChatHistory())
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
        on_token=lambda token: print(token, end="", flush=True),
        on_generation_finish=lambda: print("\n"),
    )
    pipeline = pipeline.run()

    return [f"Agent `{agent_to_dispatch}` ran successfully!"]
