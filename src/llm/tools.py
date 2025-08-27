import httpx
import urllib.parse
import json
import os
import uuid

from src.llm.session.actor.system import SystemActor
from src.llm.spawner.tool import create_tool_actor_spawner
from src.llm.spawner.assistant import spawn_assistant_actor
from src.llm.session.session import Session
from src.llm.utils import LLMGenerationConfig
from src.constants import (
    AGENT_LLM,
    DISPATCHED_AGENT_PROMPT,
    EVOLVED_AGENT_DIR,
    SEARXNG_ENDPOINT,
    SUMMARIZER_LLM,
    SUMMARIZER_SYSTEM_PROMPT,
)
from src.llm.client import llm_client
from src.llm.evolution import get_tools_from, tool
from src.llm.pipeline import SeaConfig, SeaPipeline
from src.vector_db.client import knowledge_base_client
from src.utils import html_to_text


@tool
def categorize_prompt(category: str) -> str:
    """Tool used to categorize the user's prompt.
    The prompt can be one of:
        1. "conversational" - mainly includes chatty behaviour, e.g. greetings, general conversation, etc...
        2. "search" - mainly includes requests to search for information, either on the web, or for info you already have in your knowledge base
        3. "agentic" - mainly includes requests to perform tasks, like listing files in a directory, opening applications, etc...

    Args:
        category (Literal["conversational"] | Literal["search"] | Literal["agentic"]): The category that matches the prompt. Can be one of `conversational`, `search` or `agentic`

    Returns:
        Literal["conversational"] | Literal["search"] | Literal["agentic"]: The category that matches the prompt. Can be one of `conversational`, `search` or `agentic`
    """
    assert category in ["conversational", "search", "agentic"]
    return category


@tool
def search_for_information_on_the_web(
    query: str, should_summarize: bool, max_results: int = 10
) -> list[dict[str, str]]:
    """Tool used to TEXTUAL search for a given query on the web

    Args:
        query (str): The query to perform on the web
        should_summarize (bool): Whether the results you find should be summarized so that they fit within your context window.
            IMPORTANT: THIS NEEDS TO BE FALSE WHEN LOOKING UP CODE!!!
        max_results (int): Max search results. Defaults to 10.

    Returns:
        list[dict[str, str]]: A list of objects of shape {url: <URL>, content: <CONTENT>, title: <TITLE>}
    """
    query = urllib.parse.quote(query)

    spoofed_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    r = httpx.get(
        f"{SEARXNG_ENDPOINT}/search?q={query}&format=json",
        headers={"User-Agent": spoofed_user_agent},
    )
    r.raise_for_status()

    results = r.json().get("results", [])

    contexts: list[dict[str, str]] = []

    visited_index = 0
    while len(contexts) < min(max_results, len(results)) and visited_index < len(
        results
    ):
        if len(results) == 0:
            break

        result = results[visited_index]
        visited_index += 1

        try:
            search_response = httpx.get(result["url"])
            if search_response.status_code != 200:
                continue

            html = search_response.text
            contexts.append(
                {
                    "url": result["url"],
                    "title": result["title"],
                    "content": summarize.invoke(html_to_text(html))
                    if should_summarize
                    else html_to_text(html),
                }
            )
        except Exception:
            continue

    return contexts


@tool
def get_available_collections_in_knowledge_base() -> list[str]:
    """Tool used to check available collections in knowledge base. Useful to know the category of information you have available to you.
    Returns:
        list[str]: The collections in the knowledge base, available to you.
    """
    collections = knowledge_base_client.list_collections()
    return [collection.name for collection in collections]


@tool
def add_to_knowledge_base(collection: str, info: str) -> None:
    """Tool used to add a particular piece of info to the knowledge base
    TIP: You can also use this to store facts as you progress in the conversation.
    TIP: You can also use this to store notes in a `scratchpad` collection as you progress in the conversation.

    Args:
        collection (str): The collection to store data in.
            For example, you may have a `facts_about_user` where you store their birthday, or their name, you may have a `things_i_learned`
            where you store things you learned during your interactions with the user, etc...
            NOTE: This is also really useful for storing data that you found in a web search.
            For example, you would use your "things_from_the_web" collection or something to store the data there...
        info (str): The info to add to the knowledge base,
    """
    c = knowledge_base_client.get_or_create_collection(name=collection)
    c.upsert(ids=[str(uuid.uuid4())], documents=[info])


@tool
def update_data_in_knowledge_base(
    collection: str, queries: list[str], replacement: str
) -> None:
    """ Tool used to update a particular piece of info to the knowledge base,
    TIP: You can also use this to update facts as you progress in the conversation.
    TIP: You can also use this to update notes in a `scratchpad` collection as you progress in the conversation.

    Args:
        collection (str): The collection to store data in.
            For example, you may have a `facts_about_user` where you store their birthday,
            or their name, you may have a `things_i_learned` where you store things you learned during your interactions with the user,
            etc...
            NOTE: This is also really useful for storing data that you found in a web search.
            For example, you would use your "things_from_the_web" collection or something to store the data there...
        queries (list[str]): The queries to use in the knowledge base search, so that we can get the entries and update them. Please use multiple queries for a better search result.
        replacement (list[str]): The info to replace the data with.
    """
    c = knowledge_base_client.get_or_create_collection(name=collection)
    query_results = c.query(query_texts=queries, include=["documents"])
    doc_id = query_results["ids"][0][0]
    c.upsert(ids=[doc_id], documents=[replacement])


@tool
def forget_data_from_knowledge_base(collection: str, queries: list[str]) -> None:
    """Tool used to forget a particular piece of info to the knowledge base,
    TIP: You can also use this to forget facts as you progress in the conversation.
    TIP: You can also use this to forget notes from a `scratchpad` collection as you progress in the conversation.

    Args:
        collection (str): The collection where data to forget lives in.
            For example, you may have a `facts_about_user` where you store their birthday, or their name,
            you may have a `things_i_learned` where you store things you learned during your interactions with the user,
            etc...
            NOTE: This is also really useful for storing data that you found in a web search.
            For example, you would use your "things_from_the_web" collection or something to store the data there...
        queries (list[str]): The queries to use in the knowledge base search, so that we can get the entries and delete them. Please use multiple queries for a better search result.,
    """
    c = knowledge_base_client.get_or_create_collection(name=collection)
    query_results = c.query(query_texts=queries, include=["documents"])
    ids = [ids for ids_cluster in query_results["ids"] for ids in ids_cluster]
    c.delete(ids=ids)

@tool
def dump_knowledge_base_collection(collection: str):
    """Tool used to dump information from a collection in your knowledge base.
    TIP: You can also use this to dump facts that you stored during the current or previous conversations.
    TIP: You can also use this to dump notes that you stored in a `scratchpad` collection over time.

    Args:
        collection (str): The collection to look for data in.
            For example, you may have a `facts_about_user` where you store their birthday, or their name,
            you may have a `things_i_learned` where you store things you learned during your interactions with the user,
            etc...
            NOTE: This is really useful for getting data that you previously searched for, in a summarized manner.
            For example, you would use your "things_from_the_web" collection or something...
    """
    c = knowledge_base_client.get_collection(collection)
    results = c.get()
    docs = results['documents'] or []
    return docs

@tool
def query_knowledge_base(collection: str, queries: list[str], max_results: int = 10):
    """Tool used to look for information related to the query in the knowledge base,
    TIP: You can also use this to look for facts that you stored during the current or previous conversations.
    TIP: You can also use this to look for notes that you stored in a `scratchpad` collection over time.

    Args:
        collection (str): The collection to look for data in.
            For example, you may have a `facts_about_user` where you store their birthday, or their name,
            you may have a `things_i_learned` where you store things you learned during your interactions with the user,
            etc...
            NOTE: This is really useful for getting data that you previously searched for, in a summarized manner.
            For example, you would use your "things_from_the_web" collection or something...
        queries (list[str]): The queries to use in the knowledge base search. Try to use multiple queries for better results.
        max_results (int): Max search results. Defaults to 10.

    Returns:
        list[str]: List of results from the knowledge base
    """
    c = knowledge_base_client.get_collection(collection)
    query_results = c.query(
        query_texts=queries, n_results=max_results, include=["documents"]
    )

    doc_cluster = query_results["documents"] or []
    if len(doc_cluster) == 0:
        return []

    return [doc for docs in doc_cluster for doc in docs]


@tool
def get_available_agents() -> list[str]:
    """Tool used to get all available agents in the collection,
    Returns:
        list[str]: The agents available. Always call this before starting to work on a task, if appropriate
    """
    if not os.path.exists(EVOLVED_AGENT_DIR):
        return []

    agent_scripts = os.listdir(EVOLVED_AGENT_DIR)
    return [script[:-3] for script in agent_scripts if script.endswith(".py")]


@tool
def summarize(text: str) -> str:
    """Tool used to summarize a piece of text into a shorter piece of text
    Args:
        text (str): The text to summarize
    Returns:
        str: The summarized text
    """
    llm_generation_config = LLMGenerationConfig(
        model=SUMMARIZER_LLM,
        on_content_token=lambda token: print(token, end="", flush=True),
        on_generation_finish=lambda: print("\n"),
    )
    session = Session(
        looped=False,
        static_actors=[
            SystemActor.with_message(
                SUMMARIZER_SYSTEM_PROMPT(text),
            )
        ],
        main_assistant_actor=spawn_assistant_actor(
            llm_client=llm_client,
            config=llm_generation_config,
        ),
    )
    sea_config = SeaConfig(llm_client=llm_client, session=session)
    chat_histories = SeaPipeline(config=sea_config).run()

    if len(chat_histories) == 0:
        return ""

    if len(chat_histories[-1]) == 0:
        return ""

    return chat_histories[-1][-1]["content"]


@tool
def retrieve_agent_implementation(agent: str) -> str | None:
    """Tool used to retrieve a given agent's implementation
    Args:
        agent (str): The agent's whose implementation we want to retrieve
    Returns:
        str: The implementation of the agent.
    """
    if not os.path.exists(EVOLVED_AGENT_DIR):
        raise FileNotFoundError(f"This `{agent}` agent file does not exist.")

    agent_scripts = os.listdir(EVOLVED_AGENT_DIR)
    if f"{agent}.py" not in agent_scripts:
        raise FileNotFoundError(f"This `{agent}` agent file does not exist.")

    implementation = ""
    with open(os.path.join(EVOLVED_AGENT_DIR, f"{agent}.py"), "r") as f:
        implementation = f.read()

    return json.dumps(implementation)


@tool
def register_agent(agent_name: str, implementation: str) -> None:
    """Tool used to register an agent.
    An agent is just a file containing a bunch of functions.
    Those functions are also called tools.
    IMPORTANT: DO NOT FORGET THAT `@` SYMBOL WHEN USING THE DECORATOR!!!
    IMPORTANT: DO NOT TRY TO REGISTER AN AGENT THAT ALREADY EXISTS! If you need to modify it, call the `modify_agent` tool. If you need to fix it, call the `fix_agent` tool.

    <example>
        # Creating an agent named "calculator"
        from src.llm.evolution import tool

        @tool
        def calculator__add(numbers: list[int]) -> int:
            \"\"\"A tool that adds multiple numbers
            Args:
                numbers (list[int]): The numbers to add
            Returns:
                int: The sum of the given numbers
            \"\"\"
            return sum(numbers)

        @tool
        def calculator__subtract(numbers: list[int]) -> int:
            \"\"\"A tool that subtracts multiple numbers
            Args:
                numbers (list[int]): The numbers to subtract
            Returns:
                int: The difference of the given numbers
            \"\"\"
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

        @tool
        def calculator__complex_calculation(numbers: list[int]) -> int:
            \"\"\"A tool that performs a complex calculation on multiple numbers
            Args:
                numbers (list[int]): The numbers to perform the calculation on
            Returns:
                int: The result of the calculation
            \"\"\"
            return helper1(*numbers) + helper2()
    </example>

    1. Always fit your solution in a single file.
    2. Always decorate your tools with the `@tool` decorator from `src.llm.evolution`
    3. Never decorate helper functions with the `@tool` decorator
    4. Never implement tools as generators or anything complex - follow the KISS (keep it simple, stupid) principle

    Args:
        agent_name (str): The name of the agent you want to register (make sure it's a snake_case string)
        implementation (str): The python code that contains the tools you want the agent to use.
    """
    os.makedirs(EVOLVED_AGENT_DIR, exist_ok=True)

    with open(os.path.join(EVOLVED_AGENT_DIR, f"{agent_name}.py"), "w") as f:
        f.write(implementation)


@tool
def fix_agent(
    agent_to_fix: str,
    fixed_implementation: str,
) -> None:
    """Tool used to fix an agent if the task fails
    Args:
        agent_to_fix (str): The name of the agent you want to fix
        fixed_implementation (str): The python code that should replace the faulty implementation.
    """
    if not os.path.exists(EVOLVED_AGENT_DIR):
        raise FileNotFoundError(f"This `{agent_to_fix}` agent file does not exist.")

    agent_scripts = os.listdir(EVOLVED_AGENT_DIR)
    if f"{agent_to_fix}.py" not in agent_scripts:
        raise FileNotFoundError(f"This `{agent_to_fix}` agent file does not exist.")

    with open(os.path.join(EVOLVED_AGENT_DIR, f"{agent_to_fix}.py"), "w") as f:
        f.write(fixed_implementation)


@tool
def modify_agent(agent_to_modify: str, new_implementation: str) -> None:
    """Tool used to modify an agent
    Args:
        agent_to_modify (str): The name of the agent you want to modify
        new_implementation (str): The python code that should replace the current implementation.
    """
    if not os.path.exists(EVOLVED_AGENT_DIR):
        raise FileNotFoundError(f"This `{agent_to_modify}` agent file does not exist.")

    agent_scripts = os.listdir(EVOLVED_AGENT_DIR)
    if f"{agent_to_modify}.py" not in agent_scripts:
        raise FileNotFoundError(f"This `{agent_to_modify}` agent file does not exist.")

    with open(os.path.join(EVOLVED_AGENT_DIR, f"{agent_to_modify}.py"), "w") as f:
        f.write(new_implementation)


@tool
def dispatch_agent(
    original_request: str,
    context: str,
    agent_to_dispatch: str,
) -> list[str]:
    """Tool used to ask an agent to perform a task for you
    NOTE: After calling this, do not perform additional operations, unless explicitely instructed by the user.

    Args:
        original_request (str): The request from the user. Can be rephrased so that it makes sense for the agent.
        context (str): Context regarding things relevant to the user\'s request. E.g. "I did so and so and eventually created you so you take care of this task for me"
        agent_to_dispatch (str): The agent that is responsible with fulfilling the task

    Returns:
        list[str]: A summary of how executing the agent went.
    """
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

    llm_generation_config = LLMGenerationConfig(
        model=AGENT_LLM,
        on_content_token=lambda token: print(token, end="", flush=True),
        on_tool_call_token=lambda token: print(token, end="", flush=True),
        on_generation_finish=lambda: print("\n"),
    )
    session = Session(
        looped=False,
        static_actors=[
            SystemActor.with_message(
                DISPATCHED_AGENT_PROMPT(agent_to_dispatch, original_request, context),
            )
        ],
        main_assistant_actor=spawn_assistant_actor(
            llm_client=llm_client,
            config=llm_generation_config,
            tools_factory=lambda: tools,
        ),
        tool_actor_spawner=create_tool_actor_spawner(),
    )
    sea_config = SeaConfig(llm_client=llm_client, session=session)
    chat_histories = SeaPipeline(config=sea_config).run()
    chat_history: list[str] = [message["content"] for chat_history in chat_histories for message in chat_history]

    return [f"Results from running the {agent_to_dispatch} agent: {chat_history}"]

categorize_prompt.standalone = True
add_to_knowledge_base.requires_hitl = True
register_agent.requires_hitl = True
dispatch_agent.requires_hitl = True
