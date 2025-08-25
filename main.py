from src.llm.session.actor.system import SystemActor
from src.llm.spawner.assistant import spawn_assistant_actor
from src.llm.spawner.tool import create_tool_actor_spawner
from src.llm.session.session import InteractiveSession
from src.llm.evolution import get_tools_from
from src.llm.utils import LLMGenerationConfig
from src.constants import GENERALIST_LLM, PRIMITIVE_TOOLS_DIR, SEMANTIC_ROUTER_SYSTEM_PROMPT
from src.llm.client import llm_client
from src.llm.pipeline import SeaConfig, SeaPipeline


def main():
    llm_generation_config = LLMGenerationConfig(
        model=GENERALIST_LLM,
        on_content_token=lambda token: print(token, end="", flush=True),
        on_tool_call_token=lambda token: print(
            token.replace("\\n", "\n").replace('\\"', '"'), end="", flush=True
        ),
        on_generation_finish=lambda: print("\n"),
    )

    session = InteractiveSession(
        looped=True,
        static_actors=[
            SystemActor.with_message(
                SEMANTIC_ROUTER_SYSTEM_PROMPT(), turns_allowed="unlimited"
            )
        ],
        main_assistant_actor=spawn_assistant_actor(
            llm_client=llm_client,
            config=llm_generation_config,
            tools_factory=lambda: get_tools_from(
                dir=PRIMITIVE_TOOLS_DIR, module_name="tools", evolved=False
            ),
        ),
        tool_actor_spawner=create_tool_actor_spawner(),
    )
    sea_config = SeaConfig(llm_client=llm_client, session=session)
    pipeline = SeaPipeline(config=sea_config)
    pipeline = pipeline.with_semantic_router(config=llm_generation_config)
    pipeline = pipeline.with_short_term_memory_summary()
    pipeline.run()


if __name__ == "__main__":
    main()
