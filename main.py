from src.llm.evolution import get_tools_from
from src.llm.utils import LLMGenerationConfig
from src.constants import AGENTIC_SYSTEM_PROMPT, LLM, PRIMITIVE_TOOLS_DIR
from src.llm.client import llm_client
from src.llm.pipeline import SeaConfig, SeaPipeline


def main():
    llm_generation_config = LLMGenerationConfig(
        model=LLM,
        on_content_token=lambda token: print(token, end="", flush=True),
        on_tool_call_token=lambda token: print(
            token.replace("\\n", "\n").replace('\\"', '"'), end="", flush=True
        ),
        on_generation_finish=lambda: print("\n"),
    )

    sea_config = SeaConfig(llm_client=llm_client, use_fragmented_context=False)
    pipeline = SeaPipeline(config=sea_config)
    pipeline = pipeline.with_system_message(system_prompt=AGENTIC_SYSTEM_PROMPT())
    pipeline = pipeline.with_user_prompt()
    pipeline = pipeline.with_available_knowledge_base_collections_injection()
    pipeline = pipeline.with_available_agents_injection()
    pipeline = pipeline.force_llm_to_think()
    pipeline = pipeline.with_tools(
        tools_factory=lambda: get_tools_from(
            dir=PRIMITIVE_TOOLS_DIR, module_name="tools", evolved=False
        )
    )
    # pipeline = pipeline.with_semantic_router(config=llm_generation_config)
    pipeline = pipeline.generate(config=llm_generation_config)
    pipeline.chat_loop(debug=False)


if __name__ == "__main__":
    main()
