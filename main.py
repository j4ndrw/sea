from src.constants import LLM, MASTER_SYSTEM_PROMPT, PRIMITIVE_TOOLS_DIR
from src.llm.client import llm_client
from src.llm.evolution import get_tools_from
from src.llm.history import ChatHistory
from src.llm.pipeline import SeaPipeline


def main():
    pipeline = SeaPipeline(llm_client=llm_client, chat_history=ChatHistory())
    pipeline = pipeline.with_system_message(system_prompt=MASTER_SYSTEM_PROMPT())
    pipeline = pipeline.prompt_user()
    pipeline = pipeline.force_llm_to_think()
    pipeline = pipeline.generate_llm_answer(
        model=LLM,
        tools=get_tools_from(
            dir=PRIMITIVE_TOOLS_DIR, module_name="tools", evolved=False
        ),
        on_token=lambda token: print(token, end="", flush=True),
        on_generation_finish=lambda: print("\n"),
    )
    pipeline.chat_loop()


if __name__ == "__main__":
    main()
