from src.constants import LLM, MASTER_SYSTEM_PROMPT
from src.llm.client import llm_client
from src.llm.history import ChatHistory
from src.llm.pipeline import Pipeline


def main():
    pipeline = Pipeline(llm_client=llm_client, chat_history=ChatHistory())
    pipeline = pipeline.with_system_message(system_prompt=MASTER_SYSTEM_PROMPT)
    pipeline = pipeline.prompt_user()
    pipeline = pipeline.generate_llm_answer(
        model=LLM,
        on_token=lambda token: print(token, end="", flush=True),
        on_generation_finish=lambda: print("\n"),
    )
    pipeline.chat_loop()


if __name__ == "__main__":
    main()
