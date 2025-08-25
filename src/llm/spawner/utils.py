def human_in_the_loop(*, tool_name: str) -> tuple[bool, str | None]:
    prompt = input(
        f"[Allow Sea to run the `{tool_name}` tool with those arguments]\n[y/N/<reason_for_refusal>] >>> "
    )

    if prompt.lower() == "y":
        return True, None

    return False, None if prompt == "N" else prompt
