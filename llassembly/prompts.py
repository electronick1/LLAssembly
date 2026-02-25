import os

from llassembly.asm_emulator import ExternCall

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(MODULE_PATH, "prompts_md")


def get_asm_prompt(
    system_message: str, extern_calls: list[ExternCall], prompt_path: str | None = None
) -> str:
    extern_calls_prompt = "\n".join(
        [
            extern_call.get_signature_prompt(call_index)
            for call_index, extern_call in enumerate(extern_calls)
        ]
    )

    prompt_path = prompt_path or os.path.join(PROMPTS_PATH, "base.md")
    with open(prompt_path, "r") as f:
        return f.read().format(
            context=system_message,
            extern_functions=extern_calls_prompt,
        )
