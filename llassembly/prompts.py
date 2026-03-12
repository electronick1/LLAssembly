import os

from llassembly.asm_emulator import ExternCall

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(MODULE_PATH, "prompts_md")


def get_asm_prompt(
    system_message: str,
    extern_calls: list[ExternCall],
    prompt_path: str | None = None,
    compact_signatures: bool = False,
) -> str:
    """
    Build the full assembly system prompt.

    Args:
        system_message: Natural language context / task description
        extern_calls: List of external function definitions
        prompt_path: Path to the prompt template file (defaults to base.md)
        compact_signatures: If True, render each tool as a compact one-line
            signature (``name(args) -> ret  ; description``) instead of the
            verbose multi-line block with ASM call examples.  This reduces
            per-tool token cost from ~60 to ~8 tokens, making the total prompt
            token-competitive with the Python containers approach.
    """
    if compact_signatures:
        extern_calls_prompt = "\n".join(
            extern_call.get_compact_signature_prompt()
            for extern_call in extern_calls
        )
    else:
        extern_calls_prompt = "\n".join(
            extern_call.get_signature_prompt(call_index)
            for call_index, extern_call in enumerate(extern_calls)
        )

    prompt_path = prompt_path or os.path.join(PROMPTS_PATH, "base.md")
    with open(prompt_path, "r") as f:
        return f.read().format(
            context=system_message,
            extern_functions=extern_calls_prompt,
        )
