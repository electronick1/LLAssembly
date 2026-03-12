Generate assembly code for the given task. Output ONLY assembly code, no explanation.

Context: {context}

Rules:
- Use only: MOV, PUSH, POP, ADD, SUB, CMP, JMP, JE, JNE, JLT, JLE, JGT, JGE, CALL, RET, label:
- Registers: R0 (throwaway), R1..R100 (general purpose)
- Call convention: PUSH args in order → CALL fn → POP return values (fn pops its own args)
- Never POP after CALL if fn returns nothing; POP to R0 to discard unwanted return values
- Loops/conditionals: use CMP + conditional jump to label
- String constants: section .rodata with `name db "value"`, then PUSH name
- Never invent functions — only use those listed below
- Each instruction must have an inline comment

Functions:
{extern_functions}
