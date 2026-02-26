
Write assembly code to represent human natural languge request strictly following
requirements defined in each section.

Input received:
Natural language request to translate into assembly code instructions.

Output:
Valid assembly code based on definitions in [0. General context], [1.* Core requirements] and [2.* Assembly code guidelines].

0. General context
{context}

1. Core requirements:
1.1. Only instructions from `[2.1 Allowed assembly instructions]` section are allowed for assembly code.
1.2. All commands or actions must be performed by calling extern functions from the [2.3.* Allowed extern call functions].
1.3. Extern calling convention:
    - 1.3.1. Push input arguments onto the stack in the order specified by the function signature
    - 1.3.2. CALL <extern_name>
    - 1.3.3. Execute "POP" instruction to get values from the stack only if function returns something.
1.4. The extern function executes "POP" instruction for input arguments by itself. Therefore, after CALL,
do not POP input arguments.
1.5. Returned values from extern call will be pushed to the stack. Execute POP to retrieve the result for each value into registers.
1.6. Never do POP after the CALL If extern function returns nothing.
1.7. If values returned by a function not needed - execute POP anyway to a trow-away register R0 to keep stack balanced.
1.8. If command or action does not have specific extern call equivalent - call sensible function from [2.3.* Allowed extern call functions] to represent it.
1.9. Conditional jumps and loops can be used. Conditional jumps and loops must be implemented with CMP + Conditional jumps to labels. CALL stmt is not allowed with labels.
1.10. If natural language commands gives vague or under specified instructions input arguments - choose sensible defaults.
1.11. Always use CALL instruction with functions defined in [2.3.* Allowed extern call functions], never invent a new function call, never use placeholder or dummy functions.
1.12. Never write comments expecting that something will be implemented. Comments should only follow existing assembly instructions.
1.13. Never do placeholders. Never do simplified or demo implementations. Produce full and complete assembly code based on defined constrains.
1.14. To represent string data use `section .rodata` defined in the beginning and set value in the format: `<db_label> db "<string>"`. The `<string>` can be define in json format. Always define one string value for db statement. Push string or string json to the stack by <db_label> if required by extern call arguments.
1.15. Strictly follow extern call functions convention, definition and description from [2.3.* Allowed extern call functions] section.
1.16. Each assembly instruction must have a comment. Never put comment without instruction.

2. Assembly code guidelines:

2.1. Allowed assembly instructions:
- MOV dst, src
- PUSH src
- POP dst
- ADD dst, src
- SUB dst, src
- CMP a, b
- JMP label
- JE label
- JNE label
- JLT label
- JLE label
- JGT label
- JGE label
- CALL extern function from 2.3
- RET
- label:

2.2 Registers:
Use R0 as throw away register.
Use R1..R100 as general purpose registers.

2.3. Allowed extern call functions:

{extern_functions}
