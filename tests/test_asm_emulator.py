import json

import pydantic
import pytest
from langchain.tools import tool

from llassembly.asm_emulator import (
    ASMEmulator,
    AsmInstruction,
    ExternCall,
    ExternCallContext,
    Flag,
    Register,
)


def test_extern_call_from_langchain_tool():
    @tool
    def sample_tool_func(arg1: str, arg2: int) -> str:
        """doc string"""
        return "result"

    extern_call = ExternCall.from_langchain_tool(sample_tool_func)
    assert extern_call.name == "sample_tool_func"
    assert extern_call.description == "doc string"
    assert len(extern_call.input_args) == 2
    assert "arg1" in extern_call.input_args
    assert "arg2" in extern_call.input_args


def test_extern_call_get_input_arg_prompt():
    @tool
    def sample_tool_func(arg1: str, arg2: int) -> str:
        """doc string"""
        return "result"

    extern_call = ExternCall.from_langchain_tool(sample_tool_func)

    prompt = extern_call.get_input_arg_prompt("arg1")
    assert "arg1" in prompt
    assert "String" in prompt

    with pytest.raises(RuntimeError):
        extern_call.get_input_arg_prompt("nonexistent")


def test_extern_call_get_signature_prompt():
    @tool
    def sample_tool_func(arg1: str, arg2: int) -> str:
        """doc string"""
        return "result"

    extern_call = ExternCall.from_langchain_tool(sample_tool_func)

    # Test signature prompt
    prompt = extern_call.get_signature_prompt(1)
    assert "sample_tool_func" in prompt
    assert "Input arguments" in prompt
    assert "Returns 1 value" in prompt


def test_extern_call_context_from_asm_stack_supplier():
    @tool
    def sample_tool_func(arg1: str, arg2: int) -> str:
        """doc string"""
        return "result"

    extern_call = ExternCall.from_langchain_tool(sample_tool_func)

    def arguments_supplier() -> tuple[str, int]:
        return "test_value", 42

    def infer_result_hook(value):
        pass

    context = ExternCallContext.from_asm_stack_supplier(
        extern_call=extern_call,
        arguments_supplier=arguments_supplier,
        infer_result_hook=infer_result_hook,
    )

    assert context.extern_call == extern_call
    assert "arg1" in context.call_kwargs
    assert "arg2" in context.call_kwargs


def test_extern_call_from_pydantic_input():
    class Animal(pydantic.BaseModel):
        legs: int = pydantic.Field(description="number of legs")
        other_parts: dict[str, int]

    @tool
    def sample_tool_func(arg1: Animal) -> str:
        """doc string"""
        return "result"

    extern_call = ExternCall.from_langchain_tool(sample_tool_func)
    assert json.dumps(Animal.model_json_schema()) in extern_call.get_signature_prompt(1)


def test_asm_instruction_from_row_line():
    instruction = AsmInstruction.from_row_line("mov eax, ebx")
    assert instruction
    assert instruction.command == "mov"
    assert instruction.operands == ["eax", "ebx"]

    instruction = AsmInstruction.from_row_line("add eax, ebx  ; comment")
    assert instruction
    assert instruction.command == "add"
    assert instruction.operands == ["eax", "ebx"]

    instruction = AsmInstruction.from_row_line("my_label: db 0x1234")
    assert instruction
    assert instruction.command == "db"
    assert instruction.operands == ["my_label:", "0x1234"]

    instruction = AsmInstruction.from_row_line("")
    assert instruction is None

    instruction = AsmInstruction.from_row_line("; just a comment")
    assert instruction is None


def test_asm_emulator_register_operations():
    emulator = ASMEmulator([])

    value = emulator._get_register_value("eax")
    assert value == 0

    emulator._set_register_value("eax", 42)
    assert emulator._state.registers[Register.EAX] == 42

    with pytest.raises(RuntimeError):
        emulator._set_register_value("invalid_reg", 42)


def test_asm_emulator_add_sub_instructions():
    asm_code = """
    mov eax, 10
    mov ebx, 5
    add eax, ebx
    sub eax, ebx
    """

    emulator = ASMEmulator.from_asm_code(asm_code)

    emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 10

    emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EBX] == 5

    emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 15
    assert emulator._state.flags[Flag.ZERO] is False
    assert emulator._state.flags[Flag.SIGN] is False

    emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 10
    assert emulator._state.flags[Flag.ZERO] is False
    assert emulator._state.flags[Flag.SIGN] is False


def test_asm_emulator_cmp_instruction():
    asm_code = """
    mov eax, 10
    mov ebx, 5
    cmp eax, ebx
    """

    emulator = ASMEmulator.from_asm_code(asm_code)

    emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 10

    emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EBX] == 5

    emulator.execute_current_instruction()
    assert emulator._state.flags[Flag.ZERO] is False
    assert emulator._state.flags[Flag.SIGN] is False
    assert emulator._state.flags[Flag.CARRY] is False


def test_asm_emulator_jmp_instructions():
    asm_code = """
    mov eax, 10
    mov ebx, 5
    cmp eax, ebx
    jne not_equal_label
    jmp end_label
    not_equal_label:
    mov eax, 100
    end_label:
    """

    emulator = ASMEmulator.from_asm_code(asm_code)

    emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 10

    emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EBX] == 5

    emulator.execute_current_instruction()
    assert emulator._state.flags[Flag.ZERO] is False

    emulator.execute_current_instruction()
    assert emulator._state.eip == 5

    emulator.execute_current_instruction()
    emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 100


def test_asm_emulator_call_instruction():
    @tool
    def sample_tool_func(arg1: str, arg2: int) -> str:
        """doc string"""
        return "result"

    extern_call = ExternCall.from_langchain_tool(sample_tool_func)
    emulator = ASMEmulator([])
    emulator.add_extern_call(extern_call)

    emulator._state.stack = ["test_value", "42"]

    asm_code = """
    call sample_tool_func
    """

    emulator._instructions = ASMEmulator.from_asm_code(asm_code)._instructions
    context = emulator.execute_current_instruction()

    assert context is not None
    assert context.extern_call == extern_call


def test_ret_exists_program():
    code = """
    mov eax, 5
    ret
    mov ebx, -5
    add eax, ebx
    """
    emulator = ASMEmulator.from_asm_code(code)
    emulator.execute_current_instruction()  # mov eax
    emulator.execute_current_instruction()  # ret
    assert emulator._state.registers[Register.EAX] == 5
    assert emulator._state.registers[Register.EBX] == 0


def test_asm_emulator_execution_limit():
    asm_code = """
    start:
    mov eax, 1
    add eax, 1
    jmp start
    """

    emulator = ASMEmulator.from_asm_code(asm_code)

    with pytest.raises(RuntimeError, match="Execution limit reached"):
        for _ in range(2001):
            emulator.execute_current_instruction()


def test_asm_emulator_special_registers():
    emulator = ASMEmulator([])

    emulator._set_register_value("r42", 100)
    value = emulator._get_register_value("r42")
    assert value == 100
    assert "r42" in emulator._state.storage
    assert emulator._state.storage["r42"] == 100


def test_asm_emulator_complex_assembly():
    asm_code = """
    mov eax, 10
    mov ebx, 5
    add eax, ebx
    cmp eax, 15
    je equal
    mov eax, 0
    equal:
    push eax
    pop ebx
    """

    emulator = ASMEmulator.from_asm_code(asm_code)

    while not emulator.is_finished():
        emulator.execute_current_instruction()

    assert emulator._state.registers[Register.EAX] == 15
    assert emulator._state.registers[Register.EBX] == 15
    assert emulator._state.stack == []
    assert emulator._state.flags[Flag.ZERO] is True


def test_unconditional_jmp():
    asm_code = """
    mov eax, 1
    jmp skip
    mov eax, 2
    skip:
    """
    emulator = ASMEmulator.from_asm_code(asm_code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 1


def test_je_and_jne():
    je_code = """
    mov eax, 5
    mov ebx, 5
    cmp eax, ebx
    je equal
    mov eax, 10
    equal:
    """
    emulator = ASMEmulator.from_asm_code(je_code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 5

    jne_code = """
    mov eax, 5
    mov ebx, 6
    cmp eax, ebx
    jne not_equal
    mov eax, 10
    not_equal:
    """
    emulator = ASMEmulator.from_asm_code(jne_code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 5


def test_jl_and_jle():
    jl_code = """
    mov eax, 3
    mov ebx, 5
    cmp eax, ebx
    jl less
    mov eax, 10
    less:
    """
    emulator = ASMEmulator.from_asm_code(jl_code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 3

    jle_code = """
    mov eax, 5
    mov ebx, 5
    cmp eax, ebx
    jle equal_or_less
    mov eax, 10
    equal_or_less:
    """
    emulator = ASMEmulator.from_asm_code(jle_code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 5


def test_jg_and_jge():
    jg_code = """
    mov eax, 8
    mov ebx, 5
    cmp eax, ebx
    jg greater
    mov eax, 10
    greater:
    """
    emulator = ASMEmulator.from_asm_code(jg_code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 8

    jge_code = """
    mov eax, 5
    mov ebx, 5
    cmp eax, ebx
    jge equal_or_greater
    mov eax, 10
    equal_or_greater:
    """
    emulator = ASMEmulator.from_asm_code(jge_code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 5


def test_js_and_jns():
    js_code = """
    mov eax, -1
    js sign
    mov ebx, 1
    sign:
    mov ebx, 2
    """
    emulator = ASMEmulator.from_asm_code(js_code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EBX] == 2

    jns_code = """
    mov eax, 1
    jns non_negative
    mov ebx, 1
    non_negative:
    mov ebx, 2
    """
    emulator = ASMEmulator.from_asm_code(jns_code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EBX] == 2


def test_jump_chain():
    chain_code = """
    jmp first
    first:
    jmp second
    second:
    mov eax, 5
    """
    emulator = ASMEmulator.from_asm_code(chain_code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 5


def test_add_carry_flag():
    add_carry_code = """
    mov eax, 4294967295
    mov ebx, 1
    add eax, ebx
    """
    emulator = ASMEmulator.from_asm_code(add_carry_code)
    emulator.execute_current_instruction()  # mov eax
    emulator.execute_current_instruction()  # mov ebx
    emulator.execute_current_instruction()  # add
    # CARRY flag is unlimited in python
    assert emulator._state.flags[Flag.CARRY] is False
    assert emulator._state.flags[Flag.ZERO] is False
    assert emulator._state.flags[Flag.SIGN] is False


def test_add_zero_and_sign_flags():
    add_code = """
    mov eax, 5
    mov ebx, -5
    add eax, ebx
    """
    emulator = ASMEmulator.from_asm_code(add_code)
    emulator.execute_current_instruction()  # mov eax
    emulator.execute_current_instruction()  # mov ebx
    emulator.execute_current_instruction()  # add
    assert emulator._state.flags[Flag.ZERO] is True
    assert emulator._state.flags[Flag.SIGN] is False


def test_sub_zero_and_sign_flags():
    sub_zero_code = """
    mov eax, 5
    mov ebx, 5
    sub eax, ebx
    """
    emulator = ASMEmulator.from_asm_code(sub_zero_code)
    emulator.execute_current_instruction()  # mov eax
    emulator.execute_current_instruction()  # mov ebx
    emulator.execute_current_instruction()  # sub
    assert emulator._state.flags[Flag.ZERO] is True
    assert emulator._state.flags[Flag.SIGN] is False

    sub_sign_code = """
    mov eax, 5
    mov ebx, 10
    sub eax, ebx
    """
    emulator = ASMEmulator.from_asm_code(sub_sign_code)
    emulator.execute_current_instruction()  # mov eax
    emulator.execute_current_instruction()  # mov ebx
    emulator.execute_current_instruction()  # sub
    assert emulator._state.flags[Flag.SIGN] is True


def test_cmp_flags():
    # Zero
    zero_cmp = ASMEmulator.from_asm_code("mov eax, 5\nmov ebx, 5\ncmp eax, ebx")
    zero_cmp.execute_current_instruction()
    zero_cmp.execute_current_instruction()
    zero_cmp.execute_current_instruction()
    assert zero_cmp._state.flags[Flag.ZERO] is True
    assert zero_cmp._state.flags[Flag.SIGN] is False
    assert zero_cmp._state.flags[Flag.CARRY] is False

    # Sign
    sign_cmp = ASMEmulator.from_asm_code("mov eax, 5\nmov ebx, 10\ncmp eax, ebx")
    sign_cmp.execute_current_instruction()
    sign_cmp.execute_current_instruction()
    sign_cmp.execute_current_instruction()
    assert sign_cmp._state.flags[Flag.SIGN] is True
    assert sign_cmp._state.flags[Flag.ZERO] is False
    assert sign_cmp._state.flags[Flag.CARRY] is True  # 5 < 10

    # Carry
    carry_cmp = ASMEmulator.from_asm_code("mov eax, 10\nmov ebx, 5\ncmp eax, ebx")
    carry_cmp.execute_current_instruction()
    carry_cmp.execute_current_instruction()
    carry_cmp.execute_current_instruction()
    assert carry_cmp._state.flags[Flag.CARRY] is False


def test_get_jmp_index_and_no_jump():
    code = """
    start:
    jmp end
    mov eax, 1
    end:
    """
    emulator = ASMEmulator.from_asm_code(code)
    assert emulator.get_jmp_index(1) == 3
    assert emulator.get_jmp_index(0) is None
    assert emulator.get_jmp_index(2) is None


def test_sub_sign_flag():
    code = """
    mov eax, 5
    mov ebx, 10
    sub eax, ebx
    """
    emulator = ASMEmulator.from_asm_code(code)
    emulator.execute_current_instruction()  # mov eax
    emulator.execute_current_instruction()  # mov ebx
    emulator.execute_current_instruction()  # sub
    assert emulator._state.registers[Register.EAX] == -5
    assert emulator._state.flags[Flag.SIGN] is True


def test_cmp_zero_and_carry_flags():
    code = """
    mov eax, 5
    mov ebx, 5
    cmp eax, ebx
    """
    emulator = ASMEmulator.from_asm_code(code)
    emulator.execute_current_instruction()  # mov eax
    emulator.execute_current_instruction()  # mov ebx
    emulator.execute_current_instruction()  # cmp
    assert emulator._state.flags[Flag.ZERO] is True
    assert emulator._state.flags[Flag.CARRY] is False

    code = """
    mov eax, 3
    mov ebx, 5
    cmp eax, ebx
    """
    emulator = ASMEmulator.from_asm_code(code)
    emulator.execute_current_instruction()  # mov eax
    emulator.execute_current_instruction()  # mov ebx
    emulator.execute_current_instruction()  # cmp
    assert emulator._state.flags[Flag.ZERO] is False
    assert emulator._state.flags[Flag.CARRY] is True


def test_conditional_je_jump():
    code = """
    mov eax, 5
    mov ebx, 5
    cmp eax, ebx
    je target
    mov eax, 0
    jmp end
    target:
    mov eax, 1
    end:
    """
    emulator = ASMEmulator.from_asm_code(code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 1


def test_conditional_jne_jump():
    code = """
    mov eax, 5
    mov ebx, 10
    cmp eax, ebx
    jne target
    mov eax, 0
    jmp end
    target:
    mov eax, 1
    end:
    """
    emulator = ASMEmulator.from_asm_code(code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 1


def test_conditional_jl_jump():
    code = """
    mov eax, -1
    mov ebx, 0
    cmp eax, ebx
    jl target
    mov eax, 0
    jmp end
    target:
    mov eax, 1
    end:
    """
    emulator = ASMEmulator.from_asm_code(code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 1


def test_conditional_jle_jump():
    code = """
    mov eax, -1
    mov ebx, 0
    cmp eax, ebx
    jle target
    mov eax, 0
    jmp end
    target:
    mov eax, 1
    end:
    """
    emulator = ASMEmulator.from_asm_code(code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 1


def test_conditional_jg_jump():
    code = """
    mov eax, 5
    mov ebx, 0
    cmp eax, ebx
    jg target
    mov eax, 0
    jmp end
    target:
    mov eax, 1
    end:
    """
    emulator = ASMEmulator.from_asm_code(code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 1


def test_conditional_jge_jump():
    code = """
    mov eax, 5
    mov ebx, 0
    cmp eax, ebx
    jge target
    mov eax, 0
    jmp end
    target:
    mov eax, 1
    end:
    """
    emulator = ASMEmulator.from_asm_code(code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 1


def test_conditional_js_jump():
    code = """
    mov eax, -1
    mov ebx, 0
    cmp eax, ebx
    js target
    mov eax, 0
    jmp end
    target:
    mov eax, 1
    end:
    """
    emulator = ASMEmulator.from_asm_code(code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 1


def test_conditional_jns_jump():
    code = """
    mov eax, 5
    mov ebx, 0
    cmp eax, ebx
    jns target
    mov eax, 0
    jmp end
    target:
    mov eax, 1
    end:
    """
    emulator = ASMEmulator.from_asm_code(code)
    while not emulator.is_finished():
        emulator.execute_current_instruction()
    assert emulator._state.registers[Register.EAX] == 1


def test_get_jmp_index_returns_correct_value():
    code = """
    mov eax, 5
    jne label
    mov eax, 0
    label:
    mov eax, 1
    """
    emulator = ASMEmulator.from_asm_code(code)
    assert emulator.get_jmp_index(1) == 3
    assert emulator.get_jmp_index(3) is None


def test_label_parsing_and_skipping():
    code = """
    start:
    mov eax, 5
    end:
    """
    emulator = ASMEmulator.from_asm_code(code)
    emulator.execute_current_instruction()  # skip 'start:'
    emulator.execute_current_instruction()  # mov
    emulator.execute_current_instruction()  # skip 'end:'
    assert emulator._state.registers[Register.EAX] == 5
    assert emulator.is_finished()


def test_reset_state_clears_registers_and_flags():
    code = """
    mov eax, 5
    mov ebx, 10
    """
    emulator = ASMEmulator.from_asm_code(code)
    emulator.execute_current_instruction()
    emulator.execute_current_instruction()
    emulator.reset_state()
    assert emulator._state.registers[Register.EAX] == 0
    assert emulator._state.registers[Register.EBX] == 0
    for flag in Flag:
        assert emulator._state.flags[flag] is False


def test_is_finished_returns_true_at_end():
    code = """
    mov eax, 5
    """
    emulator = ASMEmulator.from_asm_code(code)
    assert not emulator.is_finished()
    emulator.execute_current_instruction()
    assert emulator.is_finished()


def test_push_and_pop_update_esp_and_stack():
    code = """
    push 10
    push 20
    pop eax
    pop ebx
    """
    emulator = ASMEmulator.from_asm_code(code)
    emulator.execute_current_instruction()
    emulator.execute_current_instruction()
    # After pushes, ESP should be 0x1000 - 8
    assert emulator._state.registers[Register.ESP] == 0x1000 - 8
    emulator.execute_current_instruction()
    emulator.execute_current_instruction()
    # After pops, stack should be empty
    assert emulator._state.stack == []
    assert emulator._state.registers[Register.ESP] == 0x1000
    # Registers should hold popped values
    assert emulator._state.registers[Register.EAX] == 20
    assert emulator._state.registers[Register.EBX] == 10


def test_asm_emulator_db_instruction_basic_string():
    asm_code = """
    my_label db "hello"
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == "hello"


def test_asm_emulator_db_instruction_basic_int():
    asm_code = """
    my_label db 42
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == 42


def test_asm_emulator_db_instruction_hex_value():
    asm_code = """
    my_label db 0xFF
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == 255


def test_asm_emulator_db_instruction_binary_value():
    asm_code = """
    my_label db 0b1010
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == 5


def test_asm_emulator_db_instruction_char_value():
    asm_code = """
    my_label db 'A'
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == "A"


def test_asm_emulator_db_instruction_string_with_spaces():
    asm_code = """
    my_label db "hello world"
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == "hello world"


def test_asm_emulator_db_instruction_json_string():
    asm_code = """
    my_label db "{\\"key\\": \\"value\\"}"
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == {"key": "value"}


def test_asm_emulator_db_instruction_empty_string():
    asm_code = """
    my_label db ""
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == ""


def test_asm_emulator_db_instruction_zero_value():
    asm_code = """
    my_label db 0
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == 0


def test_asm_emulator_db_instruction_negative_value():
    asm_code = """
    my_label db -42
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == -42


def test_asm_emulator_db_instruction_large_number():
    asm_code = """
    my_label db 123456789
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == 123456789


def test_asm_emulator_db_instruction_float_value():
    asm_code = """
    my_label db 3.14
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == 3.14


def test_asm_emulator_db_instruction_mixed_types():
    asm_code = """
    my_label db "string", 42, 0xFF
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == '"string", 42, 0xFF'


def test_asm_emulator_db_instruction_with_escaped_chars():
    asm_code = """
    my_label db "hello\\nworld"
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == "hello\nworld"


def test_asm_emulator_db_instruction_with_special_chars():
    asm_code = """
    my_label db "hello\\tworld\\r\\n"
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == "hello\tworld\r\n"


def test_asm_emulator_db_instruction_hex_with_prefix():
    asm_code = """
    my_label db 0x1234
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == 4660


def test_asm_emulator_db_instruction_complex_json():
    json_data = {
        "test": json.dumps(["test,\n\r"]),
        "test2": None,
        "test3": json.dumps(json.dumps({"a": 1})),
    }
    asm_code = f"""
    my_label db {json.dumps(json_data)}
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == json_data


def test_asm_emulator_db_instruction_complex_json_quoted():
    json_data = {
        "test": json.dumps(["test,\n\r"]),
        "test2": None,
        "test3": json.dumps(json.dumps({"a": 1})),
    }
    asm_code = f"""
    my_label db "{json.dumps(json_data)}"
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == json_data


def test_asm_emulator_db_instruction_complex_string():
    asm_code = """
    my_label db "This is a complex string with 'quotes' and \\"double quotes\\""
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert "complex string" in emulator._state.storage["my_label"]


def test_asm_emulator_db_instruction_unicode():
    asm_code = """
    my_label db "café"
    """

    emulator = ASMEmulator.from_asm_code(asm_code)
    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == "café"


def test_nested_calls_return_correctly():
    asm = """
    call outer
    mov R1, 1
    ret

    outer:
        call inner
        mov R2, 1
        ret

    inner:
        mov R3, 1
        ret
    """
    emu = ASMEmulator.from_asm_code(asm)
    while not emu.is_finished():
        emu.execute_current_instruction()
    assert emu.is_finished()
    assert emu._state.stack == []
    assert emu._get_register_value("r1") == 1
    assert emu._get_register_value("r2") == 1
    assert emu._get_register_value("r3") == 1


def test_ret_after_call_with_jmp_and_label():
    asm = """
    call sub
    jmp skip
    mov eax, 99        ; This should never execute
    skip:
        ret

    sub:
        ret
    """
    emu = ASMEmulator.from_asm_code(asm)
    while not emu.is_finished():
        emu.execute_current_instruction()
    assert emu.is_finished()
    # The MOV in the main body was never executed
    assert emu._state.registers[Register.EAX] == 0


def test_ret_and_jne_jmp_combination():
    asm = """
    mov eax, 0
    call sub
    ret

    sub:
        cmp eax, 0
        jne label
        mov eax, 1
        ret

    label:
        mov eax, 2
        ret
    """
    emu = ASMEmulator.from_asm_code(asm)
    while not emu.is_finished():
        emu.execute_current_instruction()
    assert emu._state.registers[Register.EAX] == 1


def test_ret_with_label_and_jmp_back_to_call_site():
    asm = """
    call sub
    ret

    sub:
        jmp back
        ret

    back:
        mov eax, 99
        ret
    """
    emu = ASMEmulator.from_asm_code(asm)
    while not emu.is_finished():
        emu.execute_current_instruction()
    assert emu._state.registers[Register.EAX] == 99
