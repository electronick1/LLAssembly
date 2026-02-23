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
    """Test CMP instruction"""

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


def test_asm_emulator_db_instruction():
    asm_code = """
    db my_label, 0x1234
    """

    emulator = ASMEmulator.from_asm_code(asm_code)

    emulator.execute_current_instruction()

    assert "my_label" in emulator._state.storage
    assert emulator._state.storage["my_label"] == "0x1234"


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
