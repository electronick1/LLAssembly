import json
import logging
import textwrap
from enum import Enum
from typing import Any, Callable, Self, get_args, get_origin

import pydantic
from langchain.tools import ToolRuntime
from langchain_core import tools as langchain_tools
from pydantic_core import PydanticUndefined

logger = logging.getLogger("llassembly_asm")
logger.addHandler(logging.NullHandler())


class ExternCall(pydantic.BaseModel):
    """
    Wrapper around a LangChain ``StructuredTool`` or any other type of tools.

    Used to define prompt for a tool ("extern call" in Assembly terms) based
    on the input/output types defined in the tool handler.
    """

    tool_handler: langchain_tools.BaseTool
    name: str
    description: str
    input_args: dict[str, pydantic.fields.FieldInfo]
    output_annotations: list[pydantic.fields.FieldInfo]

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_langchain_tool(
        cls,
        tool_handler: langchain_tools.BaseTool,
    ) -> Self:
        # Parse input arguments based on the pydantic definition
        input_args: dict[str, pydantic.fields.FieldInfo] = {}
        tool_call_schema = tool_handler.tool_call_schema
        if isinstance(tool_call_schema, dict):
            tool_fields = tool_call_schema
        else:
            tool_fields = tool_call_schema.model_fields
        for arg_name, arg_field in tool_fields.items():
            if get_origin(arg_field.annotation) is ToolRuntime:
                continue
            input_args[arg_name] = arg_field

        # Parse output arguments based return type of the tool handler.
        output_annotations: list[pydantic.fields.FieldInfo] = []
        if isinstance(
            tool_handler,
            (langchain_tools.StructuredTool, langchain_tools.Tool),
        ):
            return_type = tool_handler.func.__annotations__.get("return")
            # tuple return type will be unpacked in Assembly language
            # into multiple registers.
            if get_origin(return_type) is tuple:
                for arg in get_args(return_type):
                    if arg is not None:
                        output_annotations.append(
                            pydantic.fields.FieldInfo(annotation=arg)
                        )
            elif return_type and return_type is not None:
                output_annotations.append(
                    pydantic.fields.FieldInfo(annotation=return_type)
                )

        return cls(
            tool_handler=tool_handler,
            name=tool_handler.name,
            description=tool_handler.description,
            input_args=input_args,
            output_annotations=output_annotations,
        )

    def get_input_arg_prompt(self, arg_name: str) -> str:
        """
        Returns LLM prompt definition for the single tool argument
        as a guidelines for Assembly of what arguments to supply for
        external function.
        """
        field = self.input_args.get(arg_name)
        if field is None:
            raise RuntimeError(f"Field {arg_name} not found")

        arg_prompt = f"{arg_name}"
        if field.description:
            arg_prompt = field.description
        if field.default is not PydanticUndefined:
            arg_prompt = f"{arg_prompt}, default value is: {field.default}"
        if field.default_factory:
            raise RuntimeError("Factory default is not supported")

        arg_type = "String in json format"
        if field.annotation is str:
            arg_type = "String"
        elif field.annotation is int:
            arg_type = "Integer"
        elif field.annotation and issubclass(field.annotation, pydantic.BaseModel):
            # TODO: handle optional and nested types
            arg_type = f"String in json format: `{json.dumps(field.annotation.model_json_schema())}`"

        arg_prompt = f"{arg_prompt}. Argument type: {arg_type}."

        return arg_prompt

    def get_signature_prompt(self, extern_call_index: int) -> str:
        """
        Returns specification for the tool function that used in the LLM prompt
        to describe when and how Assembly can call this function.
        """
        arguments_to_push: list[str] = []
        input_arg_names: list[str] = []
        for input_arg_name, input_arg in self.input_args.items():
            input_arg_alias = input_arg.alias or input_arg_name
            input_arg_desc = self.get_input_arg_prompt(input_arg_name)
            arguments_to_push.append(f"PUSH <{input_arg_alias}>\t;{input_arg_desc}")
            input_arg_names.append(input_arg_alias)

        pop_arguments: list[str] = []
        for value_i, _ in enumerate(self.output_annotations):
            pop_arguments.append(
                f"POP <register>\t; Sets return value {value_i} to <register>"
            )

        output_definition = "Returns nothing"
        if len(pop_arguments):
            output_definition = f"Returns {len(pop_arguments)} values"

        return textwrap.dedent(f"""
        - 2.3.{extern_call_index}. Extern function `{self.name}`:
        `{self.name}` Input arguments: ({", ".join(input_arg_names)}).
        `{self.name}` Output: {output_definition}.
        `{self.name}` Description and purpose:    
        '''
        {self.description}
        '''
        `{self.name}` Call example:
        '''
        {"\n".join(arguments_to_push)}
        CALL {self.name}
        {"\n".join(pop_arguments)}
        '''
        """)


class ExternCallContext(pydantic.BaseModel):
    """
    Context passed to the emulator when an external call is executed.

    External call context supposed to be built based on the Assembly
    emulator stack values, and used to communicate back the result of
    extern call execution by supplying result to the stack.
    """

    extern_call: ExternCall
    call_kwargs: dict[str, Any]
    infer_result_hook: Callable[[Any], None]

    @classmethod
    def from_asm_stack_supplier(
        cls,
        extern_call: ExternCall,
        arguments_supplier: Callable[[], Any],
        infer_result_hook: Callable[[Any], None],
    ) -> Self:
        """
        Takes input arguments from the emulator stack (mutating emulator object)
        and prepares the execution context for the tool call.
        """
        kwargs: dict[str, Any] = {}
        for arg_name, arg_field in reversed(extern_call.input_args.items()):
            arg_alias = arg_field.alias or arg_name
            kwargs[arg_alias] = arguments_supplier()
            if arg_field.annotation not in (str, int) and isinstance(
                kwargs[arg_alias], str
            ):
                kwargs[arg_alias] = json.loads(f"[{kwargs[arg_alias]}]")[0]

        return cls(
            extern_call=extern_call,
            call_kwargs=kwargs,
            infer_result_hook=infer_result_hook,
        )

    def infer_result(self, result: Any):
        if len(self.extern_call.output_annotations) > 1:
            for item in result:
                self.infer_result_hook(item)
        else:
            self.infer_result_hook(result)


class AsmInstruction(pydantic.BaseModel):
    """
    Representation of a single assembly instruction.
    """

    origin: str
    command: str
    operands: list[str]

    @classmethod
    def from_row_line(cls, line: str) -> Self | None:
        origin = line
        line = line.strip().split(";")[0].strip()

        if not line:
            return None

        parts = line.split()
        if len(parts) > 2 and parts[1] == "db":
            return cls(
                origin=origin, command="db", operands=[parts[0], " ".join(parts[2:])]
            )

        # remove `,` and handle cases like `move eax,ebcx`
        parts = " ".join(parts).lower().replace(",", " ").split()
        operands = [op.strip() for op in parts[1:]]
        return cls(origin=origin, command=parts[0], operands=operands)


class Register(Enum):
    """
    Enumeration of the registers understood by the emulator.
    ``r0`` … ``r100`` are treated as special storage keys.
    """

    EAX = "eax"
    EBX = "ebx"
    ECX = "ecx"
    EDX = "edx"
    ESI = "esi"
    EDI = "edi"
    ESP = "esp"
    EBP = "ebp"
    EIP = "eip"
    FLAGS = "flags"


class Flag(Enum):
    ZERO = "zero"
    SIGN = "sign"
    OVERFLOW = "overflow"
    CARRY = "carry"


class ASMEmulatorState:
    def __init__(self, max_instructions_to_exec=1000):
        self.registers: dict[Register, Any] = {
            Register.EAX: 0,
            Register.EBX: 0,
            Register.ECX: 0,
            Register.EDX: 0,
            Register.ESI: 0,
            Register.EDI: 0,
            Register.ESP: 0x1000,
            Register.EBP: 0,
            Register.FLAGS: 0,
        }
        self.flags: dict[Flag, bool] = {
            Flag.ZERO: False,
            Flag.SIGN: False,
            Flag.CARRY: False,
            Flag.OVERFLOW: False,
        }
        self.eip: int = 0
        self.stack: list[Any] = []
        self.call_stack: list[int] = []
        self.storage: dict[str, Any] = {}
        self.instruction_count: int = 0
        self.max_instructions_to_exec: int = max_instructions_to_exec


class ASMEmulator:
    """
    Lightweight emulator for assembly-like code.

    Designed specifically for a prompt that used to generate tools
    execution plan:
    - Supports different types in registers, stack and storage
    - Supports different types on CMP
    - Has limited number of instructions
    - Simplified specifications for operands

    It supports a small subset of instructions needed for the
    tools execution: data movement, arithmetic, comparisons,
    conditional jumps, procedure calls, and the special
    ``db`` directive.  External tools are invoked via a callback
    mechanism that uses the `ExternCallContext` to communicate
    between emulator and LLM agent framework (e.g. LangChain).
    """

    def __init__(
        self, asm_instructions: list[AsmInstruction], max_instructions_to_exec=1000
    ):
        self._state = ASMEmulatorState(max_instructions_to_exec)
        self._instructions: list[AsmInstruction] = asm_instructions
        self._extern_calls: dict[str, ExternCall] = {}
        self._labels: dict[str, int] = {}
        self.instruction_map: dict[str, Callable] = {
            "mov": self._mov,
            "push": self._push,
            "pop": self._pop,
            "add": self._add,
            "sub": self._sub,
            "cmp": self._cmp,
            "call": self._call,
            "ret": self._ret,
            "jmp": self._jmp,
            "je": self._je,
            "jne": self._jne,
            "jl": self._jl,
            "jlt": self._jl,
            "jle": self._jle,
            "jg": self._jg,
            "jgt": self._jg,
            "jge": self._jge,
            "js": self._js,
            "jns": self._jns,
            "db": self._db,
        }
        self._parse_labels()

    @classmethod
    def from_asm_code(cls, asm_code: str) -> Self:
        instructions = []
        for line in asm_code.strip().splitlines():
            if instruction := AsmInstruction.from_row_line(line):
                instructions.append(instruction)

        return cls(instructions)

    def get_instructions(self) -> list[AsmInstruction]:
        return self._instructions.copy()

    def get_instruction(self, inst_index: int) -> AsmInstruction | None:
        if inst_index < 0 or inst_index >= len(self._instructions):
            return None
        return self._instructions[inst_index].copy()

    def get_current_instruction_index(self) -> int:
        return self._state.eip

    def get_extern_calls(self) -> dict[str, ExternCall]:
        return self._extern_calls.copy()

    def add_extern_call(self, extern_call: ExternCall) -> None:
        self._extern_calls[extern_call.name] = extern_call

    def execute_current_instruction(self) -> ExternCallContext | None:
        if self.is_finished():
            raise RuntimeError("ASM emulator finished execution")
        if self._state.eip < 0:
            raise RuntimeError("ASM stack pointer is less than 0")

        instruction = self._instructions[self._state.eip]
        logger.debug("exec_instruction: %s", str(instruction.origin).strip())

        # Skip labels
        if instruction.command.endswith(":") and not instruction.operands:
            self._state.eip += 1
            return None

        if instruction_handler := self.instruction_map.get(instruction.command):
            instruction_result = instruction_handler(*instruction.operands)
            if isinstance(instruction_result, ExternCallContext):
                return instruction_result
        else:
            self._state.eip += 1

        self._state.instruction_count += 1
        if self._state.instruction_count > self._state.max_instructions_to_exec:
            raise RuntimeError("Execution limit reached")

        return None

    def get_call_jmp_index(self, instruction_index: int) -> list[int] | None:
        if instruction_index < 0 or instruction_index >= len(self._instructions):
            return None

        instruction = self._instructions[instruction_index]
        if instruction.command == "ret":
            label_call_indexes = self.get_instruction_indexes_when_label_called()
            return [index + 1 for index in label_call_indexes] + [
                len(self._instructions)
            ]

        if (
            instruction.command == "call"
            and instruction.operands
            and instruction.operands[0] not in self._extern_calls
            and instruction.operands[0] in self._labels
        ):
            return [self._labels[instruction.operands[0]]]

        return None

    def get_jmp_index(self, instruction_index: int) -> int | None:
        if instruction_index < 0 or instruction_index >= len(self._instructions):
            return None

        instruction = self._instructions[instruction_index]
        if (
            instruction.command
            in {
                "jmp",
                "je",
                "jne",
                "jl",
                "jle",
                "jg",
                "jge",
                "js",
                "jns",
                "jlt",
                "jgt",
            }
            and instruction.operands
        ):
            return self._labels.get(instruction.operands[0])

        return None

    def get_tool_call(self, instruction_index: int) -> ExternCall | None:
        if instruction_index < 0 or instruction_index >= len(self._instructions):
            return None
        instruction = self._instructions[instruction_index]
        if instruction.command == "call" and instruction.operands:
            return self._extern_calls.get(instruction.operands[0])
        return None

    def get_instruction_indexes_when_label_called(self) -> list[int]:
        indexes_when_label_called = []
        for instruction_index, instruction in enumerate(self._instructions):
            if (
                instruction.command == "call"
                and instruction.operands
                and instruction.operands[0] in self._labels
            ):
                indexes_when_label_called.append(instruction_index)
        return indexes_when_label_called

    def is_finished(self) -> bool:
        return self._state.eip >= len(self._instructions)

    def reset_state(
        self,
        new_state: ASMEmulatorState | None = None,
        max_instructions_to_exec: int = 1000,
    ):
        self._state = new_state or ASMEmulatorState(
            max_instructions_to_exec=max_instructions_to_exec
        )

    def _get_register_value(self, reg_name: str) -> Any:
        if reg_name.startswith("r"):
            # R0..R100 - are special registers added in the prompt message
            # to extend storage space and simplify ASM logic.
            return self._state.storage.get(reg_name, 0)
        if reg_name not in Register:
            return None
        reg = Register(reg_name)
        return self._state.registers[reg]

    def _set_register_value(self, reg_name: str, value: Any):
        try:
            value = try_convert_to_numbers(value)
        except ValueError:
            pass
        if reg_name.startswith("r"):
            # R0..R100 - are special registers added in the prompt message
            # to extend storage space and simplify ASM logic.
            self._state.storage[reg_name] = value
            return
        if reg_name not in Register:
            raise RuntimeError("Unknown register")
        reg = Register(reg_name)
        self._state.registers[reg] = value

    def _get_flag_value(self, flag: Flag) -> bool:
        return self._state.flags[flag]

    def _set_flag_value(self, flag: Flag, value: bool):
        self._state.flags[flag] = value

    def _get_operand_value(self, operand: Any) -> Any:
        operand = operand.replace("[", "").replace("]", "")
        if (result := self._get_register_value(operand)) is not None:
            return result
        if (result := self._state.storage.get(operand)) is not None:
            return result
        try:
            return try_convert_to_numbers(operand)
        except ValueError:
            raise RuntimeError(
                "Can't find operand value in registers/storage or convert to int"
            )

    def _set_operand_value(self, operand: str, value: Any):
        self._set_register_value(operand, value)

    def _db(self, key_name: str, values_str: str):
        values_str = values_str.strip()
        # By the prompt definition llm must use one string or json-string like format,
        # but following parsing extends this rules by a bit to cover hallucinations.
        try:
            if (
                (comma_separated_values := values_str.split(","))
                and try_convert_to_numbers(comma_separated_values[-1]) == 0
                and len(comma_separated_values) > 1
            ):
                values_str = ",".join(comma_separated_values[:-1]).strip()
        except ValueError:
            pass

        undefined = object()
        value = undefined
        try:
            # Trying to convert `db` stmt as a list of values
            value = json.loads(f"[{values_str.strip()}]")[0]
        except json.JSONDecodeError:
            pass
        if value is undefined:
            # Trying to convert `db` stmt as a list of values but considering
            # that LLM may add additional quotes
            try:
                strip_quotes = values_str
                while strip_quotes.startswith(('"', "'")) and strip_quotes.endswith(
                    ('"', "'")
                ):
                    strip_quotes = strip_quotes[1:-1]
                value = json.loads(f"[{strip_quotes}]")[0]
            except json.JSONDecodeError:
                pass
        if isinstance(value, str):
            # Try to parse nested json
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        if value is undefined:
            # Can't convert to json
            strip_quotes = values_str
            while strip_quotes.startswith(('"', "'")) and strip_quotes.endswith(
                ('"', "'")
            ):
                strip_quotes = strip_quotes[1:-1]
            value = strip_quotes
        try:
            # Try to convert to numbers by default
            value = try_convert_to_numbers(value)
        except ValueError:
            pass
        self._state.storage[key_name] = value
        self._state.eip += 1

    def _mov(self, dest: str, src: str):
        src_value = self._get_operand_value(src)
        self._set_operand_value(dest, src_value)
        self._state.eip += 1

    def _push(self, operand: Any):
        value = self._get_operand_value(operand)
        self._state.stack.append(value)
        if esp_value := self._get_register_value("esp"):
            self._set_register_value("esp", esp_value - 4)
        self._state.eip += 1

    def _pop(self, dest: str):
        if not self._state.stack:
            raise RuntimeError("Stack underflow")

        value = self._state.stack.pop()
        self._set_operand_value(dest, value)
        if esp_value := self._get_register_value("esp"):
            self._set_register_value("esp", esp_value + 4)
        self._state.eip += 1

    def _add(self, dest: str, src: str):
        src_value = self._get_operand_value(src)
        dest_value = self._get_operand_value(dest)
        try:
            src_value = try_convert_to_numbers(src_value)
            dest_value = try_convert_to_numbers(dest_value)
        except ValueError:
            raise RuntimeError("Can't apply ADD command for none int/float operands")

        result = dest_value + src_value

        self._set_flag_value(Flag.ZERO, result == 0)
        self._set_flag_value(Flag.SIGN, result < 0)
        self._set_flag_value(Flag.CARRY, False)
        self._set_operand_value(dest, result)
        self._state.eip += 1

    def _sub(self, dest: str, src: str):
        src_value = self._get_operand_value(src)
        dest_value = self._get_operand_value(dest)
        try:
            src_value = try_convert_to_numbers(src_value)
            dest_value = try_convert_to_numbers(dest_value)
        except ValueError:
            raise RuntimeError("Can't apply SUB command for none int/float operands")
        result = dest_value - src_value

        self._set_flag_value(Flag.ZERO, result == 0)
        self._set_flag_value(Flag.SIGN, result < 0)
        self._set_flag_value(Flag.CARRY, result < 0)
        self._set_operand_value(dest, result)
        self._state.eip += 1

    def _cmp(self, src1: str, src2: str):
        src1_value = self._get_operand_value(src1)
        src2_value = self._get_operand_value(src2)
        try:
            src1_value = try_convert_to_numbers(src1_value)
            src2_value = try_convert_to_numbers(src2_value)
            result = src1_value - src2_value
        except ValueError:
            self._set_flag_value(Flag.ZERO, src1_value == src2_value)
            self._set_flag_value(Flag.SIGN, src1_value < src2_value)
            self._set_flag_value(Flag.CARRY, src1_value < src2_value)
            self._state.eip += 1
            return

        self._set_flag_value(Flag.ZERO, result == 0)
        self._set_flag_value(Flag.SIGN, result < 0)
        self._set_flag_value(Flag.CARRY, src1_value < src2_value)
        self._state.eip += 1

    def _call(self, dest: str) -> ExternCallContext | None:
        # Jump to destination
        self._state.eip += 1
        if extern_call := self._extern_calls.get(dest):
            return ExternCallContext.from_asm_stack_supplier(
                extern_call=extern_call,
                arguments_supplier=self._pop_to_extern_call,
                infer_result_hook=self._push_from_extern_call,
            )

        if dest in self._labels:
            self._state.call_stack.append(self._state.eip)
            self._state.eip = self._labels[dest]

        return None

    def _ret(self):
        if self._state.call_stack:
            self._state.eip = self._state.call_stack.pop()
            return
        self._state.eip = len(self._instructions)

    def _jmp(self, dest: str):
        if dest in self._labels:
            self._state.eip = self._labels[dest]
            return
        raise RuntimeError("Label address not found")

    def _je(self, dest: str):
        # Jump if equal instruction.
        if self._get_flag_value(Flag.ZERO):
            self._jmp(dest)
            return
        self._state.eip += 1

    def _jne(self, dest: str):
        # Jump if not equal instruction.
        if not self._get_flag_value(Flag.ZERO):
            self._jmp(dest)
            return
        self._state.eip += 1

    def _jl(self, dest: str):
        # Jump if less instruction.
        sign = self._get_flag_value(Flag.SIGN)
        if sign:
            self._jmp(dest)
            return
        self._state.eip += 1

    def _jle(self, dest: str):
        # Jump if less or equal instruction.
        sign = self._get_flag_value(Flag.SIGN)
        if self._get_flag_value(Flag.ZERO) or sign:
            self._jmp(dest)
            return
        self._state.eip += 1

    def _jg(self, dest: str):
        # Jump if greater instruction.
        sign = self._get_flag_value(Flag.SIGN)
        if not self._get_flag_value(Flag.ZERO) and not sign:
            self._jmp(dest)
            return
        self._state.eip += 1

    def _jge(self, dest: str):
        # Jump if greater or equal instruction.
        sign = self._get_flag_value(Flag.SIGN)
        if not sign:
            self._jmp(dest)
            return
        self._state.eip += 1

    def _js(self, dest: str):
        # Jump if sign instruction.
        if self._get_flag_value(Flag.SIGN):
            self._jmp(dest)
            return
        self._state.eip += 1

    def _jns(self, dest: str):
        # Jump if not sign instruction.
        if not self._get_flag_value(Flag.SIGN):
            self._jmp(dest)
            return
        self._state.eip += 1

    def _parse_labels(self):
        for inst_index, inst in enumerate(self._instructions):
            if inst.command.endswith(":") and not inst.operands:
                self._labels[inst.command[:-1]] = inst_index

    def _pop_to_extern_call(self) -> Any:
        return self._state.stack.pop()

    def _push_from_extern_call(self, value: Any):
        self._state.stack.append(value)


def try_convert_to_numbers(operand: Any) -> float | int:
    if isinstance(operand, (int, float)):
        return operand

    operand = str(operand).lower()
    if operand in ("nan", "+nan", "-nan"):
        return float("nan")
    if operand in ("inf", "+inf", "infinity", "+infinity"):
        return float("inf")
    if operand in ("-inf", "-infinity"):
        return float("-inf")

    if operand.endswith("h"):
        return int(operand[:-1], 16)
    if operand.endswith(("o", "q")):
        return int(operand[:-1], 8)
    if operand.endswith("b") or operand.startswith("0b"):
        return int(operand[:-1], 2)

    try:
        return int(operand, 0)
    except ValueError:
        pass
    return float(operand)
