import json
import typing
import uuid

from langchain.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph import state as graph_state
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tool_node

from llassembly.asm_emulator import ASMEmulator, ExternCallContext


class PlannerState(MessagesState):
    plan_emulator: ASMEmulator


def build_graph_from_asm(
    emulator: ASMEmulator,
) -> graph_state.CompiledStateGraph:
    """
    Builds a LangGraph sub‑graph that mimics the execution of an assembly
    program.

    This function constructs a state graph that represents the execution flow
    of an assembly program by mapping instructions to nodes and handling
    conditional jumps and tool calls.
    """
    graph = StateGraph(PlannerState)

    tool_nodes_by_name: dict[str, ToolNode] = {}
    for extern_call in emulator.get_extern_calls().values():
        tool_nodes_by_name[extern_call.name] = ToolNode(
            [extern_call.tool_handler],
            wrap_tool_call=infer_tool_result_wrap,
        )

    emulator_instructions = emulator.get_instructions()

    # Populate graph with nodes based on the instructions set
    for inst_i, _ in enumerate(emulator_instructions):
        graph.add_node(_get_node_name(inst_i), PlanEmulatorNode(inst_i))
        if (tool_call := emulator.get_tool_call(inst_i)) and (
            tool_node := tool_nodes_by_name.get(tool_call.name)
        ):
            graph.add_node(
                _get_tool_name(tool_call.name, inst_i),
                tool_node,
            )

    # Connect nodes based on ASM instructions
    graph.add_edge(START, _get_node_name(0))
    for inst_i, _ in enumerate(emulator_instructions):
        inst_name = _get_node_name(inst_i)
        next_inst_name = _get_node_name(inst_i + 1)
        if inst_i >= len(emulator_instructions) - 1:
            next_inst_name = END

        # Adds conditional jump to the label
        if jmp_to_index := emulator.get_jmp_index(inst_i):
            jmp_to = _get_node_name(jmp_to_index)
            if jmp_to_index >= len(emulator_instructions):
                jmp_to = END

            graph.add_conditional_edges(
                _get_node_name(inst_i),
                ConditionalJMP(inst_i),
                {True: jmp_to, False: next_inst_name},
            )

        # Possible jumps to the subrutines and back (with RET)
        elif call_jmp_indexes := emulator.get_call_jmp_index(inst_i):
            call_jmp_names = set()
            for jmp_index in call_jmp_indexes:
                if jmp_index >= len(emulator_instructions):
                    call_jmp_names.add(END)
                    continue
                call_jmp_names.add(_get_node_name(jmp_index))

            if len(call_jmp_names) > 1:
                graph.add_conditional_edges(
                    _get_node_name(inst_i),
                    CallJMP(inst_i),
                    dict(zip(call_jmp_names, call_jmp_names)),
                )
            elif len(call_jmp_names) == 1:
                graph.add_edge(inst_name, list(call_jmp_names)[0])

        # Adds tool call for asm "CALL" instruction
        elif tool_call := emulator.get_tool_call(inst_i):
            tool_name = _get_tool_name(tool_call.name, inst_i)
            graph.add_edge(inst_name, tool_name)
            graph.add_edge(tool_name, next_inst_name)

        # Continue Asm execution for the next instruction
        else:
            graph.add_edge(inst_name, next_inst_name)

    return graph.compile()


class PlanEmulatorNode:
    """
    Node that represents an instruction in the assembly execution plan.

    This node handles the execution of a single instruction, checking if it
    is a tool call and preparing the necessary messages for execution.
    """

    def __init__(self, instruction_index: int):
        self.instruction_index = instruction_index

    def __call__(self, state: PlannerState) -> PlannerState:
        plan_emulator: ASMEmulator = state["plan_emulator"]
        if plan_emulator.get_current_instruction_index() != self.instruction_index:
            raise RuntimeError("Emulator out of sync with graph state")

        if extern_call_ctx := state["plan_emulator"].execute_current_instruction():
            # Executed insruction is a tool call, prepare messages state to
            # run a tool call on the next LangGraph execution step.
            tool_id = str(uuid.uuid4())
            tool_calls = [
                {
                    "id": tool_id,
                    "name": extern_call_ctx.extern_call.name,
                    "args": extern_call_ctx.call_kwargs,
                }
            ]

            asm_instruction_str = ""
            if asm_instruction := plan_emulator.get_instruction(self.instruction_index):
                asm_instruction_str = str(asm_instruction.origin)
            state["messages"].append(
                AIMessage(
                    "",
                    tool_calls=tool_calls,
                    response_metadata={
                        "extern_call_ctx": extern_call_ctx,
                        "extern_call_tool_id": tool_id,
                        "asm_instruction": asm_instruction_str,
                    },
                )
            )

        return state


def infer_tool_result_wrap(
    request: tool_node.ToolCallRequest, handler: typing.Callable
):
    result = handler(request)
    if result.status == "error":
        # TODO: consider to return result
        raise RuntimeError("Tool error")
    for message in request.state["messages"]:
        extern_call_ctx: ExternCallContext
        if not (extern_call_ctx := message.response_metadata.get("extern_call_ctx")):
            continue
        if (
            message.response_metadata.get("extern_call_tool_id")
            == request.tool_call["id"]
        ):

            if isinstance(result.content, str):
                try:
                    infer_value = json.loads(result.content)
                except json.JSONDecodeError:
                    infer_value = result.content
                extern_call_ctx.infer_result(infer_value)
            else:
                extern_call_ctx.infer_result(result.content)

            break

    return result


class ConditionalJMP:
    def __init__(self, instruction_index: int):
        self.instruction_index = instruction_index

    def __call__(self, state: PlannerState) -> bool:
        plan_emulator: ASMEmulator = state["plan_emulator"]
        if (
            abs(self.instruction_index - plan_emulator.get_current_instruction_index())
            > 1
        ):
            return True
        return False


class CallJMP:
    def __init__(self, instruction_index: int):
        self.instruction_index = instruction_index

    def __call__(self, state: PlannerState) -> str:
        plan_emulator: ASMEmulator = state["plan_emulator"]
        if plan_emulator.is_finished():
            return END
        return _get_node_name(plan_emulator.get_current_instruction_index())


def _get_node_name(index: int) -> str:
    return f"tools_planner_inst_#{index}"


def _get_tool_name(tool_name: str, index: int) -> str:
    return f"{tool_name}_inst_#{index}"
