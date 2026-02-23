import typing

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import BaseTool as LangChainBaseTool
from langgraph.graph import state as graph_state
from langgraph.graph.message import MessagesState

from llassembly import prompts
from llassembly.asm_emulator import ASMEmulator, ExternCall
from llassembly.langgraph_plan import PlannerState, build_graph_from_asm


class BaseToolsPlannerNode:
    """
    A base langgraph node that orchestrates the planning process for tools based on
    assembly code provided by LLMs.

    Attributes:
    - `model`: Langchain model used for generating assembly code from tool specifications
    - `tools`: List of tools that can be invoked during plan execution
    - `max_instructions_to_exec`: The maximum number of assembly instructions to emulate
    - `infer_tools_messages`: If True, includes messages generated during the assembly
    emulation (e.g., intermediate tool responses) in the final model response.
    - `prompt_path`: Path to a custom prompt file. If None, a default prompt is used.
    The LLM must generate valid assembly code compatible with the internal emulator.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[LangChainBaseTool],
        max_instructions_to_exec: int = 1000,
        infer_tools_messages: bool = True,
        prompt_path: str | None = None,
    ):
        self.model = model
        self.extern_calls = [ExternCall.from_langchain_tool(tool) for tool in tools]
        self.max_instructions_to_exec = max_instructions_to_exec
        self.infer_tools_messages = infer_tools_messages
        self.prompt_path = prompt_path

    def _get_sub_graph_with_state(
        self, llm_response: AnyMessage
    ) -> tuple[graph_state.CompiledStateGraph, PlannerState]:
        """
        Makes langgraph's "Sub-graph" by transforming each assembly instruction
        into langgraph node and connecting them together based on the the assembly
        defined logic (including conditional branches and loops).
        """
        asm_emulator = ASMEmulator.from_asm_code(llm_response.text)
        asm_emulator.reset_state(max_instructions_to_exec=self.max_instructions_to_exec)
        for ext_call in self.extern_calls:
            asm_emulator.add_extern_call(ext_call)

        return (
            build_graph_from_asm(asm_emulator),
            PlannerState(plan_emulator=asm_emulator, messages=[]),
        )

    def get_system_message(self, messages: list[AnyMessage]) -> str:
        existing_sys_message: str | None = None
        for message in reversed(messages):
            if isinstance(message, SystemMessage) and isinstance(message.content, str):
                existing_sys_message = message.content
                break
        return prompts.get_asm_prompt(
            system_message=existing_sys_message or "",
            extern_calls=self.extern_calls,
            prompt_path=self.prompt_path,
        )

    def filter_system_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        return [msg for msg in messages if not isinstance(msg, SystemMessage)]


class ToolsPlannerNode(BaseToolsPlannerNode):
    def __call__(self, state: MessagesState) -> dict[str, typing.Any]:
        """
        Executes the planning process for the given state.

        Processes the input messages, generates assembly code using the
        language model, executes the assembly code through an emulator, and returns
        the resulting graph execution.
        """
        if not state["messages"]:
            raise RuntimeError("AIMessage not found in handler response")

        system_message = self.get_system_message(state["messages"])

        llm_response = self.model.invoke(
            [system_message, *self.filter_system_messages(state["messages"])],
            tools=[],
        )

        sub_graph, state = self._get_sub_graph_with_state(llm_response)
        graph_messages = sub_graph.invoke(state)

        if not self.infer_tools_messages:
            return {"messages": [llm_response]}

        graph_messages["messages"].append(AIMessage(content=""))
        return graph_messages


class AToolsPlannerNode(BaseToolsPlannerNode):
    async def __call__(self, state: MessagesState) -> dict[str, typing.Any]:
        """
        Executes the planning process for the given state in async mode.

        Processes the input messages, generates assembly code using the
        language model, executes the assembly code through an emulator, and returns
        the resulting graph execution.
        """
        if not state["messages"]:
            raise RuntimeError("AIMessage not found in handler response")

        system_message = self.get_system_message(state["messages"])

        llm_response = await self.model.ainvoke(
            [system_message, *self.filter_system_messages(state["messages"])],
            tools=[],
        )

        sub_graph, state = self._get_sub_graph_with_state(llm_response)
        graph_messages = await sub_graph.ainvoke(state)

        if not self.infer_tools_messages:
            return {"messages": [llm_response]}

        graph_messages["messages"].append(AIMessage(content=""))
        return graph_messages
