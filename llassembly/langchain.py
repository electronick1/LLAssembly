from typing import Awaitable, Callable, Generator

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import AIMessage, SystemMessage
from langchain_core.tools.structured import BaseTool as LangChainBaseTool
from langgraph.graph import state as graph_state

from llassembly import prompts
from llassembly.asm_emulator import ASMEmulator, ExternCall
from llassembly.langgraph_plan import PlannerState, build_graph_from_asm


class ToolsPlannerMiddleware(AgentMiddleware):
    """
    A middleware for LangChain agents that enables complex multi-step tools usage
    by generating and executing assembly-like code.

    Instead of relying on the LLM to directly invoke tools in sequence, this middleware
    uses the LLM to generate assembly code that is then emulated. The emulation process
    supports conditional logic, loops, and complex control flow, allowing for dynamic
    tool invocation within a single agent call.

    The middleware replaces the original tool calls with a system prompt that guides
    the LLM to produce valid assembly code. This code is then transformed into a
    LangGraph sub-graph, which executes the tool calls.

    Usage:
    ```
    agent = create_agent(
        model=model,
        tools=[...],
        middleware=[ToolsPlannerMiddleware()],
    )
    ```

    In this case:
    - The system message is injected into a prompt template.
    - Tool calls directly provided to the agent replaced by assembly code that
    will be emulated.
    - Assembly code is interpreted by the emulator to dynamically execute
    tool calls.

    Middleware configuration:
    - `infer_tools_messages` (bool): If True, includes messages generated
      during the assembly emulation (e.g., intermediate tool responses) in the
      final model response.
    - `prompt_path` (str | None): Path to a custom prompt file. If None, a default
      prompt is used. The LLM must generate valid assembly code compatible with
      the internal emulator.
    - `max_asm_instructions_to_exec` (int): The maximum number of Assembly
    instructions to emulate.
    """

    def __init__(
        self,
        infer_tools_messages: bool = True,
        prompt_path: str | None = None,
        max_asm_instructions_to_exec: int = 1000,
    ):
        self.infer_tools_messages = infer_tools_messages
        self.prompt_path = prompt_path
        self.max_asm_instructions_to_exec = max_asm_instructions_to_exec

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        extern_calls: list[ExternCall] = list(self._get_extern_tools(request))

        request = self._before_handler_call(request, extern_calls)
        handler_response = handler(request)
        sub_graph, state = self._get_sub_graph_with_state(
            handler_response, extern_calls
        )
        self._after_handler_call(handler_response, sub_graph.invoke(state))

        return handler_response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        extern_calls: list[ExternCall] = list(self._get_extern_tools(request))

        request = self._before_handler_call(request, extern_calls)
        handler_response = await handler(request)

        sub_graph, state = self._get_sub_graph_with_state(
            handler_response, extern_calls
        )
        self._after_handler_call(handler_response, await sub_graph.ainvoke(state))

        return handler_response

    def _get_extern_tools(self, request: ModelRequest) -> Generator[ExternCall]:
        for tool in request.tools:
            if isinstance(tool, LangChainBaseTool):
                yield ExternCall.from_langchain_tool(tool)
            else:
                raise RuntimeError("Tool must be instance of langchain BaseTool")

    def _before_handler_call(
        self, request: ModelRequest, extern_calls: list[ExternCall]
    ) -> ModelRequest:
        """
        Transforms the user request so that the LLM receives a *system prompt*
        asking to produce assembly instead of direct tool calls.

        For example when agent invoked like this:
        ```
        agent.invoke(
            {
                "messages": [
                    SystemMessage(
                        "Control in-game NPC unit based on the provided commands."
                    ),
                    HumanMessage("Move to 5,5 then dance and go to 1,1"),
                ],
            },
        )
        ```
        System message will be injected to `prompts_src/base.md` and tool calls
        will be replaced by assembly code that invokes langchain tools directly
        during assembly emulation.
        """
        current_system_msg_content = (
            request.system_message.text if request.system_message else ""
        )
        system_message = SystemMessage(
            content=prompts.get_asm_prompt(
                system_message=current_system_msg_content,
                extern_calls=extern_calls,
                prompt_path=self.prompt_path,
            )
        )
        request = request.override(tools=[])
        request = request.override(system_message=system_message)
        return request

    def _after_handler_call(
        self, handler_response: ModelResponse, graph_messages: PlannerState
    ):
        if self.infer_tools_messages:
            handler_response.result.extend(graph_messages["messages"])
            # Prevent tools to be running after model execution
            handler_response.result.append(AIMessage(content=""))

    def _get_sub_graph_with_state(
        self, handler_response: ModelResponse, extern_calls: list[ExternCall]
    ) -> tuple[graph_state.CompiledStateGraph, PlannerState]:
        """
        Makes langgraph's "Sub-graph" by transforming each assembly instruction
        into langgraph node and connecting them together based on the the assembly
        defined logic (including conditional branches and loops).
        """
        if not handler_response.result:
            raise RuntimeError("AIMessage not found in handler response")

        asm_emulator = ASMEmulator.from_asm_code(
            handler_response.result[-1].text.lower()
        )
        asm_emulator.reset_state(
            max_instructions_to_exec=self.max_asm_instructions_to_exec
        )
        for ext_call in extern_calls:
            asm_emulator.add_extern_call(ext_call)

        return (
            build_graph_from_asm(asm_emulator),
            PlannerState(plan_emulator=asm_emulator, messages=[]),
        )
