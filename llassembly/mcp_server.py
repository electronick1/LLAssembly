"""
LLAssembly MCP Server

Exposes LLAssembly as an MCP (Model Context Protocol) server so that
AI assistants (Claude Desktop, Cursor, etc.) can use it to orchestrate
tool calls with assembly-style planning instead of Python code execution.

Usage:
    python -m llassembly.mcp_server
    # or
    mcp run llassembly/mcp_server.py

Configure in claude_desktop_config.json:
    {
        "mcpServers": {
            "llassembly": {
                "command": "python",
                "args": ["-m", "llassembly.mcp_server"],
                "cwd": "/path/to/LLAssembly"
            }
        }
    }
"""

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from llassembly.asm_emulator import ASMEmulator, ExternCall, ExternCallContext
from llassembly.prompts import get_asm_prompt

logger = logging.getLogger("llassembly_mcp")
logger.addHandler(logging.NullHandler())

mcp = FastMCP(
    "llassembly",
    instructions=(
        "LLAssembly orchestrates tool calls using assembly-style execution plans. "
        "Use get_assembly_system_prompt to get a system prompt that instructs the LLM "
        "to generate assembly code for a task, then call execute_assembly with the "
        "generated code and registered tool results."
    ),
)

# ---------------------------------------------------------------------------
# In-memory tool registry: maps tool_name -> callable
# Populated when the MCP client registers tools via register_tool.
# ---------------------------------------------------------------------------
MAX_REGISTRY_SIZE = 500
"""Maximum number of tools that can be registered before clear_registered_tools must be called.

Prevents unbounded memory growth in long-running MCP server processes where multiple
clients may register tools without explicitly clearing them. 500 is well above any
realistic use-case while still providing a safety bound.
"""

_tool_registry: dict[str, Any] = {}
_extern_call_registry: dict[str, ExternCall] = {}


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def register_tool(
    tool_name: str,
    tool_description: str,
    input_schema: str,
    output_schema: str,
) -> str:
    """
    Register a tool with LLAssembly so it can be called during assembly execution.

    This is used to declare tools available for assembly code to CALL.
    The tool will be invoked by the MCP client when execute_assembly yields
    a tool_call event for this tool name.

    Args:
        tool_name: Unique name of the tool (used in CALL instructions)
        tool_description: Human-readable description of what the tool does
        input_schema: JSON schema string describing input arguments
        output_schema: JSON schema string describing output values

    Returns:
        Confirmation message
    """
    try:
        input_schema_dict = json.loads(input_schema) if input_schema else {}
        output_schema_dict = json.loads(output_schema) if output_schema else {}
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON schema — {e}"

    import pydantic

    input_args: dict[str, pydantic.fields.FieldInfo] = {}
    for arg_name, arg_info in input_schema_dict.get("properties", {}).items():
        annotation = str  # default to str
        arg_type = arg_info.get("type", "string")
        if arg_type == "integer":
            annotation = int
        elif arg_type == "number":
            annotation = float
        input_args[arg_name] = pydantic.fields.FieldInfo(
            annotation=annotation,
            description=arg_info.get("description", arg_name),
        )

    output_annotations: list[pydantic.fields.FieldInfo] = []
    for out_info in output_schema_dict.get("items", [output_schema_dict]):
        out_type = out_info.get("type", "string") if isinstance(out_info, dict) else "string"
        annotation = str
        if out_type == "integer":
            annotation = int
        elif out_type == "number":
            annotation = float
        output_annotations.append(pydantic.fields.FieldInfo(annotation=annotation))

    # Create a dummy langchain tool handler for ExternCall compatibility
    from langchain_core.tools import StructuredTool

    def _placeholder(**kwargs):
        """Placeholder — actual execution handled by MCP client."""
        return None

    _placeholder.__name__ = tool_name
    _placeholder.__doc__ = tool_description

    lc_tool = StructuredTool.from_function(
        func=_placeholder,
        name=tool_name,
        description=tool_description,
    )

    extern_call = ExternCall(
        tool_handler=lc_tool,
        name=tool_name,
        description=tool_description,
        input_args=input_args,
        output_annotations=output_annotations,
    )
    if tool_name not in _extern_call_registry and len(_extern_call_registry) >= MAX_REGISTRY_SIZE:
        return (
            f"Error: Tool registry is full ({MAX_REGISTRY_SIZE} tools). "
            "Call clear_registered_tools first to remove existing tools."
        )
    _extern_call_registry[tool_name] = extern_call
    return f"Tool '{tool_name}' registered successfully."


@mcp.tool()
def get_assembly_system_prompt(context: str) -> str:
    """
    Returns the system prompt to use when asking an LLM to generate assembly code
    for tool orchestration using LLAssembly.

    The returned prompt should be set as the SYSTEM message when calling the LLM.
    The LLM will then respond with assembly code that can be passed to execute_assembly.

    Args:
        context: Description of the task / system context. Include what the agent
                 should accomplish. Registered tools (via register_tool) will be
                 automatically included in the prompt.

    Returns:
        System prompt string to pass to the LLM
    """
    extern_calls = list(_extern_call_registry.values())
    return get_asm_prompt(context, extern_calls)


@mcp.tool()
def execute_assembly(
    asm_code: str,
    tool_results: str = "{}",
    max_instructions: int = 1000,
) -> str:
    """
    Execute an assembly plan generated by an LLM using the LLAssembly emulator.

    The assembly code should have been produced by an LLM using the system prompt
    returned by get_assembly_system_prompt. Tools registered via register_tool will
    be called when CALL instructions are encountered.

    When a tool call is pending (needs external execution), this tool returns a
    JSON object with status="tool_call_pending" and details of the tool to call.
    The caller should execute the tool and call execute_assembly again with
    tool_results containing the result.

    Args:
        asm_code: Assembly code string generated by an LLM
        tool_results: JSON string mapping tool_call_id -> result for pending tool calls.
                      Pass "{}" on the first call, then supply results on subsequent calls.
        max_instructions: Maximum number of instructions to execute (default 1000)

    Returns:
        JSON string with one of:
        - {"status": "completed", "tool_calls": [...], "summary": "..."}
        - {"status": "tool_call_pending", "tool_name": "...", "kwargs": {...}, "tool_call_id": "..."}
        - {"status": "error", "message": "..."}
    """
    try:
        results_map: dict = json.loads(tool_results) if tool_results.strip() else {}
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "message": f"Invalid tool_results JSON: {e}"})

    emulator = ASMEmulator.from_asm_code(asm_code, max_instructions_to_exec=max_instructions)

    for name, extern_call in _extern_call_registry.items():
        emulator.add_extern_call(extern_call)

    executed_tool_calls: list[dict] = []

    try:
        for tool_ctx in emulator.iter_tool_calls():
            tool_name = tool_ctx.extern_call.name
            call_kwargs = tool_ctx.call_kwargs
            tool_call_id = f"{tool_name}_{len(executed_tool_calls)}"

            if tool_call_id in results_map:
                # Result already supplied by caller — inject into stack
                tool_ctx.infer_result(results_map[tool_call_id])
                executed_tool_calls.append({
                    "tool_call_id": tool_call_id,
                    "tool": tool_name,
                    "kwargs": call_kwargs,
                    "result": results_map[tool_call_id],
                })
            else:
                # Tool result not yet available — ask MCP client to execute it
                return json.dumps({
                    "status": "tool_call_pending",
                    "tool_name": tool_name,
                    "kwargs": call_kwargs,
                    "tool_call_id": tool_call_id,
                    "executed_so_far": executed_tool_calls,
                    "instruction": (
                        "Execute the tool and call execute_assembly again, "
                        "adding the result to tool_results as: "
                        f'{{"{tool_call_id}": <your_result>}}'
                    ),
                })

    except RuntimeError as e:
        return json.dumps({"status": "error", "message": str(e)})

    summary_parts = [
        f"Tool '{c['tool']}' called with {c['kwargs']} → {c['result']}"
        for c in executed_tool_calls
    ]
    return json.dumps({
        "status": "completed",
        "tool_calls": executed_tool_calls,
        "summary": "\n".join(summary_parts) if summary_parts else "No tool calls executed.",
    })


@mcp.tool()
def list_registered_tools() -> str:
    """
    List all tools currently registered with LLAssembly.

    Returns:
        JSON string with list of registered tool names and descriptions
    """
    tools = [
        {"name": name, "description": ec.description}
        for name, ec in _extern_call_registry.items()
    ]
    return json.dumps({"registered_tools": tools, "count": len(tools)})


@mcp.tool()
def clear_registered_tools() -> str:
    """
    Clear all registered tools from LLAssembly.
    Use this to reset between different benchmark runs.

    Returns:
        Confirmation message
    """
    count = len(_extern_call_registry)
    _extern_call_registry.clear()
    _tool_registry.clear()
    return f"Cleared {count} registered tools."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Entry point for llassembly-mcp CLI script and python -m llassembly.mcp_server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
