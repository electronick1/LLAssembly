"""
Benchmark runner: Claude Anthropic tool-use agent (containers) vs Claude + LLAssembly (Assembly).

Architecture comparison:
  - Anthropic tool-use (agentic): LLM makes N round-trips (one per tool call decision group)
  - LLAssembly: LLM generates assembly code once; emulator executes all tool calls locally (1 LLM request)

This script measures:
  - LLM request count per task  ← KEY METRIC: LLAssembly always = 1; containers = N
  - Wall-clock latency
  - Tool call correctness (did the right tools get called the right number of times?)
  - Token usage (total tokens across all LLM requests — cost proxy)

Usage:
    # Run all scenarios with LLAssembly approach only (no Anthropic API key needed for dry-run)
    python benchmark/run_benchmark.py --approach llassembly --dry-run

    # Run all scenarios comparing both approaches (requires ANTHROPIC_API_KEY)
    # LLAssembly uses --compact by default (token-competitive with containers)
    python benchmark/run_benchmark.py --approach both

    # LLAssembly full prompt (verbose) vs containers
    python benchmark/run_benchmark.py --approach both --no-compact

    # With prompt caching (reduces repeat LLAssembly token cost ~80%)
    python benchmark/run_benchmark.py --approach both --cache

    # Re-execution demo: generate ASM once, run 5 times (0 extra LLM requests)
    python benchmark/run_benchmark.py --scenario s1_sequential --reuse --reuse-n 5

    # Run a specific scenario
    python benchmark/run_benchmark.py --scenario s1_sequential --approach both

Requirements:
    pip install anthropic mcp[cli] langchain-anthropic

Environment variables:
    ANTHROPIC_API_KEY  — Anthropic API key
    OLLAMA_BASE_URL    — Ollama server URL (optional, for local LLM testing)
    OLLAMA_MODEL       — Ollama model name (optional, default: gpt-oss:20b)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.scenarios import (
    ALL_SCENARIOS, SCENARIOS_BY_ID, BenchmarkScenario, ToolDefinition, OutputAssertion,
)
from llassembly import ASMEmulator, ExternCall, get_asm_prompt
from llassembly.asm_emulator import ExternCallContext


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    tool_name: str
    kwargs: dict
    result: Any


@dataclass
class BenchmarkResult:
    scenario_id: str
    approach: str            # "llassembly" or "containers/tool-use"
    llm_requests: int
    latency_seconds: float
    tool_calls: list[ToolCallRecord]
    correct: bool            # Did the tool call pattern match expectations?
    output_assertions_passed: int = 0
    output_assertions_total: int = 0
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Context window growth per round (containers only — shows bloat)
    context_tokens_per_round: list[int] = field(default_factory=list)
    notes: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_tool_call(self) -> float:
        """Total tokens divided by number of tool calls — efficiency metric."""
        n = len(self.tool_calls)
        if n == 0:
            return float("inf")
        return self.total_tokens / n

    @property
    def max_context_size(self) -> int:
        """Peak context window size (containers context grows per round)."""
        return max(self.context_tokens_per_round) if self.context_tokens_per_round else self.prompt_tokens

    @property
    def peak_tokens_per_request(self) -> int:
        """Maximum tokens in any single LLM request.

        For LLAssembly: always equals prompt_tokens (one request).
        For containers: equals the last (largest) context window size, since
        context grows with every round and the final round is the most expensive.
        This metric determines what *model tier* you need — a 128k context window
        model may be required if containers bloat past 32k tokens.
        """
        if self.context_tokens_per_round:
            return max(self.context_tokens_per_round)
        return self.prompt_tokens


# ---------------------------------------------------------------------------
# LLAssembly approach
# ---------------------------------------------------------------------------


def _build_extern_calls(tools: list[ToolDefinition]) -> list[ExternCall]:
    """Convert ToolDefinitions to ExternCall objects for LLAssembly.

    Uses tool_def.timed_func as the callable so that simulated_latency_ms
    is respected during emulation, making latency benchmarks meaningful.
    """
    from langchain_core.tools import StructuredTool
    import pydantic

    extern_calls = []
    for tool_def in tools:
        # Build a plain callable that wraps timed_func.
        # Using a closure to capture tool_def per iteration.
        _td = tool_def  # capture current tool_def in closure

        def _make_timed_wrapper(td):
            def _wrapper(**kwargs):
                return td.timed_func(**kwargs)
            _wrapper.__name__ = td.name
            _wrapper.__doc__ = td.description
            return _wrapper

        func = _make_timed_wrapper(_td)

        # Create a LangChain StructuredTool wrapper — required by ExternCall.tool_handler type
        lc_tool = StructuredTool.from_function(
            func=func,
            name=tool_def.name,
            description=tool_def.description,
        )

        # Parse input args from schema
        input_args: dict[str, pydantic.fields.FieldInfo] = {}
        for arg_name, arg_info in tool_def.input_schema.get("properties", {}).items():
            annotation = str
            if arg_info.get("type") == "integer":
                annotation = int
            elif arg_info.get("type") == "number":
                annotation = float
            input_args[arg_name] = pydantic.fields.FieldInfo(
                annotation=annotation,
                description=arg_info.get("description", arg_name),
            )

        # Parse output annotations
        output_annotations: list[pydantic.fields.FieldInfo] = []
        out_schema = tool_def.output_schema
        if out_schema.get("type") == "array" and "items" in out_schema:
            for item in out_schema["items"]:
                ann = int if item.get("type") == "integer" else str
                output_annotations.append(pydantic.fields.FieldInfo(annotation=ann))
        else:
            ann = int if out_schema.get("type") == "integer" else str
            output_annotations.append(pydantic.fields.FieldInfo(annotation=ann))

        # ExternCall.tool_handler must be a BaseTool for construction (type validation),
        # but ExternCallContext.call_tool_handler() calls it as a callable.
        # Mirror what ExternCall.from_callable() does: construct with BaseTool,
        # then replace tool_handler with the raw callable for actual invocation.
        extern_call = ExternCall(
            tool_handler=lc_tool,
            name=tool_def.name,
            description=tool_def.description,
            input_args=input_args,
            output_annotations=output_annotations,
        )
        extern_call.tool_handler = func  # replace with raw callable (same as from_callable)
        extern_calls.append(extern_call)

    return extern_calls


def _get_compact_prompt_path() -> str:
    """Returns path to the compact assembly prompt (~316 tokens for S1)."""
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(module_dir, "llassembly", "prompts_md", "base_compact.md")


def _call_llm_for_assembly(
    task: str,
    extern_calls: list[ExternCall],
    dry_run: bool = False,
    compact: bool = False,
    use_cache: bool = False,
    model: str = "claude-opus-4-5",
) -> tuple[str, int, int, int]:
    """
    Call LLM to generate assembly code.

    Args:
        task: Natural language task description
        extern_calls: List of tools available for the assembly code
        dry_run: If True, return placeholder code without calling LLM
        compact: If True, use the compact prompt with compact signatures (~316 tokens).
                 Default ON — reduces prompt 2.7× vs base.md with no correctness loss.
        use_cache: If True, use Anthropic prompt caching for static prompt parts
        model: Anthropic model name to use

    Returns:
        (asm_code, llm_requests, prompt_tokens, completion_tokens)
    """
    if compact:
        prompt_path = _get_compact_prompt_path()
        compact_signatures = True
    else:
        prompt_path = None
        compact_signatures = False

    system_prompt = get_asm_prompt(
        task,
        extern_calls,
        prompt_path=prompt_path,
        compact_signatures=compact_signatures,
    )

    if dry_run:
        # Return a minimal working assembly for dry-run testing
        tool_names = [ec.name for ec in extern_calls]
        calls = "\n".join([f"    CALL {name}  ; call {name}" for name in tool_names[:1]])
        asm_code = f"section .text\nglobal _start\n_start:\n{calls}\n    RET\n"
        print(f"  [DRY RUN] Generated placeholder assembly ({len(asm_code)} chars)")
        return asm_code, 1, len(system_prompt) // 4, 50

    # Try Anthropic Claude first
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            t0 = time.monotonic()

            if use_cache:
                # Split system prompt into static (cacheable) + dynamic (tool descriptions)
                # The static part is everything before the {extern_functions} section
                # Split on "2.3. Allowed extern call functions:" or the last section header
                split_marker = "2.3. Allowed extern call functions:"
                compact_split_marker = "Functions:"
                marker = compact_split_marker if compact else split_marker
                if marker in system_prompt:
                    static_part, dynamic_part = system_prompt.split(marker, 1)
                    system_blocks = [
                        {
                            "type": "text",
                            "text": static_part + marker,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": dynamic_part,
                        },
                    ]
                    cache_label = " [CACHED]"
                else:
                    system_blocks = system_prompt
                    cache_label = ""
            else:
                system_blocks = system_prompt
                cache_label = ""

            resp = client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_blocks,
                messages=[{"role": "user", "content": task}],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"} if use_cache else {},
            )
            elapsed = time.monotonic() - t0
            cache_info = ""
            if use_cache and hasattr(resp.usage, "cache_read_input_tokens"):
                cache_read = resp.usage.cache_read_input_tokens or 0
                cache_creation = resp.usage.cache_creation_input_tokens or 0
                cache_info = f" | cache_read={cache_read} cache_creation={cache_creation}"
            print(f"  [Anthropic/{model}{cache_label}] LLM responded in {elapsed:.2f}s{cache_info}")
            return (
                resp.content[0].text,
                1,
                resp.usage.input_tokens,
                resp.usage.output_tokens,
            )
        except Exception as e:
            print(f"  [Anthropic] Error: {e} — falling back to Ollama")

    # Try Ollama
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
    try:
        import ollama
        client = ollama.Client(host=ollama_url)
        t0 = time.monotonic()
        resp = client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            model=ollama_model,
            stream=False,
            think=False,
        )
        print(f"  [Ollama:{ollama_model}] LLM responded in {time.monotonic() - t0:.2f}s")
        return resp.message.content, 1, 0, 0
    except Exception as e:
        raise RuntimeError(
            f"No LLM available. Set ANTHROPIC_API_KEY or run Ollama at {ollama_url}. Error: {e}"
        )


def run_llassembly_scenario(
    scenario: BenchmarkScenario,
    dry_run: bool = False,
    compact: bool = True,   # Default: compact prompt + compact signatures for fair comparison
    use_cache: bool = False,
    model: str = "claude-opus-4-5",
    parallel: bool = False,
) -> BenchmarkResult:
    """
    Run a benchmark scenario using the LLAssembly approach.

    LLAssembly makes exactly 1 LLM request regardless of how many tool calls
    the task requires. The emulator runs all tool calls locally.

    With compact=True (default), the prompt uses one-line function signatures
    (~316-474 tokens total) vs the verbose full prompt (~1200 tokens). This
    makes the per-request token count comparable to a single containers round-trip.

    With parallel=True, uses iter_tool_calls_parallel() to batch independent
    tool calls and execute them concurrently (requires simulated_latency_ms > 0
    on tools to show a meaningful latency improvement).
    """
    approach_label = "llassembly"
    if compact:
        approach_label += "+compact"
    if parallel:
        approach_label += "+parallel"
    if use_cache:
        approach_label += "+cache"
    if model != "claude-opus-4-5":
        approach_label += f"/{model.split('-')[1]}"

    print(f"\n[LLAssembly:{approach_label}] Running scenario: {scenario.name}")

    # S6 (re-execution) is handled by run_llassembly_reuse_scenario — skip here
    if "reuse" in scenario.tags:
        print(f"  [INFO] S6 re-execution scenario — use --reuse flag or run_llassembly_reuse_scenario()")
        return BenchmarkResult(
            scenario_id=scenario.id,
            approach=approach_label,
            llm_requests=0,
            latency_seconds=0.0,
            tool_calls=[],
            correct=False,
            notes="Use --reuse to run this scenario properly",
        )

    # Reset scenario state before each run
    if scenario.reset_fn:
        scenario.reset_fn()

    extern_calls = _build_extern_calls(scenario.tools)
    t_start = time.monotonic()
    error = None
    tool_calls_executed: list[ToolCallRecord] = []
    prompt_tokens = completion_tokens = 0
    llm_requests = 0

    try:
        asm_code, llm_req, pt, ct = _call_llm_for_assembly(
            scenario.task,
            extern_calls,
            dry_run=dry_run,
            compact=compact,
            use_cache=use_cache,
            model=model,
        )
        llm_requests += llm_req
        prompt_tokens += pt
        completion_tokens += ct

        print(f"  Generated assembly ({asm_code.count(chr(10))} lines)")
        if dry_run:
            print(f"  [DRY RUN ASM]\n{asm_code}")

        emulator = ASMEmulator.from_asm_code(asm_code)
        emulator.add_extern_calls(extern_calls)

        if parallel:
            for batch_results in emulator.iter_tool_calls_parallel():
                for tool_ctx, result in batch_results:
                    tool_calls_executed.append(ToolCallRecord(
                        tool_name=tool_ctx.extern_call.name,
                        kwargs=dict(tool_ctx.call_kwargs),
                        result=result,
                    ))
                    print(f"  [PARALLEL] CALL {tool_ctx.extern_call.name}({tool_ctx.call_kwargs}) → {result}")
        else:
            for tool_ctx in emulator.iter_tool_calls():
                result = tool_ctx.call_tool_handler()
                tool_calls_executed.append(ToolCallRecord(
                    tool_name=tool_ctx.extern_call.name,
                    kwargs=dict(tool_ctx.call_kwargs),
                    result=result,
                ))
                print(f"  CALL {tool_ctx.extern_call.name}({tool_ctx.call_kwargs}) → {result}")

    except Exception as e:
        error = str(e)
        print(f"  ERROR: {e}")

    latency = time.monotonic() - t_start

    # Check correctness + output assertions
    correct = _check_correctness(tool_calls_executed, scenario)
    assertions_passed, assertions_total = _check_output_assertions(tool_calls_executed, scenario)

    return BenchmarkResult(
        scenario_id=scenario.id,
        approach=approach_label,
        llm_requests=llm_requests,
        latency_seconds=latency,
        tool_calls=tool_calls_executed,
        correct=correct,
        output_assertions_passed=assertions_passed,
        output_assertions_total=assertions_total,
        error=error,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ---------------------------------------------------------------------------
# LLAssembly re-execution (--reuse): generate ASM once, run N times
# ---------------------------------------------------------------------------


def run_llassembly_reuse_scenario(
    scenario: BenchmarkScenario,
    n_runs: int = 3,
    compact: bool = True,
    use_cache: bool = False,
    model: str = "claude-opus-4-5",
    dry_run: bool = False,
) -> list[BenchmarkResult]:
    """
    Demonstrate LLAssembly re-execution advantage: generate assembly code ONCE with
    1 LLM request, then run it N times without any additional LLM calls.

    This is a unique LLAssembly capability that containers CANNOT replicate:
    - LLAssembly: 1 LLM request total for N executions
    - Containers: N × (N tool calls each) = N × K LLM requests

    Use cases:
    - Re-running a workflow with different inputs (change data source, retry)
    - Scheduled execution (generate plan once, run hourly without LLM cost)
    - Batch processing (same orchestration logic on different datasets)

    Returns:
        List of BenchmarkResult, one per run. First run includes LLM request count=1,
        subsequent runs have llm_requests=0 (ASM already generated).
    """
    approach_label = "llassembly+reuse"
    if compact:
        approach_label += "+compact"

    print(f"\n[LLAssembly:reuse] Running scenario: {scenario.name} × {n_runs} times")

    extern_calls = _build_extern_calls(scenario.tools)
    results: list[BenchmarkResult] = []

    # Phase 1: Generate ASM with 1 LLM call
    if scenario.reset_fn:
        scenario.reset_fn()

    try:
        asm_code, llm_req, prompt_tokens, completion_tokens = _call_llm_for_assembly(
            scenario.task,
            extern_calls,
            dry_run=dry_run,
            compact=compact,
            use_cache=use_cache,
            model=model,
        )
        print(f"  [Phase 1] ASM generated: {asm_code.count(chr(10))} lines — will reuse {n_runs} times")
    except Exception as e:
        print(f"  ERROR generating ASM: {e}")
        return []

    # Phase 2: Execute N times without additional LLM calls
    for run_idx in range(n_runs):
        if scenario.reset_fn:
            scenario.reset_fn()

        tool_calls_executed: list[ToolCallRecord] = []
        error = None
        t_start = time.monotonic()

        try:
            emulator = ASMEmulator.from_asm_code(asm_code)
            emulator.add_extern_calls(extern_calls)

            for tool_ctx in emulator.iter_tool_calls():
                result = tool_ctx.call_tool_handler()
                tool_calls_executed.append(ToolCallRecord(
                    tool_name=tool_ctx.extern_call.name,
                    kwargs=dict(tool_ctx.call_kwargs),
                    result=result,
                ))
                print(f"  [Run {run_idx+1}] CALL {tool_ctx.extern_call.name}({tool_ctx.call_kwargs}) → {result}")

        except Exception as e:
            error = str(e)
            print(f"  [Run {run_idx+1}] ERROR: {e}")

        latency = time.monotonic() - t_start
        correct = _check_correctness(tool_calls_executed, scenario)
        assertions_passed, assertions_total = _check_output_assertions(tool_calls_executed, scenario)

        results.append(BenchmarkResult(
            scenario_id=scenario.id,
            approach=f"{approach_label}/run{run_idx+1}",
            # Only the first run "counts" the LLM request; subsequent runs are free
            llm_requests=llm_req if run_idx == 0 else 0,
            latency_seconds=latency,
            tool_calls=tool_calls_executed,
            correct=correct,
            output_assertions_passed=assertions_passed,
            output_assertions_total=assertions_total,
            error=error,
            # Token cost amortizes across runs: distribute tokens across all runs
            prompt_tokens=prompt_tokens if run_idx == 0 else 0,
            completion_tokens=completion_tokens if run_idx == 0 else 0,
            notes=f"ASM reuse: 1 LLM call shared across {n_runs} runs",
        ))

    total_llm = llm_req
    total_tool_calls = sum(len(r.tool_calls) for r in results)
    print(
        f"  [Reuse summary] {total_llm} LLM request(s) for {n_runs} runs "
        f"({total_tool_calls} total tool calls) — "
        f"{total_tool_calls / max(total_llm, 1):.0f} tool calls per LLM request"
    )
    return results


# ---------------------------------------------------------------------------
# Anthropic tool-use agentic loop approach (real containers model)
# ---------------------------------------------------------------------------


def _build_anthropic_tool_specs(tools: list[ToolDefinition]) -> list[dict]:
    """Convert ToolDefinitions to Anthropic tool-use API format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in tools
    ]


def _build_tool_signatures(tools: list[ToolDefinition]) -> str:
    """Build Python-style function signatures (used for token counting only)."""
    lines = []
    for t in tools:
        props = t.input_schema.get("properties", {})
        args = ", ".join(
            f"{k}: {v.get('type', 'str')}" for k, v in props.items()
        )
        out_type = t.output_schema.get("type", "any")
        lines.append(f"def {t.name}({args}) -> {out_type}:")
        lines.append(f'    """{t.description}"""')
        lines.append("    ...")
        lines.append("")
    return "\n".join(lines)


def run_containers_scenario(
    scenario: BenchmarkScenario,
    dry_run: bool = False,
    model: str = "claude-opus-4-5",
) -> BenchmarkResult:
    """
    Run a benchmark scenario using the Anthropic real tool-use agentic loop.

    This is how Claude computer use / agent tool-use actually works:
    1. LLM receives task + tool specs
    2. LLM responds with tool_use blocks
    3. We execute the tools and send tool_result blocks back
    4. LLM decides next action (another tool call or end_turn)
    5. Repeat until stop_reason == "end_turn"

    This means N tool calls = N LLM round-trips (or fewer if LLM batches calls).
    Compare to LLAssembly which always uses exactly 1 LLM request.

    Args:
        scenario: The benchmark scenario to run
        dry_run: If True, simulate one tool call per tool without calling the LLM
        model: Anthropic model to use (default: claude-opus-4-5). Set to the same
               model as LLAssembly for a fair apples-to-apples comparison.
    """
    print(f"\n[Containers/tool-use:{model}] Running scenario: {scenario.name}")

    # Reset scenario state before each run
    if scenario.reset_fn:
        scenario.reset_fn()

    t_start = time.monotonic()
    error = None
    tool_calls_executed: list[ToolCallRecord] = []
    prompt_tokens = completion_tokens = 0
    llm_requests = 0
    context_tokens_per_round: list[int] = []

    # Build tool lookup: name → ToolDefinition
    tool_map = {t.name: t for t in scenario.tools}

    try:
        if dry_run:
            # Simulate one tool call per tool for dry-run
            for tool_def in scenario.tools[:1]:
                props = tool_def.input_schema.get("properties", {})
                dummy_kwargs = {k: "" for k in props}
                result = tool_def.func(**dummy_kwargs)
                tool_calls_executed.append(ToolCallRecord(
                    tool_name=tool_def.name, kwargs=dummy_kwargs, result=result
                ))
            llm_requests = len(scenario.tools[:1])
            prompt_tokens = sum(len(t.description) // 4 for t in scenario.tools) * llm_requests
            completion_tokens = 10 * llm_requests
            print(f"  [DRY RUN] Simulated {llm_requests} LLM request(s)")
        else:
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if not anthropic_key:
                raise RuntimeError("ANTHROPIC_API_KEY required for containers approach")

            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)

            # Build Anthropic tool specs from ToolDefinition list
            tools_spec = _build_anthropic_tool_specs(scenario.tools)
            messages = [{"role": "user", "content": scenario.task}]

            # Agentic loop: LLM → tools → LLM → ... until end_turn
            max_rounds = 50  # safety limit
            for round_num in range(max_rounds):
                t0 = time.monotonic()
                resp = client.messages.create(
                        model=model,
                        max_tokens=1024,
                        tools=tools_spec,
                        messages=messages,
                    )
                elapsed = time.monotonic() - t0
                llm_requests += 1
                prompt_tokens += resp.usage.input_tokens
                completion_tokens += resp.usage.output_tokens
                context_tokens_per_round.append(resp.usage.input_tokens)
                print(
                    f"  [Anthropic round {round_num+1}] {resp.stop_reason} "
                    f"in {elapsed:.2f}s | ctx={resp.usage.input_tokens} "
                    f"output={resp.usage.output_tokens} "
                    f"[ctx grows: +{resp.usage.input_tokens - (context_tokens_per_round[-2] if len(context_tokens_per_round) > 1 else resp.usage.input_tokens)} tokens]"
                )

                # Append assistant response to message history
                messages.append({"role": "assistant", "content": resp.content})

                if resp.stop_reason == "end_turn":
                    break

                if resp.stop_reason != "tool_use":
                    # Unexpected stop — treat as done
                    print(f"  [Containers] Unexpected stop_reason: {resp.stop_reason}")
                    break

                # Execute all tool_use blocks from this response
                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        tool_def = tool_map.get(block.name)
                        if tool_def is None:
                            result_str = f"ERROR: unknown tool {block.name}"
                        else:
                            try:
                                result = tool_def.func(**block.input)
                                tool_calls_executed.append(ToolCallRecord(
                                    tool_name=block.name,
                                    kwargs=dict(block.input),
                                    result=result,
                                ))
                                result_str = str(result)
                                print(f"  CALL {block.name}({block.input}) → {result}")
                            except Exception as e:
                                result_str = f"ERROR: {e}"
                                print(f"  CALL {block.name}({block.input}) → ERROR: {e}")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })

                # Send tool results back to LLM
                messages.append({"role": "user", "content": tool_results})

    except Exception as e:
        error = str(e)
        print(f"  ERROR: {e}")

    latency = time.monotonic() - t_start
    correct = _check_correctness(tool_calls_executed, scenario)
    assertions_passed, assertions_total = _check_output_assertions(tool_calls_executed, scenario)

    return BenchmarkResult(
        scenario_id=scenario.id,
        approach=f"containers/tool-use/{model.split('-')[1]}",
        llm_requests=llm_requests,
        latency_seconds=latency,
        tool_calls=tool_calls_executed,
        correct=correct,
        output_assertions_passed=assertions_passed,
        output_assertions_total=assertions_total,
        error=error,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        context_tokens_per_round=context_tokens_per_round,
        notes="Real Anthropic tool-use agentic loop",
    )


# ---------------------------------------------------------------------------
# Correctness checks + output assertion validation
# ---------------------------------------------------------------------------


def _check_correctness(
    executed: list[ToolCallRecord],
    scenario: BenchmarkScenario,
) -> bool:
    """Check whether the executed tool calls match scenario expectations."""
    call_counts: dict[str, int] = {}
    for rec in executed:
        call_counts[rec.tool_name] = call_counts.get(rec.tool_name, 0) + 1

    for expected in scenario.expected_tool_calls:
        actual = call_counts.get(expected.tool_name, 0)
        if actual < expected.times_called_min:
            print(
                f"  ✗ CALL_COUNT {expected.tool_name}: called {actual}x, "
                f"expected at least {expected.times_called_min}x"
            )
            return False
        if expected.times_called_max is not None and actual > expected.times_called_max:
            print(
                f"  ✗ CALL_COUNT {expected.tool_name}: called {actual}x, "
                f"expected at most {expected.times_called_max}x"
            )
            return False
    return True


def _check_output_assertions(
    executed: list[ToolCallRecord],
    scenario: BenchmarkScenario,
) -> tuple[int, int]:
    """
    Validate output values against scenario assertions.
    Returns (passed_count, total_count).
    """
    if not scenario.output_assertions:
        return 0, 0

    # Group calls by tool name
    calls_by_tool: dict[str, list[ToolCallRecord]] = {}
    for rec in executed:
        calls_by_tool.setdefault(rec.tool_name, []).append(rec)

    passed = 0
    total = len(scenario.output_assertions)

    for assertion in scenario.output_assertions:
        tool_calls = calls_by_tool.get(assertion.tool_name, [])
        if not tool_calls:
            print(f"  ⚠ OUTPUT [{assertion.tool_name}]: no calls recorded — cannot validate")
            continue

        # Get the target call
        try:
            rec = tool_calls[assertion.call_index]
        except IndexError:
            print(
                f"  ✗ OUTPUT [{assertion.tool_name}[{assertion.call_index}]]: "
                f"only {len(tool_calls)} call(s) recorded"
            )
            continue

        result = rec.result
        fail_reason = None

        # Type check
        if assertion.expected_type is not None and not isinstance(result, assertion.expected_type):
            fail_reason = f"expected type {assertion.expected_type.__name__}, got {type(result).__name__}"

        # Exact value check
        elif assertion.expected_value is not None:
            # Handle case where result might be a tuple/list and we want one element
            actual = result
            if isinstance(result, (tuple, list)) and len(result) == 1:
                actual = result[0]
            if actual != assertion.expected_value:
                fail_reason = f"expected {assertion.expected_value!r}, got {actual!r}"

        # Numeric range checks
        elif assertion.min_value is not None and result < assertion.min_value:
            fail_reason = f"expected >= {assertion.min_value}, got {result}"
        elif assertion.max_value is not None and result > assertion.max_value:
            fail_reason = f"expected <= {assertion.max_value}, got {result}"

        # String contains check
        elif assertion.contains is not None:
            result_str = str(result).lower()
            if assertion.contains.lower() not in result_str:
                fail_reason = f"expected to contain {assertion.contains!r}, got {str(result)!r}"

        if fail_reason:
            desc = f" ({assertion.description})" if assertion.description else ""
            print(f"  ✗ OUTPUT [{assertion.tool_name}[{assertion.call_index}]]{desc}: {fail_reason}")
        else:
            passed += 1
            desc = f" ({assertion.description})" if assertion.description else ""
            print(f"  ✓ OUTPUT [{assertion.tool_name}[{assertion.call_index}]]{desc}: {result!r}")

    return passed, total


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[BenchmarkResult]):
    """Print a summary table of benchmark results with efficiency metrics."""
    print("\n" + "=" * 130)
    print("BENCHMARK RESULTS")
    print("=" * 130)
    print(
        f"{'Scenario':<22} {'Approach':<30} {'LLM':<5} {'Latency':<9} "
        f"{'OK':<4} {'Outputs':<9} {'Tokens':<8} {'Tok/Call':<9} {'PeakTok':<9} {'MaxCtx':<8} Error"
    )
    print("-" * 130)
    for r in results:
        correct_str = "✓" if r.correct else "✗"
        outputs_str = (
            f"{r.output_assertions_passed}/{r.output_assertions_total}"
            if r.output_assertions_total > 0
            else "N/A"
        )
        tok_per_call = (
            f"{r.tokens_per_tool_call:.0f}" if len(r.tool_calls) > 0 else "—"
        )
        peak_tok = f"{r.peak_tokens_per_request}"
        max_ctx = f"{r.max_context_size}" if r.context_tokens_per_round else "fixed"
        error_str = r.error[:20] if r.error else ""
        print(
            f"{r.scenario_id:<22} {r.approach:<30} {r.llm_requests:<5} "
            f"{r.latency_seconds:<9.2f} {correct_str:<4} {outputs_str:<9} "
            f"{r.total_tokens:<8} {tok_per_call:<9} {peak_tok:<9} {max_ctx:<8} {error_str}"
        )
    print("=" * 130)

    # Print context window growth analysis for containers runs
    containers_results = [r for r in results if r.context_tokens_per_round]
    if containers_results:
        print("\n📈 Context Window Growth (containers/tool-use approach):")
        for r in containers_results:
            growth = [str(t) for t in r.context_tokens_per_round]
            projected_20_tools = None
            if len(r.context_tokens_per_round) >= 2:
                avg_growth = (r.context_tokens_per_round[-1] - r.context_tokens_per_round[0]) / (len(r.context_tokens_per_round) - 1)
                start = r.context_tokens_per_round[0]
                projected_20_tools = int(start + avg_growth * 20)
            proj_str = f" → projected @20 tools: ~{projected_20_tools:,} tokens" if projected_20_tools else ""
            print(f"  {r.scenario_id}: {' → '.join(growth)} tokens{proj_str}")

    # Print efficiency comparison summary
    llassembly_results = {r.scenario_id: r for r in results if r.approach.startswith("llassembly")}
    container_results_map = {r.scenario_id: r for r in results if r.approach.startswith("containers/")}
    common = set(llassembly_results) & set(container_results_map)
    if common:
        print("\n🚀 LLAssembly vs Containers Efficiency Summary:")
        print(f"  {'Scenario':<20} {'LLM requests':<20} {'Total tokens':<20} {'Tokens/call':<20}")
        print(f"  {'-'*80}")
        for sid in sorted(common):
            la = llassembly_results[sid]
            ct = container_results_map[sid]
            req_ratio = ct.llm_requests / max(la.llm_requests, 1)
            tok_ratio = ct.total_tokens / max(la.total_tokens, 1)
            print(
                f"  {sid:<20} "
                f"LLAssembly={la.llm_requests} vs containers={ct.llm_requests} ({req_ratio:.1f}x more) | "
                f"Tokens: {la.total_tokens} vs {ct.total_tokens} ({tok_ratio:.1f}x more)"
            )

    # Save JSON report
    report_path = "benchmark/results.json"
    report_data = [
        {
            "scenario_id": r.scenario_id,
            "approach": r.approach,
            "llm_requests": r.llm_requests,
            "latency_seconds": r.latency_seconds,
            "tool_calls_count": len(r.tool_calls),
            "correct": r.correct,
            "output_assertions_passed": r.output_assertions_passed,
            "output_assertions_total": r.output_assertions_total,
            "total_tokens": r.total_tokens,
            "tokens_per_tool_call": round(r.tokens_per_tool_call, 1),
            "max_context_size": r.max_context_size,
            "context_tokens_per_round": r.context_tokens_per_round,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "error": r.error,
            "notes": r.notes,
        }
        for r in results
    ]
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nDetailed results saved to {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Claude containers vs Claude + LLAssembly"
    )
    parser.add_argument(
        "--approach",
        choices=["llassembly", "containers", "both"],
        default="llassembly",
        help="Which approach to benchmark (default: llassembly)",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS_BY_ID.keys()) + ["all"],
        default="all",
        help="Which scenario to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with placeholder LLM responses (no API key required)",
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help=(
            "Disable compact mode for LLAssembly (use verbose full prompt, ~1208 tokens). "
            "By default, compact mode is ON: compact prompt + one-line signatures (~316-474 tokens). "
            "Compact mode makes LLAssembly token-competitive with containers per round-trip."
        ),
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable Anthropic prompt caching for LLAssembly (reduces token cost ~80% after first call)",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-5",
        help="Anthropic model to use (default: claude-opus-4-5). Try claude-haiku-3-5 for speed.",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Run LLAssembly with multiple model variants for comparison",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help=(
            "Demonstrate LLAssembly re-execution: generate assembly code ONCE with 1 LLM request, "
            "then execute it 3 times without additional LLM calls. "
            "Shows containers cannot replicate this: N re-runs × K tool calls = N×K LLM requests."
        ),
    )
    parser.add_argument(
        "--reuse-n",
        type=int,
        default=3,
        help="Number of re-execution runs for --reuse mode (default: 3)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Use iter_tool_calls_parallel() for LLAssembly execution. "
            "Groups independent tool calls into batches and runs them concurrently. "
            "Best used with S7 (--scenario s7_parallel_latency) to see a meaningful "
            "latency difference: 20 calls × 100ms → ~200ms parallel vs ~2,000ms sequential."
        ),
    )
    args = parser.parse_args()

    scenarios = (
        ALL_SCENARIOS
        if args.scenario == "all"
        else [SCENARIOS_BY_ID[args.scenario]]
    )

    results: list[BenchmarkResult] = []

    for scenario in scenarios:
        if args.reuse and args.approach in ("llassembly", "both"):
            # --reuse mode: generate ASM once, execute N times without additional LLM calls
            use_compact = not args.no_compact
            reuse_results = run_llassembly_reuse_scenario(
                scenario,
                n_runs=args.reuse_n,
                compact=use_compact,
                use_cache=args.cache,
                model=args.model,
            )
            results.extend(reuse_results)
            total_llm = sum(r.llm_requests for r in reuse_results)
            total_calls = sum(len(r.tool_calls) for r in reuse_results)
            print(
                f"  → {args.reuse_n} runs completed | "
                f"{total_llm} LLM request(s) total | "
                f"{total_calls} tool calls across all runs"
            )
            continue

        if args.approach in ("llassembly", "both"):
            # LLAssembly run (compact is ON by default for fair token comparison)
            use_compact = not args.no_compact

            # S6 "reuse" scenario: auto-route to run_llassembly_reuse_scenario()
            if "reuse" in scenario.tags:
                reuse_results = run_llassembly_reuse_scenario(
                    scenario,
                    n_runs=args.reuse_n,
                    compact=use_compact,
                    use_cache=args.cache,
                    model=args.model,
                    dry_run=args.dry_run,
                )
                results.extend(reuse_results)
                total_llm = sum(r.llm_requests for r in reuse_results)
                total_calls = sum(len(r.tool_calls) for r in reuse_results)
                print(
                    f"  → {args.reuse_n} runs completed | "
                    f"{total_llm} LLM request(s) total | "
                    f"{total_calls} tool calls across all runs"
                )
            else:
                # S7 "parallel" tag: auto-enable parallel execution regardless of --parallel flag
                use_parallel = args.parallel or ("parallel" in scenario.tags)
                result = run_llassembly_scenario(
                    scenario,
                    dry_run=args.dry_run,
                    compact=use_compact,
                    use_cache=args.cache,
                    model=args.model,
                    parallel=use_parallel,
                )
                results.append(result)
                status = "✓ CORRECT" if result.correct else "✗ INCORRECT"
                tokens = result.prompt_tokens + result.completion_tokens
                print(
                    f"  → {status} | {result.llm_requests} LLM request(s) | "
                    f"{result.latency_seconds:.2f}s | {tokens} tokens"
                )

            # If compare-models flag: also run with compact+cache+haiku for full comparison
            if args.compare_models and not args.dry_run:
                result_opt = run_llassembly_scenario(
                    scenario,
                    dry_run=False,
                    compact=True,
                    use_cache=True,
                    model="claude-haiku-3-5",
                )
                results.append(result_opt)
                status = "✓ CORRECT" if result_opt.correct else "✗ INCORRECT"
                tokens_opt = result_opt.prompt_tokens + result_opt.completion_tokens
                print(
                    f"  → {status} | {result_opt.llm_requests} LLM request(s) | "
                    f"{result_opt.latency_seconds:.2f}s | {tokens_opt} tokens"
                )

        if args.approach in ("containers", "both"):
            result = run_containers_scenario(scenario, dry_run=args.dry_run, model=args.model)
            results.append(result)
            status = "✓ CORRECT" if result.correct else "✗ INCORRECT"
            tokens = result.prompt_tokens + result.completion_tokens
            print(
                f"  → {status} | {result.llm_requests} LLM request(s) | "
                f"{result.latency_seconds:.2f}s | {tokens} tokens"
            )

    print_report(results)


if __name__ == "__main__":
    main()
