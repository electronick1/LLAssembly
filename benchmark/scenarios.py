"""
Benchmark scenarios for comparing:
  1. Claude computer use containers (Python code execution)
  2. Claude + LLAssembly (Assembly-based tool orchestration via MCP)

Each scenario defines:
  - A natural language task (unambiguous, explicit)
  - A set of mock tool functions with optional simulated latency
  - Expected tool calls (call count verification)
  - Output validators (verify actual result values make sense)
  - State reset function (so scenarios can run cleanly multiple times)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    """Describes a tool available to the agent."""
    name: str
    description: str
    func: Callable
    input_schema: dict
    output_schema: dict
    simulated_latency_ms: float = 0.0
    """Simulated network/IO latency in milliseconds.

    Use this to make latency comparisons meaningful between sequential
    and parallel execution approaches.  When > 0, each tool call sleeps
    for this many milliseconds before returning, mimicking a real API call.
    """

    def timed_func(self, **kwargs) -> Any:
        """Call func with optional simulated latency."""
        if self.simulated_latency_ms > 0:
            time.sleep(self.simulated_latency_ms / 1000.0)
        return self.func(**kwargs)


@dataclass
class ExpectedToolCall:
    """Describes an expected tool call for correctness checking."""
    tool_name: str
    times_called_min: int = 1
    times_called_max: int | None = None  # None = at least min, no upper bound


@dataclass
class OutputAssertion:
    """Validates an output value from a specific tool call."""
    tool_name: str              # Tool whose output to check
    call_index: int = 0         # Which call (0 = first, -1 = last)
    expected_value: Any = None  # Exact value to match (None = skip)
    expected_type: type = None  # Expected Python type
    min_value: Any = None       # For numeric: minimum acceptable value
    max_value: Any = None       # For numeric: maximum acceptable value
    contains: str = None        # For strings: must contain this substring
    description: str = ""       # Human-readable description of what's being checked


@dataclass
class BenchmarkScenario:
    """A single benchmark scenario."""
    id: str
    name: str
    description: str
    task: str                           # Natural language task for the agent
    tools: list[ToolDefinition]
    expected_tool_calls: list[ExpectedToolCall]
    output_assertions: list[OutputAssertion] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    reset_fn: Callable | None = None    # Called before each run to reset state


# ---------------------------------------------------------------------------
# Scenario S1: Simple sequential tool calls
# ---------------------------------------------------------------------------

def _get_ski_resort() -> str:
    """Returns a ski resort name."""
    return "Courchevel"


def _get_snow_coverage(resort: str) -> int:
    """Returns snow coverage in cm for a ski resort."""
    coverage = {"courchevel": 45, "val thorens": 80, "chamonix": 20}
    return coverage.get(resort.lower(), 0)


def _print_weather(resort: str, days_from_now: int) -> str:
    """Returns weather forecast for a ski resort for a given number of days ahead."""
    return f"Weather for {resort} over the next {days_from_now} days: Sunny, -5°C"


S1_SEQUENTIAL = BenchmarkScenario(
    id="s1_sequential",
    name="Sequential Tool Calls",
    description=(
        "Calls three tools exactly once each in sequence: "
        "get resort → get snow coverage → print weather. "
        "Tests basic sequential tool orchestration with no branching."
    ),
    task=(
        "Step 1: Call get_ski_resort exactly once to get the resort name. "
        "Step 2: Call get_snow_coverage exactly once with that resort name. "
        "Step 3: Call print_weather exactly once with the resort name and days_from_now=7. "
        "Do not call any tool more than once."
    ),
    tools=[
        ToolDefinition(
            name="get_ski_resort",
            description="Returns the name of a ski resort. Call this first.",
            func=_get_ski_resort,
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "string"},
        ),
        ToolDefinition(
            name="get_snow_coverage",
            description="Returns snow coverage in cm for the given resort name (string argument).",
            func=_get_snow_coverage,
            input_schema={
                "type": "object",
                "properties": {"resort": {"type": "string", "description": "Resort name from get_ski_resort"}},
                "required": ["resort"],
            },
            output_schema={"type": "integer"},
        ),
        ToolDefinition(
            name="print_weather",
            description="Returns weather forecast. Takes resort (string) and days_from_now (integer).",
            func=_print_weather,
            input_schema={
                "type": "object",
                "properties": {
                    "resort": {"type": "string"},
                    "days_from_now": {"type": "integer"},
                },
                "required": ["resort", "days_from_now"],
            },
            output_schema={"type": "string"},
        ),
    ],
    expected_tool_calls=[
        ExpectedToolCall("get_ski_resort", times_called_min=1, times_called_max=1),
        ExpectedToolCall("get_snow_coverage", times_called_min=1, times_called_max=1),
        ExpectedToolCall("print_weather", times_called_min=1, times_called_max=1),
    ],
    output_assertions=[
        OutputAssertion("get_ski_resort", call_index=0, expected_value="Courchevel",
                        description="Resort name should be Courchevel"),
        OutputAssertion("get_snow_coverage", call_index=0, expected_value=45,
                        description="Snow coverage should be 45cm for Courchevel"),
        OutputAssertion("print_weather", call_index=0, contains="Courchevel",
                        description="Weather forecast should mention Courchevel"),
        OutputAssertion("print_weather", call_index=0, contains="7",
                        description="Weather forecast should mention 7 days"),
    ],
    tags=["sequential"],
)


# ---------------------------------------------------------------------------
# Scenario S2: Conditional branching
# ---------------------------------------------------------------------------

_snow_data = {"paris": 0, "london": 0, "zurich": 30, "oslo": 60, "madrid": 0}


def _get_city_snow(city: str) -> int:
    """Returns snow coverage in cm for a city."""
    return _snow_data.get(city.lower(), 0)


def _book_ski_trip(city: str) -> str:
    """Books a ski trip to the given city."""
    return f"Ski trip to {city} booked successfully!"


def _send_no_snow_alert(city: str) -> str:
    """Sends an alert that there is no snow in a city."""
    return f"Alert sent: No snow in {city}"


S2_CONDITIONAL = BenchmarkScenario(
    id="s2_conditional",
    name="Conditional Branching",
    description=(
        "Checks snow in Zurich (returns 30cm > 20cm threshold). "
        "Expected: book_ski_trip called. send_no_snow_alert NOT called. "
        "Tests conditional control flow (JNE/JE)."
    ),
    task=(
        "Call get_city_snow with city='Zurich' to get snow coverage. "
        "If the result is greater than 20, call book_ski_trip with city='Zurich'. "
        "If the result is 20 or less, call send_no_snow_alert with city='Zurich'. "
        "Call exactly one of book_ski_trip or send_no_snow_alert."
    ),
    tools=[
        ToolDefinition(
            name="get_city_snow",
            description="Returns snow coverage in cm (integer) for a city name (string).",
            func=_get_city_snow,
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            output_schema={"type": "integer"},
        ),
        ToolDefinition(
            name="book_ski_trip",
            description="Books a ski trip to the given city name (string). Call when snow > 20cm.",
            func=_book_ski_trip,
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            output_schema={"type": "string"},
        ),
        ToolDefinition(
            name="send_no_snow_alert",
            description="Sends a no-snow alert for the city name (string). Call when snow <= 20cm.",
            func=_send_no_snow_alert,
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            output_schema={"type": "string"},
        ),
    ],
    expected_tool_calls=[
        ExpectedToolCall("get_city_snow", times_called_min=1, times_called_max=1),
        # book_ski_trip expected since Zurich has 30cm > 20cm threshold
        ExpectedToolCall("book_ski_trip", times_called_min=1, times_called_max=1),
    ],
    output_assertions=[
        OutputAssertion("get_city_snow", call_index=0, expected_value=30,
                        description="Zurich snow should be 30cm"),
        OutputAssertion("book_ski_trip", call_index=0, contains="Zurich",
                        description="Booking confirmation should mention Zurich"),
        OutputAssertion("book_ski_trip", call_index=0, contains="booked",
                        description="Booking confirmation should say 'booked'"),
    ],
    tags=["conditional"],
)


# ---------------------------------------------------------------------------
# Scenario S3: Loop — check multiple cities until snow found
# ---------------------------------------------------------------------------

_cities_to_check = ["madrid", "paris", "london", "oslo", "lisbon"]
_city_index = 0


def _reset_s3():
    global _city_index
    _city_index = 0


def _get_next_city() -> str:
    """Returns the next city to check from the list (stateful iterator)."""
    global _city_index
    city = _cities_to_check[_city_index % len(_cities_to_check)]
    _city_index += 1
    return city


def _check_city_snow(city: str) -> int:
    """Returns snow coverage in cm for a city. Oslo has 60cm, others 0."""
    data = {"madrid": 0, "paris": 0, "london": 0, "oslo": 60, "lisbon": 0}
    return data.get(city.lower(), 0)


def _report_snow_city(city: str, snow_cm: int) -> str:
    """Reports a city has snow. Returns a confirmation string."""
    return f"Found snow! {city} has {snow_cm}cm of snow."


S3_LOOP = BenchmarkScenario(
    id="s3_loop",
    name="Loop Until Condition Met",
    description=(
        "Loops through cities (madrid→paris→london→oslo) checking snow. "
        "Oslo (4th city) has 60cm. Expected: get_next_city + check_city_snow called 4x, "
        "report_snow_city called once with oslo/60. Tests loop control flow."
    ),
    task=(
        "Loop: call get_next_city (no arguments) to get a city name, "
        "then call check_city_snow with that city name. "
        "If check_city_snow returns 0, continue the loop and get the next city. "
        "If check_city_snow returns greater than 0, stop looping and call report_snow_city "
        "with that city name and the snow amount. "
        "Check at most 5 cities. Stop as soon as you find snow."
    ),
    tools=[
        ToolDefinition(
            name="get_next_city",
            description="Returns the next city name (string) to check. No arguments. Stateful iterator.",
            func=_get_next_city,
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "string"},
        ),
        ToolDefinition(
            name="check_city_snow",
            description="Returns snow coverage in cm (integer) for a given city name (string).",
            func=_check_city_snow,
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            output_schema={"type": "integer"},
        ),
        ToolDefinition(
            name="report_snow_city",
            description=(
                "Reports that a city has snow. Takes city name (string) "
                "and snow_cm (integer). Call exactly once when snow is found."
            ),
            func=_report_snow_city,
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "snow_cm": {"type": "integer"},
                },
                "required": ["city", "snow_cm"],
            },
            output_schema={"type": "string"},
        ),
    ],
    expected_tool_calls=[
        ExpectedToolCall("get_next_city", times_called_min=4, times_called_max=5),
        ExpectedToolCall("check_city_snow", times_called_min=4, times_called_max=5),
        ExpectedToolCall("report_snow_city", times_called_min=1, times_called_max=1),
    ],
    output_assertions=[
        OutputAssertion("report_snow_city", call_index=0, contains="oslo",
                        description="Reported city should be oslo"),
        OutputAssertion("report_snow_city", call_index=0, contains="60",
                        description="Reported snow amount should be 60cm"),
        OutputAssertion("report_snow_city", call_index=0, contains="Found snow",
                        description="Report should start with 'Found snow'"),
    ],
    reset_fn=_reset_s3,
    tags=["loop"],
)


# ---------------------------------------------------------------------------
# Scenario S4: Complex — loop + conditional (NPC unit movement)
# ---------------------------------------------------------------------------

_position = [0, 0]
_enemies = {(2, 2): "goblin", (4, 3): "orc"}
_s4_has_sword = [False]


def _reset_s4():
    _position[0] = 0
    _position[1] = 0
    _s4_has_sword[0] = False


def _get_position() -> tuple[int, int]:
    """Returns current unit position as (x, y)."""
    return (_position[0], _position[1])


def _make_step(target_x: int, target_y: int) -> str:
    """Moves unit one step towards target_x, target_y. Returns new position as 'x,y'."""
    dx = target_x - _position[0]
    dy = target_y - _position[1]
    if dx != 0:
        _position[0] += 1 if dx > 0 else -1
    elif dy != 0:
        _position[1] += 1 if dy > 0 else -1
    return f"Moved to {_position[0]},{_position[1]}"


def _get_enemies_around() -> int:
    """Returns 1 if enemy at current position, 0 if none."""
    pos = (_position[0], _position[1])
    return 1 if pos in _enemies else 0


def _pick_sword() -> str:
    """Picks up a sword. Returns 'Sword picked up'."""
    _s4_has_sword[0] = True
    return "Sword picked up"


def _attack_enemy(enemy_id: int) -> str:
    """Attacks enemy with given enemy_id (integer 1 or higher). Returns result."""
    return f"Attacked enemy {enemy_id}"


S4_COMPLEX = BenchmarkScenario(
    id="s4_complex",
    name="Complex: Loop + Conditional (NPC Control)",
    description=(
        "NPC starts at (0,0) and must reach (3,3). "
        "At each step check for enemies. The path is clear (no enemies on 0→3,3 path). "
        "Tests loop + conditional without enemy encounters (clean path)."
    ),
    task=(
        "Move an NPC unit from its current position to target position x=3, y=3. "
        "Algorithm: "
        "(1) Call get_position to get current x,y. "
        "(2) If current position is x=3 and y=3, stop — you have arrived. "
        "(3) Call get_enemies_around. If it returns 1 (enemy found): "
        "    call pick_sword (no arguments), then call attack_enemy with enemy_id=1. "
        "(4) Call make_step with target_x=3 and target_y=3. "
        "(5) Go back to step 1. "
        "Repeat until you arrive at x=3, y=3."
    ),
    tools=[
        ToolDefinition(
            name="get_position",
            description=(
                "Returns current unit position. "
                "Returns two integer values: x and y (use POP twice to get both)."
            ),
            func=_get_position,
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "array", "items": [{"type": "integer"}, {"type": "integer"}]},
        ),
        ToolDefinition(
            name="make_step",
            description=(
                "Moves unit one step towards target_x (integer), target_y (integer). "
                "Returns new position as string 'Moved to x,y'."
            ),
            func=_make_step,
            input_schema={
                "type": "object",
                "properties": {
                    "target_x": {"type": "integer"},
                    "target_y": {"type": "integer"},
                },
                "required": ["target_x", "target_y"],
            },
            output_schema={"type": "string"},
        ),
        ToolDefinition(
            name="get_enemies_around",
            description="Returns 1 (integer) if enemy at current position, 0 if none.",
            func=_get_enemies_around,
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "integer"},
        ),
        ToolDefinition(
            name="pick_sword",
            description="Picks up a sword. No arguments. Returns 'Sword picked up'.",
            func=_pick_sword,
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "string"},
        ),
        ToolDefinition(
            name="attack_enemy",
            description="Attacks enemy. Takes enemy_id (integer, use 1). Returns attack result.",
            func=_attack_enemy,
            input_schema={
                "type": "object",
                "properties": {"enemy_id": {"type": "integer"}},
                "required": ["enemy_id"],
            },
            output_schema={"type": "string"},
        ),
    ],
    expected_tool_calls=[
        ExpectedToolCall("get_position", times_called_min=6),   # called once per loop iteration + final check
        ExpectedToolCall("make_step", times_called_min=6),       # 6 steps from (0,0) to (3,3): 3 right + 3 up
        ExpectedToolCall("get_enemies_around", times_called_min=6),
    ],
    output_assertions=[
        OutputAssertion("make_step", call_index=-1, contains="3,3",
                        description="Last step should arrive at position 3,3"),
    ],
    reset_fn=_reset_s4,
    tags=["loop", "conditional", "complex", "npc"],
)


# ---------------------------------------------------------------------------
# Scenario S5: Many tool calls (10 cities, showing 1-request advantage)
# ---------------------------------------------------------------------------

_CITIES_10 = ["paris", "london", "berlin", "madrid", "rome",
               "amsterdam", "vienna", "prague", "warsaw", "zurich"]
_EXPECTED_TEMPS = {
    "paris": 12, "london": 8, "berlin": 5, "madrid": 18, "rome": 20,
    "amsterdam": 9, "vienna": 6, "prague": 4, "warsaw": 2, "zurich": 7,
}


def _get_city_temperature(city: str) -> int:
    """Returns temperature in Celsius for a city."""
    return _EXPECTED_TEMPS.get(city.lower(), 15)


def _log_temperature(city: str, temp: int) -> str:
    """Logs a temperature reading for a city. Returns confirmation."""
    return f"Logged: {city}={temp}°C"


S5_MANY_TOOLS = BenchmarkScenario(
    id="s5_many_tools",
    name="Many Tool Calls (10 Cities × 2 = 20 calls)",
    description=(
        "For each of 10 cities: get temperature then log it. "
        "20 total tool calls. Shows LLAssembly vs containers at scale."
    ),
    task=(
        "For each of these 10 cities in this exact order — "
        "Paris, London, Berlin, Madrid, Rome, Amsterdam, Vienna, Prague, Warsaw, Zurich — "
        "do the following two steps: "
        "(1) Call get_city_temperature with the city name to get its temperature. "
        "(2) Call log_temperature with that city name and the temperature value you just got. "
        "Repeat for all 10 cities. Do not skip any city."
    ),
    tools=[
        ToolDefinition(
            name="get_city_temperature",
            description="Returns temperature in Celsius (integer) for a given city name (string).",
            func=_get_city_temperature,
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
            output_schema={"type": "integer"},
        ),
        ToolDefinition(
            name="log_temperature",
            description=(
                "Logs a temperature reading. "
                "Takes city (string) and temp (integer). "
                "Returns 'Logged: city=temp°C'."
            ),
            func=_log_temperature,
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "temp": {"type": "integer"},
                },
                "required": ["city", "temp"],
            },
            output_schema={"type": "string"},
        ),
    ],
    expected_tool_calls=[
        ExpectedToolCall("get_city_temperature", times_called_min=10, times_called_max=10),
        ExpectedToolCall("log_temperature", times_called_min=10, times_called_max=10),
    ],
    output_assertions=[
        # First city: Paris = 12°C
        OutputAssertion("get_city_temperature", call_index=0, expected_value=12,
                        description="Paris temperature should be 12°C"),
        # Last city: Zurich = 7°C
        OutputAssertion("get_city_temperature", call_index=-1, expected_value=7,
                        description="Zurich temperature should be 7°C"),
        # Log entries should contain city names
        OutputAssertion("log_temperature", call_index=0, contains="Paris",
                        description="First log entry should mention Paris"),
        OutputAssertion("log_temperature", call_index=-1, contains="Zurich",
                        description="Last log entry should mention Zurich"),
        OutputAssertion("log_temperature", call_index=-1, contains="7",
                        description="Last log entry should show 7°C"),
    ],
    tags=["many_tools", "sequential"],
)


# ---------------------------------------------------------------------------
# Scenario S6: Re-execution advantage — same plan, many runs
# ---------------------------------------------------------------------------
# Same tools as S1 but showcasing that LLAssembly generates ONE plan
# and the runner can execute it N times without extra LLM calls.
# The BenchmarkScenario here is used as a marker; the runner handles
# re-execution logic separately via --reuse / run_llassembly_reuse_scenario.

S6_REUSE = BenchmarkScenario(
    id="s6_reuse",
    name="Re-execution: 1 LLM request → 10 runs",
    description=(
        "Same 3-call workflow (get_resort→get_snow→print_weather) run 10 times. "
        "LLAssembly: 1 LLM request for all 10 runs. "
        "Containers: 3 LLM round-trips × 10 = 30 total. "
        "This scenario is always run with --reuse-n 10 by default."
    ),
    task=(
        "Step 1: Call get_ski_resort exactly once to get the resort name. "
        "Step 2: Call get_snow_coverage exactly once with that resort name. "
        "Step 3: Call print_weather exactly once with the resort name and days_from_now=7. "
        "Do not call any tool more than once."
    ),
    tools=[
        ToolDefinition(
            name="get_ski_resort",
            description="Returns the name of a ski resort. Call this first.",
            func=_get_ski_resort,
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "string"},
        ),
        ToolDefinition(
            name="get_snow_coverage",
            description="Returns snow coverage in cm for the given resort name (string argument).",
            func=_get_snow_coverage,
            input_schema={
                "type": "object",
                "properties": {"resort": {"type": "string", "description": "Resort name from get_ski_resort"}},
                "required": ["resort"],
            },
            output_schema={"type": "integer"},
        ),
        ToolDefinition(
            name="print_weather",
            description="Returns weather forecast. Takes resort (string) and days_from_now (integer).",
            func=_print_weather,
            input_schema={
                "type": "object",
                "properties": {
                    "resort": {"type": "string"},
                    "days_from_now": {"type": "integer"},
                },
                "required": ["resort", "days_from_now"],
            },
            output_schema={"type": "string"},
        ),
    ],
    expected_tool_calls=[
        ExpectedToolCall("get_ski_resort", times_called_min=1, times_called_max=1),
        ExpectedToolCall("get_snow_coverage", times_called_min=1, times_called_max=1),
        ExpectedToolCall("print_weather", times_called_min=1, times_called_max=1),
    ],
    output_assertions=[
        OutputAssertion("get_ski_resort", call_index=0, expected_value="Courchevel",
                        description="Resort name should be Courchevel"),
    ],
    tags=["reuse", "sequential"],
)


# ---------------------------------------------------------------------------
# Scenario S7: Parallel latency — 10 cities with simulated 100ms API latency
# ---------------------------------------------------------------------------
# This scenario demonstrates iter_tool_calls_parallel():
#   - Sequential:   20 tool calls × 100ms = ~2,000ms wall-clock
#   - Parallel:     2 batches (10 fetches + 10 logs) × 100ms = ~200ms
#   - Containers:   10 LLM rounds, each with 1-2 tool calls × 100ms = ~1,000ms
#
# The simulated_latency_ms=100 on each tool makes these differences visible.

def _get_city_temperature_slow(city: str) -> int:
    """Returns temperature in Celsius for a city (simulates 100ms API call)."""
    return _EXPECTED_TEMPS.get(city.lower(), 15)


def _log_temperature_slow(city: str, temp: int) -> str:
    """Logs a temperature reading for a city (simulates 100ms write). Returns confirmation."""
    return f"Logged: {city}={temp}°C"


S7_PARALLEL_LATENCY = BenchmarkScenario(
    id="s7_parallel_latency",
    name="Parallel Latency (10 Cities, 100ms/call)",
    description=(
        "Same as S5 but each tool has 100ms simulated latency. "
        "Sequential: 20 × 100ms = 2,000ms. "
        "LLAssembly parallel: ~200ms (10 fetches || 10 logs). "
        "Containers: ~1,000ms (10 LLM rounds × 100ms)."
    ),
    task=(
        "For each of these 10 cities in this exact order — "
        "Paris, London, Berlin, Madrid, Rome, Amsterdam, Vienna, Prague, Warsaw, Zurich — "
        "do the following two steps: "
        "(1) Call get_city_temperature with the city name to get its temperature. "
        "(2) Call log_temperature with that city name and the temperature value you just got. "
        "Repeat for all 10 cities. Do not skip any city."
    ),
    tools=[
        ToolDefinition(
            name="get_city_temperature",
            description="Returns temperature in Celsius (integer) for a given city name (string).",
            func=_get_city_temperature_slow,
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
            output_schema={"type": "integer"},
            simulated_latency_ms=100.0,
        ),
        ToolDefinition(
            name="log_temperature",
            description=(
                "Logs a temperature reading. "
                "Takes city (string) and temp (integer). "
                "Returns 'Logged: city=temp°C'."
            ),
            func=_log_temperature_slow,
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "temp": {"type": "integer"},
                },
                "required": ["city", "temp"],
            },
            output_schema={"type": "string"},
            simulated_latency_ms=100.0,
        ),
    ],
    expected_tool_calls=[
        ExpectedToolCall("get_city_temperature", times_called_min=10, times_called_max=10),
        ExpectedToolCall("log_temperature", times_called_min=10, times_called_max=10),
    ],
    output_assertions=[
        OutputAssertion("get_city_temperature", call_index=0, expected_value=12,
                        description="Paris temperature should be 12°C"),
        OutputAssertion("get_city_temperature", call_index=-1, expected_value=7,
                        description="Zurich temperature should be 7°C"),
    ],
    tags=["many_tools", "parallel", "latency"],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_SCENARIOS: list[BenchmarkScenario] = [
    S1_SEQUENTIAL,
    S2_CONDITIONAL,
    S3_LOOP,
    S4_COMPLEX,
    S5_MANY_TOOLS,
    S6_REUSE,
    S7_PARALLEL_LATENCY,
]

SCENARIOS_BY_ID: dict[str, BenchmarkScenario] = {s.id: s for s in ALL_SCENARIOS}
