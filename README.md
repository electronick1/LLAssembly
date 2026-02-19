
## About

LLAssembly is a tool-orchestration library for LLM agents.

Rather than having the LLM invoke tools repeatedly in a fixed sequence, LLAssembly asks the model to write complete execution plan up-front that includes conditionals, loops, and state tracking in assembly-like program that then initiates tools during emulation process, enabling complex control flow within a single agent invocation.

Below is an updated version of the diagram from the [official LangChain documentation](https://docs.langchain.com/oss/python/langchain/agents), extended with the LLAssembly execution plan:
<details>
<summary>Diagram (click to expand)</summary>
	
```mermaid  theme={null}
%%{
  init: {
    "fontFamily": "monospace",
    "flowchart": {
      "curve": "curve"
    },
    "themeVariables": {"edgeLabelBackground": "transparent"}
  }
}%%
graph TD
  %% Outside the agent
  QUERY([input])
  LLM{model}
  TOOL(tools)
  ANSWER([output])

  %% Main flows (no inline labels)
  QUERY --> LLM
  LLM --"action"--> TOOL
  TOOL --"observation"--> LLM
  TOOL --> A
  F --> TOOL
  LLM --"finish"--> ANSWER

 subgraph LLAsembly tools plan
  A([Start]) --> AA[Executes Tool #1]
  AA --> B[What Tool #1 replied?]
  B -- Answer #1 --> C[Executes Tool #2]
  B -- Answer #2  --> D[Executes Tool #3]
  C --> E[Repeat Tool #2?]
  E -- Yes --> C
  E -- No  --> F([End])
  D --> F
end

  classDef blueHighlight fill:#0a1c25,stroke:#0a455f,color:#bae6fd;
  classDef greenHighlight fill:#0b1e1a,stroke:#0c4c39,color:#9ce4c4;
  class QUERY blueHighlight;
  class ANSWER blueHighlight;
```
</details>

⚠️ Work in progress! LLAssembly is under active development, some parts not tested well and could be unstable. Feedback and PRs are welcome. If you hit issues, please open a ticket.

## Use Cases

Most tool-using agents work like following: call a tool, show the result to the model, ask the model what to do next, repeat. It’s easy to manage and great for a simple flows, but it gets expensive and slow as soon as the task turns into a long sequence of actions or when you need the agent to react quickly to changing conditions.

LLAssembly was originally designed for in-game NPC unit control throught natural language commands. A command like:
`Go to 5,5 if you see enemy on the road attack him and run to 7,7` is a sequence of actions with conditions ("if you see enemy ...") and repeated checks (“look for an enemy at each step”). A traditional “next tool call” approach often needs an LLM round trip at each step to decide what to do next, which can quickly balloon into hundreds of requests per unit and introduce latency. With LLAssembly, you make only one request that generates a complete execution plan to react on environment change, implement conditions, loops and track state between tool calls.

This approach is particularly useful in scenarios where you need to reduce the number of requests to LLMs, and when context/environment between tool calls changes rapidly. For instance:

- **Robotics**: When decisions depend on sensor input and must happen quickly, minimizing LLM round trips is crucial.
- **Code Assistants**: When execution requires complex control-flow and number of LLM requests is what you are paying for
- **Game AI**: When you want to control NPC depending on the rapidly changing environment and there is no time to wait for a next action from LLM
- **Automated Workflows**: When you need to orchestrate multiple tools with a branching logic


## Why Assembly?

When you want tool orchestration with branching logic or loops, there are a few common approaches—each with tradeoffs:
- The traditional approach when you ask the LLM for the next tool to run on every step - this may result in many LLM requests and additional delays to get reply from LLM.
- Creating your own DSL (doman specific language) that will describe the logic for the tool calls - often leads to LLM hallucination, as it tends to make things up due to the luck of context (training set) about this custom DSL.
- Asking the model for a high-level language code (e.g. Python, JS, Lua, ...) for execution plan to invoke tool calls - this could be a more stable approach because LLMs are better at producing python code than Assembly. But it's quite unsafe and complicated to emulate high-level programming languages based on the user input. Assembly code (the light version of it) can be emulated in 300 lines of python code in a very strict and easy to control environment.

An Assembly or SQL instructions set is a middle ground between custom DSL and high-level programming code - it can be emulated in a strict environment (in fact it's converted to a LangGraph sub-graph) and most LLMs have more than enough context about Assembly to handle tool calls, for example `gpt-oss:20b` that fits in 16G GPU getting things done in handling NPC unit commands.

## How It Works

The system works by using a LangChain agent with a custom middleware or LangGraph nodes. When you invoke the agent, it:
1. Your request is wrapped in a system prompt that instructs the LLM to generate assembly-like instructions rather than directly calling tools.
2. The LLM returns a sequence of instructions describing the intended behavior and control flow.
3. The assembly code is parsed and executed through a lightweight emulator, converting each Assembly instruction to LangGraph nodes
4. During execution, the emulator performs the actual tool calls, stores intermediate results, and evaluates branches/loops based on tool outputs and tracked state.
5. The results are returned to the user, including all the intermediate tool responses

## Installation

```
uv add git+https://github.com/electronick1/LLAssembly.git
```
or
```
git clone git+https://github.com/electronick1/LLAssembly.git
cd LLAssembly
uv pip install .
```



## Examples

See more examples in `tests/use_cases`. Here's how you might use it to control a in-game NPC unit:

Langchain example:
```python
from langchain.agents import create_agent
from llassembly import ToolsPlannerMiddleware

agent = create_agent(
    model=ollama_model,
    tools=[
        make_one_step,
        get_current_position,
        pick_sword,
        attack_enemy,
        has_sword,
        get_enemies_around
    ],
    middleware=[ToolsPlannerMiddleware()],
)

result = agent.invoke({
    "messages": [
        SystemMessage("Control in-game NPC unit based on the provided commands."),
        HumanMessage("Go to 5,5 if you see enemy on the road attack him and run to 7,7"),
    ]
})
```

LangGraph example:
```python
from llassembly import ToolsPlannerNode

agent_graph = StateGraph(MessagesState)
agent_graph.add_node("llm_tools_planner", ToolsPlannerNode(ollama_model, tools=[...]))
agent_graph.add_edge(START, "llm_tools_planner")
agent_graph.add_edge("llm_tools_planner", END)
agent = agent_graph.compile()
result = agent.invoke({
    "messages": [
        SystemMessage("Control in-game NPC unit based on the provided commands."),
        HumanMessage("Go to 5,5 if you see enemy on the road attack him and run to 7,7"),
    ]
})
```

This will result in assembly code like this (generated by gpt-oss:20b): 
```assembly 
; Target coordinates for first phase
MOV R1, 5         ; R1 = target_x
MOV R2, 5         ; R2 = target_y
loop_to_5_5:
    CALL get_current_position   ; Get current position (x,y)
    POP  R3                     ; R3 = current_x
    POP  R4                     ; R4 = current_y

    CMP  R3, R1                 ; Is current_x == target_x ?
    JNE  not_at_5_5             ; If not, still moving
    CMP  R4, R2                 ; Is current_y == target_y ?
    JNE  not_at_5_5             ; If not, still moving
    JMP  after_5_5              ; Destination reached

not_at_5_5:
    CALL get_enemies_around     ; Check for enemy around
    POP  R5                     ; R5 = enemy_id (0 if none)
    CMP  R5, 0                  ; Any enemy found?
    JNE  enemy_found
    JMP  step_5_5                ; No enemy, just step

enemy_found:
    CALL has_sword              ; Does unit already have a sword?
    POP  R6                     ; R6 = has_sword flag (1 or 0)
    CMP  R6, 0                  ; No sword?
    JNE  have_sword
    CALL pick_sword             ; Pick a sword
have_sword:
    PUSH R5                     ; Push enemy_id for attack
    CALL attack_enemy          ; Attack the enemy
    JMP  step_5_5                ; Continue moving after attack

step_5_5:
    PUSH R1                     ; Push target_x
    PUSH R2                     ; Push target_y
    CALL make_one_step          ; Take one step towards (5,5)
    JMP  loop_to_5_5            ; Repeat loop

after_5_5:
    ; Set new target coordinates for second phase
    MOV R1, 7                   ; R1 = target_x (7)
    MOV R2, 7                   ; R2 = target_y (7)
    JMP  loop_to_7_7

loop_to_7_7:
    CALL get_current_position   ; Get current position (x,y)
    POP  R3                     ; R3 = current_x
    POP  R4                     ; R4 = current_y

    CMP  R3, R1                 ; Is current_x == target_x ?
    JNE  not_at_7_7             ; If not, still moving
    CMP  R4, R2                 ; Is current_y == target_y ?
    JNE  not_at_7_7             ; If not, still moving
    JMP  end_program            ; Destination reached

not_at_7_7:
    PUSH R1                     ; Push target_x
    PUSH R2                     ; Push target_y
    CALL make_one_step          ; Take one step towards (7,7)
    JMP  loop_to_7_7            ; Repeat loop

end_program:
    RET                         ; Return from program
```

Which then will be converted to the LangGraph sub-graph where each ASM instruction is
a separate node. Such sub-graph will be executed with a provided context/state
and each "CALL" instruction will result in a tool call produce `ToolMessage`.

LangGraph with `get_graph().draw_mermaid()`:
<details>
<summary>Diagram (click to expand)</summary>
	
```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	tools_planner_inst_\231(MOV R1, 5)
	tools_planner_inst_\232(MOV R2, 5)
	tools_planner_inst_\233(loop_to_5_5)
	tools_planner_inst_\234(CALL get_current_position)
	get_current_position_inst_\234(get_current_position tool instruction_#4)
	tools_planner_inst_\235(POP R3)
	tools_planner_inst_\236(POP R4)
	tools_planner_inst_\237(CMP R3, R1)
	tools_planner_inst_\238(JNE not_at_5_5)
	tools_planner_inst_\239(CMP R4, R2)
	tools_planner_inst_\2310(JNE not_at_5_5)
	tools_planner_inst_\2311(JMP after_5_5)
	tools_planner_inst_\2312(not_at_5_5)
	tools_planner_inst_\2313(CALL get_enemies_around)
	get_enemies_around_inst_\2313(get_enemies_around tool instruction_#13)
	tools_planner_inst_\2314(POP R5)
	tools_planner_inst_\2315(CMP R5, 0)
	tools_planner_inst_\2316(JNE enemy_found)
	tools_planner_inst_\2317(JMP step_5_5)
	tools_planner_inst_\2318(enemy_found:)
	tools_planner_inst_\2319(CALL has_sword)
	has_sword_inst_\2319(has_sword tool instruction_#19)
	tools_planner_inst_\2320(POP R6)
	tools_planner_inst_\2321(CMP R6, 0)
	tools_planner_inst_\2322(JNE have_sword)
	tools_planner_inst_\2323(CALL pick sword)
	pick_sword_inst_\2323(pick_sword tool instruction_#23)
	tools_planner_inst_\2324(have_sword:)
	tools_planner_inst_\2325(PUSH R5)
	tools_planner_inst_\2326(CALL attack_enemy)
	attack_enemy_inst_\2326(attack_enemy tool instruction_#26)
	tools_planner_inst_\2327(JMP step_5_5)
	tools_planner_inst_\2328(step_5_5:)
	tools_planner_inst_\2329(PUSH R1)
	tools_planner_inst_\2330(PUSH R2)
	tools_planner_inst_\2331(CALL make_one_step)
	make_one_step_inst_\2331(make_one_step tool instruction_#31)
	tools_planner_inst_\2332(JMP loop_to_5_5)
	tools_planner_inst_\2333(after_5_5:)
	tools_planner_inst_\2334(MOV R1, 7)
	tools_planner_inst_\2335(MOV R2, 7)
	tools_planner_inst_\2336(JMP loop_to_7_7)
	tools_planner_inst_\2337(loop_to_7_7:)
	tools_planner_inst_\2338(CALL get_current_position)
	get_current_position_inst_\2338(get_current_position tool instruction_#38)
	tools_planner_inst_\2339(POP R3)
	tools_planner_inst_\2340(POP R4)
	tools_planner_inst_\2341(CMP R3, R1)
	tools_planner_inst_\2342(JNE not_at_7_7)
	tools_planner_inst_\2343(CMP R4, R2)
	tools_planner_inst_\2344(JNE not_at_7_7)
	tools_planner_inst_\2345(JMP end_program)
	tools_planner_inst_\2346(not_at_7_7)
	tools_planner_inst_\2347(PUSH R1)
	tools_planner_inst_\2348(PUSH R2)
	tools_planner_inst_\2349(CALL make_one_step)
	make_one_step_inst_\2349(make_one_step_inst_#49)
	tools_planner_inst_\2350(JMP loop_to_7_7)
	tools_planner_inst_\2351(end_program:)
	tools_planner_inst_\2352(RET)
	__end__([<p>__end__</p>]):::last
	__start__ --> tools_planner_inst_\230;
	attack_enemy_inst_\2326 --> tools_planner_inst_\2327;
	get_current_position_inst_\2338 --> tools_planner_inst_\2339;
	get_current_position_inst_\234 --> tools_planner_inst_\235;
	get_enemies_around_inst_\2313 --> tools_planner_inst_\2314;
	has_sword_inst_\2319 --> tools_planner_inst_\2320;
	make_one_step_inst_\2331 --> tools_planner_inst_\2332;
	make_one_step_inst_\2349 --> tools_planner_inst_\2350;
	pick_sword_inst_\2323 --> tools_planner_inst_\2324;
	tools_planner_inst_\230 --> tools_planner_inst_\231;
	tools_planner_inst_\231 --> tools_planner_inst_\232;
	tools_planner_inst_\2310 -. &nbsp;False&nbsp; .-> tools_planner_inst_\2311;
	tools_planner_inst_\2310 -. &nbsp;True&nbsp; .-> tools_planner_inst_\2312;
	tools_planner_inst_\2311 -. &nbsp;False&nbsp; .-> tools_planner_inst_\2312;
	tools_planner_inst_\2311 -. &nbsp;True&nbsp; .-> tools_planner_inst_\2333;
	tools_planner_inst_\2312 --> tools_planner_inst_\2313;
	tools_planner_inst_\2313 --> get_enemies_around_inst_\2313;
	tools_planner_inst_\2314 --> tools_planner_inst_\2315;
	tools_planner_inst_\2315 --> tools_planner_inst_\2316;
	tools_planner_inst_\2316 -. &nbsp;False&nbsp; .-> tools_planner_inst_\2317;
	tools_planner_inst_\2316 -. &nbsp;True&nbsp; .-> tools_planner_inst_\2318;
	tools_planner_inst_\2317 -. &nbsp;False&nbsp; .-> tools_planner_inst_\2318;
	tools_planner_inst_\2317 -. &nbsp;True&nbsp; .-> tools_planner_inst_\2328;
	tools_planner_inst_\2318 --> tools_planner_inst_\2319;
	tools_planner_inst_\2319 --> has_sword_inst_\2319;
	tools_planner_inst_\232 --> tools_planner_inst_\233;
	tools_planner_inst_\2320 --> tools_planner_inst_\2321;
	tools_planner_inst_\2321 --> tools_planner_inst_\2322;
	tools_planner_inst_\2322 -. &nbsp;False&nbsp; .-> tools_planner_inst_\2323;
	tools_planner_inst_\2322 -. &nbsp;True&nbsp; .-> tools_planner_inst_\2324;
	tools_planner_inst_\2323 --> pick_sword_inst_\2323;
	tools_planner_inst_\2324 --> tools_planner_inst_\2325;
	tools_planner_inst_\2325 --> tools_planner_inst_\2326;
	tools_planner_inst_\2326 --> attack_enemy_inst_\2326;
	tools_planner_inst_\2327 -. &nbsp;False&nbsp; .-> tools_planner_inst_\2328;
	tools_planner_inst_\2328 --> tools_planner_inst_\2329;
	tools_planner_inst_\2329 --> tools_planner_inst_\2330;
	tools_planner_inst_\233 --> tools_planner_inst_\234;
	tools_planner_inst_\2330 --> tools_planner_inst_\2331;
	tools_planner_inst_\2331 --> make_one_step_inst_\2331;
	tools_planner_inst_\2332 -. &nbsp;True&nbsp; .-> tools_planner_inst_\233;
	tools_planner_inst_\2332 -. &nbsp;False&nbsp; .-> tools_planner_inst_\2333;
	tools_planner_inst_\2333 --> tools_planner_inst_\2334;
	tools_planner_inst_\2334 --> tools_planner_inst_\2335;
	tools_planner_inst_\2335 --> tools_planner_inst_\2336;
	tools_planner_inst_\2336 -. &nbsp;False&nbsp; .-> tools_planner_inst_\2337;
	tools_planner_inst_\2337 --> tools_planner_inst_\2338;
	tools_planner_inst_\2338 --> get_current_position_inst_\2338;
	tools_planner_inst_\2339 --> tools_planner_inst_\2340;
	tools_planner_inst_\234 --> get_current_position_inst_\234;
	tools_planner_inst_\2340 --> tools_planner_inst_\2341;
	tools_planner_inst_\2341 --> tools_planner_inst_\2342;
	tools_planner_inst_\2342 -. &nbsp;False&nbsp; .-> tools_planner_inst_\2343;
	tools_planner_inst_\2342 -. &nbsp;True&nbsp; .-> tools_planner_inst_\2346;
	tools_planner_inst_\2343 --> tools_planner_inst_\2344;
	tools_planner_inst_\2344 -. &nbsp;False&nbsp; .-> tools_planner_inst_\2345;
	tools_planner_inst_\2344 -. &nbsp;True&nbsp; .-> tools_planner_inst_\2346;
	tools_planner_inst_\2345 -. &nbsp;False&nbsp; .-> tools_planner_inst_\2346;
	tools_planner_inst_\2345 -. &nbsp;True&nbsp; .-> tools_planner_inst_\2351;
	tools_planner_inst_\2346 --> tools_planner_inst_\2347;
	tools_planner_inst_\2347 --> tools_planner_inst_\2348;
	tools_planner_inst_\2348 --> tools_planner_inst_\2349;
	tools_planner_inst_\2349 --> make_one_step_inst_\2349;
	tools_planner_inst_\235 --> tools_planner_inst_\236;
	tools_planner_inst_\2350 -. &nbsp;True&nbsp; .-> tools_planner_inst_\2337;
	tools_planner_inst_\2350 -. &nbsp;False&nbsp; .-> tools_planner_inst_\2351;
	tools_planner_inst_\2351 --> tools_planner_inst_\2352;
	tools_planner_inst_\2352 --> tools_planner_inst_\2353;
	tools_planner_inst_\236 --> tools_planner_inst_\237;
	tools_planner_inst_\237 --> tools_planner_inst_\238;
	tools_planner_inst_\238 -. &nbsp;True&nbsp; .-> tools_planner_inst_\2312;
	tools_planner_inst_\238 -. &nbsp;False&nbsp; .-> tools_planner_inst_\239;
	tools_planner_inst_\239 --> tools_planner_inst_\2310;
	tools_planner_inst_\2353 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
</details>

See more examples in `tests/use_cases`, as well as example of Assembly code produced
for the tool calls in `cassettes` folder for each test module.

## Implementation details:

Assembly calls a tool by pushing its input arguments onto the stack, then popping the tool’s output back off the stack. If your tool returns multiple values, declare the tool-call function’s return type as a tuple[...] so the assembly can read each result value correctly.
```python
@tool
def get_current_position(unit_id: int) -> tuple[int, int]:
	return 5,5
```
This translates to assembly:
```Assembly
PUSH 1
CALL get_current_position
POP R1	; x coordinate
POP R2	; y coordinate
```

The assembly emulator can store values of any type in its registers and stack, which makes it possible to write tool functions that return arbitrary objects. For Assembly, any non-integer value is coerced to a string (or to a JSON-formatted string where appropriate). When a value is stored as a string, comparison (CMP) and arithmetic instructions are not supported for that operand.
```Assembly
CALL get_unit
POP R1		; a json string representing unit object from get_unit
CMP R1 100	; Not possible, R1 is json string
PUSH R1
CALL heal_unit	; heal_unit will receive json string representing unit
```
