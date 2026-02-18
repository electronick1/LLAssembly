import json

import pytest
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage, ToolMessage

from llassembly.langchain import ToolsPlannerMiddleware
from tests.conftest import vcr_config_ollama
from tests.use_cases.npc_in_game_unit.fixtures.unit_controls import (
    UnitContext,
    attack_enemy,
    drop_sword,
    get_current_position,
    get_enemies_around,
    has_sword,
    make_one_step,
    pick_sword,
)


def _run_agent(
    ollama_model,
    tools,
    human_msg,
    context: UnitContext,
):
    agent = create_agent(
        model=ollama_model,
        tools=tools,
        middleware=[ToolsPlannerMiddleware()],
    )
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(
                    "Control in-game MPC unit based on the provided commands."
                ),
                HumanMessage(human_msg),
            ]
        },
        context=context,
    )
    return result


@pytest.mark.vcr(**vcr_config_ollama())
def test_unit_moves(ollama_test_model):
    """
    In this test case unit goes step by step (in the loop) to the coordinates 5,5
    then decide by itself how to "dance" (there is no such tool as dance) and
    then continue moving to 7,7

    All in one request to LLM.

    You can find assembly instructions generated to emulate tool calls in:
    `tests/use_cases/npc_in_game_unit/cassettes/test_unit_controls/test_unit_moves.yaml`
    """
    result = _run_agent(
        ollama_test_model,
        tools=[make_one_step, get_current_position],
        human_msg="Move to 5,5 then dance and go to 7,7",
        context=UnitContext(),
    )
    coords_unit_must_visit = {(pos, pos) for pos in range(1, 8)}
    coords_visited = set()
    for message in result["messages"]:
        if isinstance(message, ToolMessage) and message.name == "get_current_position":
            x, y = json.loads(message.content)
            coords_visited.add((x, y))

    # All coordinates (from 1,1 to 7,7) visited by unit
    assert coords_unit_must_visit & coords_visited == coords_unit_must_visit


@pytest.mark.vcr(**vcr_config_ollama())
def test_unit_condition_stmt(ollama_test_model):
    """
    In this test case unit goes step by step (in the loop) to 5,5 and then
    based on the condition checks if enemy is around on each step to 5,5
    decides if unit should pick a sword and attack enemy, after that goes to 7,7.

    all in one request to llm.

    You can find assembly instructions generated to emulate tool calls in:
    `tests/use_cases/npc_in_game_unit/cassettes/test_unit_controls/test_unit_condition_stmt.yaml`
    """
    result = _run_agent(
        ollama_test_model,
        tools=[
            make_one_step,
            get_current_position,
            pick_sword,
            drop_sword,
            has_sword,
            get_enemies_around,
            attack_enemy,
        ],
        human_msg="Go to 5,5 if you see enemy on the road attack him and run to 7,7",
        context=UnitContext(),
    )
    coords_unit_must_visit = {(pos, pos) for pos in range(1, 8)}
    coords_visited = set()
    checked_enemies_around = 0
    attacked_enemy = False
    has_sword_in_hands = False
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            match message.name:
                case "get_current_position":
                    x, y = json.loads(message.content)
                    coords_visited.add((x, y))
                case "get_enemies_around":
                    checked_enemies_around += 1
                case "attack_enemy":
                    attacked_enemy = True
                case "pick_sword":
                    has_sword_in_hands = True

    # Check enemies around on each step to 5,5
    assert checked_enemies_around >= 5
    # Attacked enemies with a sword
    assert attacked_enemy
    assert has_sword_in_hands
    # All coordinates visited by unit
    assert coords_unit_must_visit & coords_visited == coords_unit_must_visit


@pytest.mark.vcr(**vcr_config_ollama())
def test_unit_conditional_loop(ollama_test_model):
    """
    In this test case unit we check conditional loop (attack until none remain).

    Expected behaviour:
        - pick_sword called exactly once
        - attack_enemy called at least 2 times (one per enemy)
        - drop_sword called exactly once
        - has_sword flag is reset after drop
        - The unit visits every coordinate from (1,1) to (4,4)

    All in one request to LLM.

    You can find assembly instructions generated to emulate tool calls in:
    `tests/use_cases/npc_in_game_unit/cassettes/test_unit_controls/test_unit_conditional_loop.yaml`
    """
    result = _run_agent(
        ollama_test_model,
        tools=[
            make_one_step,
            get_current_position,
            pick_sword,
            drop_sword,
            has_sword,
            get_enemies_around,
            attack_enemy,
        ],
        human_msg="""if you don't have a sword pick one,
        attack all enemies until none remain, drop the sword, then go to 4,4
        """,
        context=UnitContext(),
    )

    coords_unit_must_visit = {(pos, pos) for pos in range(1, 4)}
    coords_visited = set()
    sword_picked = False
    sword_dropped = False
    attacked = 0

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            match msg.name:
                case "get_current_position":
                    x, y = json.loads(msg.content)
                    coords_visited.add((x, y))
                case "pick_sword":
                    sword_picked = True
                case "drop_sword":
                    sword_dropped = True
                case "attack_enemy":
                    attacked += 1

    assert sword_picked, "Sword should have been picked"
    assert sword_dropped, "Sword should have been dropped"
    assert attacked >= 2, "At least two attacks expected (one per enemy)"
    # All coordinates visited by unit
    assert coords_unit_must_visit & coords_visited == coords_unit_must_visit
