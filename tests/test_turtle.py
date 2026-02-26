import pytest
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage

from llassembly.langchain import ToolsPlannerMiddleware
from tests.conftest import vcr_config_ollama


@pytest.mark.vcr(**vcr_config_ollama())
def test_square_drawing_with_position_tracking(ollama_test_model, patched_turtle):
    navigator, positions_visited = patched_turtle

    agent = create_agent(
        model=ollama_test_model,
        tools=[
            navigator.forward,
            navigator.backward,
            navigator.left,
            navigator.right,
            navigator.penup,
            navigator.pendown,
            navigator.home,
        ],
        middleware=[ToolsPlannerMiddleware()],
    )
    agent.invoke(
        {
            "messages": [
                SystemMessage(
                    "Control a turtle based on the provided commands. Turtle step is 100"
                ),
                HumanMessage("Draw a square"),
            ]
        },
    )

    positions_visited = [
        (abs(int(pos[0])), abs(int(pos[1]))) for pos in positions_visited
    ]

    assert (0, 0) in positions_visited
    assert (100, 0) in positions_visited
    assert (100, 100) in positions_visited
    assert (0, 100) in positions_visited


@pytest.mark.vcr(**vcr_config_ollama())
def test_condition_stmt_when_drawing(ollama_test_model, patched_turtle):
    hunger_levels = ["low", "high"]

    def get_hunger() -> str:
        """
        Returns turtle hunger level as a string "low" or "high"
        """
        if not hunger_levels:
            return "low"
        return hunger_levels.pop()

    navigator, positions_visited = patched_turtle

    agent = create_agent(
        model=ollama_test_model,
        tools=[
            get_hunger,
            navigator.forward,
            navigator.backward,
            navigator.left,
            navigator.right,
            navigator.penup,
            navigator.pendown,
            navigator.home,
        ],
        middleware=[ToolsPlannerMiddleware()],
    )
    agent.invoke(
        {
            "messages": [
                SystemMessage(
                    "Control a turtle based on the provided commands. Turtle step is 100"
                ),
                HumanMessage(
                    "Draw a square, before each step check hunger level, if low go home"
                ),
            ]
        },
    )

    assert len(positions_visited) == 3
    positions_visited = [
        (abs(int(pos[0])), abs(int(pos[1]))) for pos in positions_visited
    ]

    assert positions_visited[0] == (0, 0)
    assert positions_visited[1] == (100, 0)
    assert positions_visited[2] == (0, 0)


@pytest.fixture
def patched_turtle(monkeypatch):
    from turtle import TNavigator

    positions_visited = []

    def position_getter(self):
        return self.__dict__["_position"]

    def position_setter(self, value):
        if not hasattr(self, "_drawing") or self._drawing:
            positions_visited.append(value)
        self.__dict__["_position"] = value

    def penup(self):
        """Pull the pen up -- no drawing when moving.

        Aliases: penup | pu | up

        No argument

        Example (for a Turtle instance named turtle):
        >>> turtle.penup()
        """
        self._drawing = False

    def pendown(self):
        """Pull the pen down -- drawing when moving.

        Aliases: pendown | pd | down

        No argument.

        Example (for a Turtle instance named turtle):
        >>> turtle.pendown()
        """
        self._drawing = True

    monkeypatch.setattr(
        TNavigator,
        "_position",
        property(position_getter, position_setter),
        raising=False,
    )
    monkeypatch.setattr(TNavigator, "penup", penup, raising=False)
    monkeypatch.setattr(TNavigator, "pendown", pendown, raising=False)
    navigator = TNavigator()
    yield navigator, positions_visited
