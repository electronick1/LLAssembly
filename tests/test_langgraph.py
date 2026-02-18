import pytest
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

from llassembly import AToolsPlannerNode, ToolsPlannerNode
from tests.conftest import vcr_config_ollama


@tool
def sum_a_b(a: int, b: int) -> int:
    """
    Takes two ints and returns sum
    """
    return a + b


@pytest.mark.vcr(**vcr_config_ollama())
def test_langgraph_node(ollama_test_model):
    agent_graph = StateGraph(MessagesState)
    agent_graph.add_node(
        "llm_tools_planner", ToolsPlannerNode(ollama_test_model, [sum_a_b])
    )

    agent_graph.add_edge(START, "llm_tools_planner")
    agent_graph.add_edge("llm_tools_planner", END)

    agent = agent_graph.compile()

    messages = agent.invoke({"messages": [HumanMessage("Do sum of 5 and 5")]})[
        "messages"
    ]
    assert messages
    assert isinstance(messages[-2], ToolMessage)
    assert int(messages[-2].content) == 10


@pytest.mark.vcr(**vcr_config_ollama())
@pytest.mark.asyncio
async def test_async_langgraph_node(ollama_test_model):
    agent_graph = StateGraph(MessagesState)
    agent_graph.add_node(
        "llm_tools_planner", AToolsPlannerNode(ollama_test_model, [sum_a_b])
    )

    agent_graph.add_edge(START, "llm_tools_planner")
    agent_graph.add_edge("llm_tools_planner", END)

    agent = agent_graph.compile()

    messages = (await agent.ainvoke({"messages": [HumanMessage("Do sum of 5 and 5")]}))[
        "messages"
    ]
    assert isinstance(messages[-2], ToolMessage)
    assert int(messages[-2].content) == 10
