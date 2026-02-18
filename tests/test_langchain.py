import pytest
from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool

from llassembly.langchain import ToolsPlannerMiddleware
from tests.conftest import vcr_config_ollama


@tool
def sum_a_b(a: int, b: int) -> int:
    """
    Takes two ints and returns sum
    """
    return a + b


@pytest.mark.vcr(**vcr_config_ollama())
def test_langchain_middleware(ollama_test_model):
    agent = create_agent(
        model=ollama_test_model,
        tools=[sum_a_b],
        middleware=[ToolsPlannerMiddleware()],
    )
    result = agent.invoke(
        {"messages": [HumanMessage("Do sum of 5 and 5")]},
    )
    messages = result["messages"]
    assert messages
    assert isinstance(messages[-2], ToolMessage)
    assert int(messages[-2].content) == 10


@pytest.mark.vcr(**vcr_config_ollama())
@pytest.mark.asyncio
async def test_async_langchain_middleware(ollama_test_model):
    agent = create_agent(
        model=ollama_test_model,
        tools=[sum_a_b],
        middleware=[ToolsPlannerMiddleware()],
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("Do sum of 5 and 5")]},
    )
    messages = result["messages"]
    assert messages
    assert isinstance(messages[-2], ToolMessage)
    assert int(messages[-2].content) == 10
