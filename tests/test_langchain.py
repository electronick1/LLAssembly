import random

import pydantic
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


@pytest.mark.vcr(**vcr_config_ollama())
def test_weather_prompt(ollama_test_model):
    @tool
    def get_weather(city: str) -> int:
        """
        Returns weather in celsius for a city.
        """
        if city.lower() == "paris":
            return -10
        return random.randint(0, 20)

    agent = create_agent(
        model=ollama_test_model,
        tools=[get_weather],
        middleware=[ToolsPlannerMiddleware()],
    )
    result = agent.invoke(
        {"messages": [HumanMessage("""
                    Check weather in Paris and if lower than 0 
                    check weather in 10 other European cities
                    """)]},
    )
    messages = result["messages"]
    called_tools = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "get_weather":
            called_tools.append(msg)

    assert len(called_tools) == 11  # Paris + 10 others
    assert messages[2].response_metadata["extern_call_ctx"].call_kwargs == {
        "city": "paris"
    }


@pytest.mark.vcr(**vcr_config_ollama())
def test_weather_pydantic_prompt(ollama_test_model):
    class Location(pydantic.BaseModel):
        city: str
        country: str
        longitude: float
        attitude: float

    @tool
    def get_weather(loc: Location) -> int:
        """
        Returns weather in celsius for a city.
        """
        if loc.city.lower() == "paris":
            return -10
        return random.randint(0, 20)

    agent = create_agent(
        model=ollama_test_model,
        tools=[get_weather],
        middleware=[ToolsPlannerMiddleware()],
    )
    result = agent.invoke(
        {"messages": [HumanMessage("""
                    Check weather in Paris and if lower than 0 
                    check weather in 10 other European cities
                    """)]},
    )
    messages = result["messages"]
    called_tools = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "get_weather":
            called_tools.append(msg)

    assert len(called_tools) == 11  # Paris + 10 others
    assert (
        messages[2].response_metadata["extern_call_ctx"].call_kwargs["loc"]["city"]
        == "paris"
    )
