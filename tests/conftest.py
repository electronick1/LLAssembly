import json
from typing import Any, Dict

import pydantic
import pydantic_settings
import pytest
import vcr
from langchain_ollama import ChatOllama


class TestConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(env_prefix="test_llassembly_")

    ollama_host: pydantic.HttpUrl


@pytest.fixture
def config():
    return TestConfig()


@pytest.fixture
def ollama_test_model(config):
    return ChatOllama(
        base_url=str(config.ollama_host),
        model="gpt-oss:20b",
    )


def vcr_config_ollama():
    return {
        "before_record_request": ollama_prettify_vcr_record_request,
        "before_record_response": ollama_prettify_vcr_record_response,
        "match_on": ["raw_body"],
    }


def ollama_prettify_vcr_record_response(response: Dict[str, Any]) -> Dict[str, Any]:
    if not response or "body" not in response or "string" not in response["body"]:
        raise RuntimeError("Invalid ollama response")
    collapsed_content = None
    for line in response["body"]["string"].splitlines():
        line = json.loads(line)
        if "model" not in line:
            raise RuntimeError("Invalid ollama response")
        if collapsed_content is None:
            collapsed_content = line
            continue
        collapsed_content["message"]["content"] += line["message"]["content"]

    if collapsed_content is None:
        raise RuntimeError("Invalid ollama response")
    return {
        "ollama_content": [
            f"{row} " for row in collapsed_content["message"]["content"].splitlines()
        ],
        "body": {
            "string": json.dumps(collapsed_content).encode("utf-8"),
        },
        "status": response["status"],
        "headers": {},
    }


def ollama_prettify_vcr_record_request(request):
    return vcr.request.Request(
        body=request.body, method=request.method, uri="", headers={}
    )
