import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.chat import _build_agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel

def test_build_agent_openai():
    agent = _build_agent(
        provider_slug="openai",
        model_name="gpt-4o",
        provider_url="https://api.openai.com/v1",
        api_key="sk-test",
        variant="medium",
    )
    assert isinstance(agent.model, OpenAIChatModel)
    print("OpenAI agent build test passed!")

def test_build_agent_anthropic():
    agent = _build_agent(
        provider_slug="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        provider_url="https://api.anthropic.com/v1",
        api_key="sk-ant-test",
        variant="medium",
    )
    assert isinstance(agent.model, AnthropicModel)
    print("Anthropic agent build test passed!")

def test_build_agent_openrouter():
    agent = _build_agent(
        provider_slug="openrouter",
        model_name="anthropic/claude-3.5-sonnet",
        provider_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        variant="medium",
    )
    assert isinstance(agent.model, OpenAIChatModel)
    print("OpenRouter agent build test passed!")

if __name__ == "__main__":
    test_build_agent_openai()
    test_build_agent_anthropic()
    test_build_agent_openrouter()
