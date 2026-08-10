from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    ModelRequestContext,
    RunContext,
    AgentStreamEvent,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_ai.capabilities import Hooks

hooks = Hooks()


def tool():
    """use this when asked to say hello"""
    return "Halla ismail"


@hooks.on.before_tool_execute
async def log_tool_call(ctx, *, call, tool_def, args):
    print(f"🔧 Calling {call.tool_name}({args})")
    return args


@hooks.on.after_tool_execute
async def after_tool_call(ctx, *, call, tool_def, args, result):
    print(f"🔧 Results {call.tool_name} {result}")
    return result


provider = OpenAIProvider(
    base_url="https://opencode.ai/zen/v1",
    api_key="sk-KEzLkDC9IkYiDlRRr4KjX0tvoaUsQDNw2gg0b88PgUJTVemSFGGNSOpc9ABZWNqO",
)
model = OpenAIChatModel("deepseek-v4-flash-free", provider=provider)

agent = Agent(model, tools=[tool])

event_test = ""


async def chat():
    global event_test
    async with agent.run_stream_events("Hello") as run:
        async for event in run:
            event_test = event
            print(event)
            print("---" * 10)


await chat()
# (await agent.run("Hello")).output

print(isinstance(event_test, AgentRunResultEvent))
print(event_test)