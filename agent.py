# %%
from pydantic_ai.capabilities.hooks import Hooks

from pathlib import Path
import yaml

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.capabilities import Capability
from retrieve_function import (
    get_ayahs,
    get_tafsir,
    get_hadith,
    get_aqeedah,
    hybrid_quran_search,
    hybrid_tafsir_search,
    hybrid_hadith_search,
    hybrid_aqeedah_search,
)

from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent

# %%
# Point to your local or custom endpoint
provider = OpenAIProvider(
    base_url="https://opencode.ai/zen/v1",
    api_key="sk-KEzLkDC9IkYiDlRRr4KjX0tvoaUsQDNw2gg0b88PgUJTVemSFGGNSOpc9ABZWNqO",
)
model = OpenAIChatModel("deepseek-v4-flash-free", provider=provider)

# %%

system_prompt = """
You are a Salafi Sunni Islamic scholar assistant. You answer Islamic questions using the provided retrieval tools — Quran, hadith, tafsir, and aqeedah books.

Core principles:
- The Salaf (first three generations) are the authoritative reference for understanding Islam.
- You follow their aqeedah and defend it against opposing views.
- You always respond in Arabic unless asked otherwise.
- Never fabricate quotes or knowledge. Use the tools to retrieve verified texts.
- Every claim must have a citation from the tool results.
"""


def load_skill(path: Path) -> Capability | None:
    parts = path.read_text().split("---", 2)
    if len(parts) < 3:
        print(f"⚠️  Skipping {path.name}: no YAML front matter")
        return None
    _, header, body = parts
    meta = yaml.safe_load(header)
    if not meta or "description" not in meta:
        print(f"⚠️  Skipping {path.name}: missing description in front matter")
        return None
    return Capability(
        id=meta["id"],
        description=meta["description"],
        instructions=body.strip(),
    )


# %%
skills = [s for p in Path("skills").glob("*.md") if (s := load_skill(p)) is not None]


hooks = Hooks()


@hooks.on.before_tool_execute
async def log_tool_call(ctx, *, call, tool_def, args):
    print(f"🔧 Calling {call.tool_name}({args})")
    return args


@hooks.on.after_tool_execute
async def log_tool_result(ctx, *, call, tool_def, args, result):
    print(f"✅ {call.tool_name} returned")


agent = Agent(
    model,
    name="islamic_scholar_agent",
    system_prompt=system_prompt,
    capabilities=[*skills, hooks],
    tools=[
        get_ayahs,
        get_tafsir,
        get_hadith,
        get_aqeedah,
        hybrid_quran_search,
        hybrid_tafsir_search,
        hybrid_hadith_search,
        hybrid_aqeedah_search,
    ],
)


# %%


# resp = agent.run_sync("ماهو القول المبين في من قال ان كلام الله مخلو")

# print(resp.output)
# resp.all_messages()

hybrid_aqeedah_search("الله في السماء", k=20)
