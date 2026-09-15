import os
from dotenv import load_dotenv

load_dotenv()

from pydantic_ai import Agent
from pydantic_ai.agent.abstract import Instructions
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pathlib import Path
import yaml


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


provider = OpenAIProvider(
    base_url="https://opencode.ai/zen/v1",
    api_key=os.environ["PROVIDER_API_KEY"],
)
model = OpenAIChatModel("deepseek-v4-flash-free", provider=provider)


def load_skill(path: Path) -> Capability | None:
    parts = path.read_text().split("---", 2)
    if len(parts) < 3:
        return None
    _, header, body = parts
    meta = yaml.safe_load(header)
    if not meta or "description" not in meta:
        return None
    return Capability(
        id=meta["id"],
        description=meta["description"],
        instructions=body.strip(),
    )


skills_dir = Path("skills")
skills = [s for p in skills_dir.glob("*.md") if (s := load_skill(p)) is not None]


system_prompt = """
You are a Salafi Sunni Islamic scholar assistant. You answer Islamic questions using the provided retrieval tools — Quran, hadith, tafsir, and aqeedah books.

Core principles:
- The Salaf (first three generations) are the authoritative reference for understanding Islam.
- You follow their aqeedah and defend it against opposing views.
- You always respond in Arabic unless asked otherwise.
- Never fabricate quotes or knowledge. Use the tools to retrieve verified texts.
- Every claim must have a citation from the tool results.
"""

agent = Agent(
    model,
    name="islamic_scholar_agent",
    system_prompt=system_prompt,
    capabilities=[*skills],
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

app = agent.to_web(instructions=system_prompt)
