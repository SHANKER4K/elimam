"""
FastAPI server for the Islamic scholar agent.
Run: uvicorn backend.server:app --reload
"""

import sys
from pathlib import Path


import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# from retrieve_function import (
#     get_ayahs,
#     get_tafsir,
#     get_hadith,
#     get_aqeedah,
#     hybrid_quran_search,
#     hybrid_tafsir_search,
#     hybrid_hadith_search,
#     hybrid_aqeedah_search,
# )

# ── Agent setup ──────────────────────────────────────────────────────────────

provider = OpenAIProvider(
    base_url="https://opencode.ai/zen/v1",
    api_key="sk-KEzLkDC9IkYiDlRRr4KjX0tvoaUsQDNw2gg0b88PgUJTVemSFGGNSOpc9ABZWNqO",
)
model = OpenAIChatModel("deepseek-v4-flash-free", provider=provider)

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
# skills = [s for p in skills_dir.glob("*.md") if (s := load_skill(p)) is not None]

hooks = Hooks()


@hooks.on.before_tool_execute
async def log_tool_call(ctx, *, call, tool_def, args):
    print(f"🔧 Calling {call.tool_name}({args})")
    return args


agent = Agent(
    model,
    name="islamic_scholar_agent",
    model_settings={"thinking": "high"},
    system_prompt=system_prompt,
    # capabilities=[*skills, hooks],
    # tools=[
    #     get_ayahs,
    #     get_tafsir,
    #     get_hadith,
    #     get_aqeedah,
    #     hybrid_quran_search,
    #     hybrid_tafsir_search,
    #     hybrid_hadith_search,
    #     hybrid_aqeedah_search,
    # ],
)

# ── FastAPI app ──────────────────────────────────────────────────────────────
messages: list | None = None

app = FastAPI(title="Islamic Scholar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ponytail: lock to your Next.js origin in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    # ponytail: add tool_calls list here if the UI needs to show them


@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    global messages
    result = await agent.run(req.message, message_history=messages)
    messages = result.all_messages()
    return ChatResponse(response=result.output)


@app.get("/health")
async def health():
    return {"status": "ok"}
