"""
FastAPI server for the Islamic scholar agent.
Run: uvicorn backend.server:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from api.chat import router as chat_router
from api.sessions import router as sessions_router
from api.users import router as users_router
from api.messages import router as messages_router
from api.keys import router as keys_router


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Islamic Scholar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ponytail: lock to your Next.js origin in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(sessions_router)
app.include_router(users_router)
app.include_router(messages_router)
app.include_router(keys_router)
