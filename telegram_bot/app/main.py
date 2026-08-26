from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from app.backend import BackendClient
from app.config import Settings
from app.handlers import chat, model, reset, start
from app.handlers.help import router as test_format_router


async def main() -> None:
    # ponytail: plain text is enough here; the JSON pipeline lives in the backend.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings.from_env()

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(storage=MemoryStorage())

    backend = BackendClient(
        base_url=settings.backend_url,
        bot_shared_secret=settings.bot_shared_secret,
        timeout=settings.request_timeout,
    )

    dp.include_router(start.router)
    dp.include_router(model.router)
    dp.include_router(reset.router)
    dp.include_router(test_format_router)
    dp.include_router(chat.router)

    dp["settings"] = settings
    dp["backend"] = backend

    try:
        await dp.start_polling(bot)
    finally:
        await backend.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
