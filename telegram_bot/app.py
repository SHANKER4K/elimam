import json
import os
from collections.abc import AsyncIterator

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, MessageEntity
from aiogram.enums import ChatAction
from aiogram.utils.chat_action import ChatActionSender
from telegramify_markdown import telegramify
from telegramify_markdown.content import ContentType

from dotenv import load_dotenv

load_dotenv()


async def stream_chat(
    message: str,
    session_id: str,
    *,
    base_url: str = "",
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[str]:
    """Stream the agent's reply from the server's /chat SSE endpoint."""
    base_url = base_url or os.environ.get("BACKEND_URL", "http://localhost:8000")
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=60.0)
    message = f"You are in Telegram so ignore the printing formats of quran and hadith use them in block quotes instead \n{message}"
    try:
        async with client.stream(
            "POST",
            f"{base_url}/chat",
            json={"message": message, "session_id": session_id},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                if "tool_call_id" in payload:  # tool / tool_result events
                    continue
                text = payload.get("text")
                if text:
                    yield text
    finally:
        if own_client:
            await client.aclose()


sessions: dict[int, int] = {}  # chat_id -> generation, bumped by /reset


def session_id(chat_id: int) -> str:
    return f"{chat_id}:{sessions.get(chat_id, 0)}"


async def handle_text(message: Message, bot: Bot) -> None:
    if not message.text:
        return
    chat_id = message.chat.id
    async with ChatActionSender(bot=bot, chat_id=chat_id):
        try:
            text = "".join(
                chunk
                async for chunk in stream_chat(message.text, session_id(chat_id))
            )
            results = await telegramify(text, max_message_length=4090)
        except Exception:
            await message.answer("حدث خطأ في الاتصال بالخادم، حاول مرة أخرى.")
            return
    if not text or not results:
        await message.answer("لم أستطع توليد رد.")
        return
    for item in results:
        if item.content_type == ContentType.TEXT:
            entities = [MessageEntity(**e.to_dict()) for e in item.entities]
            await message.answer(item.text, entities=entities or None)
        elif item.content_type == ContentType.FILE:
            await message.answer_document(
                document=item.file_data,
                filename=item.file_name,
                caption=item.caption_text or None,
            )
        elif item.content_type == ContentType.PHOTO:
            await message.answer_photo(
                photo=item.file_data,
                caption=item.caption_text or None,
            )


async def cmd_start(message: Message) -> None:
    await message.answer(
        "السلام عليكم، أنا مساعدك في الأسئلة الإسلامية.\n"
        "اسألني عن القرآن والحديث والتفسير والعقيدة، وسأجيب مع ذكر المصادر.\n"
        "استخدم /reset لبدء محادثة جديدة."
    )


async def cmd_reset(message: Message) -> None:
    chat_id = message.chat.id
    sessions[chat_id] = sessions.get(chat_id, 0) + 1
    await message.answer("تم تصفير المحادثة، ابدأ سؤالاً جديداً.")


def main() -> None:
    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    dp = Dispatcher()
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_reset, Command("reset"))
    dp.message.register(handle_text, F.text)
    dp.run_polling(bot)


if __name__ == "__main__":
    main()