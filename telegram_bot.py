import json
import os
from collections.abc import AsyncIterator
from telegram import MessageEntity, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegramify_markdown import telegramify
from telegramify_markdown.content import ContentType


import httpx

from dotenv import load_dotenv  # Run `pip install python-dotenv` first

# Load variables from a .env file into os.environ
load_dotenv()


async def stream_chat(
    message: str,
    session_id: str,
    *,
    base_url: str = "",
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[str]:
    """Stream the agent's reply from the server's /chat SSE endpoint."""
    base_url = base_url or os.environ.get("ISLAM_SERVER_URL", "http://localhost:8000")
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


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    chat_id = update.message.chat_id
    await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        text = "".join(
            [
                chunk
                async for chunk in stream_chat(update.message.text, session_id(chat_id))
            ]
        )
        results = await telegramify(text, max_message_length=4090)
    except Exception:
        await update.message.reply_text("حدث خطأ في الاتصال بالخادم، حاول مرة أخرى.")
        return
    if not text or not results:
        await update.message.reply_text("لم أستطع توليد رد.")
        return
    for item in results:
        if item.content_type == ContentType.TEXT:
            entities = [MessageEntity(**e.to_dict()) for e in item.entities]
            await update.message.reply_text(item.text, entities=entities or None)
        elif item.content_type == ContentType.FILE:
            await update.message.reply_document(
                document=item.file_data,
                filename=item.file_name,
                caption=item.caption_text or None,
            )
        elif item.content_type == ContentType.PHOTO:
            await update.message.reply_photo(
                photo=item.file_data,
                caption=item.caption_text or None,
            )


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "السلام عليكم، أنا مساعدك في الأسئلة الإسلامية.\n"
        "اسألني عن القرآن والحديث والتفسير والعقيدة، وسأجيب مع ذكر المصادر.\n"
        "استخدم /reset لبدء محادثة جديدة."
    )


async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    sessions[chat_id] = sessions.get(chat_id, 0) + 1
    await update.message.reply_text("تم تصفير المحادثة، ابدأ سؤالاً جديداً.")


def main() -> None:
    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling()


if __name__ == "__main__":
    main()
