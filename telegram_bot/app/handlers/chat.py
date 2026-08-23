import logging
from aiogram import Bot, F, Router
from aiogram.enums import parse_mode
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender
from app.backend import BackendClient, BackendError
from app.config import Settings
from telegramify_markdown import markdownify, split_markdownv2


router = Router()

logger = logging.getLogger(__name__)

TELEGRAM_MAX_LENGTH = 4096


def split_text(text: str, max_length: int = TELEGRAM_MAX_LENGTH) -> list[str]:
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""

    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(candidate) <= max_length:
            current = candidate
        else:
            if current:
                chunks.append(current)

            # Paragraph itself is too large
            while len(paragraph) > max_length:
                # Prefer splitting on a newline or space
                split_at = paragraph.rfind("\n", 0, max_length)

                if split_at == -1:
                    split_at = paragraph.rfind(" ", 0, max_length)

                if split_at == -1:
                    split_at = max_length

                chunks.append(paragraph[:split_at])
                paragraph = paragraph[split_at:].lstrip()

            current = paragraph

    if current:
        chunks.append(current)

    return chunks


@router.message(F.text)
async def handle_text(
    message: Message,
    bot: Bot,
    backend: BackendClient,
    settings: Settings,
) -> None:
    if not message.text or message.text.startswith("/"):
        return

    user = message.from_user
    if user is None:
        return

    async with ChatActionSender.typing(
        bot=bot,
        chat_id=message.chat.id,
    ):
        try:
            text_parts: list[str] = []

            async for chunk in backend.stream_chat(
                path=settings.chat_path,
                sender=message,
                message=message.text,
                telegram_id=str(user.id),
                username=user.username,
                display_name=user.first_name,
            ):
                text_parts.append(chunk)

            text = "".join(text_parts)

            if not text:
                await message.answer("لم أستطع توليد رد.")
                return

            mdv2 = markdownify(text)

            for chunk in split_text(mdv2):
                await message.answer(
                    chunk,
                    parse_mode="MarkdownV2",
                )

        except BackendError:
            await message.answer("حدث خطأ في الاتصال بالخادم، حاول مرة أخرى.")

        except Exception:
            logger.exception("Error while processing text message")
            await message.answer("حدث خطأ أثناء تجهيز الرد، حاول مرة أخرى.")
