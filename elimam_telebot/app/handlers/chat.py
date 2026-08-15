from __future__ import annotations

from aiogram import F, Router
from aiogram import Bot
from aiogram.types import Message, MessageEntity
from aiogram.utils.chat_action import ChatActionSender
from telegramify_markdown import telegramify
from telegramify_markdown.content import ContentType

from app.backend import BackendClient, BackendError
from app.config import Settings

router = Router()


@router.message(F.text)
async def handle_text(message: Message, bot: Bot, backend: BackendClient, settings: Settings) -> None:
    if not message.text or message.text.startswith("/"):
        return

    user = message.from_user
    if user is None:
        return

    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        try:
            text_parts: list[str] = []
            async for chunk in backend.stream_chat(
                path=settings.chat_path,
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

            results = await telegramify(text, max_message_length=4090)
        except BackendError:
            await message.answer("حدث خطأ في الاتصال بالخادم، حاول مرة أخرى.")
            return
        except Exception:
            await message.answer("حدث خطأ أثناء تجهيز الرد، حاول مرة أخرى.")
            return

    for item in results:
        if item.content_type == ContentType.TEXT:
            entities = [MessageEntity(**entity.to_dict()) for entity in item.entities]
            await message.answer(item.text, entities=entities or None)
        elif item.content_type == ContentType.FILE:
            await message.answer_document(
                document=item.file_data, filename=item.file_name, caption=item.caption_text or None
            )
        elif item.content_type == ContentType.PHOTO:
            await message.answer_photo(photo=item.file_data, caption=item.caption_text or None)
