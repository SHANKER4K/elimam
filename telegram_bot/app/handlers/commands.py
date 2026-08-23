import logging

from aiogram import Router
from aiogram.filters.command import Command
from aiogram.types import Message
from telegramify_markdown import markdownify

router = Router()

logger = logging.getLogger(__name__)


@router.message(Command("help"))
async def test_format(message: Message) -> None:
    text = """
> Commands:

- /start: بدأ و تصطيب محادثة جديدة
- /model: اختيار و تغيير اعدادات النموذج
- /reset: بدأ محادثة جديدة
- /help: مساعدة هذا المشروع للنمو

    """

    mdv2 = markdownify(text)

    logger.info("MarkdownV2: %r", mdv2)

    await message.answer(
        mdv2,
        parse_mode="MarkdownV2",
    )
