import logging

from aiogram import Router
from aiogram.filters.command import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from app.backend import BackendClient, BackendError
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
    await message.answer(
        "القائمة الكاملة للمزودين: /providers",
        parse_mode="MarkdownV2",
    )

@router.message(Command("providers"))
async def cmd_providers(
    message: Message, backend: BackendClient
) -> None:
    telegram_id = str(message.from_user.id)
    try:
        result = await backend.list_my_providers(telegram_id)
        connections = result.get("connections", [])
    except BackendError:
        await message.answer("تعذر جلب قائمة المزودين. حاول مرة أخرى.")
        return

    if not connections:
        await message.answer("لا تملك أي اتصالات حالية بمزودي خدمة.")
        return

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=c["name"], callback_data=f"manage_provider:{c['id']}")]
            for c in connections
        ]
    )
    await message.answer("إليك قائمة بمزودي الخدمة المتصلين بحسابك:", reply_markup=keyboard)

    await message.answer(
        "القائمة الكاملة للمزودين: /providers",
        parse_mode="MarkdownV2",
    )
