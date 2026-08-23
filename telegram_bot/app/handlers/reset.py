from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.backend import BackendClient, BackendError
from app.config import Settings

router = Router()


@router.message(Command("reset"))
async def cmd_reset(message: Message, backend: BackendClient, settings: Settings) -> None:
    user = message.from_user
    if user is None:
        return

    try:
        user_result = await backend.find_user(str(user.id), settings.user_lookup_path)
        if not user_result:
            await message.answer("حسابك غير مسجل بعد. استخدم /start أولاً.")
            return

        # Atomic on the backend: deactivate current active session + create
        # a new one carrying over the same model configuration, in one call.
        await backend.reset_session(path_template=settings.session_reset_path, user_id=user_result["id"])
    except BackendError:
        await message.answer("حدث خطأ أثناء إعادة ضبط الجلسة.")
        return

    await message.answer("تم إغلاق الجلسة الحالية وإنشاء جلسة جديدة. ابدأ بسؤالك.")
