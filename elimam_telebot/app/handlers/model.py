from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.backend import BackendClient, BackendError
from app.config import PROVIDERS, Settings
from app.handlers.start import models_keyboard, providers_keyboard, variants_keyboard
from app.states import ModelChange

router = Router()


@router.message(Command("model"))
async def cmd_model(message: Message, state: FSMContext, backend: BackendClient, settings: Settings) -> None:
    user = message.from_user
    if user is None:
        return
    telegram_id = str(user.id)

    try:
        existing = await backend.find_user(telegram_id, settings.user_lookup_path)
    except BackendError:
        await message.answer("تعذر الاتصال بالخادم. حاول مرة أخرى.")
        return

    if not existing:
        await message.answer("حسابك غير مسجل بعد. استخدم /start أولاً.")
        return

    if not PROVIDERS:
        await message.answer("لا توجد مزودات مضبوطة حالياً.")
        return

    await state.clear()
    await state.update_data(user_id=existing["id"])
    await state.set_state(ModelChange.provider)
    await message.answer("اختر مزود النموذج الجديد:", reply_markup=providers_keyboard())


@router.callback_query(ModelChange.provider, F.data.startswith("provider:"))
async def model_choose_provider(
    callback: CallbackQuery, state: FSMContext, backend: BackendClient, settings: Settings
) -> None:
    provider = callback.data.split(":", 1)[1]
    if provider not in PROVIDERS:
        await callback.answer("مزود غير صالح.", show_alert=True)
        return

    data = await state.get_data()
    user_id = data["user_id"]
    await state.update_data(provider=provider)

    # Only ask for a key if the user doesn't already have one for this
    # provider - never re-ask on every /model call.
    try:
        has_key = await backend.key_exists(
            path_template=settings.key_exists_path, user_id=user_id, provider=provider
        )
    except BackendError:
        await callback.message.answer("تعذر التحقق من مفتاح API. حاول مرة أخرى.")
        await callback.answer()
        return

    if has_key:
        await state.set_state(ModelChange.model)
        await callback.message.edit_text(f"المزود: {provider}\n\nاختر النموذج:")
        await callback.message.answer("اختر النموذج:", reply_markup=models_keyboard(provider))
    else:
        await state.set_state(ModelChange.api_key)
        await callback.message.edit_text(f"اخترت: {provider}\n\nأرسل الآن API key الخاص بك.")
    await callback.answer()


@router.message(ModelChange.api_key, F.text)
async def model_receive_api_key(
    message: Message, state: FSMContext, backend: BackendClient, settings: Settings
) -> None:
    api_key = message.text.strip()
    if not api_key:
        await message.answer("أرسل API key صالحاً.")
        return

    data = await state.get_data()
    user_id = data["user_id"]
    provider = data["provider"]

    try:
        await backend.store_key(path=settings.key_add_path, user_id=user_id, provider=provider, api_key=api_key)
    except BackendError:
        await message.answer("تعذر حفظ مفتاح API. حاول مرة أخرى.")
        return
    finally:
        # Clear the key from FSM immediately - it must never linger here.
        await state.update_data(api_key=None)

    try:
        await message.delete()
    except Exception:
        pass

    await state.set_state(ModelChange.model)
    await message.answer(f"المزود: {provider}\n\nاختر النموذج:", reply_markup=models_keyboard(provider))


@router.callback_query(ModelChange.model, F.data.startswith("model:"))
async def model_choose_model(callback: CallbackQuery, state: FSMContext) -> None:
    model = callback.data.split(":", 1)[1]
    data = await state.get_data()
    provider = data.get("provider")

    if not provider or model not in PROVIDERS.get(provider, {}).get("models", {}):
        await callback.answer("نموذج غير صالح.", show_alert=True)
        return

    await state.update_data(model=model)
    await state.set_state(ModelChange.variant)
    await callback.message.edit_text(f"المزود: {provider}\nالنموذج: {model}\n\nاختر مستوى التفكير:")
    await callback.message.answer("اختر variant:", reply_markup=variants_keyboard(provider, model))
    await callback.answer()


@router.callback_query(ModelChange.variant, F.data.startswith("variant:"))
async def model_choose_variant(
    callback: CallbackQuery, state: FSMContext, backend: BackendClient, settings: Settings
) -> None:
    variant = callback.data.split(":", 1)[1]
    data = await state.get_data()
    user_id = data.get("user_id")
    provider = data.get("provider")
    model = data.get("model")

    valid_variants = PROVIDERS.get(provider, {}).get("models", {}).get(model, {}).get(
        "variants", ("low", "high", "max")
    )
    if not user_id or not provider or not model or variant not in valid_variants:
        await state.clear()
        await callback.message.answer("انتهت العملية أو القيم غير صالحة. استخدم /model مرة أخرى.")
        await callback.answer()
        return

    try:
        session = await backend.get_active_session(
            path_template=settings.active_session_path, user_id=user_id
        )
        if session is None:
            await backend.create_session(
                path=settings.session_create_path,
                user_id=user_id,
                model_provider=provider,
                model_name=model,
                model_variant=variant,
            )
        else:
            # /model updates the CURRENT active session in place - it never
            # creates a new session.
            await backend.update_session_model(
                path_template=settings.session_model_update_path,
                session_id=session["id"],
                model_provider=provider,
                model_name=model,
                model_variant=variant,
            )
    except BackendError:
        await callback.message.answer("تعذر تحديث النموذج. حاول مرة أخرى.")
        await callback.answer()
        return

    await state.clear()
    await callback.message.edit_text(
        f"✅ تم تحديث النموذج.\n\nالمزود: {provider}\nالنموذج: {model}\nVariant: {variant}"
    )
    await callback.answer()
