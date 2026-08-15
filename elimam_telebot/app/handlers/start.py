from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.backend import BackendClient, BackendError
from app.config import PROVIDERS, Settings
from app.states import Setup

router = Router()


def providers_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=name, callback_data=f"provider:{name}")]
            for name in PROVIDERS
        ]
    )


def models_keyboard(provider: str) -> InlineKeyboardMarkup:
    models = PROVIDERS.get(provider, {}).get("models", {})
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=model, callback_data=f"model:{model}")]
            for model in models
        ]
    )


def variants_keyboard(provider: str, model: str) -> InlineKeyboardMarkup:
    variants = (
        PROVIDERS.get(provider, {}).get("models", {}).get(model, {}).get("variants")
        or ("low", "high", "max")
    )
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=variant.title(), callback_data=f"variant:{variant}")
                for variant in variants
            ]
        ]
    )


@router.message(Command("start"), StateFilter("*"))
async def cmd_start(message: Message, state: FSMContext, backend: BackendClient, settings: Settings) -> None:
    await state.clear()

    user = message.from_user
    if user is None:
        return
    telegram_id = str(user.id)

    try:
        existing = await backend.find_user(telegram_id, settings.user_lookup_path)
    except BackendError:
        await message.answer("تعذر الاتصال بالخادم. حاول مرة أخرى.")
        return

    if existing:
        # Existing user: do NOT run the wizard again. Ensure an active
        # session exists (backend self-heals this on /chat too, but we
        # check here so /start always feels consistent).
        try:
            session = await backend.get_active_session(
                path_template=settings.active_session_path, user_id=existing["id"]
            )
        except BackendError:
            session = None

        if session is None:
            await message.answer(
                "أهلاً بك مجدداً! 👋\nلم يتم إعداد نموذج بعد. استخدم /model لاختيار مزود ونموذج."
            )
            return

        await message.answer(
            f"أهلاً بك مجدداً، {user.first_name}! 👋\n"
            "جلسة حسابك الحالية جاهزة. أرسل سؤالك مباشرة.\n\n"
            "استخدم /reset لبدء محادثة جديدة أو /model لتغيير النموذج."
        )
        return

    if not PROVIDERS:
        await message.answer("لم يتم إعداد أي مزود نماذج بعد. أضف المزودات إلى PROVIDERS في config.py.")
        return

    await state.set_state(Setup.provider)
    await message.answer(
        "السلام عليكم! 👋\nلنجهز حسابك أولاً. اختر مزود النموذج:",
        reply_markup=providers_keyboard(),
    )


@router.callback_query(Setup.provider, F.data.startswith("provider:"))
async def choose_provider(callback: CallbackQuery, state: FSMContext) -> None:
    provider = callback.data.split(":", 1)[1]
    if provider not in PROVIDERS:
        await callback.answer("مزود غير صالح.", show_alert=True)
        return

    await state.update_data(provider=provider)
    await state.set_state(Setup.api_key)
    await callback.message.edit_text(f"اخترت: {provider}\n\nأرسل الآن API key الخاص بك.")
    await callback.answer()


@router.message(Setup.api_key, F.text)
async def receive_api_key(message: Message, state: FSMContext) -> None:
    api_key = message.text.strip()
    if not api_key:
        await message.answer("أرسل API key صالحاً.")
        return

    # Held in FSM only transiently, cleared as soon as it is sent to the
    # backend for storage (see choose_variant below).
    await state.update_data(api_key=api_key)
    try:
        await message.delete()
    except Exception:
        pass

    data = await state.get_data()
    provider = data["provider"]
    models = PROVIDERS[provider].get("models", {})

    if not models:
        await state.clear()
        await message.answer("لا توجد نماذج مضبوطة لهذا المزود.")
        return

    await state.set_state(Setup.model)
    await message.answer(f"المزود: {provider}\n\nاختر النموذج:", reply_markup=models_keyboard(provider))


@router.callback_query(Setup.model, F.data.startswith("model:"))
async def choose_model(callback: CallbackQuery, state: FSMContext) -> None:
    model = callback.data.split(":", 1)[1]
    data = await state.get_data()
    provider = data.get("provider")

    if not provider or model not in PROVIDERS.get(provider, {}).get("models", {}):
        await callback.answer("نموذج غير صالح.", show_alert=True)
        return

    await state.update_data(model=model)
    await state.set_state(Setup.variant)
    await callback.message.edit_text(f"المزود: {provider}\nالنموذج: {model}\n\nاختر مستوى التفكير:")
    await callback.message.answer("اختر variant:", reply_markup=variants_keyboard(provider, model))
    await callback.answer()


@router.callback_query(Setup.variant, F.data.startswith("variant:"))
async def choose_variant(
    callback: CallbackQuery, state: FSMContext, backend: BackendClient, settings: Settings
) -> None:
    variant = callback.data.split(":", 1)[1]
    user = callback.from_user
    data = await state.get_data()
    provider = data.get("provider")
    model = data.get("model")
    api_key = data.get("api_key")

    valid_variants = PROVIDERS.get(provider, {}).get("models", {}).get(model, {}).get(
        "variants", ("low", "high", "max")
    )
    if not provider or not model or not api_key or variant not in valid_variants:
        await state.clear()
        await callback.message.answer("انتهت عملية الإعداد أو القيم غير صالحة. استخدم /start مرة أخرى.")
        await callback.answer()
        return

    username = user.username or ""
    display_name = " ".join(part for part in [user.first_name, user.last_name] if part) or username or str(user.id)
    telegram_id = str(user.id)

    try:
        # 1. create user, 2. store api key, 3. create active session -
        # this whole flow is atomic on the backend side per-call, and the
        # bot clears the API key from FSM state immediately after use.
        user_result = await backend.create_user(
            path=settings.user_create_path,
            username=username,
            display_name=display_name,
            telegram_id=telegram_id,
        )
        user_id = user_result["id"]

        await backend.store_key(
            path=settings.key_add_path, user_id=user_id, provider=provider, api_key=api_key
        )
        await state.update_data(api_key=None)

        await backend.create_session(
            path=settings.session_create_path,
            user_id=user_id,
            model_provider=provider,
            model_name=model,
            model_variant=variant,
        )
    except BackendError:
        await callback.message.answer("تعذر حفظ إعدادات الحساب في الخادم. تحقق من البيانات وحاول مرة أخرى.")
        await callback.answer()
        return

    await state.clear()
    await callback.message.edit_text(
        "✅ تم إعداد حسابك بنجاح.\n\n"
        f"المزود: {provider}\nالنموذج: {model}\nVariant: {variant}\n\n"
        "أرسل سؤالك الآن."
    )
    await callback.answer()
