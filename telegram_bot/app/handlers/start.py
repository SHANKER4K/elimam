from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.backend import BackendClient, BackendError
from app.config import Settings
from app.states import Setup

router = Router()


def providers_keyboard(providers: dict) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text=name, callback_data=f"provider:{name}")]
        for name in providers
    ]
    buttons.append([InlineKeyboardButton(text="➕ Add Custom", callback_data="provider:custom")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def models_keyboard(providers: dict, provider: str) -> InlineKeyboardMarkup:
    models = providers.get(provider, {}).get("models", {})
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=model, callback_data=f"model:{model}")] for model in models
        ]
    )


def variants_keyboard(providers: dict, provider: str, model: str) -> InlineKeyboardMarkup:
    variants = providers.get(provider, {}).get("models", {}).get(model, {}).get("variants") or (
        "low",
        "high",
        "max",
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
async def cmd_start(
    message: Message, state: FSMContext, backend: BackendClient, settings: Settings
) -> None:
    await state.clear()

    user = message.from_user
    if user is None:
        return
    telegram_id = str(user.id)

    try:
        existing = await backend.find_user(telegram_id, settings.user_lookup_path)
    except BackendError:
        await message.answer("Server connection failed. Try again.")
        return

    if existing:
        try:
            session = await backend.get_active_session(
                path_template=settings.active_session_path,
                user_id=existing["id"],
                telegram_id=telegram_id,
            )
        except BackendError:
            session = None

        if session is None:
            await message.answer(
                "مرحبا بك مجددا انت لم تختر النموذج الذي تريد العمل به. اكتب /model لكي تختار نموذج للعمل."
            )
            return

        await message.answer(
            f"مرحبا بك مجددا، {user.first_name}! المحادثة موجودة بالفعل. اكتب ما تريد السؤال او البحث عنه \n\n"
            "يمكنك بدأ محادثة جديدة باستخدام /reset او /model لتغيير النموذج"
        )
        return

    try:
        providers = await backend.fetch_providers(settings.providers_path)
    except BackendError:
        await message.answer("Could not fetch provider list from server. Try again.")
        return

    if not providers:
        await message.answer("انت لم تختر نموذجا بعد اكتب /model للاختيار")
        return

    await state.update_data(providers=providers)
    await state.set_state(Setup.provider)
    await message.answer(
        "مرحبًا! هيا بنا ننشئ حسابك. اختر مزود الخدمة:",
        reply_markup=providers_keyboard(providers),
    )


@router.callback_query(Setup.provider, F.data.startswith("provider:"))
async def choose_provider(callback: CallbackQuery, state: FSMContext) -> None:
    provider = callback.data.split(":", 1)[1]
    data = await state.get_data()
    providers = data.get("providers")

    if not providers or provider not in providers:
        await callback.answer("Invalid provider.", show_alert=True)
        return

    await state.update_data(provider=provider)
    await state.set_state(Setup.api_key)
    await callback.message.edit_text(f"""المزود المختار: {provider}

اضف الAPI key الخاص بك

لا تملك واحدا؟
سجل في هذه المنصات و خذ واحدا مجّانيّا
- https://opencode.ai/auth
- https://inference.dahl.global/#api-key
- https://portal.nousrese""")
    await callback.answer()


@router.message(Setup.api_key, F.text)
async def receive_api_key(message: Message, state: FSMContext) -> None:
    api_key = message.text.strip()
    if not api_key:
        await message.answer("ارسل API key صحيح.")
        return

    await state.update_data(api_key=api_key)
    try:
        await message.delete()
    except Exception:
        pass

    data = await state.get_data()
    provider = data["provider"]
    providers = data.get("providers", {})
    models = providers.get(provider, {}).get("models", {})

    if not models:
        await state.clear()
        await message.answer("لا توجد نماذج مهيأة لمزود الخدمة هذا.")
        return

    await state.set_state(Setup.model)
    await message.answer(
        f"مزود الخدمة: {provider}\n\nاختر نموذجًا:",
        reply_markup=models_keyboard(providers, provider),
    )


@router.callback_query(Setup.model, F.data.startswith("model:"))
async def choose_model(callback: CallbackQuery, state: FSMContext) -> None:
    model = callback.data.split(":", 1)[1]
    data = await state.get_data()
    provider = data.get("provider")
    providers = data.get("providers", {})

    if not provider or model not in providers.get(provider, {}).get("models", {}):
        await callback.answer("نموذج غير صالحة.", show_alert=True)
        return

    await state.update_data(model=model)
    await state.set_state(Setup.variant)
    await callback.message.edit_text(
        f"مزود الخدمة: {provider}\nالنموذج: {model}\n\nاختر مستوى التفكير:"
    )
    await callback.message.answer(
        "اختر الإضافة (Variant):", reply_markup=variants_keyboard(providers, provider, model)
    )
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
    providers = data.get("providers", {})

    valid_variants = providers.get(provider, {}).get("models", {}).get(model, {}).get(
        "variants"
    ) or ("low", "high", "max")
    if not provider or not model or not api_key or variant not in valid_variants:
        await state.clear()
        await callback.message.answer(
            "انتهت جلسة الإعداد أو البيانات غير صالحة. استخدم /start مجددًا."
        )
        await callback.answer()
        return

    username = user.username or ""
    display_name = (
        " ".join(part for part in [user.first_name, user.last_name] if part)
        or username
        or str(user.id)
    )
    telegram_id = str(user.id)

    try:
        user_result = await backend.create_user(
            path=settings.user_create_path,
            username=username,
            display_name=display_name,
            telegram_id=telegram_id,
        )
        user_id = user_result["id"]

        await backend.store_key(
            path=settings.key_add_path,
            user_id=user_id,
            provider=provider,
            api_key=api_key,
            telegram_id=telegram_id,
        )
        await state.update_data(api_key=None)

        await backend.create_session(
            path=settings.session_create_path,
            user_id=user_id,
            telegram_id=telegram_id,
            model_provider=provider,
            model_name=model,
            model_variant=variant,
        )
    except BackendError:
        await callback.message.answer("تعذر حفظ إعدادات الحساب. تحقق من بياناتك وحاول مرة أخرى.")
        await callback.answer()
        return

    await state.clear()
    await callback.message.edit_text(
        "تم إعداد الحساب بنجاح.\n\n"
        f"مزود الخدمة: {provider}\nالنموذج: {model}\nالإضافة: {variant}\n\n"
        "أرسل سؤالك الآن."
    )
    await callback.answer()
