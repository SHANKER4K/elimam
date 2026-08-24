from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from app.backend import BackendClient, BackendError
from app.config import Settings
from app.handlers.start import models_keyboard, providers_keyboard, variants_keyboard
from app.states import ModelChange

router = Router()


async def _load_providers_or_notify(
    backend: BackendClient, settings: Settings, message: Message, state: FSMContext
) -> dict | None:
    try:
        providers = await backend.fetch_providers(settings.providers_path)
    except BackendError:
        await message.answer("تعذر جلب قائمة مزودي الخدمة من الخادم. حاول مرة أخرى.")
        return None
    if providers:
        await state.update_data(providers=providers)
    return providers


@router.message(Command("model"))
async def cmd_model(
    message: Message, state: FSMContext, backend: BackendClient, settings: Settings
) -> None:
    user = message.from_user
    if user is None:
        return
    telegram_id = str(user.id)

    try:
        existing = await backend.find_user(telegram_id, settings.user_lookup_path)
    except BackendError:
        await message.answer("فشل الاتصال بالخادم. حاول مرة أخرى.")
        return

    if not existing:
        await message.answer("لم يتم العثور على حساب. استخدم الأمر /start أولاً.")
        return

    providers = await _load_providers_or_notify(backend, settings, message, state)
    if providers is None:
        return

    if not providers:
        await message.answer("لا يوجد مزودو خدمة مهيأون حاليًا.")
        return

    await state.clear()
    await state.update_data(user_id=existing["id"], providers=providers)
    await state.set_state(ModelChange.provider)
    await message.answer("اختر مزود خدمة جديدًا:", reply_markup=providers_keyboard(providers))


@router.callback_query(ModelChange.provider, F.data.startswith("provider:"))
async def model_choose_provider(
    callback: CallbackQuery, state: FSMContext, backend: BackendClient, settings: Settings
) -> None:
    provider = callback.data.split(":", 1)[1]
    data = await state.get_data()
    user_id = data["user_id"]
    providers = data.get("providers", {})

    if not providers or provider not in providers:
        await callback.answer("مزود خدمة غير صالح.", show_alert=True)
        return

    await state.update_data(provider=provider)

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
        await callback.message.edit_text(f"مزود الخدمة: {provider}\n\nاختر نموذجًا:")
        await callback.message.answer(
            "اختر نموذجًا:", reply_markup=models_keyboard(providers, provider)
        )
    else:
        await state.set_state(ModelChange.api_key)
        await callback.message.edit_text(f"""المزود المختار: {provider}

اضف الAPI key الخاص بك

لا تملك واحدا؟
سجل في هذه المنصات و خذ واحدا مجّانيّا
- https://opencode.ai/auth
- https://inference.dahl.global/#api-key
- https://portal.nousrese""")

    await callback.answer()


@router.message(ModelChange.api_key, F.text)
async def model_receive_api_key(
    message: Message, state: FSMContext, backend: BackendClient, settings: Settings
) -> None:
    api_key = message.text.strip()
    if not api_key:
        await message.answer("أرسل مفتاح API صالحًا.")
        return

    data = await state.get_data()
    user_id = data["user_id"]
    provider = data["provider"]

    try:
        await backend.store_key(
            path=settings.key_add_path, user_id=user_id, provider=provider, api_key=api_key
        )
    except BackendError:
        await message.answer("تعذر حفظ مفتاح API. حاول مرة أخرى.")
        return
    finally:
        await state.update_data(api_key=None)

    try:
        await message.delete()
    except Exception:
        pass

    providers = data.get("providers", {})
    await state.set_state(ModelChange.model)
    await message.answer(
        f"مزود الخدمة: {provider}\n\nاختر نموذجًا:",
        reply_markup=models_keyboard(providers, provider),
    )


@router.callback_query(ModelChange.model, F.data.startswith("model:"))
async def model_choose_model(callback: CallbackQuery, state: FSMContext) -> None:
    model = callback.data.split(":", 1)[1]
    data = await state.get_data()
    provider = data.get("provider")
    providers = data.get("providers", {})

    if not provider or model not in providers.get(provider, {}).get("models", {}):
        await callback.answer("نموذج غير صالح.", show_alert=True)
        return

    await state.update_data(model=model)
    await state.set_state(ModelChange.variant)
    await callback.message.edit_text(
        f"مزود الخدمة: {provider}\nالنموذج: {model}\n\nاختر مستوى التفكير:"
    )
    await callback.message.answer(
        "اختر درجة التفكير للنموذج:", reply_markup=variants_keyboard(providers, provider, model)
    )
    await callback.answer()


@router.callback_query(ModelChange.variant, F.data.startswith("variant:"))
async def model_choose_variant(
    callback: CallbackQuery, state: FSMContext, backend: BackendClient, settings: Settings
) -> None:
    variant = callback.data.split(":", 1)[1]
    data = await state.get_data()
    user_id = data["user_id"]
    provider = data.get("provider")
    model = data.get("model")
    providers = data.get("providers", {})

    valid_variants = providers.get(provider, {}).get("models", {}).get(model, {}).get(
        "variants"
    ) or ("low", "high", "max")
    if not user_id or not provider or not model or variant not in valid_variants:
        await state.clear()
        await callback.message.answer(
            "انتهت صلاحية العملية أو البيانات غير صالحة. استخدم الأمر /model مجددًا."
        )
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
        f"تم تحديث النموذج.\n\nمزود الخدمة: {provider}\nالنموذج: {model}\nالإضافة: {variant}"
    )
    await callback.answer()
