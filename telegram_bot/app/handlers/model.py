from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message, InlineKeyboardButton, InlineKeyboardMarkup

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

    if provider == "custom":
        await state.set_state(ModelChange.custom_name)
        await callback.message.edit_text("أدخل اسمًا مخصصًا لمزود الخدمة:")
        await callback.answer()
        return

    if not providers or provider not in providers:
        await callback.answer("مزود خدمة غير صالح.", show_alert=True)
        return

    await state.update_data(provider=provider)

    try:
        has_key = await backend.key_exists(
            path_template=settings.key_exists_path,
            user_id=user_id,
            provider=provider,
            telegram_id=str(callback.from_user.id),
        )
    except BackendError:
        await callback.message.answer("تعذر التحقق من مفتاح API. حاول مرة أخرى.")
        await callback.answer()
        return

    if has_key:
        await state.set_state(ModelChange.model)
        await callback.message.edit_text(
            f"مزود الخدمة: {provider}\n\nلديك مفتاح API مسجل بالفعل. هل تريد تحديثه؟",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="تحديث المفتاح", callback_data="update_key")],
                    [InlineKeyboardButton(text="تجاوز والذهاب للنماذج", callback_data="proceed_to_models")],
                ]
            )
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
    message: Message, state: FSMContext, backend: BackendClient
) -> None:
    """First key for a builtin provider: connect it, which stores the key and
    seeds/discovers its models in the same call."""
    api_key = message.text.strip()
    if not api_key:
        await message.answer("أرسل مفتاح API صالحًا.")
        return

    data = await state.get_data()
    provider = data.get("provider")
    providers = data.get("providers", {})
    telegram_id = str(message.from_user.id)

    if not provider:
        await state.clear()
        await message.answer("انتهت صلاحية العملية. استخدم الأمر /model مجددًا.")
        return

    try:
        await message.delete()
    except Exception:
        pass

    try:
        catalog = (await backend.list_my_providers(telegram_id)).get("catalog", [])
        provider_row = next((p for p in catalog if p["slug"] == provider), None)
        if provider_row is None:
            raise BackendError(f"مزود خدمة غير معروف: {provider}")
        await backend.connect_provider(
            telegram_id, provider_row["id"], api_key=api_key
        )
    except BackendError as exc:
        await message.answer(f"تعذر حفظ مفتاح API: {exc}")
        return

    await state.set_state(ModelChange.model)
    await message.answer(
        f"مزود الخدمة: {provider}\n\nاختر نموذجًا:",
        reply_markup=models_keyboard(providers, provider),
    )


@router.callback_query(ModelChange.model, F.data == "update_key")
async def model_request_key_update(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(ModelChange.update_api_key)
    await callback.message.edit_text("أدخل مفتاح API الجديد:")
    await callback.answer()


@router.callback_query(ModelChange.model, F.data == "proceed_to_models")
async def model_proceed_to_models(callback: CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()
    providers = data.get("providers", {})
    provider = data.get("provider")
    
    await callback.message.edit_text(f"مزود الخدمة: {provider}\n\nاختر نموذجًا:")
    await callback.message.answer(
        "اختر نموذجًا:", reply_markup=models_keyboard(providers, provider)
    )
    await callback.answer()


@router.message(ModelChange.custom_name, F.text)
async def model_custom_name(
    message: Message, state: FSMContext
) -> None:
    name = message.text.strip()
    if not name:
        await message.answer("أرسل اسمًا صالحًا.")
        return

    await state.update_data(custom_name=name)
    await state.set_state(ModelChange.custom_url)
    await message.answer("أدخل عنوان URL لمزود الخدمة المخصص:")

@router.message(ModelChange.custom_url, F.text)
async def model_custom_url(
    message: Message, state: FSMContext
) -> None:
    url = message.text.strip()
    if not url:
        await message.answer("أرسل عنوان URL صالحًا.")
        return

    await state.update_data(custom_url=url)
    await state.set_state(ModelChange.custom_style)
    await message.answer(
        "اختر نمط واجهة API:",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="OpenAI-compatible",
                        callback_data="api_style:openai_compatible"
                    ),
                    InlineKeyboardButton(
                        text="Anthropic-compatible",
                        callback_data="api_style:anthropic_compatible"
                    )
                ]
            ]
        )
    )

@router.callback_query(ModelChange.custom_style, F.data.startswith("api_style:"))
async def model_custom_style(
    callback: CallbackQuery, state: FSMContext
) -> None:
    api_style = callback.data.split(":", 1)[1]
    await state.update_data(custom_style=api_style)
    await state.set_state(ModelChange.custom_api_key)
    await callback.message.edit_text("أدخل مفتاح API (اختياري):")
    await callback.answer()

@router.message(ModelChange.custom_api_key, F.text)
async def model_custom_api_key(
    message: Message, state: FSMContext, backend: BackendClient
) -> None:
    api_key = message.text.strip()
    data = await state.get_data()
    name = data.get("custom_name")
    url = data.get("custom_url")
    api_style = data.get("custom_style")

    if not name or not url or not api_style:
        await state.clear()
        await message.answer("انتهت جلسة الإعداد. استخدم الأمر /model مجددًا.")
        return

    try:
        await message.delete()
    except Exception:
        pass

    try:
        connection = await backend.add_custom_provider(
            telegram_id=str(message.from_user.id),
            name=name,
            base_url=url,
            api_style=api_style,
            api_key=api_key or None,
        )
    except BackendError as exc:
        await message.answer(f"تعذر إضافة مزود مخصص: {str(exc)}")
        return

    await state.clear()
    await message.answer(
        f"تم إضافة مزود مخصص بنجاح: {name}\n\n" +
        f"عنوان: {url}\n" +
        f"نمط واجهة: {api_style}\n" +
        (f"مفتاح API: {api_key[:4]}..." if api_key else "لا يوجد مفتاح API") +
        "\n\nاختر نموذجًا من القائمة التالية:",
        reply_markup=models_keyboard({"custom": {"models": connection.get("models", {})}}, "custom"),
    )

@router.message(ModelChange.update_api_key, F.text)
async def model_update_api_key(
    message: Message, state: FSMContext, backend: BackendClient
) -> None:
    api_key = message.text.strip()
    if not api_key:
        await message.answer("أرسل مفتاح API صالحًا.")
        return

    data = await state.get_data()
    provider = data.get("provider")
    telegram_id = str(message.from_user.id)

    if not provider:
        await state.clear()
        await message.answer("انتهت جلسة التحديث. استخدم الأمر /model مجددًا.")
        return

    try:
        await message.delete()
    except Exception:
        pass

    try:
        # Find connection id for the provider
        my_providers = await backend.list_my_providers(telegram_id)
        connections = my_providers.get("connections", [])
        connection = next((c for c in connections if c["slug"] == provider), None)
        
        if not connection:
            raise BackendError("لم يتم العثور على اتصال لهذا المزود.")

        await backend.update_provider(
            telegram_id=telegram_id,
            connection_id=connection["id"],
            body={"api_key": api_key},
        )
    except BackendError as exc:
        await message.answer(f"تعذر تحديث مفتاح API: {str(exc)}")
        return

    await state.clear()
    await message.answer(
        f"تم تحديث مفتاح API لمزود الخدمة: {provider} بنجاح.\n\n"
        "يمكنك الآن العودة واختيار النموذج.",
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
    telegram_id = str(callback.from_user.id)
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
            path_template=settings.active_session_path,
            user_id=user_id,
            telegram_id=telegram_id,
        )
        if session is None:
            await backend.create_session(
                path=settings.session_create_path,
                user_id=user_id,
                telegram_id=telegram_id,
                model_provider=provider,
                model_name=model,
                model_variant=variant,
            )
        else:
            await backend.update_session_model(
                path_template=settings.session_model_update_path,
                session_id=session["id"],
                telegram_id=telegram_id,
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

@router.callback_query(F.data.startswith("manage_provider:"))
async def manage_provider_menu(
    callback: CallbackQuery, backend: BackendClient
) -> None:
    cid = callback.data.split(":", 1)[1]
    
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 مزامنة النماذج", callback_data=f"manage_provider:sync:{cid}")],
            [InlineKeyboardButton(text="🗑️ حذف الاتصال", callback_data=f"manage_provider:delete:{cid}")],
            [InlineKeyboardButton(text="⬅️ عودة", callback_data="back_to_providers")],
        ]
    )
    
    await callback.message.edit_text(
        "إدارة الاتصال بمزود الخدمة:\n\nاختر إجراءً:",
        reply_markup=keyboard
    )
    await callback.answer()

@router.callback_query(F.data.startswith("manage_provider:sync:"))
async def manage_provider_sync(
    callback: CallbackQuery, backend: BackendClient
) -> None:
    cid = callback.data.split(":", 2)[2]
    try:
        await backend.sync_provider(str(callback.from_user.id), cid)
        await callback.answer("تمت مزامنة النماذج بنجاح ✅", show_alert=True)
    except BackendError as exc:
        await callback.answer(f"فشلت المزامنة: {str(exc)}", show_alert=True)

@router.callback_query(F.data.startswith("manage_provider:delete:"))
async def manage_provider_delete(
    callback: CallbackQuery, backend: BackendClient
) -> None:
    cid = callback.data.split(":", 2)[2]
    try:
        await backend.delete_provider(str(callback.from_user.id), cid)
        await callback.message.edit_text("تم حذف الاتصال بنجاح. يمكنك إضافة مزود جديد باستخدام /model.")
        await callback.answer()
    except BackendError as exc:
        await callback.answer(f"فشل الحذف: {str(exc)}", show_alert=True)

@router.callback_query(F.data == "back_to_providers")
async def manage_provider_back(
    callback: CallbackQuery, backend: BackendClient
) -> None:
    telegram_id = str(callback.from_user.id)
    try:
        result = await backend.list_my_providers(telegram_id)
        connections = result.get("connections", [])
    except BackendError:
        await callback.answer("تعذر جلب القائمة.", show_alert=True)
        return

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=c["name"], callback_data=f"manage_provider:{c['id']}")]
            for c in connections
        ]
    )
    await callback.message.edit_text("إليك قائمة بمزودي الخدمة المتصلين بحسابك:", reply_markup=keyboard)
    await callback.answer()
