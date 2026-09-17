from aiogram.fsm.state import State, StatesGroup


class Setup(StatesGroup):
    provider = State()
    api_key = State()
    model = State()
    variant = State()


class ModelChange(StatesGroup):
    provider = State()
    api_key = State()
    model = State()
    variant = State()
    custom_name = State()
    custom_url = State()
    custom_style = State()
    custom_api_key = State()
    update_api_key = State()
    custom_name = State()
    custom_url = State()
    custom_style = State()
    custom_key = State()
    update_key = State()
    custom_name = State()
    custom_url = State()
    custom_style = State()
