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
    # Custom provider wizard: name -> url -> style -> key.
    custom_name = State()
    custom_url = State()
    custom_style = State()
    custom_api_key = State()
    # Replacing the key on an existing connection.
    update_api_key = State()
