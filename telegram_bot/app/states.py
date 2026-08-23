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
