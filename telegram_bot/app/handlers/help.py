import logging

from aiogram import Router
from aiogram.filters.command import Command
from aiogram.types import Message
from telegramify_markdown import markdownify

router = Router()

logger = logging.getLogger(__name__)


@router.message(Command("help"))
async def test_format(message: Message) -> None:
    text = """> Help me continue on this project

**Paypal:** `medjahdiismail1998@gmail.com`

**Binance:** `582355344`
• BSC: `0x95f983d483c9aec9ffc198491fff598317abe8e8`
• TRX: `TEfeRsKniPqWqjNgi3mZEzkjysmoHoBthm`
• ETH: `0x95f983d483c9aec9ffc198491fff598317abe8e8`
• TON: `UQDezY2u0Ujg3nbckKy_1JX71c2CB4zP_7oiMlyQcwzPv2m-`


**RedotPay:** `199250977`
• BSC: `0x76e4668FaBaf27Ad69bEa1De71e73C09391Ace2E`
• TRX: `TSWmeYgCwEYHgbHtZmvome6Q9HvdcPTWSS`
• ETH: `0x76e4668FaBaf27Ad69bEa1De71e73C09391Ace2E`

    """

    mdv2 = markdownify(text)

    await message.answer(
        mdv2,
        parse_mode="MarkdownV2",
    )
