"""
Test Quranic Tafsir format with PydanticAI and DeepSeek model.
This tests the actual model output against the schema.
"""

import asyncio
import os
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from search import (
    get_quran,
    get_tafsir,
    get_books_tafsir,
    setup_indexes,
    dense_search,
    sparse_search,
    hybrid_search,
    hybrid_search_weighted,
)


tools = [
    get_quran,
    get_tafsir,
    get_books_tafsir,
    dense_search,
    sparse_search,
    hybrid_search,
    hybrid_search_weighted,
]


class Citation(BaseModel):
    """Represents a detected citation from the Quran."""

    order: int = Field(..., description="Order of the citation in the response")
    surah_number: int = Field(..., description="Surah number (1-114)")
    surah_name: str = Field(..., description="Name of the Surah in Arabic")
    ayah_number: int = Field(..., description="Ayah number within the Surah")
    citation: str = Field(..., description="Citation text in format [surah:ayah]")
    verse_fragment_text: str = Field(
        ..., description="The Arabic text fragment being cited"
    )


class Resource(BaseModel):
    """Represents a resource link (Quran or Tafsir)."""

    source_type: str = Field(..., description="Type of source: 'quran' or 'tafsir'")
    title: str = Field(..., description="Human-readable title of the resource")
    url: str = Field(..., description="URL to the resource")
    surah_number: int = Field(..., description="Surah number")
    ayah_number: int = Field(..., description="Ayah number")
    tafsir_book: Optional[str] = Field(
        None,
        description="Tafsir book name if applicable (e.g., 'tabary', 'katheer', 'saadi', 'moyassar')",
    )


async def main():
    api_key = os.getenv("API_KEY", "sk-aa1af21fba8146cc9f689caeb906b532")
    if not api_key:
        print("Set DEEPSEEK_API_KEY to run this test.")
        return

    model = OpenAIChatModel(
        "deepseek-chat",
        provider=DeepSeekProvider(api_key=api_key),
    )

    agent = Agent(
        model=model,
        output_type=QuranTafsirResponse,
        system_prompt=(
            "أنت مفسر قرآني خبير. اكتب تفسيرًا بالعربية. "
            "اذكر مصادر التفسير: الميسر، السعدي، ابن كثير، الطبري. "
            "استشهد بالآيات بصيغة [سورة:آية] مثل [2:282]."
        ),
        tools=tools,
    )

    questions = [
        TafsirQuestion(
            question="ما تفسير آية الكرسي (2:255)؟", surah_number=2, ayah_number=255
        ),
        TafsirQuestion(
            question="ما تفسير آية الدين (2:282)؟", surah_number=2, ayah_number=282
        ),
    ]

    for idx, q in enumerate(questions, 1):
        print(f"\nQ{idx}: {q.question}")
        result = await agent.run(q.question)
        resp: QuranTafsirResponse = result.output
        print(f"  text: {len(resp.raw_response_text)} chars")
        print(f"  citations: {len(resp.detected_citations)}")
        print(f"  resources: {len(resp.resources)}")

        out = f"deepseek_test_{idx}.json"
        with open(out, "w", encoding="utf-8") as f:
            f.write(resp.model_dump_json(indent=2, ensure_ascii=False))
        print(f"  saved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
