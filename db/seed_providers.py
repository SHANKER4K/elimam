import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
)

from db.connection import get_conn

PROVIDERS_TO_SEED = [
    {
        "slug": "openai",
        "name": "OpenAI",
        "api_style": "openai_compatible",
        "default_base_url": "https://api.openai.com/v1",
        "requires_key": True,
        "default_variants": ["none", "low", "medium", "high", "xhigh", "max"],
        "models": {
            "gpt-5.6": {"variants": ["none", "low", "medium", "high", "xhigh", "max"]},
            "gpt-5.6-sol": {
                "variants": ["none", "low", "medium", "high", "xhigh", "max"]
            },
            "gpt-5.6-terra": {
                "variants": ["none", "low", "medium", "high", "xhigh", "max"]
            },
            "gpt-5.6-luna": {
                "variants": ["none", "low", "medium", "high", "xhigh", "max"]
            },
        },
    },
    {
        "slug": "anthropic",
        "name": "Anthropic",
        "api_style": "anthropic_compatible",
        "default_base_url": "https://api.anthropic.com/v1",
        "requires_key": True,
        "default_variants": ["low", "medium", "high", "max"],
        "models": {
            "claude-opus-5": {"variants": ["low", "medium", "high", "max"]},
            "claude-sonnet-5": {"variants": ["low", "medium", "high", "max"]},
            "claude-fable-5": {"variants": ["low", "medium", "high", "max"]},
            "claude-mythos-5": {"variants": ["low", "medium", "high", "max"]},
            "claude-opus-4-8": {"variants": ["low", "medium", "high", "max"]},
            "claude-opus-4-7": {"variants": ["low", "medium", "high", "max"]},
            "claude-opus-4-6": {"variants": ["low", "medium", "high", "max"]},
            "claude-sonnet-4-6": {"variants": ["low", "medium", "high", "max"]},
            "claude-haiku-4-5-20251001": {"variants": ["low", "medium", "high", "max"]},
        },
    },
    {
        "slug": "deepseek",
        "name": "DeepSeek",
        "api_style": "openai_compatible",
        "default_base_url": "https://api.deepseek.com/v1",
        "requires_key": True,
        "default_variants": ["none", "low", "high", "max"],
        "models": {
            "deepseek-flash": {"variants": ["none", "low", "high", "max"]},
            "deepseek-v4-pro": {"variants": ["none", "low", "high", "max"]},
        },
    },
    {
        "slug": "openrouter",
        "name": "OpenRouter",
        "api_style": "openai_compatible",
        "default_base_url": "https://openrouter.ai/api/v1",
        "requires_key": True,
        "default_variants": ["low", "medium", "high", "xhigh", "max"],
        "models": {
            "openrouter/auto": {"variants": ["low", "medium", "high", "xhigh", "max"]},
            "openrouter/free": {"variants": ["none"]},
        },
    },
    {
        "slug": "nvidia",
        "name": "NVIDIA NIM",
        "api_style": "openai_compatible",
        "default_base_url": "https://integrate.api.nvidia.com/v1",
        "requires_key": True,
        "default_variants": ["low", "high"],
        "models": {
            "deepseek-ai/deepseek-r1": {"variants": ["low", "high"]},
            "meta/llama-3.3-70b-instruct": {"variants": ["low", "high"]},
            "google/gemini-2.0-flash-exp": {"variants": ["low", "high"]},
        },
    },
]


def seed_providers():
    print("Seeding providers into database...")
    with get_conn() as conn:
        with conn.cursor() as cur:
            for p in PROVIDERS_TO_SEED:
                cur.execute(
                    """
                    INSERT INTO providers
                        (slug, name, api_style, default_base_url, requires_key, default_variants, models)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (slug) DO UPDATE
                        SET name = EXCLUDED.name,
                            api_style = EXCLUDED.api_style,
                            default_base_url = EXCLUDED.default_base_url,
                            requires_key = EXCLUDED.requires_key,
                            default_variants = EXCLUDED.default_variants,
                            models = EXCLUDED.models
                    """,
                    (
                        p["slug"],
                        p["name"],
                        p["api_style"],
                        p["default_base_url"],
                        p["requires_key"],
                        p["default_variants"],
                        json.dumps(p["models"]),
                    ),
                )
                print(f"Upserted provider: {p['name']} ({p['slug']})")
        conn.commit()
    print("Provider seeding completed successfully.")


if __name__ == "__main__":
    seed_providers()
