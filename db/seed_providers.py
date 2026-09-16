import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from db.connection import get_conn

PROVIDERS_TO_SEED = [
    {
        "slug": "openai",
        "name": "OpenAI",
        "api_style": "openai_compatible",
        "default_base_url": "https://api.openai.com/v1",
        "requires_key": True,
        "default_variants": ["low", "medium", "high"],
        "models": {
            "gpt-4o": {"variants": ["low", "medium", "high"]},
            "gpt-4o-mini": {"variants": ["low", "medium", "high"]},
            "o1": {"variants": ["low", "medium", "high"]},
            "o3-mini": {"variants": ["low", "medium", "high"]},
        },
    },
    {
        "slug": "anthropic",
        "name": "Anthropic",
        "api_style": "anthropic_compatible",
        "default_base_url": "https://api.anthropic.com/v1",
        "requires_key": True,
        "default_variants": ["low", "high"],
        "models": {
            "claude-3-5-sonnet-20241022": {"variants": ["low", "high"]},
            "claude-3-5-haiku-20241022": {"variants": ["low", "high"]},
            "claude-3-opus-20240229": {"variants": ["low", "high"]},
        },
    },
    {
        "slug": "deepseek",
        "name": "DeepSeek",
        "api_style": "openai_compatible",
        "default_base_url": "https://api.deepseek.com/v1",
        "requires_key": True,
        "default_variants": ["low", "high"],
        "models": {
            "deepseek-chat": {"variants": ["low", "high"]},
            "deepseek-reasoner": {"variants": ["low", "high"]},
        },
    },
    {
        "slug": "openrouter",
        "name": "OpenRouter",
        "api_style": "openai_compatible",
        "default_base_url": "https://openrouter.ai/api/v1",
        "requires_key": True,
        "default_variants": ["low", "high"],
        "models": {
            "anthropic/claude-3.5-sonnet": {"variants": ["low", "high"]},
            "openai/gpt-4o": {"variants": ["low", "high"]},
            "deepseek/deepseek-chat": {"variants": ["low", "high"]},
            "meta-llama/llama-3.3-70b-instruct": {"variants": ["low", "high"]},
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
