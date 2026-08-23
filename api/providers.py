from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Header

router = APIRouter(prefix="/providers", tags=["Providers"])


BOT_SHARED_SECRET = os.environ.get("BOT_SHARED_SECRET", "")


class ProviderConfigError(RuntimeError):
    pass


def _config_path() -> Path:
    return Path(
        os.getenv(
            "PROVIDERS_CONFIG_PATH",
            "/app/config/providers.yaml",
        )
    )


def load_providers(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    config_path = Path(path) if path else _config_path()

    if not config_path.is_file():
        raise ProviderConfigError(
            f"Provider configuration file does not exist: {config_path}"
        )

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ProviderConfigError(
            f"Invalid YAML in provider configuration: {config_path}"
        ) from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("providers"), dict):
        raise ProviderConfigError(
            "Configuration must contain a top-level 'providers' mapping"
        )

    providers = raw["providers"]

    for provider_name, provider_config in providers.items():
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ProviderConfigError("Provider names must be non-empty strings")

        if not isinstance(provider_config, dict):
            raise ProviderConfigError(
                f"Provider '{provider_name}' must be a mapping"
            )

        url = provider_config.get("url")
        models = provider_config.get("models")

        if not isinstance(url, str) or not url.strip():
            raise ProviderConfigError(
                f"Provider '{provider_name}' must define a non-empty url"
            )

        if not isinstance(models, dict) or not models:
            raise ProviderConfigError(
                f"Provider '{provider_name}' must define at least one model"
            )

        for model_name, model_config in models.items():
            if not isinstance(model_name, str) or not model_name.strip():
                raise ProviderConfigError(
                    f"Provider '{provider_name}' contains an empty model name"
                )

            if not isinstance(model_config, dict):
                raise ProviderConfigError(
                    f"Model '{provider_name}/{model_name}' must be a mapping"
                )

            variants = model_config.get("variants")

            if not isinstance(variants, list) or not variants:
                raise ProviderConfigError(
                    f"Model '{provider_name}/{model_name}' must define variants"
                )

            if any(
                not isinstance(v, str) or not v.strip()
                for v in variants
            ):
                raise ProviderConfigError(
                    f"Variants for '{provider_name}/{model_name}' "
                    "must be non-empty strings"
                )

    return providers


def resolve_model_config(
    provider: str,
    model: str,
    variant: str,
) -> dict[str, str]:
    provider_config = PROVIDERS.get(provider)

    if provider_config is None:
        raise ValueError(f"Unknown model provider: {provider}")

    url = provider_config.get("url")
    models = provider_config.get("models", {})
    model_config = models.get(model)

    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"Provider '{provider}' has no valid URL")

    if not isinstance(model_config, dict):
        raise ValueError(f"Unknown model: {provider}/{model}")

    variants = model_config.get("variants", [])

    if variant not in variants:
        raise ValueError(
            f"Invalid variant '{variant}' for {provider}/{model}"
        )

    return {
        "provider": provider,
        "url": url,
        "model": model,
        "variant": variant,
    }


PROVIDERS = load_providers()


@router.get("")
async def list_providers(
    x_bot_secret: str | None = Header(default=None, alias="X-Bot-Secret"),
) -> dict[str, Any]:
    """Return the full validated provider configuration.
    The Telegram bot calls this once at startup to build keyboards."""
    if not BOT_SHARED_SECRET or x_bot_secret != BOT_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # ponytail: return the already-loaded dict; if the file changed on disk,
    # the user must restart the backend. Hot-reload is a zero-value feature here.
    return {"providers": PROVIDERS}
