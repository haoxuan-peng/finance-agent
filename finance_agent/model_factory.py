import os

from model_library.base import DelegateConfig, LLM, LLMConfig
from model_library.providers.openai import OpenAIModel
from model_library.registry_utils import get_registry_model
from pydantic import SecretStr


PROXY_URL_ENV_VARS = ("AGENT_BASE_URL", "AGENT_URL")
PROXY_KEY_ENV_VARS = ("AGENT_API_KEY", "AGENT_KEY")


def _get_first_env(names: tuple[str, ...]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip(), name
    return None, None


def _normalize_base_url(base_url: str) -> str:
    """Accept either an API base URL or a full chat-completions URL."""
    normalized = base_url.rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized + "/"


def get_model(model_name: str, config: LLMConfig) -> LLM:
    """Create a model client.

    Provider-qualified names (for example ``openai/gpt-5``) keep using the
    model-library registry. Bare names (for example ``glm-5.2``) use the
    OpenAI-compatible proxy configured through AGENT_BASE_URL/AGENT_API_KEY.
    """
    if "/" in model_name:
        return get_registry_model(model_name, config)

    base_url, base_url_env = _get_first_env(PROXY_URL_ENV_VARS)
    api_key, api_key_env = _get_first_env(PROXY_KEY_ENV_VARS)

    missing = []
    if not base_url:
        missing.append("AGENT_BASE_URL (or AGENT_URL)")
    if not api_key:
        missing.append("AGENT_API_KEY (or AGENT_KEY)")
    if missing:
        raise ValueError(
            f"Bare model name '{model_name}' uses the custom proxy, but "
            f"{', '.join(missing)} is not set."
        )
    assert base_url is not None
    assert api_key is not None

    proxy_config = config.model_copy(
        update={
            "supports_tools": True,
            "supports_output_schema": False,
        }
    )
    model = OpenAIModel(
        model_name=model_name,
        provider="agent_proxy",
        config=proxy_config,
        use_completions=True,
        delegate_config=DelegateConfig(
            base_url=_normalize_base_url(base_url),
            api_key=SecretStr(api_key),
        ),
    )
    model.instance_logger.info(
        "Using custom OpenAI-compatible proxy configured by %s and %s",
        base_url_env,
        api_key_env,
    )
    return model
