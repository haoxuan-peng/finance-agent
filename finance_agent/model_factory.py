import hashlib
import os
import re

from model_library.base import DelegateConfig, LLM, LLMConfig
from model_library.providers.openai import OpenAIModel
from model_library.registry_utils import get_registry_model
from pydantic import SecretStr


DEFAULT_PROXY_URL_ENV_VARS = ("AGENT_BASE_URL", "AGENT_URL")
DEFAULT_PROXY_KEY_ENV_VARS = ("AGENT_API_KEY", "AGENT_KEY")
QWEN_PROXY_URL_ENV_VARS = ("QWEN3_API_URL", "QWEN3_BASE_URL")
QWEN_PROXY_KEY_ENV_VARS = ("QWEN3_API_KEY",)


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


def _model_env_prefix(model_name: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", model_name.upper()).strip("_")
    return f"MODEL_{normalized}"


def _proxy_env_vars_for_model(model_name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return proxy env vars in descending precedence for a model."""
    prefix = _model_env_prefix(model_name)
    url_vars = (f"{prefix}_BASE_URL", f"{prefix}_API_URL")
    key_vars = (f"{prefix}_API_KEY",)

    if model_name.lower().startswith("qwen"):
        url_vars += QWEN_PROXY_URL_ENV_VARS
        key_vars += QWEN_PROXY_KEY_ENV_VARS

    return (
        url_vars + DEFAULT_PROXY_URL_ENV_VARS,
        key_vars + DEFAULT_PROXY_KEY_ENV_VARS,
    )


def _api_model_name_for_model(model_name: str) -> tuple[str, str | None]:
    """Return the server-advertised model ID, falling back to the CLI name."""
    env_var = f"{_model_env_prefix(model_name)}_MODEL_ID"
    value = os.getenv(env_var)
    if value and value.strip():
        return value.strip(), env_var
    return model_name, None


def get_model(model_name: str, config: LLMConfig) -> LLM:
    """Create a model client.

    Provider-qualified names (for example ``openai/gpt-5``) keep using the
    model-library registry. Bare names (for example ``glm-5.2``) use the
    OpenAI-compatible proxy. Configuration can be model-specific, family-specific
    (QWEN3_*), or use the general AGENT_* fallback.
    """
    if "/" in model_name:
        return get_registry_model(model_name, config)

    url_env_vars, key_env_vars = _proxy_env_vars_for_model(model_name)
    base_url, base_url_env = _get_first_env(url_env_vars)
    api_key, api_key_env = _get_first_env(key_env_vars)
    api_model_name, api_model_name_env = _api_model_name_for_model(model_name)

    missing = []
    if not base_url:
        missing.append(f"a base URL ({', '.join(url_env_vars)})")
    if not api_key:
        missing.append(f"an API key ({', '.join(key_env_vars)})")
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
    normalized_base_url = _normalize_base_url(base_url)
    endpoint_id = hashlib.sha256(normalized_base_url.encode()).hexdigest()[:12]
    model = OpenAIModel(
        model_name=api_model_name,
        # model-library caches clients by provider and key. Including an opaque
        # endpoint id prevents two servers that share a key from reusing the
        # wrong HTTP client inside the same process.
        provider=f"agent_proxy_{endpoint_id}",
        config=proxy_config,
        use_completions=True,
        delegate_config=DelegateConfig(
            base_url=normalized_base_url,
            api_key=SecretStr(api_key),
        ),
    )
    model.instance_logger.info(
        "Using custom OpenAI-compatible proxy configured by %s and %s; API model from %s",
        base_url_env,
        api_key_env,
        api_model_name_env or "--model",
    )
    return model
