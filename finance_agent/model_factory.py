import hashlib
import os
from copy import deepcopy
from typing import Any, Sequence

from model_library.base import DelegateConfig, InputItem, LLM, LLMConfig, ToolDefinition
from model_library.providers.openai import OpenAIModel
from pydantic import BaseModel, SecretStr


DEFAULT_PROXY_URL_ENV_VARS = ("AGENT_BASE_URL", "AGENT_URL")
DEFAULT_PROXY_KEY_ENV_VARS = ("AGENT_API_KEY", "AGENT_KEY")


class _AgentProxyModel(OpenAIModel):
    """Apply service-specific JSON fields to every request, including tool queries."""

    def __init__(
        self,
        model_name: str,
        *,
        provider: str,
        config: LLMConfig,
        delegate_config: DelegateConfig,
        extra_body: dict[str, Any] | None = None,
    ):
        self.extra_body = deepcopy(extra_body or {})
        super().__init__(
            model_name=model_name,
            provider=provider,
            config=config,
            use_completions=True,
            delegate_config=delegate_config,
        )

    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        body = await super().build_body(
            input, tools=tools, output_schema=output_schema, **kwargs
        )
        if self.extra_body:
            # Keep these under the SDK's extra_body argument so nonstandard
            # fields reach the HTTP JSON without becoming unknown SDK kwargs.
            # Per-call overrides take precedence over the model defaults.
            body["extra_body"] = {
                **deepcopy(self.extra_body),
                **(body.get("extra_body") or {}),
            }
        return body


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


def get_model(
    model_name: str,
    config: LLMConfig,
    *,
    extra_body: dict[str, Any] | None = None,
) -> LLM:
    """Send any model ID verbatim to the shared OpenAI-compatible Agent API."""
    base_url, base_url_env = _get_first_env(DEFAULT_PROXY_URL_ENV_VARS)
    api_key, api_key_env = _get_first_env(DEFAULT_PROXY_KEY_ENV_VARS)

    missing = []
    if not base_url:
        missing.append(f"a base URL ({', '.join(DEFAULT_PROXY_URL_ENV_VARS)})")
    if not api_key:
        missing.append(f"an API key ({', '.join(DEFAULT_PROXY_KEY_ENV_VARS)})")
    if missing:
        raise ValueError(
            f"Model '{model_name}' uses the shared Agent API, but "
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
    model = _AgentProxyModel(
        model_name=model_name,
        # model-library caches clients by provider and key. Including an opaque
        # endpoint id prevents two servers that share a key from reusing the
        # wrong HTTP client inside the same process.
        provider=f"agent_proxy_{endpoint_id}",
        config=proxy_config,
        delegate_config=DelegateConfig(
            base_url=normalized_base_url,
            api_key=SecretStr(api_key),
        ),
        extra_body=extra_body,
    )
    model.instance_logger.info(
        "Using shared OpenAI-compatible Agent API configured by %s and %s; model from --model",
        base_url_env,
        api_key_env,
    )
    return model
