import json
import os
import unittest
from unittest.mock import patch

import httpx
from model_library.base import LLMConfig, TextInput, ToolBody, ToolDefinition
from model_library.base.base import client_registry
from openai import AsyncOpenAI

from finance_agent.model_factory import get_model


class ModelRequestTests(unittest.IsolatedAsyncioTestCase):
    async def send_request(self, model_name, config=None, extra_body=None, env=None):
        requests = []

        def respond(request):
            requests.append(request)
            chunk = {
                "id": "test-completion",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Done"},
                        "finish_reason": "stop",
                    }
                ],
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )

        environment = {
            "AGENT_BASE_URL": "https://agent.example/v1",
            "AGENT_API_KEY": "shared-test-key",
        }
        if env is not None:
            environment = env

        def create_client(*, base_url, api_key):
            return AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            )

        with (
            patch.dict(os.environ, environment, clear=True),
            patch.dict(client_registry, {}, clear=True),
            patch(
                "model_library.providers.openai.create_openai_client_with_defaults",
                side_effect=create_client,
            ),
        ):
            model = get_model(
                model_name,
                config or LLMConfig(max_tokens=32000, temperature=0.0),
                extra_body=extra_body,
            )
            try:
                result = await model._query_completions(
                    [TextInput(text="Test question")],
                    tools=[
                        ToolDefinition(
                            name="submit_final_result",
                            body=ToolBody(
                                name="submit_final_result",
                                description="Submit an answer",
                                properties={"answer": {"type": "string"}},
                                required=["answer"],
                            ),
                        )
                    ],
                )
                self.assertEqual(result.output_text, "Done")
            finally:
                await model.get_client().close()

        self.assertEqual(len(requests), 1)
        return requests[0], json.loads(requests[0].content)

    async def test_all_model_names_use_shared_endpoint_without_remapping(self):
        env = {
            "AGENT_BASE_URL": "https://agent.example/v1",
            "AGENT_API_KEY": "shared-test-key",
            "QWEN3_API_URL": "https://old-qwen.example/v1",
            "QWEN3_API_KEY": "old-qwen-key",
            "MODEL_QWEN3_5_9B_BASE_URL": "https://old-model.example/v1",
            "MODEL_QWEN3_5_9B_API_KEY": "old-model-key",
            "MODEL_QWEN3_5_9B_MODEL_ID": "old-model-id",
        }
        for name in ("qwen3.5-9b", "openai/my-model", "/srv/models/my-model"):
            with self.subTest(model=name):
                request, body = await self.send_request(name, env=env)
                self.assertEqual(
                    str(request.url), "https://agent.example/v1/chat/completions"
                )
                self.assertEqual(
                    request.headers["authorization"], "Bearer shared-test-key"
                )
                self.assertEqual(body["model"], name)
                self.assertTrue(body["stream"])
                self.assertEqual(
                    body["tools"][0]["function"]["name"], "submit_final_result"
                )

    async def test_legacy_shared_aliases_and_full_completion_url_are_supported(self):
        request, body = await self.send_request(
            "my-model",
            env={
                "AGENT_URL": "https://agent.example/v1/chat/completions/",
                "AGENT_KEY": "alias-test-key",
            },
        )
        self.assertEqual(str(request.url), "https://agent.example/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer alias-test-key")
        self.assertEqual(body["model"], "my-model")

    async def test_reasoning_effort_is_omitted_by_default(self):
        _, body = await self.send_request("my-model")
        self.assertNotIn("reasoning_effort", body)
        self.assertNotIn("extra_body", body)
        self.assertNotIn("chat_template_kwargs", body)
        self.assertEqual(body["max_tokens"], 32000)
        self.assertNotIn("max_completion_tokens", body)

    async def test_service_defined_reasoning_effort_is_sent_verbatim(self):
        _, body = await self.send_request(
            "my-model",
            config=LLMConfig(max_tokens=1234, reasoning_effort="custom-high"),
        )
        self.assertEqual(body["reasoning_effort"], "custom-high")
        self.assertEqual(body["max_tokens"], 1234)
        self.assertNotIn("max_completion_tokens", body)

    async def test_extra_body_sends_thinking_switch_and_budget_at_top_level(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                _, body = await self.send_request(
                    "qwen3.5-9b",
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": enabled},
                        "thinking_budget": 2048,
                        "thinking": {"type": "enabled" if enabled else "disabled"},
                    },
                )
                self.assertEqual(
                    body["chat_template_kwargs"], {"enable_thinking": enabled}
                )
                self.assertEqual(body["thinking_budget"], 2048)
                self.assertEqual(
                    body["thinking"]["type"], "enabled" if enabled else "disabled"
                )
                self.assertNotIn("extra_body", body)

    async def test_extra_body_overrides_fields_generated_from_config(self):
        _, body = await self.send_request(
            "my-model",
            config=LLMConfig(max_tokens=1000, temperature=0.0, reasoning_effort="high"),
            extra_body={"reasoning_effort": "low", "temperature": 0.7},
        )
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["max_tokens"], 1000)

    def test_old_model_specific_config_does_not_replace_shared_api(self):
        with patch.dict(
            os.environ,
            {
                "QWEN3_API_URL": "https://old-qwen.example/v1",
                "QWEN3_API_KEY": "old-qwen-key",
                "MODEL_QWEN3_5_9B_BASE_URL": "https://old-model.example/v1",
                "MODEL_QWEN3_5_9B_API_KEY": "old-model-key",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "AGENT_BASE_URL"):
                get_model("qwen3.5-9b", LLMConfig())


if __name__ == "__main__":
    unittest.main()
