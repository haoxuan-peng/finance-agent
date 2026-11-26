import io
import json
import unittest
from unittest.mock import AsyncMock, patch

from finance_agent import run_agent


class RunAgentCliTests(unittest.IsolatedAsyncioTestCase):
    async def run_cli(self, *arguments):
        with (
            patch(
                "sys.argv",
                [
                    "finance-agent",
                    "--model",
                    "my-model",
                    "--questions",
                    "Question",
                    *arguments,
                ],
            ),
            patch("finance_agent.run_agent.load_dotenv"),
            patch(
                "finance_agent.run_agent.run_tests_parallel", new_callable=AsyncMock
            ) as run,
        ):
            await run_agent.main()
        run.assert_awaited_once()
        return run.call_args.kwargs["parameters"]

    async def test_reasoning_effort_and_service_specific_extra_body_are_accepted(self):
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_budget": 1024,
        }
        parameters = await self.run_cli(
            "--reasoning-effort",
            "custom-max",
            "--extra-body",
            json.dumps(extra_body),
        )
        self.assertEqual(parameters.llm_config.reasoning_effort, "custom-max")
        self.assertEqual(parameters.extra_body, extra_body)

    async def test_omitted_options_preserve_default_generation_config(self):
        parameters = await self.run_cli()
        self.assertIsNone(parameters.llm_config.reasoning_effort)
        self.assertEqual(parameters.extra_body, {})
        self.assertEqual(parameters.llm_config.max_tokens, 32000)
        self.assertEqual(parameters.llm_config.temperature, 0.0)
        self.assertFalse(parameters.llm_config.reasoning)

    async def test_extra_body_preserves_false_null_and_nested_values(self):
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking_budget": 0,
            "custom_value": None,
        }
        parameters = await self.run_cli("--extra-body", json.dumps(extra_body))
        self.assertEqual(parameters.extra_body, extra_body)

    async def test_malformed_or_nonobject_extra_body_fails_before_running(self):
        for value in ("{", "[]", "null", '"string"', "true", "1"):
            with self.subTest(value=value):
                stderr = io.StringIO()
                with (
                    patch(
                        "sys.argv",
                        [
                            "finance-agent",
                            "--questions",
                            "Question",
                            "--extra-body",
                            value,
                        ],
                    ),
                    patch("finance_agent.run_agent.load_dotenv"),
                    patch(
                        "finance_agent.run_agent.run_tests_parallel",
                        new_callable=AsyncMock,
                    ) as run,
                    patch("sys.stderr", stderr),
                ):
                    with self.assertRaises(SystemExit) as error:
                        await run_agent.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn("--extra-body", stderr.getvalue())
                run.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
